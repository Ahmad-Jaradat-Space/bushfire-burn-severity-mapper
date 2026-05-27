"""Train XGBoost on per-pixel features. Mirrors train_rf.py with the XGB API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from xgboost import XGBClassifier

from src.evaluation.metrics import summary
from src.models.tabular_dataset import predict_full_event, stack_events
from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)


def _write_uint8(path: Path, arr: np.ndarray, transform, crs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
        "nodata": 255, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr[np.newaxis, ...])


def train_xgb(config_path: str = "configs/experiments/xgb_multiclass.yaml") -> dict:
    cfg = load_config(config_path)
    if cfg.experiment.split_mode == "vertical_slice":
        train_events = [cfg.experiment.event]
        val_events = [cfg.experiment.event]
        test_events = [cfg.experiment.event]
        splits_train = ("train",); splits_val = ("val",); splits_test = ("test",)
    else:
        train_events = list(cfg.events.train)
        val_events = list(cfg.events.val)
        test_events = list(cfg.events.test)
        splits_train = splits_val = splits_test = ("train", "val", "test")

    Xtr, ytr, feat_names = stack_events(train_events, splits_train,
                                        pixels_per_class=cfg.sampling.pixels_per_class)
    log.info("Training matrix: X=%s y=%s", Xtr.shape, ytr.shape)

    clf = XGBClassifier(
        n_estimators=cfg.xgb.n_estimators,
        max_depth=cfg.xgb.max_depth,
        learning_rate=cfg.xgb.learning_rate,
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        reg_lambda=cfg.xgb.reg_lambda,
        tree_method=cfg.xgb.tree_method,
        random_state=cfg.xgb.random_state,
        objective="multi:softprob",
        num_class=4,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    log.info("Fitted XGBoost.")

    model_dir = REPO_ROOT / cfg.outputs.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(clf, model_path)

    importance = pd.DataFrame({
        "feature": feat_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    metrics_all: dict = {"train_events": train_events, "val_events": val_events,
                         "test_events": test_events, "model": "xgboost"}
    pred_dir = REPO_ROOT / cfg.outputs.prediction_dir
    metrics_dir = REPO_ROOT / cfg.outputs.metrics_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for label, events in (("val", val_events), ("test", test_events)):
        metrics_all[label] = {}
        for ev in events:
            pred, transform, crs = predict_full_event(clf, ev)
            with rasterio.open(REPO_ROOT / "data" / "interim" / ev / "label_10m.tif") as ds:
                lab = ds.read(1)
            metrics_all[label][ev] = summary(pred, lab, num_classes=4)
            out_pred = pred_dir / f"{ev}.tif"
            _write_uint8(out_pred, pred, transform, crs)
            write_manifest(
                out_pred, event_id=ev, pipeline_step="xgb.predict",
                inputs={"model_path": str(model_path.relative_to(REPO_ROOT))},
                crs=str(crs), resampling=None,
            )
            log.info("[%s/%s] macro-IoU=%.3f macro-F1=%.3f",
                     label, ev,
                     metrics_all[label][ev]["macro_iou"],
                     metrics_all[label][ev]["macro_f1"])

    (metrics_dir / "metrics.json").write_text(json.dumps(metrics_all, indent=2))
    return metrics_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/xgb_multiclass.yaml")
    args = p.parse_args()
    train_xgb(args.config)


if __name__ == "__main__":
    main()
