"""Train a RandomForest on per-pixel features and evaluate it event-wise.

Vertical-slice mode (default): trains and tests on the same event using a
random tile split. M10 switches to event-wise splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier

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


def train_rf(config_path: str = "configs/experiments/rf_multiclass.yaml") -> dict:
    cfg = load_config(config_path)
    if cfg.experiment.split_mode == "vertical_slice":
        train_events = [cfg.experiment.event]
        val_events = [cfg.experiment.event]
        test_events = [cfg.experiment.event]
        splits_train, splits_val, splits_test = ("train",), ("val",), ("test",)
    else:
        train_events = list(cfg.events.train)
        val_events = list(cfg.events.val)
        test_events = list(cfg.events.test)
        splits_train = splits_val = splits_test = ("train", "val", "test")

    log.info("Train events: %s", train_events)
    Xtr, ytr, feat_names = stack_events(train_events, splits_train,
                                        pixels_per_class=cfg.sampling.pixels_per_class)
    log.info("Training matrix: X=%s y=%s class hist=%s",
             Xtr.shape, ytr.shape, dict(zip(*np.unique(ytr, return_counts=True))))

    clf = RandomForestClassifier(
        n_estimators=cfg.rf.n_estimators,
        max_depth=cfg.rf.max_depth,
        min_samples_leaf=cfg.rf.min_samples_leaf,
        class_weight=cfg.rf.class_weight,
        n_jobs=cfg.rf.n_jobs,
        random_state=cfg.rf.random_state,
    )
    clf.fit(Xtr, ytr)
    log.info("Fitted RandomForest (%d trees, max_depth=%s)",
             cfg.rf.n_estimators, cfg.rf.max_depth)

    model_dir = REPO_ROOT / cfg.outputs.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(clf, model_path)

    importance = pd.DataFrame({
        "feature": feat_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    log.info("Top 5 features: %s",
             list(importance.head(5)[["feature", "importance"]].itertuples(index=False, name=None)))

    metrics_all: dict = {"train_events": train_events,
                         "val_events": val_events, "test_events": test_events,
                         "model": "random_forest"}
    pred_dir = REPO_ROOT / cfg.outputs.prediction_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = REPO_ROOT / cfg.outputs.metrics_dir
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
                out_pred,
                event_id=ev,
                pipeline_step="rf.predict",
                inputs={"model_path": str(model_path.relative_to(REPO_ROOT))},
                crs=str(crs), resampling=None,
            )
            log.info("[%s/%s] macro-IoU=%.3f macro-F1=%.3f",
                     label, ev,
                     metrics_all[label][ev]["macro_iou"],
                     metrics_all[label][ev]["macro_f1"])

    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_all, indent=2))
    log.info("Wrote metrics to %s", metrics_path)
    return metrics_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/rf_multiclass.yaml")
    args = p.parse_args()
    train_rf(args.config)


if __name__ == "__main__":
    main()
