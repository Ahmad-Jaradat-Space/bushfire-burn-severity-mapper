"""Run inference with the trained RF / XGB / U-Net / SegFormer models on the
two training AOIs (Currowan and Gospers Mountain) and write prediction
GeoTIFFs alongside the existing Kangaroo Island outputs.

The training events already have pre/post composites + aligned GEEBAM labels
under data/interim/<event>/. We have not yet rendered the trained models'
predictions on them — only the validation event (Kangaroo Island) was
materialised during training. This script closes that gap so the notebook can
show predictions across three AOIs instead of one.

Outputs (relative to repo root):
  outputs/predictions/rf_multiclass/<event>.tif
  outputs/predictions/xgb_multiclass/<event>.tif
  outputs/predictions/unet_multiclass/<event>.tif
  outputs/predictions/segformer_multiclass/<event>.tif
  outputs/predictions/baseline_dnbr/<event>/multiclass.tif  (skipped if exists)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
import torch

from src.features.stack_features import build_stack
from src.models.baselines import dnbr as dnbr_index, dnbr_multiclass_usgs
from src.models.tabular_dataset import predict_full_event
from src.models.train_segmenter import _build_model, _sliding_window_predict
from src.utils.config import load_config
from src.utils.seed import pick_device
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)

EVENTS = ["currowan_2019_2020", "gospers_mountain_2019_2020"]


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


def _load_event_stack(event_id: str):
    interim = REPO_ROOT / "data" / "interim" / event_id
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre = ds.read().astype(np.float32)
        transform, crs = ds.transform, ds.crs
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post = ds.read().astype(np.float32)
    with rasterio.open(interim / "mask_pre_10m.tif") as ds:
        mpre = ds.read(1).astype(bool)
    with rasterio.open(interim / "mask_post_10m.tif") as ds:
        mpost = ds.read(1).astype(bool)
    pre = np.nan_to_num(pre, nan=0.0)
    post = np.nan_to_num(post, nan=0.0)
    mask = mpre & mpost
    return pre, post, mask, transform, crs


def predict_tree_model(model_path: Path, out_dir: Path, model_name: str) -> None:
    clf = joblib.load(model_path)
    log.info("[%s] loaded %s", model_name, model_path.relative_to(REPO_ROOT))
    for ev in EVENTS:
        out = out_dir / f"{ev}.tif"
        if out.exists():
            log.info("[%s/%s] exists, skipping", model_name, ev)
            continue
        pred, transform, crs = predict_full_event(clf, ev)
        _write_uint8(out, pred, transform, crs)
        write_manifest(
            out, event_id=ev, pipeline_step=f"{model_name}.predict",
            inputs={"model_path": str(model_path.relative_to(REPO_ROOT))},
            crs=str(crs), resampling=None,
        )
        hist = dict(zip(*np.unique(pred, return_counts=True)))
        log.info("[%s/%s] wrote %s   class hist=%s",
                 model_name, ev, out.relative_to(REPO_ROOT), hist)


def predict_segmentation_model(model_kind: str, config_path: str,
                               weights_path: Path, stats_path: Path,
                               out_dir: Path) -> None:
    cfg = load_config(config_path)
    device_info = pick_device(cfg.device.prefer)
    device = torch.device(device_info.name)
    model = _build_model(cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    log.info("[%s] loaded weights from %s on %s",
             model_kind, weights_path.relative_to(REPO_ROOT), device_info.name)

    stats = json.loads(stats_path.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    tile = int(cfg.data.tile_size)
    stride = tile // 2

    for ev in EVENTS:
        out = out_dir / f"{ev}.tif"
        if out.exists():
            log.info("[%s/%s] exists, skipping", model_kind, ev)
            continue
        pre, post, mask, transform, crs = _load_event_stack(ev)
        image = build_stack(pre, post).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)

        pred = _sliding_window_predict(model, image, mean, std,
                                       tile=tile, stride=stride, device=device,
                                       num_classes=4)
        pred[~mask] = 255
        _write_uint8(out, pred, transform, crs)
        write_manifest(
            out, event_id=ev, pipeline_step=f"{model_kind}.predict",
            inputs={"weights_path": str(weights_path.relative_to(REPO_ROOT))},
            crs=str(crs), resampling=None,
        )
        hist = dict(zip(*np.unique(pred, return_counts=True)))
        log.info("[%s/%s] wrote %s   class hist=%s",
                 model_kind, ev, out.relative_to(REPO_ROOT), hist)


def predict_dnbr_baseline(out_root: Path) -> None:
    for ev in EVENTS:
        out_dir = out_root / ev
        out = out_dir / "multiclass.tif"
        if out.exists():
            log.info("[dnbr/%s] exists, skipping", ev)
            continue
        pre, post, mask, transform, crs = _load_event_stack(ev)
        d = dnbr_index(pre, post)
        pred = dnbr_multiclass_usgs(d).astype(np.uint8)
        pred[~mask] = 255
        _write_uint8(out, pred, transform, crs)
        write_manifest(
            out, event_id=ev, pipeline_step="baseline_dnbr.predict",
            inputs={"thresholds": "Key & Benson 2006 USGS"},
            crs=str(crs), resampling=None,
        )
        log.info("[dnbr/%s] wrote %s", ev, out.relative_to(REPO_ROOT))


def main() -> None:
    pred_root = REPO_ROOT / "outputs" / "predictions"

    log.info("=== ΔNBR baseline ===")
    predict_dnbr_baseline(pred_root / "baseline_dnbr")

    log.info("=== RandomForest ===")
    predict_tree_model(
        REPO_ROOT / "outputs" / "models" / "rf" / "model.joblib",
        pred_root / "rf_multiclass", "rf",
    )

    log.info("=== XGBoost ===")
    predict_tree_model(
        REPO_ROOT / "outputs" / "models" / "xgb" / "model.joblib",
        pred_root / "xgb_multiclass", "xgb",
    )

    log.info("=== U-Net ===")
    predict_segmentation_model(
        "unet",
        config_path="configs/experiments/unet_multiclass.yaml",
        weights_path=REPO_ROOT / "outputs" / "models" / "unet" / "best.pt",
        stats_path=REPO_ROOT / "outputs" / "models" / "unet" / "normalization.json",
        out_dir=pred_root / "unet_multiclass",
    )

    log.info("=== SegFormer-B0 ===")
    predict_segmentation_model(
        "segformer",
        config_path="configs/experiments/segformer_multiclass.yaml",
        weights_path=REPO_ROOT / "outputs" / "models" / "segformer" / "best.pt",
        stats_path=REPO_ROOT / "outputs" / "models" / "segformer" / "normalization.json",
        out_dir=pred_root / "segformer_multiclass",
    )

    log.info("=== done ===")


if __name__ == "__main__":
    main()
