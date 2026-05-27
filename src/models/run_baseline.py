"""Run the dNBR baseline (binary + multiclass) on an event's processed tiles.

Outputs:
  outputs/predictions/baseline_dnbr/<event>/multiclass.tif
  outputs/predictions/baseline_dnbr/<event>/binary_t<thr>.tif (per threshold)
  outputs/metrics/baseline_dnbr/<event>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from src.evaluation.metrics import binary_summary, summary
from src.models.baselines import dnbr, dnbr_binary, dnbr_multiclass_usgs
from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)


def _read_band_stack(path: Path) -> tuple[np.ndarray, dict, object, object]:
    with rasterio.open(path) as ds:
        data = ds.read().astype(np.float32)
        return data, ds.meta.copy(), ds.transform, ds.crs


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


def run_baseline_event(event_id: str, config_path: str = "configs/experiments/baseline_dnbr.yaml") -> dict:
    cfg = load_config(config_path)
    interim = REPO_ROOT / "data" / "interim" / event_id
    out_pred = REPO_ROOT / "outputs" / "predictions" / "baseline_dnbr" / event_id
    out_metrics = REPO_ROOT / "outputs" / "metrics" / "baseline_dnbr"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    pre, _, transform, crs = _read_band_stack(interim / "pre_stack_10m.tif")
    post, _, _, _ = _read_band_stack(interim / "post_stack_10m.tif")
    with rasterio.open(interim / "label_10m.tif") as ds:
        label = ds.read(1)
    with rasterio.open(interim / "mask_pre_10m.tif") as ds:
        mpre = ds.read(1)
    with rasterio.open(interim / "mask_post_10m.tif") as ds:
        mpost = ds.read(1)
    clear = (mpre & mpost).astype(bool)

    d = dnbr(pre, post)

    metrics: dict = {"event_id": event_id, "model": "baseline_dnbr"}

    # Multiclass
    mc = dnbr_multiclass_usgs(d, mask=clear)
    _write_uint8(out_pred / "multiclass.tif", mc, transform, crs)
    metrics["multiclass"] = summary(mc, label, num_classes=4)
    write_manifest(
        out_pred / "multiclass.tif",
        event_id=event_id,
        pipeline_step="baseline_dnbr.multiclass",
        inputs={"thresholds": "USGS-style: 0.10/0.27/0.66"},
        crs=str(crs), resampling=None, class_remap=None,
    )

    # Binary sweep
    binary: list[dict] = []
    for t in cfg.baseline.binary_thresholds:
        b = dnbr_binary(d, threshold=float(t), mask=clear)
        _write_uint8(out_pred / f"binary_t{float(t):.2f}.tif", b, transform, crs)
        binary.append({"threshold": float(t), **binary_summary(b, label)})
    metrics["binary_threshold_sweep"] = binary

    out_metrics_path = out_metrics / f"{event_id}.json"
    out_metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Wrote metrics to %s", out_metrics_path)
    log.info("  Multiclass macro-IoU: %.3f  macro-F1: %.3f",
             metrics["multiclass"]["macro_iou"], metrics["multiclass"]["macro_f1"])
    for row in binary:
        log.info("  Binary t=%.2f  burnt IoU=%.3f  F1=%.3f",
                 row["threshold"], row["iou_burnt"], row["f1_burnt"])
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Run dNBR baseline on an event.")
    p.add_argument("--event", required=True)
    p.add_argument("--config", default="configs/experiments/baseline_dnbr.yaml")
    args = p.parse_args()
    run_baseline_event(args.event, args.config)


if __name__ == "__main__":
    main()
