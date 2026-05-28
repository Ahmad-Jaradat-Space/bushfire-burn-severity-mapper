"""Score the predictions we just wrote on Currowan + Gospers vs GEEBAM.

Loads each prediction GeoTIFF and the aligned label, computes the
src.evaluation.metrics.summary (macro-IoU, macro-F1, per-class IoU),
and writes the result to outputs/metrics/per_aoi_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from src.evaluation.metrics import summary
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

EVENTS = ["currowan_2019_2020", "gospers_mountain_2019_2020", "kangaroo_island_2019_2020"]
MODELS = {
    "baseline_dnbr":  "outputs/predictions/baseline_dnbr/{ev}/multiclass.tif",
    "rf":             "outputs/predictions/rf_multiclass/{ev}.tif",
    "xgb":            "outputs/predictions/xgb_multiclass/{ev}.tif",
    "unet":           "outputs/predictions/unet_multiclass/{ev}.tif",
    "segformer":      "outputs/predictions/segformer_multiclass/{ev}.tif",
}


def _read(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1)


def main() -> None:
    out: dict = {}
    for ev in EVENTS:
        label_path = REPO_ROOT / "data" / "interim" / ev / "label_10m.tif"
        if not label_path.exists():
            log.warning("No label for %s, skipping", ev)
            continue
        label = _read(label_path)
        out[ev] = {}
        for model_name, tmpl in MODELS.items():
            pred_path = REPO_ROOT / tmpl.format(ev=ev)
            if not pred_path.exists():
                log.info("[%s/%s] no prediction, skipping", model_name, ev)
                continue
            pred = _read(pred_path)
            m = summary(pred, label, num_classes=4)
            out[ev][model_name] = {
                "macro_iou": float(m["macro_iou"]),
                "macro_f1":  float(m["macro_f1"]),
                "per_class_iou": [float(x) for x in m["per_class_iou"]],
            }
            log.info("[%s/%s] macro-IoU=%.3f macro-F1=%.3f",
                     ev, model_name, m["macro_iou"], m["macro_f1"])

    out_path = REPO_ROOT / "outputs" / "metrics" / "per_aoi_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
