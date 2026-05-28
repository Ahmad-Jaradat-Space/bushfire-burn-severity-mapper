"""Compute real calibration data for the trained classical models on Kangaroo.

Loads the joblib RF/XGB checkpoints, runs predict_proba on the full Kangaroo
feature stack, collapses to binary burnt-class probability, and computes the
reliability-diagram bins + ECE + Brier against the AUS GEEBAM proxy labels.

Outputs:
  outputs/calibration/kangaroo_island_2019_2020/rf.json
  outputs/calibration/kangaroo_island_2019_2020/xgb.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import rasterio

from src.evaluation.calibration import reliability_data
from src.evaluation.metrics import IGNORE_ID
from src.features.stack_features import build_stack
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _load_features_and_label(event_id: str):
    interim = REPO_ROOT / "data" / "interim" / event_id
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre = ds.read().astype(np.float32)
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post = ds.read().astype(np.float32)
    with rasterio.open(interim / "mask_pre_10m.tif") as ds:
        mpre = ds.read(1).astype(bool)
    with rasterio.open(interim / "mask_post_10m.tif") as ds:
        mpost = ds.read(1).astype(bool)
    with rasterio.open(interim / "label_10m.tif") as ds:
        label = ds.read(1)
    mask = mpre & mpost
    pre = np.nan_to_num(pre, nan=0.0)
    post = np.nan_to_num(post, nan=0.0)
    image = build_stack(pre, post)
    image = np.nan_to_num(image, nan=0.0)
    return image, mask, label


def calibration_for(model_path: Path, event_id: str = "kangaroo_island_2019_2020") -> dict:
    clf = joblib.load(model_path)
    log.info("Loaded %s (%s)", model_path.name, type(clf).__name__)

    image, mask, label = _load_features_and_label(event_id)
    C, H, W = image.shape
    flat = image.reshape(C, -1).T   # [H*W, 18]
    mask_flat = mask.ravel()
    label_flat = label.ravel()

    # Only consider pixels that are clear AND have a valid label
    valid = mask_flat & (label_flat != IGNORE_ID)
    log.info("Valid pixels: %d / %d", valid.sum(), flat.shape[0])

    feat_valid = flat[valid]
    label_valid = label_flat[valid]

    proba = clf.predict_proba(feat_valid)   # [N, 4]
    # P(burnt) = 1 - P(class 0 = unburnt) = sum of P(1) + P(2) + P(3)
    p_burnt = 1.0 - proba[:, 0]

    # Reshape back into a [H, W] grid so reliability_data() can iterate over
    # 2-D arrays; pad invalid pixels with 0 prob + label 255 ignore.
    full_p = np.zeros(H * W, dtype=np.float32)
    full_t = np.full(H * W, IGNORE_ID, dtype=np.uint8)
    full_p[valid] = p_burnt
    full_t[valid] = label_valid
    full_p_grid = full_p.reshape(H, W)
    full_t_grid = full_t.reshape(H, W)

    data = reliability_data(full_p_grid, full_t_grid, n_bins=12)
    data["n_valid"] = int(valid.sum())
    data["model_path"] = str(model_path.relative_to(REPO_ROOT))
    data["event_id"] = event_id
    return data


def main():
    out_dir = REPO_ROOT / "outputs" / "calibration" / "kangaroo_island_2019_2020"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in [
        ("rf",  REPO_ROOT / "outputs" / "models" / "rf"  / "model.joblib"),
        ("xgb", REPO_ROOT / "outputs" / "models" / "xgb" / "model.joblib"),
    ]:
        if not model_path.exists():
            log.warning("Missing %s, skipping.", model_path)
            continue
        data = calibration_for(model_path)
        out = out_dir / f"{model_name}.json"
        out.write_text(json.dumps(data, indent=2))
        log.info("Wrote %s — ECE=%.3f Brier=%.3f n=%d",
                 out.relative_to(REPO_ROOT), data["ece"], data["brier"], data["n_valid"])


if __name__ == "__main__":
    main()
