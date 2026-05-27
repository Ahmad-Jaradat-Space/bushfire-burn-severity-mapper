"""Per-stratum evaluation (land-cover, slope) and reliability plots.

Inputs:
  outputs/predictions/<model>/<event>.tif        uint8 prediction
  data/interim/<event>/label_10m.tif              uint8 ground-truth proxy
  Optional:
    data/interim/<event>/landcover_10m.tif        uint8 land-cover class
    data/interim/<event>/slope_10m.tif            float32 slope (deg)

Outputs:
  outputs/metrics/<model>/<event>_stratified.json
  docs/figures/stratified_landcover_<model>_<event>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.evaluation.metrics import IGNORE_ID, summary
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# DEA Land Cover Level-3 collapsed to a coarse 6-group palette
LANDCOVER_GROUPS = {
    "woody":      [111, 112, 113, 114],
    "shrubland":  [115, 116],
    "herbaceous": [117, 118, 124],
    "cropland":   [119],
    "built_bare": [220, 215, 216],
    "water":      [221, 222, 223],
}
SLOPE_BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 90)]


def _read_band(path: Path, band: int = 1):
    with rasterio.open(path) as ds:
        return ds.read(band)


def stratify_by_landcover(pred: np.ndarray, true: np.ndarray, lc: np.ndarray,
                          num_classes: int = 4) -> dict:
    out: dict = {}
    for group, codes in LANDCOVER_GROUPS.items():
        m = np.isin(lc, codes) & (true != IGNORE_ID) & (pred != IGNORE_ID)
        if not m.any():
            out[group] = None
            continue
        out[group] = summary(pred[m], true[m], num_classes=num_classes)
    return out


def stratify_by_slope(pred: np.ndarray, true: np.ndarray, slope: np.ndarray,
                      num_classes: int = 4) -> dict:
    out: dict = {}
    for lo, hi in SLOPE_BINS:
        bin_name = f"{lo}-{hi}deg"
        m = (slope >= lo) & (slope < hi) & (true != IGNORE_ID) & (pred != IGNORE_ID)
        if not m.any():
            out[bin_name] = None
            continue
        out[bin_name] = summary(pred[m], true[m], num_classes=num_classes)
    return out


def evaluate_event(model: str, event_id: str,
                   pred_path_template: str = "outputs/predictions/{model}/{event}.tif",
                   ) -> dict:
    pred_path = REPO_ROOT / pred_path_template.format(model=model, event=event_id)
    if not pred_path.exists():
        # baseline_dnbr predictions live in a subdir
        alt = REPO_ROOT / f"outputs/predictions/{model}/{event_id}/multiclass.tif"
        if alt.exists():
            pred_path = alt
        else:
            raise FileNotFoundError(f"No prediction at {pred_path} or {alt}")
    pred = _read_band(pred_path)
    interim = REPO_ROOT / "data" / "interim" / event_id
    label = _read_band(interim / "label_10m.tif")

    report: dict = {
        "event_id": event_id,
        "model": model,
        "overall": summary(pred, label, num_classes=4),
    }

    lc_path = interim / "landcover_10m.tif"
    if lc_path.exists():
        report["by_landcover"] = stratify_by_landcover(pred, label, _read_band(lc_path))
    slope_path = interim / "slope_10m.tif"
    if slope_path.exists():
        report["by_slope"] = stratify_by_slope(pred, label, _read_band(slope_path))

    out_dir = REPO_ROOT / "outputs" / "metrics" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_id}_stratified.json"
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", out_path)

    if "by_landcover" in report:
        _plot_landcover_heatmap(report, model, event_id,
                                REPO_ROOT / "docs" / "figures" /
                                f"stratified_landcover_{model}_{event_id}.png")
    return report


def _plot_landcover_heatmap(report: dict, model: str, event_id: str,
                            out_path: Path) -> Path:
    groups = list(LANDCOVER_GROUPS.keys())
    metrics_names = ["macro_iou", "macro_f1", "accuracy"]
    M = np.zeros((len(groups), len(metrics_names)))
    for i, g in enumerate(groups):
        r = report["by_landcover"].get(g)
        if r is None:
            M[i, :] = np.nan
        else:
            for j, m in enumerate(metrics_names):
                M[i, j] = r[m]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
    ax.set_xticks(range(len(metrics_names))); ax.set_xticklabels(metrics_names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#aaa")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title(f"{model} on {event_id} — per-landcover metrics")
    fig.colorbar(im, ax=ax, shrink=0.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--event", required=True)
    args = p.parse_args()
    evaluate_event(args.model, args.event)


if __name__ == "__main__":
    main()
