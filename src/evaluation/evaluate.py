"""Aggregate evaluation across models and events.

Walks outputs/predictions/, recomputes per-event metrics against the
canonical label_10m.tif, and writes:

  outputs/metrics/aggregate_report.json          full metrics tree
  outputs/metrics/aggregate_summary.csv          one row per (model, event)
  docs/figures/per_event_scores.png              grouped bar chart
  docs/figures/confusion_<model>_<event>.png     confusion matrices
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.evaluation.metrics import IGNORE_ID, summary
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

PRED_TEMPLATES = {
    "baseline_dnbr": "outputs/predictions/baseline_dnbr/{event}/multiclass.tif",
    "rf":            "outputs/predictions/rf_multiclass/{event}.tif",
    "xgb":           "outputs/predictions/xgb_multiclass/{event}.tif",
    "unet":          "outputs/predictions/unet_multiclass/{event}.tif",
    "segformer":     "outputs/predictions/segformer_multiclass/{event}.tif",
}

ALL_EVENTS = [
    "kangaroo_island_2019_2020",
    "currowan_2019_2020",
    "gospers_mountain_2019_2020",
    "east_gippsland_2019_2020",
]


def _read_band(path: Path, band: int = 1):
    with rasterio.open(path) as ds:
        return ds.read(band)


def evaluate(events: list[str] | None = None) -> dict:
    events = events or ALL_EVENTS
    out_metrics: dict = {"models": list(PRED_TEMPLATES.keys()), "events": events,
                         "per_pair": {}}
    rows = []
    for model, tpl in PRED_TEMPLATES.items():
        out_metrics["per_pair"][model] = {}
        for ev in events:
            pred_path = REPO_ROOT / tpl.format(event=ev)
            if not pred_path.exists():
                log.info("Skipping %s/%s — no prediction yet at %s",
                         model, ev, pred_path.relative_to(REPO_ROOT))
                continue
            label_path = REPO_ROOT / "data" / "interim" / ev / "label_10m.tif"
            if not label_path.exists():
                log.info("Skipping %s/%s — no label at %s",
                         model, ev, label_path.relative_to(REPO_ROOT))
                continue
            pred = _read_band(pred_path)
            label = _read_band(label_path)
            s = summary(pred, label, num_classes=4)
            out_metrics["per_pair"][model][ev] = s
            rows.append({"model": model, "event": ev,
                         "macro_iou": s["macro_iou"], "macro_f1": s["macro_f1"],
                         "accuracy": s["accuracy"]})
            _plot_confusion(np.array(s["confusion_matrix"]), model, ev)

    out_dir = REPO_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate_report.json").write_text(json.dumps(out_metrics, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "aggregate_summary.csv", index=False)
    log.info("Wrote aggregate metrics for %d (model,event) pairs.", len(rows))
    if not df.empty:
        _plot_per_event_bar(df)
    return out_metrics


def _plot_confusion(cm: np.ndarray, model: str, event_id: str) -> Path:
    names = ["unburnt", "low_mod", "high", "very_high"]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model} on {event_id}\n(row-normalised)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    out = REPO_ROOT / "docs" / "figures" / f"confusion_{model}_{event_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _plot_per_event_bar(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    events = df["event"].unique()
    models = df["model"].unique()
    n_models = len(models)
    x = np.arange(len(events))
    width = 0.8 / max(n_models, 1)
    for i, m in enumerate(models):
        rows = df[df["model"] == m].set_index("event").reindex(events)
        vals = rows["macro_iou"].fillna(0.0).values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(events, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("macro IoU"); ax.set_ylim(0, 1)
    ax.set_title("Per-event macro IoU by model")
    ax.legend(fontsize=9, ncols=3)
    out = REPO_ROOT / "docs" / "figures" / "per_event_scores.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", nargs="*", default=None)
    p.add_argument("--all-events", action="store_true")
    args = p.parse_args()
    events = ALL_EVENTS if args.all_events else args.events
    evaluate(events)


if __name__ == "__main__":
    main()
