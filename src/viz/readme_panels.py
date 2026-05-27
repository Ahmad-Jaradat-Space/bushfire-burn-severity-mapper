"""Assemble the 5-method × N-event comparison figures for the README + docs.

Produces the M9 vertical-slice comparison panel for Kangaroo Island and the
M12 final panel across all 4 AOIs. If a particular model's prediction is
missing on disk, that cell is rendered as a "not yet trained" placeholder so
the figure is always regenerable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.maps import SEVERITY_COLORS, SEVERITY_NAMES, severity_cmap

log = get_logger(__name__)

MODELS = [
    ("baseline_dnbr_multiclass", "outputs/predictions/baseline_dnbr/{event}/multiclass.tif", "dNBR baseline"),
    ("rf",                       "outputs/predictions/rf_multiclass/{event}.tif",              "RandomForest"),
    ("xgb",                      "outputs/predictions/xgb_multiclass/{event}.tif",             "XGBoost"),
    ("unet",                     "outputs/predictions/unet_multiclass/{event}.tif",            "U-Net"),
    ("segformer",                "outputs/predictions/segformer_multiclass/{event}.tif",       "SegFormer-B0"),
]


def _read_severity(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        return ds.read(1)


def _read_truecolor(stack_path: Path) -> np.ndarray | None:
    if not stack_path.exists():
        return None
    with rasterio.open(stack_path) as ds:
        a = ds.read().astype(np.float32)
    return a[[2, 1, 0]]   # R=B04, G=B03, B=B02


def _render_truecolor(ax, rgb: np.ndarray, title: str) -> None:
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.nanpercentile(rgb[c], (2, 98))
        if hi > lo:
            out[c] = np.clip((rgb[c] - lo) / (hi - lo), 0, 1)
    ax.imshow(np.transpose(out, (1, 2, 0)))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _render_severity(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(np.ma.masked_equal(arr, 255), cmap=severity_cmap(),
              vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _render_missing(ax, title: str) -> None:
    ax.text(0.5, 0.5, "not available\n(run training)", ha="center", va="center",
            fontsize=9, color="#888", transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#fafafa")


def comparison_panel_for_event(event_id: str, out_path: Path,
                               caption: str = "") -> Path:
    """One row per panel kind, 7 panels: pre TC | post TC | label | 5 models."""
    interim = REPO_ROOT / "data" / "interim" / event_id
    pre_rgb = _read_truecolor(interim / "pre_stack_10m.tif")
    post_rgb = _read_truecolor(interim / "post_stack_10m.tif")
    label = _read_severity(interim / "label_10m.tif")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    if pre_rgb is not None:
        _render_truecolor(axes[0], pre_rgb, "Pre-fire (true-colour)")
    else:
        _render_missing(axes[0], "Pre-fire (not yet processed)")
    if post_rgb is not None:
        _render_truecolor(axes[1], post_rgb, "Post-fire (true-colour)")
    else:
        _render_missing(axes[1], "Post-fire (not yet processed)")
    if label is not None:
        _render_severity(axes[2], label, "AUS GEEBAM label (proxy)")
    else:
        _render_missing(axes[2], "Label (not yet processed)")

    for i, (key, tpl, title) in enumerate(MODELS):
        pred = _read_severity(REPO_ROOT / tpl.format(event=event_id))
        ax = axes[3 + i]
        if pred is not None:
            _render_severity(ax, pred, title)
        else:
            _render_missing(ax, title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLORS]
    fig.legend(handles, SEVERITY_NAMES, loc="lower center", ncols=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"{event_id} — five severity methods\n{caption}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def all_events_overview(events: list[str], out_path: Path) -> Path:
    """Compact one-row-per-event overview: label + best-model column."""
    fig, axes = plt.subplots(len(events), 4, figsize=(16, 4 * len(events)))
    axes = np.atleast_2d(axes)
    for r, ev in enumerate(events):
        interim = REPO_ROOT / "data" / "interim" / ev
        pre = _read_truecolor(interim / "pre_stack_10m.tif")
        post = _read_truecolor(interim / "post_stack_10m.tif")
        label = _read_severity(interim / "label_10m.tif")
        # Best-model row: prefer unet, then segformer, then rf
        for key, tpl, _ in MODELS:
            if key in ("unet", "segformer", "rf"):
                p = REPO_ROOT / tpl.format(event=ev)
                if p.exists():
                    pred = _read_severity(p)
                    pred_title = f"{key} prediction"
                    break
        else:
            pred = None
            pred_title = "no prediction yet"
        cells = [(pre, "rgb", "Pre"), (post, "rgb", "Post"),
                 (label, "severity", "Label"), (pred, "severity", pred_title)]
        for c, (arr, kind, t) in enumerate(cells):
            ax = axes[r, c]
            if arr is None:
                _render_missing(ax, f"{ev}\n{t}")
            elif kind == "rgb":
                _render_truecolor(ax, arr, f"{ev}\n{t}")
            else:
                _render_severity(ax, arr, f"{ev}\n{t}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Render README comparison panels.")
    p.add_argument("--event", default="kangaroo_island_2019_2020")
    p.add_argument("--mode", choices=["single", "overview"], default="single")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.mode == "single":
        out = Path(args.out or REPO_ROOT / "docs" / "figures" / f"{args.event}_comparison_panel.png")
        comparison_panel_for_event(args.event, out, caption="Vertical-slice smoke test")
    else:
        out = Path(args.out or REPO_ROOT / "docs" / "figures" / "all_events_overview.png")
        all_events_overview(
            ["kangaroo_island_2019_2020", "currowan_2019_2020",
             "gospers_mountain_2019_2020", "east_gippsland_2019_2020"], out,
        )


if __name__ == "__main__":
    main()
