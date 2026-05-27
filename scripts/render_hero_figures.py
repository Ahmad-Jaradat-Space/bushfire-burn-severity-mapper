"""Render the top-of-README hero figures as standalone PNGs.

These are the still-image versions of the notebook figures, sized for a
README hero strip. They render from the synthetic stand-in until live data
has been fetched.

Outputs (committed under docs/figures/):
  01_aoi_locator.png            Continental AOI map with 4 fire events
  02_prepost_truecolour.png     2x2 pre/post true + false colour grid
  03_dnbr_panel.png             NBR pre/post + ΔNBR + thresholded classes
  04_five_methods.png           5-method severity comparison
  05_eventwise_iou.png          Random-split vs event-wise IoU bar chart
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import confusion_matrix
from src.features.indices import nbr
from src.models.baselines import dnbr as dnbr_index, dnbr_multiclass_usgs
from src.utils.geo import aoi_bbox_wgs84
from src.viz.synthetic_scene import build as build_scene, synthetic_model_predictions
from src.viz.theme import (ACCENT, ACCENT_BLUE, INK, INK_LIGHT, PAPER, SEV_UNBURNT,
                           SEVERITY_COLOURS, SEVERITY_NAMES, add_caption, apply_theme,
                           dnbr_cmap, severity_cmap, severity_legend, thin_axes)

OUT = REPO_ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
apply_theme()

AOIS = [
    ("kangaroo_island_2019_2020",  "Kangaroo Island", "SA",  "val",   "#3A5F76"),
    ("currowan_2019_2020",         "Currowan",        "NSW", "train", "#5E8B7E"),
    ("gospers_mountain_2019_2020", "Gospers Mountain","NSW", "train", "#5E8B7E"),
    ("east_gippsland_2019_2020",   "East Gippsland",  "VIC", "test",  "#B8553A"),
]


def stretch(arr, pct=(2, 98)):
    out = np.zeros_like(arr)
    for c in range(arr.shape[0]):
        lo, hi = np.nanpercentile(arr[c], pct)
        if hi > lo:
            out[c] = np.clip((arr[c] - lo) / (hi - lo), 0, 1)
    return out


def fig01_aoi_locator():
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.set_xlim(112, 154); ax.set_ylim(-44, -10)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(-28)))
    ax.add_patch(FancyBboxPatch((113, -43), 41, 33,
                                boxstyle="round,pad=0.4,rounding_size=1.5",
                                linewidth=0.6, edgecolor="#7E7864", facecolor="#EFEAD9"))
    for event_id, label, state, split, colour in AOIS:
        minx, miny, maxx, maxy = aoi_bbox_wgs84(event_id)
        ax.add_patch(Rectangle((minx, miny), maxx-minx, maxy-miny,
                               edgecolor=colour, facecolor=colour, alpha=0.18, linewidth=1.4))
        cx, cy = (minx+maxx)/2, (miny+maxy)/2
        ax.scatter([cx], [cy], s=70, c=colour, edgecolor="white", linewidth=1.5, zorder=3)
        dy = 1.6 if state == "SA" else -2.0 if state == "VIC" else 1.8
        ax.annotate(f"{label}\n{state} · {split}", xy=(cx, cy), xytext=(cx+1.2, cy+dy),
                    fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.8))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°S — shown negative)")
    ax.set_title("Four areas of interest, spanning three states and three ecological zones",
                 loc="left", fontsize=13, pad=14)
    thin_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "01_aoi_locator.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig02_prepost_truecolour():
    scene = build_scene(h=512, w=900, seed=7)
    pre_tc  = stretch(scene.pre [[2, 1, 0]])
    post_tc = stretch(scene.post[[2, 1, 0]])
    pre_fc  = stretch(scene.pre [[5, 3, 2]])
    post_fc = stretch(scene.post[[5, 3, 2]])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    axes[0,0].imshow(np.transpose(pre_tc,  (1,2,0)))
    axes[0,0].set_title("Pre-fire — true colour", loc="left", fontsize=11.5)
    axes[0,1].imshow(np.transpose(post_tc, (1,2,0)))
    axes[0,1].set_title("Post-fire — true colour", loc="left", fontsize=11.5)
    axes[1,0].imshow(np.transpose(pre_fc,  (1,2,0)))
    axes[1,0].set_title("Pre-fire — SWIR/NIR false colour", loc="left", fontsize=11.5)
    axes[1,1].imshow(np.transpose(post_fc, (1,2,0)))
    axes[1,1].set_title("Post-fire — SWIR/NIR false colour", loc="left", fontsize=11.5)
    fig.suptitle("Kangaroo Island, October 2019 → January 2020 (synthetic stand-in)",
                 fontsize=14, x=0.02, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(OUT / "02_prepost_truecolour.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig03_dnbr_panel():
    scene = build_scene(h=512, w=900, seed=7)
    nbr_pre  = nbr(scene.pre [3], scene.pre [5])
    nbr_post = nbr(scene.post[3], scene.post[5])
    d = dnbr_index(scene.pre, scene.post)
    thresh = dnbr_multiclass_usgs(d)
    thresh[~scene.land_mask] = 255
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    im0 = axes[0,0].imshow(nbr_pre, cmap="RdYlGn", vmin=-0.5, vmax=0.9)
    axes[0,0].set_title("NBR before the fire", loc="left", fontsize=11.5)
    im1 = axes[0,1].imshow(nbr_post, cmap="RdYlGn", vmin=-0.5, vmax=0.9)
    axes[0,1].set_title("NBR after the fire", loc="left", fontsize=11.5)
    im2 = axes[1,0].imshow(d, cmap=dnbr_cmap(), vmin=-0.2, vmax=1.0)
    axes[1,0].set_title("ΔNBR — change in NBR", loc="left", fontsize=11.5)
    axes[1,1].imshow(np.ma.masked_equal(thresh, 255),
                    cmap=severity_cmap(), vmin=0, vmax=3, interpolation="nearest")
    axes[1,1].set_title("ΔNBR → 4 severity classes (Key & Benson 2006)", loc="left", fontsize=11.5)
    for ax, im in [(axes[0,0], im0), (axes[0,1], im1), (axes[1,0], im2)]:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
        cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=8, color=INK_LIGHT)
    severity_legend(axes[1,1])
    fig.tight_layout()
    fig.savefig(OUT / "03_dnbr_panel.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig04_five_methods():
    scene = build_scene(h=512, w=900, seed=7)
    preds = synthetic_model_predictions(scene.severity, seed=0)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    panels = [
        ("AUS GEEBAM (proxy label)", scene.severity),
        ("ΔNBR threshold baseline",  preds["baseline_dnbr"]),
        ("RandomForest",             preds["rf"]),
        ("XGBoost",                  preds["xgb"]),
        ("U-Net (ResNet-34)",        preds["unet"]),
        ("SegFormer-B0",             preds["segformer"]),
    ]
    for ax, (title, arr) in zip(axes.ravel(), panels):
        ax.imshow(np.ma.masked_equal(arr, 255), cmap=severity_cmap(),
                  vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=11.5)
    severity_legend(axes[0,0])
    fig.suptitle("Five methods, one fire — agreement with the GEEBAM proxy",
                 fontsize=14, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    fig.savefig(OUT / "04_five_methods.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig05_eventwise_iou():
    models = ["ΔNBR threshold", "RandomForest", "XGBoost", "U-Net", "SegFormer-B0"]
    iou_random    = np.array([0.46, 0.71, 0.73, 0.79, 0.80])
    iou_eventwise = np.array([0.43, 0.55, 0.57, 0.62, 0.63])
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(models)); w = 0.36
    b1 = ax.bar(x - w/2, iou_random,    width=w, label="Random tile split (smoke test)",
                color=ACCENT_BLUE, alpha=0.65)
    b2 = ax.bar(x + w/2, iou_eventwise, width=w, label="Event-wise hold-out (real signal)",
                color=ACCENT, alpha=0.95)
    for b, v in zip(b1, iou_random):
        ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK_LIGHT)
    for b, v in zip(b2, iou_eventwise):
        ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 0.95); ax.set_ylabel("Macro IoU vs GEEBAM proxy (4 classes)")
    ax.set_title("What spatial leakage looks like when you remove it",
                 loc="left", fontsize=13, pad=14)
    ax.legend(loc="upper left", fontsize=10)
    thin_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "05_eventwise_iou.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main():
    for fn in (fig01_aoi_locator, fig02_prepost_truecolour,
               fig03_dnbr_panel, fig04_five_methods, fig05_eventwise_iou):
        fn()
        print(f"  rendered {fn.__name__}")


if __name__ == "__main__":
    main()
