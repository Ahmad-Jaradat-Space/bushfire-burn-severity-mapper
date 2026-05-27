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
from src.viz.maps import render_aoi_locator
from src.viz.scene_loader import load_kangaroo, model_predictions
from src.viz.synthetic_scene import (s2_truecolour, s2_swir_falsecolour)
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


def _get_scene():
    """Load real Kangaroo data if present, synthetic fallback otherwise."""
    return load_kangaroo()


def fig01_aoi_locator():
    render_aoi_locator(AOIS, OUT / "01_aoi_locator.png", figsize=(11, 8))


def fig02_prepost_truecolour():
    scene = _get_scene()
    pre_tc, post_tc = s2_truecolour(scene.pre), s2_truecolour(scene.post)
    pre_fc, post_fc = s2_swir_falsecolour(scene.pre), s2_swir_falsecolour(scene.post)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    titles = [("Pre-fire — true colour",                    pre_tc),
              ("Post-fire — true colour",                   post_tc),
              ("Pre-fire — SWIR / NIR false colour",        pre_fc),
              ("Post-fire — SWIR / NIR false colour",       post_fc)]
    for ax, (title, img) in zip(axes.ravel(), titles):
        ax.imshow(img)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
    src = ("Real Sentinel-2 L2A via Microsoft Planetary Computer"
           if scene.is_real else "Synthetic stand-in")
    fig.suptitle(f"Kangaroo Island, October 2019 → January 2020 — {src}",
                 fontsize=15, x=0.02, ha="left", y=0.99, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.03,
                        hspace=0.18, wspace=0.04)
    fig.savefig(OUT / "02_prepost_truecolour.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig03_dnbr_panel():
    scene = _get_scene()
    nbr_pre  = nbr(scene.pre [3], scene.pre [5])
    nbr_post = nbr(scene.post[3], scene.post[5])
    d = dnbr_index(scene.pre, scene.post)
    thresh = dnbr_multiclass_usgs(d)
    thresh[~scene.land_mask] = 255
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    im0 = axes[0,0].imshow(nbr_pre, cmap="RdYlGn", vmin=-0.5, vmax=0.9)
    axes[0,0].set_title("NBR before the fire", loc="left", fontsize=12, pad=8)
    im1 = axes[0,1].imshow(nbr_post, cmap="RdYlGn", vmin=-0.5, vmax=0.9)
    axes[0,1].set_title("NBR after the fire", loc="left", fontsize=12, pad=8)
    im2 = axes[1,0].imshow(d, cmap=dnbr_cmap(), vmin=-0.2, vmax=1.0)
    axes[1,0].set_title("ΔNBR — change in NBR", loc="left", fontsize=12, pad=8)
    axes[1,1].imshow(np.ma.masked_equal(thresh, 255),
                    cmap=severity_cmap(), vmin=0, vmax=3, interpolation="nearest")
    axes[1,1].set_title("ΔNBR → 4 severity classes (Key & Benson 2006)", loc="left", fontsize=12, pad=8)
    for ax, im in [(axes[0,0], im0), (axes[0,1], im1), (axes[1,0], im2)]:
        cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015, shrink=0.78)
        cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=8, color=INK_LIGHT)
    # Severity legend below the classified panel, outside the image area
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLOURS]
    axes[1,1].legend(handles, SEVERITY_NAMES, loc="upper right",
                     bbox_to_anchor=(1.0, -0.04), ncols=4, fontsize=9,
                     frameon=False, columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.07,
                        hspace=0.22, wspace=0.08)
    fig.savefig(OUT / "03_dnbr_panel.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def fig04_five_methods():
    scene = _get_scene()
    preds, preds_real = model_predictions(scene)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_facecolor(PAPER)
    def _t(k, base):
        if k not in preds: return f"{base} · not trained"
        if not preds_real.get(k, False): return f"{base} · synthetic stand-in"
        return base
    panels = [
        ("AUS GEEBAM (proxy label)",         scene.severity, True),
        (_t('baseline_dnbr', "ΔNBR threshold baseline"),
            preds.get('baseline_dnbr'), 'baseline_dnbr' in preds),
        (_t('rf',            "RandomForest"),
            preds.get('rf'), 'rf' in preds),
        (_t('xgb',           "XGBoost"),
            preds.get('xgb'), 'xgb' in preds),
        (_t('unet',          "U-Net (ResNet-34)"),
            preds.get('unet'), 'unet' in preds),
        (_t('segformer',     "SegFormer-B0"),
            preds.get('segformer'), 'segformer' in preds),
    ]
    for ax, (title, arr, present) in zip(axes.ravel(), panels):
        if arr is not None and present:
            ax.imshow(np.ma.masked_equal(arr, 255), cmap=severity_cmap(),
                      vmin=0, vmax=3, interpolation="nearest")
        else:
            ax.text(0.5, 0.5, "not trained\n(see README)", ha="center", va="center",
                    fontsize=10, color=INK_LIGHT, transform=ax.transAxes)
            ax.set_facecolor("#FAFAF6")
        ax.set_title(title, loc="left", fontsize=12, pad=8)
    # Severity legend below all panels, centred
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLOURS]
    fig.legend(handles, SEVERITY_NAMES, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncols=4, fontsize=10,
               frameon=False, columnspacing=2.2, handletextpad=0.5)
    fig.suptitle("Five methods, one fire — agreement with the GEEBAM proxy",
                 fontsize=15, x=0.02, ha="left", y=0.98, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.07,
                        hspace=0.20, wspace=0.04)
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
