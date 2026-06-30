"""Phase-A reporting layer: turn event-wise predictions into figures with error
bars and p-values.

This is the importable library half, the notebook and
``scripts/compute_uncertainty.py`` both call :func:`compute_report` and the
``fig_*`` renderers, so the story notebook regenerates the same artifacts it
displays. The statistics live in :mod:`src.evaluation.uncertainty`,
:mod:`src.evaluation.blocks` and :mod:`src.evaluation.area_estimation`; this
module only orchestrates and draws.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.evaluation.area_estimation import olofsson_area_accuracy
from src.evaluation.blocks import block_confusions, grid_blocks
from src.evaluation.metrics import summary
from src.evaluation.uncertainty import (
    bootstrap_macro_metrics,
    mcnemar_test,
    paired_bootstrap_delta,
)
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.theme import (
    INK,
    INK_LIGHT,
    MODEL_COLOURS,
    SEVERITY_COLOURS,
    SEVERITY_NAMES,
    apply_theme,
    thin_axes,
)

log = get_logger(__name__)

EVENT = "kangaroo_island_2019_2020"
NUM_CLASSES = 4
PIXEL_AREA_M2 = 900.0  # working grid is 30 m (the *_10m filenames are legacy)

# Event-wise hold-out predictions on the val event. dNBR is parameter-free, so
# its per-event map *is* the event-wise prediction.
PRED_TEMPLATES = {
    "baseline_dnbr": "outputs/predictions/baseline_dnbr/{event}/multiclass.tif",
    "rf": "outputs/predictions/_event_wise_predictions/rf_multiclass_eventwise_kangaroo.tif",
    "xgb": "outputs/predictions/_event_wise_predictions/xgb_multiclass_eventwise_kangaroo.tif",
    "unet": "outputs/predictions/_event_wise_predictions/unet_multiclass_eventwise_kangaroo.tif",
    "segformer": "outputs/predictions/_event_wise_predictions/segformer_multiclass_eventwise_kangaroo.tif",
    "prithvi": "outputs/predictions/prithvi_eventwise/{event}.tif",
}
MODEL_LABELS = {
    "baseline_dnbr": "ΔNBR",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "unet": "U-Net",
    "segformer": "SegFormer-B0",
    "prithvi": "Prithvi-EO-2.0",
}


def _read_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(band)


def _load_predictions(event: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    label = _read_band(REPO_ROOT / "data" / "interim" / event / "label_10m.tif")
    preds: dict[str, np.ndarray] = {}
    for model, tpl in PRED_TEMPLATES.items():
        path = REPO_ROOT / tpl.format(event=event)
        if not path.exists():
            log.info("skip %s, no event-wise prediction at %s", model, path.relative_to(REPO_ROOT))
            continue
        arr = _read_band(path)
        if arr.shape != label.shape:
            log.warning("skip %s, shape %s != label %s", model, arr.shape, label.shape)
            continue
        preds[model] = arr
    if not preds:
        raise SystemExit("No event-wise predictions found, run the pipeline first.")
    return label, preds


def compute_report(
    event: str = EVENT, block_px: int = 256, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap CIs, McNemar, paired deltas, Olofsson areas and rare-class P/R."""
    label, preds = _load_predictions(event)
    log.info("loaded %d models on %s (raster %s)", len(preds), event, label.shape)
    block_id = grid_blocks(label.shape, block_px)

    report: dict = {
        "event": event,
        "protocol": "event_wise_holdout",
        "block_px": block_px,
        "n_boot": n_boot,
        "models": {},
        "pairwise_mcnemar": {},
        "paired_delta_vs_dnbr": {},
    }

    block_cms: dict[str, dict[int, np.ndarray]] = {}
    for model, pred in preds.items():
        cms = block_confusions(pred, label, block_id, NUM_CLASSES)
        block_cms[model] = cms
        boot = bootstrap_macro_metrics(cms, n_boot=n_boot, seed=seed)
        s = summary(pred, label, NUM_CLASSES)
        report["models"][model] = {
            "label": MODEL_LABELS.get(model, model),
            "bootstrap": boot,
            "per_class_precision": s["per_class_precision"],
            "per_class_recall": s["per_class_recall"],
            "per_class_iou": s["per_class_iou"],
            "macro_iou": s["macro_iou"],
            "confusion_matrix": s["confusion_matrix"],
            "olofsson": olofsson_area_accuracy(
                np.array(s["confusion_matrix"], dtype=np.float64), pixel_area_m2=PIXEL_AREA_M2
            ),
        }

    for a, b in combinations(preds, 2):
        report["pairwise_mcnemar"][f"{a}__vs__{b}"] = mcnemar_test(preds[a], preds[b], label)

    if "baseline_dnbr" in block_cms:
        for model in preds:
            if model == "baseline_dnbr":
                continue
            report["paired_delta_vs_dnbr"][model] = paired_bootstrap_delta(
                block_cms[model],
                block_cms["baseline_dnbr"],
                metric="macro_iou",
                n_boot=n_boot,
                seed=seed,
            )

    return report


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _order(models: list[str]) -> list[str]:
    pref = ["baseline_dnbr", "rf", "xgb", "unet", "segformer", "prithvi"]
    return [m for m in pref if m in models]


def fig_bootstrap_ci(report: dict, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    models = _order(list(report["models"]))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(models) + 1.8))
    for i, m in enumerate(models):
        b = report["models"][m]["bootstrap"]["metrics"]["macro_iou"]
        colour = MODEL_COLOURS.get(m, INK)
        ax.errorbar(
            b["point"],
            i,
            xerr=[[b["point"] - b["ci_low"]], [b["ci_high"] - b["point"]]],
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=2,
            capsize=4,
            markersize=8,
        )
        ax.text(
            b["ci_high"] + 0.005,
            i,
            f"{b['point']:.3f}  [{b['ci_low']:.3f}, {b['ci_high']:.3f}]",
            va="center",
            fontsize=9,
            color=INK_LIGHT,
        )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([report["models"][m]["label"] for m in models])
    ax.set_xlabel("Event-wise macro IoU  (95% spatial-block bootstrap CI)")
    ax.set_title("How big is the gap, really?", loc="left")
    ax.set_xlim(
        0,
        min(
            1.0,
            max(report["models"][m]["bootstrap"]["metrics"]["macro_iou"]["ci_high"] for m in models)
            + 0.18,
        ),
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_mcnemar(report: dict, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    models = _order(list(report["models"]))
    n = len(models)
    P = np.full((n, n), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                continue
            key = (
                f"{a}__vs__{b}"
                if f"{a}__vs__{b}" in report["pairwise_mcnemar"]
                else f"{b}__vs__{a}"
            )
            P[i, j] = report["pairwise_mcnemar"][key]["p_value"]
    with np.errstate(divide="ignore"):
        neglog = -np.log10(np.clip(P, 1e-300, 1.0))
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1.5))
    im = ax.imshow(neglog, cmap="cividis", vmin=0, vmax=min(30, np.nanmax(neglog) or 1))
    ax.set_xticks(range(n))
    ax.set_xticklabels([report["models"][m]["label"] for m in models], rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([report["models"][m]["label"] for m in models])
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, ", ", ha="center", va="center", color="#888")
                continue
            p = P[i, j]
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
            label = "p<1e-300" if p < 1e-300 else f"p={p:.1e}"
            ax.text(
                j,
                i,
                f"{label}\n{star}",
                ha="center",
                va="center",
                color="white" if neglog[i, j] > 6 else INK,
                fontsize=7.5,
            )
    ax.set_title("McNemar paired significance (per-pixel correctness)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="−log₁₀ p")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_area_adjusted(report: dict, out: Path, model: str = "unet") -> Path:
    apply_theme()
    out = Path(out)
    if model not in report["models"]:
        model = _order(list(report["models"]))[-1]
    olof = report["models"][model]["olofsson"]
    classes = olof["per_class"]
    x = np.arange(len(classes))
    mapped = [c["area_mapped_ha"] for c in classes]
    adj = [c["area_adjusted_ha"] for c in classes]
    lo = [c["area_adjusted_ha"] - c["area_ci_ha"][0] for c in classes]
    hi = [c["area_ci_ha"][1] - c["area_adjusted_ha"] for c in classes]
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, mapped, w, color="#C9C2B2", label="Naïve mapped area")
    ax.bar(x + w / 2, adj, w, color=SEVERITY_COLOURS, label="Area-adjusted (Olofsson)")
    ax.errorbar(x + w / 2, adj, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.3, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_NAMES)
    ax.set_ylabel("Area (hectares)")
    ax.set_title(
        f"A defensible number: area-adjusted severity area, {report['models'][model]['label']}",
        loc="left",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C9C2B2"),
        plt.Rectangle((0, 0), 1, 1, color=SEVERITY_COLOURS[2]),
    ]
    ax.legend(
        handles, ["Naïve mapped area", "Area-adjusted (Olofsson) ± 95% CI"], loc="upper right"
    )
    thin_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_rareclass_pr(report: dict, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    models = _order(list(report["models"]))
    focus = [(2, "High"), (3, "Very High")]
    fig, axes = plt.subplots(1, len(focus), figsize=(11, 5.0), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.08)
    x = np.arange(len(models))
    w = 0.38
    for ax, (cls, name) in zip(axes, focus, strict=False):
        prec = [report["models"][m]["per_class_precision"][cls] for m in models]
        rec = [report["models"][m]["per_class_recall"][cls] for m in models]
        bp = ax.bar(x - w / 2, prec, w, color="#3A5F76", label="Precision")
        br = ax.bar(x + w / 2, rec, w, color="#B8553A", label="Recall")
        ax.bar_label(bp, fmt="%.2f", padding=2, fontsize=7.5, color=INK_LIGHT)
        ax.bar_label(br, fmt="%.2f", padding=2, fontsize=7.5, color=INK_LIGHT)
        ax.set_xticks(x)
        ax.set_xticklabels([report["models"][m]["label"] for m in models], rotation=30, ha="right")
        ax.set_title(f"{name} severity (class {cls})", loc="left", fontsize=11.5)
        ax.set_ylim(0, 1.05)
        thin_axes(ax)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right")
    fig.suptitle(
        "The rare classes are where the maps actually disagree",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=13.5,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# trainable parameters (millions) per learned method, frozen backbones excluded
TRAINABLE_PARAMS_M = {"unet": 24.4, "segformer": 3.7, "prithvi": 2.63}


def fig_foundation(report: dict, out: Path) -> Path:
    """Trainable-params vs event-wise macro-IoU: the foundation-model efficiency story."""
    apply_theme()
    out = Path(out)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if "baseline_dnbr" in report["models"]:
        d = report["models"]["baseline_dnbr"]["bootstrap"]["metrics"]["macro_iou"]
        ax.axhline(d["point"], ls="--", color="#7E7864", lw=1.4)
        ax.text(
            0.02,
            d["point"] + 0.004,
            f"ΔNBR baseline ({d['point']:.3f}), 0 trainable params",
            color="#6b6552",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
    for m, pm in TRAINABLE_PARAMS_M.items():
        if m not in report["models"]:
            continue
        b = report["models"][m]["bootstrap"]["metrics"]["macro_iou"]
        colour = MODEL_COLOURS.get(m, INK)
        ax.errorbar(
            pm,
            b["point"],
            yerr=[[b["point"] - b["ci_low"]], [b["ci_high"] - b["point"]]],
            fmt="o",
            ms=11,
            color=colour,
            ecolor=colour,
            elinewidth=2,
            capsize=4,
        )
        ax.annotate(
            f"{report['models'][m]['label']}\n{b['point']:.3f}",
            (pm, b["point"]),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=9,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.8, 40)
    ax.set_xlabel("Trainable parameters (millions, log), frozen backbone excluded")
    ax.set_ylabel("Event-wise macro IoU (95% CI)")
    ax.set_title("Foundation models: less to train, more that travels", loc="left")
    thin_axes(ax)
    ax.grid(True, axis="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(report: dict, figdir: Path) -> dict[str, Path]:
    figdir = Path(figdir)
    return {
        "bootstrap": fig_bootstrap_ci(report, figdir / "06_bootstrap_ci.png"),
        "mcnemar": fig_mcnemar(report, figdir / "07_mcnemar.png"),
        "area": fig_area_adjusted(report, figdir / "08_area_adjusted.png"),
        "rareclass": fig_rareclass_pr(report, figdir / "08b_rareclass_pr.png"),
        "foundation": fig_foundation(report, figdir / "16_prithvi_vs_unet.png"),
    }
