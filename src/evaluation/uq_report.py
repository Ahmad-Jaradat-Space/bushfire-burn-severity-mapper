"""Phase B reporting layer: per-pixel uncertainty + conformal sets on the
event-wise U-Net (Kangaroo Island hold-out).

Loads the reproducible event-wise checkpoint trained by
``configs/experiments/unet_eventwise.yaml``, runs Monte-Carlo dropout to get
predictive / epistemic uncertainty maps, then calibrates split-conformal
prediction sets on a spatial half of the scene and measures their coverage on
the other half. The notebook and ``scripts/compute_uq.py`` both call
:func:`compute_uq`, so the story regenerates what it shows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from src.evaluation.conformal import (
    aps_calibrate,
    aps_predict_sets,
    empirical_coverage,
    mean_set_size,
    set_size_map,
)
from src.evaluation.metrics import summary
from src.evaluation.uq_maps import mc_predict
from src.features.stack_features import build_stack
from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.theme import INK, apply_theme

log = get_logger(__name__)

EVENT = "kangaroo_island_2019_2020"
MODEL_DIR = REPO_ROOT / "outputs" / "models" / "unet_eventwise"
CONFIG = "configs/experiments/unet_eventwise.yaml"


def _load_model(device):
    cfg = load_config(CONFIG)
    from src.models.unet_model import build_unet

    model = build_unet(
        in_channels=cfg.unet.in_channels,
        num_classes=cfg.unet.num_classes,
        encoder_name=cfg.unet.encoder_name,
        encoder_weights=None,
        decoder_channels=tuple(cfg.unet.decoder_channels),
        dropout=cfg.unet.dropout,
    ).to(device)
    state = torch.load(MODEL_DIR / "best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def compute_uq(
    event: str = EVENT, T: int = 10, alpha: float = 0.1, stride_frac: float = 1.0, seed: int = 42
) -> dict:
    """Return MC-dropout uncertainty maps + conformal coverage on `event`."""
    import json

    from src.utils.seed import pick_device, set_seeds

    set_seeds(seed)
    device = torch.device(pick_device("mps").name)
    model, cfg = _load_model(device)

    stats = json.loads((MODEL_DIR / "normalization.json").read_text())
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    interim = REPO_ROOT / "data" / "interim" / event
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre = ds.read().astype(np.float32)
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post = ds.read().astype(np.float32)
    with rasterio.open(interim / "label_10m.tif") as ds:
        label = ds.read(1)
    image = build_stack(pre, post)

    tile = int(cfg.data.tile_size)
    stride = max(1, int(tile * stride_frac))
    log.info("MC-dropout: T=%d tile=%d stride=%d on %s %s", T, tile, stride, event, label.shape)
    mc = mc_predict(
        model, image, mean, std, tile, stride, device, num_classes=4, T=T, mc_dropout=True
    )
    log.info("activated %d dropout module(s)", mc["n_dropout_modules"])

    # --- conformal: spatial calibration / test split (left | right halves) ----
    C, H, W = mc["mean_prob"].shape
    cols = np.broadcast_to(np.arange(W), (H, W))
    valid = label != 255
    cal_mask = (valid & (cols < W // 2)).reshape(-1)
    test_mask = (valid & (cols >= W // 2)).reshape(-1)
    probs_flat = mc["mean_prob"].reshape(C, -1).T  # [N, C]
    labels_flat = label.reshape(-1)

    probs_cal, labels_cal = probs_flat[cal_mask], labels_flat[cal_mask]
    probs_test, labels_test = probs_flat[test_mask], labels_flat[test_mask]

    # Sweep the coverage target: it is honest to show the full efficiency curve,
    # because at 90% the sets are near-maximal (the model is that unsure under
    # transfer) while at looser targets they vary and reveal spatial structure.
    alphas = sorted({alpha, 0.5, 0.4, 0.3, 0.2, 0.1}, reverse=True)
    sweep, qhats = [], {}
    for al in alphas:
        qh = aps_calibrate(probs_cal, labels_cal, alpha=al)
        sets = aps_predict_sets(probs_test, qh)
        qhats[al] = qh
        sweep.append(
            {
                "alpha": al,
                "target_coverage": 1 - al,
                "qhat": qh,
                "empirical_coverage": empirical_coverage(sets, labels_test),
                "mean_set_size": mean_set_size(sets),
            }
        )
    head = next(s for s in sweep if abs(s["alpha"] - alpha) < 1e-9)
    disp = min(sweep, key=lambda s: abs(s["mean_set_size"] - 2.0))  # most legible map
    ssize_map = set_size_map(mc["mean_prob"].transpose(1, 2, 0), qhats[disp["alpha"]])
    ssize_map = np.where(valid, ssize_map, 0).astype(np.uint8)

    s = summary(mc["pred"], label, num_classes=4)
    report = {
        "event": event,
        "T": T,
        "alpha": alpha,
        "qhat": head["qhat"],
        "target_coverage": head["target_coverage"],
        "empirical_coverage": head["empirical_coverage"],
        "mean_set_size": head["mean_set_size"],
        "conformal_sweep": sweep,
        "display_alpha": disp["alpha"],
        "n_cal": int(cal_mask.sum()),
        "n_test": int(test_mask.sum()),
        "macro_iou": s["macro_iou"],
        "mean_predictive_entropy": float(mc["predictive_entropy"][valid].mean()),
        "mean_epistemic_mi": float(mc["mutual_information"][valid].mean()),
        # arrays kept for figure rendering (not JSON-serialised by the CLI)
        "_arrays": {
            "predictive_entropy": mc["predictive_entropy"],
            "mutual_information": mc["mutual_information"],
            "set_size_map": ssize_map,
            "valid": valid,
            "display": disp,
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_entropy(report: dict, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    a = report["_arrays"]
    pe = np.ma.masked_where(~a["valid"], a["predictive_entropy"])
    mi = np.ma.masked_where(~a["valid"], a["mutual_information"])
    h, w = a["predictive_entropy"].shape
    fig_h = max(4.0, (13 / 2) * (h / w) + 1.7)
    fig, axes = plt.subplots(1, 2, figsize=(13, fig_h))
    fig.subplots_adjust(top=0.84, bottom=0.02, left=0.01, right=0.99, wspace=0.10)
    for ax, arr, cmap, title in (
        (axes[0], pe, "magma", "Total uncertainty  ·  predictive entropy"),
        (axes[1], mi, "viridis", "Epistemic uncertainty  ·  mutual information"),
    ):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.015, shrink=0.92)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        "How wrong could it be? Monte-Carlo-dropout uncertainty, U-Net on the held-out fire",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=13.5,
    )
    fig.text(
        0.01,
        0.905,
        "Epistemic uncertainty (right) is the part more or better-matched training data could reduce, "
        "it concentrates on burn edges and the unfamiliar heath.",
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
        alpha=0.7,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_conformal(report: dict, out: Path) -> Path:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    apply_theme()
    out = Path(out)
    a = report["_arrays"]
    disp = a["display"]
    ssize = np.ma.masked_where(~a["valid"], a["set_size_map"])
    colours = ["#5C8A6B", "#D8A256", "#C5683B", "#7F1F1F"]  # 1..4 classes in the set
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, (axm, axc) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.01, right=0.96, wspace=0.14)
    im = axm.imshow(ssize, cmap=cmap, norm=norm, interpolation="nearest")
    axm.axis("off")
    axm.set_anchor("NW")
    axm.set_title(
        f"Prediction-set size at {disp['target_coverage']:.0%} target  "
        f"(empirical coverage {disp['empirical_coverage']:.2f}, mean {disp['mean_set_size']:.2f} of 4)",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=8,
    )
    cbar = fig.colorbar(im, ax=axm, fraction=0.040, pad=0.02, shrink=0.78, ticks=[1, 2, 3, 4])
    cbar.set_label("classes in the set (1 = confident, 4 = no idea)", fontsize=9)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=8)

    sweep = sorted(report["conformal_sweep"], key=lambda s: s["target_coverage"])
    tgt = [s["target_coverage"] for s in sweep]
    size = [s["mean_set_size"] for s in sweep]
    axc.plot(tgt, size, "o-", color="#B8553A", lw=2)
    for s in sweep:
        axc.annotate(
            f"cov {s['empirical_coverage']:.2f}",
            (s["target_coverage"], s["mean_set_size"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            color=INK,
        )
    axc.set_xlabel("Target coverage (1 − α)")
    axc.set_ylabel("Mean prediction-set size (of 4)")
    axc.set_ylim(0.8, 4.2)
    axc.set_title("The honest cost of a guarantee", loc="left", fontsize=12)
    from src.viz.theme import thin_axes

    thin_axes(axc)

    fig.suptitle(
        "Conformal severity sets, coverage you can trust, at a visible price",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=13.5,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(report: dict, figdir: Path) -> dict[str, Path]:
    figdir = Path(figdir)
    return {
        "entropy": fig_entropy(report, figdir / "10_entropy_map.png"),
        "conformal": fig_conformal(report, figdir / "11_conformal_sets.png"),
    }
