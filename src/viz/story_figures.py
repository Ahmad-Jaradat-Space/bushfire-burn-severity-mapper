"""Story figures: the elegant, magazine-grade visuals the notebook opens and
closes on.

Everything here renders **offline** from artifacts already on disk
(``outputs/metrics/uncertainty/<event>_eventwise.json`` and
``outputs/metrics/per_aoi_summary.json``), no network, no model load, so the
notebook's hero figures regenerate in seconds and never depend on a live STAC
call. The statistics were computed upstream by
:mod:`src.evaluation.uncertainty_report`; this module only composes them into
the figures a reader meets first.

Four figures:

``fig_leaderboard``       the graphical abstract, every method ranked by
                          event-wise macro-IoU with bootstrap intervals and a
                          one-line verdict against the twenty-year-old baseline.
``fig_methods_overview``  the six methods at a glance: family, inputs, trainable
                          parameters, and the headline score, as a clean card.
``fig_split_schematic``   why an event-wise hold-out is the only honest split,
                          drawn, not described.
``fig_journey``           a navigational ribbon of the argument, for the reader
                          who wants to know where the piece is going.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.utils.geo import REPO_ROOT
from src.viz.theme import (
    ACCENT,
    INK,
    INK_LIGHT,
    MODEL_COLOURS,
    PAPER,
    RULE,
    SEV_VERY_HIGH,
    apply_theme,
)

EVENT = "kangaroo_island_2019_2020"

# verdict palette
_GOOD = "#4F7A5B"  # confident green, beats the baseline
_MEH = INK_LIGHT  # ties the baseline
_BAD = "#9A5A4A"  # muted red, below the baseline

# family label + the colour used for that method's marker
FAMILY = {
    "baseline_dnbr": "Spectral index · 1996",
    "rf": "Classical ML",
    "xgb": "Classical ML",
    "unet": "Deep net · from scratch",
    "segformer": "Deep net · from scratch",
    "prithvi": "Geospatial foundation model",
}

# what each method actually consumes, and how much of it is trainable
INPUTS = {
    "baseline_dnbr": "2 bands (NIR, SWIR2), pre + post",
    "rf": "18 per-pixel spectral + terrain features",
    "xgb": "18 per-pixel spectral + terrain features",
    "unet": "18-channel pre/post stack · ResNet-34 encoder",
    "segformer": "18-channel pre/post stack · MiT-B0 encoder",
    "prithvi": "6 bands, post-fire only · frozen 300M ViT",
}
TRAINABLE = {
    "baseline_dnbr": "0, fixed thresholds",
    "rf": "500 trees",
    "xgb": "800 trees",
    "unet": "24.4 M",
    "segformer": "3.7 M",
    "prithvi": "2.6 M (backbone frozen)",
}

_ORDER = ["baseline_dnbr", "rf", "xgb", "unet", "segformer", "prithvi"]


# --------------------------------------------------------------------------- #
# data loaders
# --------------------------------------------------------------------------- #
def _load_report(event: str = EVENT) -> dict:
    p = REPO_ROOT / "outputs" / "metrics" / "uncertainty" / f"{event}_eventwise.json"
    return json.loads(p.read_text())


def _load_per_aoi() -> dict:
    p = REPO_ROOT / "outputs" / "metrics" / "per_aoi_summary.json"
    return json.loads(p.read_text())


def _verdict(model: str, report: dict) -> tuple[str, str]:
    """Return (text, colour) describing this model vs the ΔNBR floor."""
    if model == "baseline_dnbr":
        return "the floor, a 1996 two-band index, 0 trainable params", _MEH
    d = report["paired_delta_vs_dnbr"].get(model)
    if d is None:
        return "", _MEH
    delta, p = d["delta"], d["p_value"]
    pstr = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
    if p < 0.05 and delta > 0:
        return f"beats ΔNBR  ·  Δ {delta:+.3f},  {pstr}", _GOOD
    if p < 0.05 and delta < 0:
        return f"below ΔNBR  ·  Δ {delta:+.3f},  {pstr}", _BAD
    return f"ties ΔNBR  ·  Δ {delta:+.3f},  {pstr}", _MEH


# --------------------------------------------------------------------------- #
# 1. leaderboard, the graphical abstract
# --------------------------------------------------------------------------- #
def fig_leaderboard(out: Path, event: str = EVENT) -> Path:
    """Ranked event-wise macro-IoU with bootstrap CIs and a per-row verdict.

    The single figure a hiring manager should be able to read in ten seconds:
    only the frozen foundation model clears the twenty-year-old baseline.
    """
    apply_theme()
    out = Path(out)
    report = _load_report(event)

    rows = []
    for m, info in report["models"].items():
        b = info["bootstrap"]["metrics"]["macro_iou"]
        rows.append((m, info["label"], b["point"], b["ci_low"], b["ci_high"]))
    rows.sort(key=lambda r: r[2])  # worst at bottom, best at top (y increases up)

    dnbr = report["models"]["baseline_dnbr"]["bootstrap"]["metrics"]["macro_iou"]["point"]

    fig, ax = plt.subplots(figsize=(11.2, 0.92 * len(rows) + 2.3))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.78, bottom=0.13)

    # ΔNBR reference band + line
    ax.axvspan(0, dnbr, color=RULE, alpha=0.32, zorder=0)
    ax.axvline(dnbr, color=INK_LIGHT, ls=(0, (5, 3)), lw=1.1, zorder=1)

    xmax = max(r[4] for r in rows)
    for i, (m, label, point, lo, hi) in enumerate(rows):
        colour = MODEL_COLOURS.get(m, INK)
        winner = i == len(rows) - 1
        if winner:
            ax.add_patch(
                Rectangle(
                    (0, i - 0.42),
                    xmax + 0.005,
                    0.84,
                    color=colour,
                    alpha=0.07,
                    zorder=0,
                    lw=0,
                )
            )
        # lollipop stem from 0 to the point
        ax.plot([0, point], [i, i], color=colour, lw=1.4, alpha=0.32, zorder=2)
        # CI whisker
        ax.plot([lo, hi], [i, i], color=colour, lw=2.4, alpha=0.9, zorder=3)
        ax.plot([lo, lo], [i - 0.09, i + 0.09], color=colour, lw=2.0, zorder=3)
        ax.plot([hi, hi], [i - 0.09, i + 0.09], color=colour, lw=2.0, zorder=3)
        # point
        ax.scatter(
            [point],
            [i],
            s=150 if winner else 95,
            color=colour,
            edgecolor=PAPER,
            linewidth=1.6,
            zorder=4,
        )
        # method label on the left
        ax.text(
            -0.012,
            i,
            label,
            ha="right",
            va="center",
            fontsize=12.5 if winner else 11.5,
            color=INK,
            fontweight="bold" if winner else "normal",
        )
        ax.text(
            -0.012,
            i - 0.27,
            FAMILY[m],
            ha="right",
            va="center",
            fontsize=8.3,
            color=INK_LIGHT,
            style="italic",
        )
        # score just past the point
        ax.text(
            point,
            i + 0.235,
            f"{point:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=colour,
            fontweight="bold",
        )
        # verdict on the right
        vtext, vcol = _verdict(m, report)
        ax.text(
            xmax + 0.02,
            i,
            vtext,
            ha="left",
            va="center",
            fontsize=9.2,
            color=vcol,
            fontweight="bold" if vcol == _GOOD else "normal",
        )
        if winner:
            ax.text(
                point,
                i - 0.235,
                "▲ best score",
                ha="center",
                va="top",
                fontsize=8.4,
                color=colour,
            )

    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, xmax + 0.215)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, xmax + 0.05, 0.05))
    ax.tick_params(axis="x", labelsize=9, color=INK_LIGHT)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_LIGHT)
    ax.grid(False)
    ax.set_xlabel(
        "Event-wise macro-IoU on Kangaroo Island, a fire no model trained on  "
        "(● point ·, 95% spatial-block bootstrap interval)",
        fontsize=9.5,
        color=INK_LIGHT,
    )

    # ΔNBR caption near the line
    ax.text(
        dnbr,
        len(rows) - 0.30,
        "  ΔNBR floor",
        ha="left",
        va="center",
        fontsize=8.6,
        color=INK_LIGHT,
        style="italic",
    )

    fig.suptitle(
        "Only one learned model clears a twenty-year-old bar",
        x=0.012,
        y=0.965,
        ha="left",
        fontsize=17,
        color=INK,
    )
    fig.text(
        0.012,
        0.855,
        "Four trained models plus the ΔNBR spectral index, scored on a held-out fire. Error bars that respect\n"
        "spatial autocorrelation collapse most of the apparent winners back onto the baseline, except Prithvi.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=INK_LIGHT,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 2. methods at a glance, a drawn card/table
# --------------------------------------------------------------------------- #
def fig_methods_overview(out: Path, event: str = EVENT) -> Path:
    apply_theme()
    out = Path(out)
    report = _load_report(event)
    per_aoi = _load_per_aoi()

    def score(m: str):
        info = report["models"].get(m)
        if info is not None:
            return info["bootstrap"]["metrics"]["macro_iou"]["point"], True
        # segformer never ran in the event-wise bootstrap; show its in-domain
        # number, clearly flagged, rather than fabricate an event-wise bar.
        v = per_aoi.get(event, {}).get(m, {}).get("macro_iou")
        return (v, False) if v is not None else (None, False)

    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # column anchors (axis fraction)
    cx = {"chip": 0.012, "name": 0.035, "inputs": 0.34, "params": 0.66, "score": 0.80}
    top, bottom = 0.80, 0.06
    n = len(_ORDER)
    row_h = (top - bottom) / n
    ys = [top - (k + 0.5) * row_h for k in range(n)]

    # header
    hy = top + 0.085
    for key, lab in [
        ("name", "Method"),
        ("inputs", "What it sees"),
        ("params", "Trainable"),
        ("score", "Event-wise macro-IoU"),
    ]:
        ax.text(
            cx[key],
            hy,
            lab.upper(),
            ha="left",
            va="center",
            fontsize=9,
            color=INK_LIGHT,
            fontweight="bold",
        )
    ax.plot([0.008, 0.992], [top + 0.03, top + 0.03], color=INK, lw=1.0)

    eventwise_scores = [
        report["models"][m]["bootstrap"]["metrics"]["macro_iou"]["point"]
        for m in _ORDER
        if m in report["models"]
    ]
    smax = max(eventwise_scores)
    bar_x0, bar_w = cx["score"], 0.155

    for k, (m, y) in enumerate(zip(_ORDER, ys, strict=False)):
        colour = MODEL_COLOURS.get(m, INK)
        if k % 2 == 1:
            ax.add_patch(
                Rectangle((0.008, y - row_h / 2), 0.984, row_h, color=RULE, alpha=0.22, lw=0)
            )
        # colour chip
        ax.add_patch(Rectangle((cx["chip"], y - 0.045), 0.012, 0.09, color=colour, lw=0))
        # name + family
        ax.text(
            cx["name"],
            y + 0.022,
            report["models"].get(m, {}).get("label", _label(m)),
            ha="left",
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            cx["name"],
            y - 0.052,
            FAMILY[m],
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK_LIGHT,
            style="italic",
        )
        # inputs
        ax.text(cx["inputs"], y, INPUTS[m], ha="left", va="center", fontsize=9.6, color=INK_LIGHT)
        # params
        ax.text(
            cx["params"], y, TRAINABLE[m], ha="left", va="center", fontsize=9.6, color=INK_LIGHT
        )
        # score bar
        val, is_eventwise = score(m)
        if val is not None and is_eventwise:
            w = bar_w * (val / smax)
            ax.add_patch(
                FancyBboxPatch(
                    (bar_x0, y - 0.028),
                    w,
                    0.056,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    color=colour,
                    alpha=0.85,
                    lw=0,
                )
            )
            ax.text(
                bar_x0 + w + 0.008,
                y,
                f"{val:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                color=INK,
                fontweight="bold",
            )
        else:
            tag = f"in-domain {val:.3f} only" if val is not None else ", "
            ax.text(
                bar_x0, y, tag, ha="left", va="center", fontsize=9, color=INK_LIGHT, style="italic"
            )

    fig.suptitle(
        "Six ways to read a burn scar", x=0.012, y=0.965, ha="left", fontsize=16, color=INK
    )
    fig.text(
        0.012,
        0.895,
        "From a two-band ratio to a 300-million-parameter foundation model, same imagery, same labels, same split.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=INK_LIGHT,
    )
    fig.text(
        0.012,
        0.012,
        "SegFormer-B0 was not run through the event-wise bootstrap; its in-domain score is shown for context only.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=INK_LIGHT,
        style="italic",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


def _label(m: str) -> str:
    return {
        "baseline_dnbr": "ΔNBR threshold",
        "rf": "RandomForest",
        "xgb": "XGBoost",
        "unet": "U-Net",
        "segformer": "SegFormer-B0",
        "prithvi": "Prithvi-EO-2.0",
    }.get(m, m)


# --------------------------------------------------------------------------- #
# 3. split schematic, why event-wise is the only honest protocol
# --------------------------------------------------------------------------- #
def fig_split_schematic(out: Path) -> Path:
    apply_theme()
    out = Path(out)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.6))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.80, bottom=0.06, wspace=0.14)

    train_c = "#3A5F76"
    test_c = ACCENT
    rng = np.random.default_rng(7)

    for ax in (axL, axR):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8.4)
        ax.set_aspect("equal")
        ax.axis("off")

    def _panel_head(ax, title, para):
        ax.text(0, 8.25, title, ha="left", va="top", fontsize=12.5, color=INK, fontweight="bold")
        ax.text(0, 7.55, para, ha="left", va="top", fontsize=9.4, color=INK_LIGHT, linespacing=1.3)

    # ---- left: random tile split (leaky) ----
    # one fire, tiles randomly assigned train/test -> neighbours leak
    assign = rng.random((6, 6)) < 0.72
    for r in range(6):
        for c in range(6):
            col = train_c if assign[r, c] else test_c
            axL.add_patch(Rectangle((c, r), 0.94, 0.94, color=col, alpha=0.88, lw=1.0, ec=PAPER))
    hr, hc = np.argwhere(~assign)[0]  # a test cell with a train neighbour
    axL.add_patch(Rectangle((hc, hr), 0.94, 0.94, fill=False, ec=SEV_VERY_HIGH, lw=2.6))
    axL.annotate(
        "this test pixel sits 10 m from\nits training neighbours",
        xy=(hc + 0.47, hr + 0.47),
        xytext=(3.2, -0.95),
        ha="center",
        fontsize=9.2,
        color=SEV_VERY_HIGH,
        arrowprops=dict(arrowstyle="->", color=SEV_VERY_HIGH, lw=1.2),
        annotation_clip=False,
    )
    _panel_head(
        axL,
        "Random tile split, the leak",
        "Train and test tiles are drawn from the SAME fire. Same soil,\nslope, fuel and weather, so the score flatters the model.",
    )

    # ---- right: event-wise hold-out (honest) ----
    blocks = [
        (0.2, 4.0, 2.4, 2.0, train_c, "Currowan\n(train)"),
        (3.0, 4.0, 2.6, 2.0, train_c, "Gospers Mtn\n(train)"),
        (1.4, 0.7, 3.0, 2.2, test_c, "Kangaroo Is.\n(held out)"),
    ]
    for x, y, w, h, col, lab in blocks:
        axR.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                color=col,
                alpha=0.88,
                lw=0,
            )
        )
        axR.text(
            x + w / 2,
            y + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
            linespacing=1.25,
        )
    axR.annotate(
        "",
        xy=(2.9, 2.95),
        xytext=(1.9, 4.0),
        arrowprops=dict(arrowstyle="->", color=INK_LIGHT, lw=1.3, connectionstyle="arc3,rad=-0.3"),
    )
    axR.text(
        4.2,
        3.35,
        "different fires,\nhundreds of km apart",
        ha="left",
        va="center",
        fontsize=8.8,
        color=INK_LIGHT,
        style="italic",
        linespacing=1.25,
    )
    _panel_head(
        axR,
        "Event-wise hold-out, the honest test",
        "Train on two NSW fires, test on a South Australian one. Nothing\nthe model saw in training touches the test set.",
    )

    fig.suptitle(
        "The silent killer of remote-sensing benchmarks: spatial autocorrelation",
        x=0.03,
        y=0.95,
        ha="left",
        fontsize=14.5,
        color=INK,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 4. journey ribbon, a navigational map of the argument
# --------------------------------------------------------------------------- #
_STOPS = [
    ("01", "The physics", "why fire is visible from orbit"),
    ("02", "The baseline", "ΔNBR, a 1996 index that already works"),
    ("03", "The tournament", "six methods, one fire"),
    ("04", "The leak", "spatial autocorrelation, removed"),
    ("05", "The error bars", "significance that survives the pixels"),
    ("06", "The uncertainty", "where the model is guessing"),
    ("07", "The hectares", "pixels become management units"),
    ("08", "The scale", "a laptop projected onto a continent"),
    ("09", "The foundation model", "what finally clears the bar"),
]


def fig_journey(out: Path) -> Path:
    apply_theme()
    out = Path(out)
    n = len(_STOPS)
    fig, ax = plt.subplots(figsize=(13.0, 1.9))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.04)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = np.arange(n) + 0.5
    y = 0.40
    ax.plot([xs[0], xs[-1]], [y, y], color=RULE, lw=2.2, zorder=0)
    cmap = plt.get_cmap("cividis")
    for i, (num, title, _sub) in enumerate(_STOPS):
        col = cmap(0.12 + 0.72 * i / (n - 1))
        ax.scatter([xs[i]], [y], s=540, color=col, edgecolor=PAPER, linewidth=2.4, zorder=3)
        ax.text(
            xs[i],
            y,
            num,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            xs[i],
            y + 0.27,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    ax.text(
        0.012,
        0.97,
        "How to read this piece",
        ha="left",
        va="top",
        fontsize=11.5,
        color=INK_LIGHT,
        style="italic",
        transform=ax.transAxes,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out


def render_all(figdir: Path, event: str = EVENT) -> dict[str, Path]:
    figdir = Path(figdir)
    return {
        "leaderboard": fig_leaderboard(figdir / "00_leaderboard.png", event),
        "methods": fig_methods_overview(figdir / "00_methods_overview.png", event),
        "split": fig_split_schematic(figdir / "00_split_schematic.png"),
        "journey": fig_journey(figdir / "00_journey.png"),
    }


if __name__ == "__main__":
    figs = render_all(REPO_ROOT / "docs" / "figures")
    for name, path in figs.items():
        print(f"{name:12s} -> {path.relative_to(REPO_ROOT)}")
