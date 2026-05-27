"""Scientific-magazine matplotlib theme.

Restrained palette inspired by Quanta/Nautilus/MAAP science writing:
off-white paper background, charcoal text, thin axes, no top/right spines.
Severity palette uses muted earth tones rather than the loud
red-yellow-green you see on a Kaggle notebook.

Apply with::

    from src.viz.theme import apply_theme
    apply_theme()
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- core palette ------------------------------------------------------------
PAPER       = "#F7F5EF"   # off-white background
INK         = "#222222"   # primary text / axis
INK_LIGHT   = "#4A4A4A"   # secondary text
RULE        = "#D8D3C7"   # grid / divider
ACCENT      = "#B8553A"   # call-out colour (rust)
ACCENT_BLUE = "#3A5F76"   # secondary call-out (slate blue)

# --- severity classes (always use these, in this order) ----------------------
SEV_UNBURNT   = "#5C8A6B"   # muted forest green
SEV_LOW_MOD   = "#D8A256"   # amber
SEV_HIGH      = "#C5683B"   # orange-red
SEV_VERY_HIGH = "#7F1F1F"   # deep crimson

SEVERITY_COLOURS = [SEV_UNBURNT, SEV_LOW_MOD, SEV_HIGH, SEV_VERY_HIGH]
SEVERITY_NAMES   = ["Unburnt", "Low–Moderate", "High", "Very High"]

# Per-model colour for charts (reuse across the notebook)
MODEL_COLOURS = {
    "baseline_dnbr": "#7E7864",
    "rf":            "#3A5F76",
    "xgb":           "#5E8B7E",
    "unet":          "#B8553A",
    "segformer":     "#7F1F1F",
}


def severity_cmap() -> ListedColormap:
    return ListedColormap(SEVERITY_COLOURS, name="severity4")


def dnbr_cmap() -> LinearSegmentedColormap:
    """Diverging colormap for dNBR: blue (regrowth) → cream (no change) → crimson (severe burn)."""
    stops = [
        (0.00, "#2C5F7C"),
        (0.50, "#F2EEDD"),
        (0.66, "#D8A256"),
        (0.85, "#C5683B"),
        (1.00, "#7F1F1F"),
    ]
    return LinearSegmentedColormap.from_list("dnbr", stops)


def apply_theme() -> None:
    """Push the project's matplotlib defaults. Safe to call repeatedly."""
    mpl.rcParams.update({
        # backgrounds
        "figure.facecolor":   PAPER,
        "axes.facecolor":     PAPER,
        "savefig.facecolor":  PAPER,
        "savefig.edgecolor":  PAPER,
        # text
        "font.family":        ["Iowan Old Style", "Charter", "Georgia",
                               "Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "regular",
        "axes.titlelocation": "left",
        "axes.titlepad":      10,
        "axes.labelsize":     10,
        "axes.labelcolor":    INK,
        "axes.edgecolor":     INK,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        INK_LIGHT,
        "ytick.color":        INK_LIGHT,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        # grid
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         RULE,
        "grid.linewidth":     0.6,
        "grid.alpha":         0.7,
        # legend
        "legend.frameon":     False,
        "legend.fontsize":    9,
        # figure
        "figure.dpi":         110,
        "savefig.dpi":        180,
        "savefig.bbox":       "tight",
        # default cycle (used by line plots / bar charts)
        "axes.prop_cycle":    plt.cycler(color=[
            ACCENT_BLUE, ACCENT, "#5E8B7E", "#7E7864",
            "#D8A256", "#7F1F1F", "#3F4A50",
        ]),
    })


def figure(figsize=(8, 5), **kwargs):
    """Build a figure honouring the theme (paper bg, room for left-aligned title)."""
    apply_theme()
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def add_caption(fig, text: str, *, y: float = -0.04) -> None:
    """Add a small caption under a figure (Nautilus-style)."""
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5,
             color=INK_LIGHT, wrap=True)


def thin_axes(ax) -> None:
    """Strip an axes down to just left+bottom thin spines and y-grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8); ax.spines[s].set_color(INK)
    ax.tick_params(length=3, color=INK_LIGHT, labelsize=9)
    ax.grid(True, axis="y", color=RULE, linewidth=0.6, alpha=0.7)


def severity_legend(ax, *, loc: str = "lower right",
                    fontsize: int = 9) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLOURS]
    ax.legend(handles, SEVERITY_NAMES, loc=loc, fontsize=fontsize,
              framealpha=0.92, facecolor="white", edgecolor="none",
              borderpad=0.5)
