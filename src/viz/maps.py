"""Static map rendering for README panels.

All figures use matplotlib only (no seaborn) and assume the working CRS is
already correct on the input rasters.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 4-class severity palette + ignore. Palette colours match standard burn-severity
# legends in the literature: green=unburnt, yellow=low-mod, orange=high, red=very_high.
SEVERITY_COLORS = ["#2e7d32", "#fdd835", "#ef6c00", "#c62828"]
SEVERITY_NAMES = ["Unburnt", "Low-Moderate", "High", "Very High"]


def severity_cmap() -> ListedColormap:
    return ListedColormap(SEVERITY_COLORS, name="severity4")


def render_severity_map(arr: np.ndarray, out_path: Path, title: str = "",
                        figsize: tuple[int, int] = (8, 6)) -> Path:
    """Render a uint8 4-class severity raster (with 255=ignore) to PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_equal(arr, 255)
    ax.imshow(masked, cmap=severity_cmap(), vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    # Legend chips
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLORS]
    ax.legend(handles, SEVERITY_NAMES, loc="lower right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_truecolor(rgb: np.ndarray, out_path: Path, title: str = "",
                     figsize: tuple[int, int] = (8, 6), pct=(2, 98)) -> Path:
    """Render an RGB true-colour composite (3, H, W) reflectance array."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.nanpercentile(rgb[c], pct)
        if hi > lo:
            out[c] = np.clip((rgb[c] - lo) / (hi - lo), 0, 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.transpose(out, (1, 2, 0)))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_comparison_panel(panels: list[tuple[str, np.ndarray, str]], out_path: Path,
                            ncols: int = 3, figsize: tuple[int, int] | None = None) -> Path:
    """Lay out a grid of (title, array, kind) panels where kind in {'severity','rgb'}."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    nrows = -(-n // ncols)
    figsize = figsize or (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for i, (title, arr, kind) in enumerate(panels):
        ax = axes[i // ncols, i % ncols]
        if kind == "severity":
            ax.imshow(np.ma.masked_equal(arr, 255), cmap=severity_cmap(), vmin=0, vmax=3,
                      interpolation="nearest")
        elif kind == "rgb":
            rgb = arr.astype(np.float32)
            out = np.zeros_like(rgb)
            for c in range(3):
                lo, hi = np.nanpercentile(rgb[c], (2, 98))
                if hi > lo:
                    out[c] = np.clip((rgb[c] - lo) / (hi - lo), 0, 1)
            ax.imshow(np.transpose(out, (1, 2, 0)))
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
