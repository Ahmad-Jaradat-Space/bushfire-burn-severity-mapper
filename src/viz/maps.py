"""Static map rendering for README panels.

`render_severity_map` and `render_truecolor` are used by the demo assets and
notebook. `render_aoi_locator` builds the continental-Australia locator using
cartopy + Natural Earth coastlines so the map is a real cartographic product
rather than a stand-in rectangle.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.viz.theme import (INK, INK_LIGHT, PAPER, RULE, apply_theme, thin_axes,
                           SEVERITY_COLOURS, SEVERITY_NAMES)
from src.viz.theme import severity_cmap as _theme_sev_cmap

# Compatibility aliases (the older callers expect these names)
SEVERITY_COLORS = SEVERITY_COLOURS


def severity_cmap() -> ListedColormap:
    return _theme_sev_cmap()


def render_severity_map(arr: np.ndarray, out_path: Path, title: str = "",
                        figsize: tuple[int, int] = (8, 6)) -> Path:
    apply_theme()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_equal(arr, 255)
    ax.imshow(masked, cmap=severity_cmap(), vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLOURS]
    ax.legend(handles, SEVERITY_NAMES, loc="lower right", fontsize=9,
              framealpha=0.92, facecolor="white", edgecolor="none")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out_path


def render_truecolor(rgb_or_stack: np.ndarray, out_path: Path, title: str = "",
                     figsize: tuple[int, int] = (8, 6), pct=(2, 98)) -> Path:
    """Render either a [3, H, W] RGB array or a [6, H, W] reflectance stack."""
    from src.viz.synthetic_scene import s2_truecolour
    apply_theme()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = rgb_or_stack.astype(np.float32)
    if arr.shape[0] >= 6:
        img = s2_truecolour(arr)
    else:
        # Joint linear stretch over the 3-channel array
        img = np.transpose(arr[:3], (1, 2, 0))
        img = np.clip(img * 3.5, 0, 1)
        img = np.power(img, 0.85)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out_path


def render_aoi_locator(aois: list[tuple], out_path: Path,
                       figsize: tuple[float, float] = (10.5, 8.0)) -> Path:
    """Continental Australia locator with real coastline + 4 fire-event AOIs.

    `aois` is a list of (event_id, label, state, split, colour) tuples.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from src.utils.geo import aoi_bbox_wgs84
    from matplotlib.patches import Rectangle

    apply_theme()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent([112, 154, -44, -10], crs=proj)
    ax.set_facecolor(PAPER)

    # Land + coastline + state boundaries
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#EFEAD9", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#DCE6EA", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#5C5147", linewidth=0.7)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#B0A48C", linewidth=0.5, alpha=0.7)

    # AOI patches + offset annotations to prevent overlap
    annotation_offsets = {
        "kangaroo_island_2019_2020":   (-12, -3.5),
        "currowan_2019_2020":          (4, 0.5),
        "gospers_mountain_2019_2020":  (4, 2.5),
        "east_gippsland_2019_2020":    (-1, -5),
    }
    for event_id, label, state, split, colour in aois:
        minx, miny, maxx, maxy = aoi_bbox_wgs84(event_id)
        ax.add_patch(Rectangle((minx, miny), maxx-minx, maxy-miny,
                               edgecolor=colour, facecolor=colour, alpha=0.35,
                               linewidth=1.5, transform=proj, zorder=5))
        cx, cy = (minx+maxx)/2, (miny+maxy)/2
        dx, dy = annotation_offsets.get(event_id, (4, 2.5))
        ax.annotate(f"{label}\n{state} · {split}",
                    xy=(cx, cy), xytext=(cx + dx, cy + dy),
                    fontsize=10.5, color=INK, ha="left",
                    transform=proj,
                    arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.9),
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=colour,
                              alpha=0.95, lw=0.7),
                    zorder=6)
        ax.scatter([cx], [cy], s=55, c=colour, edgecolor="white",
                   linewidth=1.4, zorder=7, transform=proj)

    # Gridlines (light, just a few)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=RULE, alpha=0.7,
                      xlocs=[120, 130, 140, 150], ylocs=[-40, -30, -20, -15])
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 9, "color": INK_LIGHT}
    gl.ylabel_style = {"size": 9, "color": INK_LIGHT}

    ax.set_title("Four areas of interest, spanning three states and three ecological zones",
                 loc="left", fontsize=13, pad=14, color=INK, fontweight="regular")

    # Legend chips for train/val/test
    handles = []
    seen = set()
    for _, _, _, split, c in aois:
        if split not in seen:
            handles.append((split, c)); seen.add(split)
    chips = [plt.Line2D([0], [0], marker="s", color=c, markersize=11,
                         linestyle="None", label=split.upper())
             for split, c in handles]
    ax.legend(handles=chips, loc="lower left", fontsize=9.5,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return out_path
