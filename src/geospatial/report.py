"""Phase D reporting layer: pixels → polygons → management-unit roll-up.

Vectorises the event-wise U-Net severity map, overlays it on a management-unit
grid, and renders the two figures the notebook embeds. Optionally pushes the
layers into PostGIS and runs the worked spatial SQL.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.geospatial.overlay import management_grid, severity_by_management_unit
from src.geospatial.vectorize import raster_to_polygons, severity_area_summary
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.theme import SEVERITY_COLOURS, SEVERITY_NAMES, apply_theme, thin_axes

log = get_logger(__name__)

EVENT = "kangaroo_island_2019_2020"
DEFAULT_PRED = "outputs/predictions/unet_eventwise/{event}.tif"
_CLASS_COLOUR = dict(zip(SEVERITY_NAMES, SEVERITY_COLOURS, strict=False))


def compute_vector_report(
    event: str = EVENT, pred_path: str | None = None, cell_m: float = 5000.0
) -> dict:
    pred = REPO_ROOT / (pred_path or DEFAULT_PRED.format(event=event))
    label = REPO_ROOT / "data" / "interim" / event / "label_10m.tif"
    gdf = raster_to_polygons(
        pred, min_pixels=4, dissolve=True, valid_mask_path=label if label.exists() else None
    )
    summary = severity_area_summary(gdf)
    units = management_grid(tuple(gdf.total_bounds), gdf.crs, cell_m=cell_m)
    pieces, unit_table = severity_by_management_unit(gdf, units)
    log.info(
        "vectorised %d class polygons; %d management units; burnt classes=%s",
        len(gdf),
        len(units),
        [c for c in unit_table.columns if c != "burnt_ha"],
    )
    return {
        "event": event,
        "severity_gdf": gdf,
        "summary": summary,
        "units": units,
        "pieces": pieces,
        "unit_table": unit_table,
    }


def fig_severity_map(report: dict, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    gdf = report["severity_gdf"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in gdf.iterrows():
        colour = _CLASS_COLOUR.get(row["class_name"], "#999999")
        gdf[gdf["class"] == row["class"]].plot(ax=ax, color=colour, linewidth=0)
    report["units"].boundary.plot(ax=ax, color="#4A4A4A", linewidth=0.4, alpha=0.5)
    ax.set_axis_off()
    ax.set_title(
        "Severity as vectors, over a 5 km management grid (QGIS-ready)", loc="left", fontsize=12
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEVERITY_COLOURS]
    ax.legend(
        handles,
        SEVERITY_NAMES,
        loc="lower right",
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_unit_areas(report: dict, out: Path, top_n: int = 12) -> Path:
    apply_theme()
    out = Path(out)
    tab = report["unit_table"].head(top_n)
    # focus the bars on burnt severity (drop Unburnt; rank already by severe area)
    classes = [c for c in ["Low–Moderate", "High", "Very High"] if c in tab.columns]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab))
    for cls in classes:
        ax.bar(x, tab[cls].values, bottom=bottom, label=cls, color=_CLASS_COLOUR.get(cls, "#999"))
        bottom += tab[cls].values
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Area (hectares)")
    ax.set_title(
        f"Burnt hectares by severity for the {top_n} hardest-hit management units",
        loc="left",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncols=4, loc="upper right")
    thin_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(report: dict, figdir: Path) -> dict[str, Path]:
    figdir = Path(figdir)
    return {
        "map": fig_severity_map(report, figdir / "12_qgis_map.png"),
        "units": fig_unit_areas(report, figdir / "13_mgmt_unit_areas.png"),
    }


def load_to_postgis(report: dict, dsn: str | None = None) -> object:
    """Push severity + units to PostGIS and run the worked queries. Returns a
    dict of query-result DataFrames. Raises if no database is reachable."""
    from src.geospatial.postgis_io import get_engine, run_sql, write_gdf

    engine = get_engine(dsn)
    sev = report["severity_gdf"][["class", "class_name", "area_ha", "geometry"]].rename_geometry(
        "geom"
    )
    units = report["units"].rename_geometry("geom")
    write_gdf(sev, "severity_polygons", engine)
    write_gdf(units, "management_units", engine)
    raw = (REPO_ROOT / "sql" / "queries.sql").read_text()
    results = {}
    for i, stmt in enumerate(raw.split(";")):
        # drop comment-only lines so the SELECT is detectable
        body = "\n".join(ln for ln in stmt.splitlines() if not ln.strip().startswith("--")).strip()
        if body.lower().startswith("select"):
            try:
                results[f"q{i + 1}"] = run_sql(engine, body)
            except Exception as exc:  # one bad query shouldn't sink the rest
                log.warning("query q%d failed: %s", i + 1, str(exc).splitlines()[0])
    return results
