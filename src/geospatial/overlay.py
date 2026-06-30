"""Spatial overlay: severity polygons × management units → per-unit hectares.

The operational question an ecology consultancy asks is not "how many pixels
burned" but "how much High-severity fire fell inside *each management unit /
bioregion / tenure parcel*". That is a vector overlay (``ST_Intersection`` in
PostGIS, ``geopandas.overlay`` here) followed by a zonal area roll-up.

``management_grid`` builds a tessellation of square units over an AOI when no
external parcel/bioregion layer is supplied — enough to demonstrate the join.
Swap it for a clipped IBRA bioregion or tenure layer in production.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def management_grid(
    bounds: tuple[float, float, float, float], crs, cell_m: float = 5000.0
) -> gpd.GeoDataFrame:
    """Square management units covering `bounds` (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    cells, ids = [], []
    n = 0
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cells.append(box(x, y, min(x + cell_m, maxx), min(y + cell_m, maxy)))
            ids.append(f"U{n:04d}")
            n += 1
            x += cell_m
        y += cell_m
    return gpd.GeoDataFrame({"unit_id": ids}, geometry=cells, crs=crs)


def severity_by_management_unit(
    severity_gdf: gpd.GeoDataFrame,
    units_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Intersect severity polygons with units; return (pieces, per-unit table)."""
    if severity_gdf.crs != units_gdf.crs:
        units_gdf = units_gdf.to_crs(severity_gdf.crs)
    pieces = gpd.overlay(severity_gdf, units_gdf, how="intersection")
    pieces["area_ha"] = pieces.geometry.area / 1e4
    table = (
        pieces.groupby(["unit_id", "class_name"], as_index=False)["area_ha"]
        .sum()
        .pivot(index="unit_id", columns="class_name", values="area_ha")
        .fillna(0.0)
    )
    # rank units by *severe* (High + Very High) burnt area
    burnt_cols = [c for c in table.columns if c != "Unburnt"]
    table["burnt_ha"] = table[burnt_cols].sum(axis=1) if burnt_cols else 0.0
    severe_cols = [c for c in ["High", "Very High"] if c in table.columns]
    table["severe_ha"] = table[severe_cols].sum(axis=1) if severe_cols else 0.0
    return pieces, table.sort_values("severe_ha", ascending=False)
