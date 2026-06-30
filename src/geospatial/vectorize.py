"""Raster severity map → vector polygons + area summaries.

The deep models emit rasters, but ecology clients work in vectors: a severity
polygon layer is what gets dropped into QGIS, intersected with management units,
and rolled up into a hectares-per-class table for a report. This module turns a
prediction GeoTIFF into a dissolved, area-attributed ``GeoDataFrame`` in the
event's UTM CRS (so areas are metric and honest).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

SEVERITY_NAMES = {0: "Unburnt", 1: "Low–Moderate", 2: "High", 3: "Very High"}


def raster_to_polygons(
    pred_path: str | Path,
    ignore_index: int = 255,
    min_pixels: int = 4,
    dissolve: bool = True,
    valid_mask_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Polygonise a uint8 severity raster into a GeoDataFrame (UTM CRS).

    Connected regions of equal class become polygons; tiny specks below
    ``min_pixels`` are dropped; with ``dissolve`` the result is one multipolygon
    per class. ``valid_mask_path`` (e.g. the GEEBAM label) restricts output to
    its non-ignore footprint, the honest way to drop ocean / unassessed pixels
    the model still labels.
    """
    with rasterio.open(pred_path) as ds:
        arr = ds.read(1)
        transform = ds.transform
        crs = ds.crs
    px_area = abs(transform.a * transform.e)

    mask = arr != ignore_index
    if valid_mask_path is not None:
        with rasterio.open(valid_mask_path) as ds:
            mask &= ds.read(1) != ignore_index
    geoms, vals = [], []
    for geom, val in shapes(arr, mask=mask, transform=transform):
        geoms.append(shape(geom))
        vals.append(int(val))
    gdf = gpd.GeoDataFrame({"class": vals}, geometry=geoms, crs=crs)
    gdf = gdf[gdf.geometry.area >= min_pixels * px_area].reset_index(drop=True)
    if dissolve:
        gdf = gdf.dissolve(by="class", as_index=False)
    gdf["class_name"] = gdf["class"].map(SEVERITY_NAMES)
    gdf["area_ha"] = gdf.geometry.area / 1e4
    return gdf


def severity_area_summary(gdf: gpd.GeoDataFrame) -> gpd.pd.DataFrame:
    """Hectares and percentage per severity class."""
    g = gdf.groupby(["class", "class_name"], as_index=False)["area_ha"].sum()
    total = g["area_ha"].sum() or 1.0
    g["pct"] = 100.0 * g["area_ha"] / total
    return g.sort_values("class").reset_index(drop=True)
