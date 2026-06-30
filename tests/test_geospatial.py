import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.geospatial.mosaic import mosaic_tiles
from src.geospatial.overlay import management_grid, severity_by_management_unit
from src.geospatial.vectorize import raster_to_polygons, severity_area_summary

CRS = "EPSG:32753"
PX = 30.0  # metres


def _write_raster(path, arr):
    transform = from_origin(0, arr.shape[0] * PX, PX, PX)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=transform,
        nodata=255,
    ) as ds:
        ds.write(arr[np.newaxis])


def test_vectorize_area_matches_pixel_count(tmp_path):
    arr = np.zeros((10, 10), np.uint8)
    arr[:, 5:] = 1  # 50 px class0, 50 px class1
    p = tmp_path / "sev.tif"
    _write_raster(p, arr)
    gdf = raster_to_polygons(p, min_pixels=1)
    summ = severity_area_summary(gdf)
    px_ha = PX * PX / 1e4
    assert abs(summ["area_ha"].sum() - 100 * px_ha) < 1e-6
    assert abs(summ.loc[summ["class"] == 1, "area_ha"].iloc[0] - 50 * px_ha) < 1e-6


def test_overlay_intersection_area(tmp_path):
    sev = gpd.GeoDataFrame(
        {"class": [2], "class_name": ["High"]}, geometry=[box(0, 0, 100, 100)], crs=CRS
    )
    sev["area_ha"] = sev.geometry.area / 1e4
    units = gpd.GeoDataFrame({"unit_id": ["U0"]}, geometry=[box(50, 0, 150, 100)], crs=CRS)
    pieces, table = severity_by_management_unit(sev, units)
    # overlap is 50x100 = 5000 m2 = 0.5 ha
    assert abs(pieces["area_ha"].sum() - 0.5) < 1e-9
    assert abs(table.loc["U0", "High"] - 0.5) < 1e-9


def test_management_grid_covers_bounds():
    g = management_grid((0, 0, 100, 100), CRS, cell_m=50)
    assert len(g) == 4  # 2x2 cells
    assert abs(g.union_all().area - 100 * 100) < 1e-6


def test_mosaic_two_tiles(tmp_path):
    left = np.full((5, 5), 1, np.uint8)
    right = np.full((5, 5), 2, np.uint8)
    lt = from_origin(0, 5 * PX, PX, PX)
    rt = from_origin(5 * PX, 5 * PX, PX, PX)
    for name, arr, tr in (("l.tif", left, lt), ("r.tif", right, rt)):
        with rasterio.open(
            tmp_path / name,
            "w",
            driver="GTiff",
            height=5,
            width=5,
            count=1,
            dtype="uint8",
            crs=CRS,
            transform=tr,
            nodata=255,
        ) as ds:
            ds.write(arr[np.newaxis])
    out = mosaic_tiles([tmp_path / "l.tif", tmp_path / "r.tif"], tmp_path / "m.tif")
    with rasterio.open(out) as ds:
        assert (ds.height, ds.width) == (5, 10)


@pytest.mark.skipif(
    not os.environ.get("POSTGIS_DSN"), reason="set POSTGIS_DSN to test the live PostGIS round-trip"
)
def test_postgis_roundtrip():
    from sqlalchemy import text

    from src.geospatial.postgis_io import get_engine

    eng = get_engine()
    with eng.connect() as c:
        v = c.execute(text("SELECT postgis_version()")).scalar()
    assert v is not None
