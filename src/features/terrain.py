"""Fetch the Copernicus GLO-30 DEM for an AOI and derive a slope raster.

Terrain is a confounder: dNBR and the learned models can mistake topographic
shadow for burn, and severity itself varies with slope/aspect. Until this runs,
the 18th channel of the feature stack (slope) is a zero plane and
:mod:`src.evaluation.stratified_reports` cannot stratify by slope. This module
materialises ``data/interim/<event>/slope_10m.tif`` on the same grid as the
reflectance composites (despite the ``_10m`` legacy suffix, the working grid is
30 m — a native match for GLO-30).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "cop-dem-glo-30"


def compute_slope(dem: np.ndarray, res_m: float = 30.0) -> np.ndarray:
    """Slope in degrees from an elevation array on a `res_m` grid."""
    gy, gx = np.gradient(dem.astype(np.float64), res_m, res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def fetch_copernicus_dem(event_id: str, out_dir: Path | None = None) -> Path:
    """Download GLO-30, align to the event grid, and write slope_10m.tif."""
    import odc.stac
    import planetary_computer
    import pystac_client

    interim = out_dir or (REPO_ROOT / "data" / "interim" / event_id)
    target_path = interim / "pre_stack_10m.tif"
    with rasterio.open(target_path) as ds:
        crs = ds.crs
        transform = ds.transform
        H, W = ds.height, ds.width
        res_m = float(abs(ds.res[0]))

    bbox = aoi_bbox_wgs84(event_id)
    log.info("Searching %s over %s …", DEM_COLLECTION, [round(b, 3) for b in bbox])
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=[DEM_COLLECTION], bbox=bbox).items())
    if not items:
        raise SystemExit(f"No {DEM_COLLECTION} items for {event_id}")
    log.info(
        "Loading %d DEM tile(s) onto the event grid (%dx%d, %s, %.0f m).",
        len(items),
        W,
        H,
        crs,
        res_m,
    )

    da = odc.stac.load(
        items,
        bands=["data"],
        crs=crs,
        resolution=res_m,
        x=(transform.c, transform.c + W * res_m),
        y=(transform.f, transform.f - H * res_m),
        chunks={},
    )["data"].squeeze()
    dem = np.asarray(da.values, dtype=np.float32)
    # Align to the exact target shape (odc may be ±1 px).
    dem = dem[:H, :W]
    if dem.shape != (H, W):
        pad = np.full((H, W), np.nan, np.float32)
        pad[: dem.shape[0], : dem.shape[1]] = dem
        dem = pad
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))

    slope = compute_slope(dem, res_m=res_m)
    out_path = interim / "slope_10m.tif"
    meta = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(slope[np.newaxis])
    write_manifest(
        out_path,
        event_id=event_id,
        pipeline_step="terrain.slope",
        inputs={
            "collection": DEM_COLLECTION,
            "n_items": len(items),
            "service": STAC_URL,
            "resolution_m": res_m,
            "licence": "Copernicus DEM — free and open (ESA/EC)",
            "slope_units": "degrees",
        },
        crs=str(crs),
        resampling="bilinear",
        notes="Slope = atan(|grad(elevation)|); fills the formerly-zero stack channel.",
    )
    log.info(
        "Wrote %s  (slope deg: min=%.1f mean=%.1f max=%.1f)",
        out_path.relative_to(REPO_ROOT),
        slope.min(),
        slope.mean(),
        slope.max(),
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default="kangaroo_island_2019_2020")
    args = p.parse_args()
    fetch_copernicus_dem(args.event)


if __name__ == "__main__":
    main()
