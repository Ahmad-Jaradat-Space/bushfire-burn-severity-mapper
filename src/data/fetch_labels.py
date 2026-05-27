"""Fetch AUS GEEBAM fire-severity labels for a given AOI.

GEEBAM is an ArcGIS REST MapServer raster. We download via the `exportImage`
operation as one or more sub-bbox tiles (server caps at ~4096 px per side),
mosaic them with rasterio.merge, then write a single GeoTIFF in EPSG:3577.

Endpoint:
  https://gis.environment.gov.au/gispubmap/rest/services/threats/
  AUS_GEEBAM_Fire_Severity/MapServer/0/exportImage

The downloaded GeoTIFF is reprojected to the AOI's UTM zone in M4 (preprocess),
not here — keeping ingest and resampling concerns separated.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pyproj
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84, load_aoi
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

GEEBAM_BASE = (
    "https://gis.environment.gov.au/gispubmap/rest/services/threats/"
    "AUS_GEEBAM_Fire_Severity/MapServer/0"
)
GEEBAM_EXPORT = f"{GEEBAM_BASE}/exportImage"
GEEBAM_REQUEST_CRS_EPSG = 3577  # GDA94 Australian Albers
GEEBAM_NATIVE_M = 40
MAX_REQUEST_PX = 4000  # stay under the 4096 server cap
HTTP_TIMEOUT = 120

log = get_logger(__name__)


def _wgs_to_3577(minx: float, miny: float, maxx: float, maxy: float) -> tuple[float, float, float, float]:
    """Reproject a WGS84 bbox to EPSG:3577 (GDA94 Australian Albers) metres."""
    tx = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)
    # Densify corners to handle the Albers curvature
    xs = []
    ys = []
    for x, y in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]:
        u, v = tx.transform(x, y)
        xs.append(u); ys.append(v)
    return (min(xs), min(ys), max(xs), max(ys))


def _request_tiles(bbox_m: tuple[float, float, float, float],
                   pixel_m: int = GEEBAM_NATIVE_M,
                   max_px: int = MAX_REQUEST_PX) -> Iterator[tuple[float, float, float, float, int, int]]:
    """Yield (minx, miny, maxx, maxy, width_px, height_px) for each request tile."""
    minx, miny, maxx, maxy = bbox_m
    max_extent_m = max_px * pixel_m
    n_x = max(1, math.ceil((maxx - minx) / max_extent_m))
    n_y = max(1, math.ceil((maxy - miny) / max_extent_m))
    step_x = (maxx - minx) / n_x
    step_y = (maxy - miny) / n_y
    for i in range(n_x):
        for j in range(n_y):
            tile_minx = minx + i * step_x
            tile_maxx = minx + (i + 1) * step_x
            tile_miny = miny + j * step_y
            tile_maxy = miny + (j + 1) * step_y
            w = max(1, round((tile_maxx - tile_minx) / pixel_m))
            h = max(1, round((tile_maxy - tile_miny) / pixel_m))
            yield (tile_minx, tile_miny, tile_maxx, tile_maxy, w, h)


def _fetch_tile(bbox: tuple[float, float, float, float], width: int, height: int,
                out_path: Path, session: requests.Session) -> Path:
    """POST exportImage for one tile and save the returned GeoTIFF.

    The server returns binary TIFF when `f=image`. We write it and then re-tag
    the CRS/transform locally because the server's embedded geotags are
    sometimes thin.
    """
    params = {
        "bbox": ",".join(f"{v:.3f}" for v in bbox),
        "bboxSR": str(GEEBAM_REQUEST_CRS_EPSG),
        "imageSR": str(GEEBAM_REQUEST_CRS_EPSG),
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "U8",
        "f": "image",
        "interpolation": "RSP_NearestNeighbor",
    }
    r = session.get(GEEBAM_EXPORT, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    out_path.write_bytes(r.content)

    # Re-tag CRS + transform from the bbox we requested (defensive).
    transform = from_bounds(*bbox, width=width, height=height)
    with rasterio.open(out_path, "r+") as ds:
        ds.crs = f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}"
        ds.transform = transform
    return out_path


def fetch_geebam(event_id: str, out_dir: Path | None = None,
                 dry_run: bool = False) -> Path:
    """Download GEEBAM for the AOI and write a mosaicked GeoTIFF in EPSG:3577."""
    feat = load_aoi(event_id)
    bbox_wgs = aoi_bbox_wgs84(event_id)
    bbox_3577 = _wgs_to_3577(*bbox_wgs)
    log.info("AOI %s: WGS84 bbox=%s -> EPSG:3577 bbox=%s", event_id, bbox_wgs, bbox_3577)

    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "labels" / "aus_geebam" / event_id
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "label_native_3577.tif"

    tiles = list(_request_tiles(bbox_3577))
    log.info("Will request %d sub-tile(s) at %d m native resolution.", len(tiles), GEEBAM_NATIVE_M)

    if dry_run:
        log.warning("dry_run=True — skipping HTTP download.")
        write_manifest(
            out_path,
            event_id=event_id,
            pipeline_step="fetch_labels.geebam",
            inputs={
                "service": GEEBAM_BASE,
                "operation": "exportImage",
                "request_crs": f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
                "native_pixel_m": GEEBAM_NATIVE_M,
                "bbox_wgs84": bbox_wgs,
                "bbox_3577": bbox_3577,
                "n_request_tiles": len(tiles),
                "licence": "CC-BY 4.0",
                "attribution": "AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0.",
            },
            crs=f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
            resampling="nearest",
            notes="DRY RUN — manifest only; no raster downloaded.",
        )
        return out_path

    session = requests.Session()
    tile_paths: list[Path] = []
    for idx, (minx, miny, maxx, maxy, w, h) in enumerate(tiles):
        tp = tile_dir / f"tile_{idx:03d}.tif"
        log.info("  tile %d/%d: bbox=%s size=%dx%d -> %s",
                 idx + 1, len(tiles), (minx, miny, maxx, maxy), w, h, tp.name)
        _fetch_tile((minx, miny, maxx, maxy), w, h, tp, session)
        tile_paths.append(tp)

    if len(tile_paths) == 1:
        tile_paths[0].rename(out_path)
    else:
        srcs = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": 1,
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic[0:1])
        for s in srcs:
            s.close()
        for p in tile_paths:
            p.unlink(missing_ok=True)

    # Sanity check: GEEBAM is U8 with classes in {1,2,3,4,5} (+0 nodata).
    with rasterio.open(out_path) as ds:
        arr = ds.read(1)
    uniq, counts = np.unique(arr, return_counts=True)
    hist = dict(zip(uniq.tolist(), counts.tolist()))
    log.info("Class histogram: %s", hist)
    if not set(hist.keys()) & {2, 3, 4, 5}:
        log.warning("No GEEBAM severity classes (2-5) found inside the AOI. "
                    "Either the AOI does not overlap GEEBAM coverage, or the request CRS is wrong.")

    write_manifest(
        out_path,
        event_id=event_id,
        pipeline_step="fetch_labels.geebam",
        inputs={
            "service": GEEBAM_BASE,
            "operation": "exportImage",
            "request_crs": f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
            "native_pixel_m": GEEBAM_NATIVE_M,
            "bbox_wgs84": bbox_wgs,
            "bbox_3577": bbox_3577,
            "n_request_tiles": len(tiles),
            "class_histogram": hist,
            "licence": "CC-BY 4.0",
            "attribution": "AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0.",
        },
        crs=f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
        resampling="nearest",
        class_remap=None,
        notes=None,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AUS GEEBAM labels for an AOI.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan the request and write the manifest, but don't fetch.")
    args = parser.parse_args()
    fetch_geebam(args.event, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
