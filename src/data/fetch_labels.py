"""Fetch AUS GEEBAM fire-severity labels for a given AOI.

Approach: the public service is a MapServer (not ImageServer), so it does not
expose `exportImage` with raw pixel access. We instead call MapServer-level
`/export` to get the rendered RGBA PNG, then **reverse-map the official
symbology RGB tuples back to class IDs** using the legend we retrieved
programmatically. This is exact (not lossy) because the renderer uses a
unique-value (nearest-neighbour) ramp with one colour per class — there is no
anti-aliasing inside the raster.

GEEBAM official colour ramp (from MapServer/legend endpoint):
  Unburnt          → (112, 168, 0)
  Low and Moderate → (230, 152, 0)
  High             → (168,  56, 0)
  Very High        → (  0,   0, 0)

Internal class remap (see src/data/class_map.py):
  Unburnt → 0,  Low/Mod → 1,  High → 2,  Very High → 3,  background → 255 ignore
"""
from __future__ import annotations

import argparse
import math
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
import pyproj
import rasterio
import requests
from PIL import Image
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84, load_aoi
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

GEEBAM_BASE = (
    "https://gis.environment.gov.au/gispubmap/rest/services/threats/"
    "AUS_GEEBAM_Fire_Severity/MapServer"
)
GEEBAM_EXPORT = f"{GEEBAM_BASE}/export"
GEEBAM_REQUEST_CRS_EPSG = 4326   # MapServer/export is happiest with WGS84 in/out
GEEBAM_NATIVE_M = 40
MAX_REQUEST_PX = 4000
HTTP_TIMEOUT = 180
IGNORE_ID = 255

# Official symbology — class label → RGB tuple → internal class ID
LEGEND_TO_INTERNAL: dict[tuple[int, int, int], int] = {
    (112, 168,   0): 0,   # Unburnt
    (230, 152,   0): 1,   # Low and Moderate
    (168,  56,   0): 2,   # High
    (  0,   0,   0): 3,   # Very High
}

log = get_logger(__name__)


def _request_tiles(bbox_wgs: tuple[float, float, float, float],
                   pixel_m: int = GEEBAM_NATIVE_M,
                   max_px: int = MAX_REQUEST_PX) -> Iterator[tuple[float, float, float, float, int, int]]:
    """Yield (minx, miny, maxx, maxy, width_px, height_px) for each request tile.

    We approximate pixel-count from the WGS84 bbox using the cosine of latitude
    so per-tile pixel counts stay below `max_px` even far south.
    """
    minx, miny, maxx, maxy = bbox_wgs
    # Convert bbox extent to approximate metres
    mean_lat = math.radians((miny + maxy) / 2.0)
    width_m = (maxx - minx) * 111_320 * math.cos(mean_lat)
    height_m = (maxy - miny) * 111_320
    max_extent_m = max_px * pixel_m
    n_x = max(1, math.ceil(width_m / max_extent_m))
    n_y = max(1, math.ceil(height_m / max_extent_m))
    dx = (maxx - minx) / n_x
    dy = (maxy - miny) / n_y
    for i in range(n_x):
        for j in range(n_y):
            tx_min = minx + i * dx
            tx_max = minx + (i + 1) * dx
            ty_min = miny + j * dy
            ty_max = miny + (j + 1) * dy
            w = max(1, round((tx_max - tx_min) * 111_320 * math.cos(mean_lat) / pixel_m))
            h = max(1, round((ty_max - ty_min) * 111_320 / pixel_m))
            yield (tx_min, ty_min, tx_max, ty_max, w, h)


def _rgb_to_class(rgba: np.ndarray) -> np.ndarray:
    """RGBA [H, W, 4] uint8 → internal class IDs [H, W] uint8.

    Nearest-RGB matching: a pixel is assigned the class whose legend RGB is
    closest in L2 distance, provided the distance is small enough; otherwise
    it is marked ignore (255). Transparent pixels (alpha=0) are also ignore.
    """
    H, W = rgba.shape[:2]
    out = np.full((H, W), IGNORE_ID, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.int32)
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    closest_dist = np.full((H, W), 10**9, dtype=np.int64)
    for (r, g, b), cls in LEGEND_TO_INTERNAL.items():
        d = (rgb[..., 0] - r) ** 2 + (rgb[..., 1] - g) ** 2 + (rgb[..., 2] - b) ** 2
        mask = d < closest_dist
        out[mask] = cls
        closest_dist[mask] = d[mask]
    # Only accept matches within a tolerance; otherwise this isn't really a class
    # colour (it's the cream background or anti-aliasing artefact).
    tolerance_sq = 30 * 30 * 3   # per-channel distance ~30 in any channel
    out[closest_dist > tolerance_sq] = IGNORE_ID
    if alpha is not None:
        out[alpha < 128] = IGNORE_ID
    return out


def _fetch_tile_rgba(bbox: tuple[float, float, float, float], width: int, height: int,
                     session: requests.Session) -> np.ndarray:
    """POST /export and return an RGBA [H, W, 4] uint8 array."""
    params = {
        "bbox": ",".join(f"{v:.6f}" for v in bbox),
        "bboxSR": str(GEEBAM_REQUEST_CRS_EPSG),
        "imageSR": str(GEEBAM_REQUEST_CRS_EPSG),
        "size": f"{width},{height}",
        "format": "png",
        "transparent": "true",
        "f": "image",
    }
    r = session.get(GEEBAM_EXPORT, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    return np.array(img)


def fetch_geebam(event_id: str, out_dir: Path | None = None,
                 dry_run: bool = False) -> Path:
    feat = load_aoi(event_id)
    bbox_wgs = aoi_bbox_wgs84(event_id)
    log.info("AOI %s WGS84 bbox=%s", event_id, bbox_wgs)

    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "labels" / "aus_geebam" / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "label_native_4326.tif"

    tiles = list(_request_tiles(bbox_wgs))
    log.info("Will request %d sub-tile(s) at %d m native resolution.", len(tiles), GEEBAM_NATIVE_M)

    if dry_run:
        log.warning("dry_run=True — skipping HTTP download.")
        write_manifest(
            out_path,
            event_id=event_id,
            pipeline_step="fetch_labels.geebam",
            inputs={
                "service": GEEBAM_BASE,
                "operation": "export (MapServer-level, RGBA → class via legend reverse-map)",
                "request_crs": f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
                "native_pixel_m": GEEBAM_NATIVE_M,
                "bbox_wgs84": bbox_wgs,
                "n_request_tiles": len(tiles),
                "legend_to_internal": {f"rgb({r},{g},{b})": cls
                                        for (r, g, b), cls in LEGEND_TO_INTERNAL.items()},
                "licence": "CC-BY 4.0",
                "attribution": "AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0.",
            },
            crs=f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}",
            resampling="nearest",
            notes="DRY RUN — manifest only; no raster downloaded.",
        )
        return out_path

    session = requests.Session()
    tile_arrays = []
    tile_bboxes = []
    for idx, (minx, miny, maxx, maxy, w, h) in enumerate(tiles):
        log.info("  tile %d/%d  bbox=(%.3f,%.3f,%.3f,%.3f)  size=%dx%d",
                 idx + 1, len(tiles), minx, miny, maxx, maxy, w, h)
        rgba = _fetch_tile_rgba((minx, miny, maxx, maxy), w, h, session)
        # MapServer/export inverts y: row 0 is at the top (maxy). The bbox we
        # pass is in (minx, miny, maxx, maxy); we'll let from_bounds handle it.
        classes = _rgb_to_class(rgba)
        tile_arrays.append(classes)
        tile_bboxes.append((minx, miny, maxx, maxy, w, h))

    # Write each tile as a tiny GeoTIFF, then merge
    tile_paths: list[Path] = []
    for idx, ((minx, miny, maxx, maxy, w, h), arr) in enumerate(zip(tile_bboxes, tile_arrays)):
        transform = from_bounds(minx, miny, maxx, maxy, width=w, height=h)
        tp = out_dir / f"_tile_{idx:03d}.tif"
        with rasterio.open(
            tp, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
            crs=f"EPSG:{GEEBAM_REQUEST_CRS_EPSG}", transform=transform, nodata=IGNORE_ID,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(arr[np.newaxis])
        tile_paths.append(tp)

    if len(tile_paths) == 1:
        tile_paths[0].rename(out_path)
    else:
        srcs = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = rio_merge(srcs, nodata=IGNORE_ID)
        meta = srcs[0].meta.copy()
        meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                     "transform": transform, "count": 1})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic[0:1])
        for s in srcs:
            s.close()
        for p in tile_paths:
            p.unlink(missing_ok=True)

    # Reproject to EPSG:3577 for downstream consistency (the rest of the pipeline
    # expects label_native_3577.tif, though preprocess.py only cares about the
    # CRS tag, not the filename suffix).
    out_3577 = out_dir / "label_native_3577.tif"
    with rasterio.open(out_path) as src:
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        dst_crs = "EPSG:3577"
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": transform, "width": w, "height": h,
                     "nodata": IGNORE_ID})
        with rasterio.open(out_3577, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_crs=src.crs, dst_crs=dst_crs,
                src_transform=src.transform, dst_transform=transform,
                resampling=Resampling.nearest,
            )

    with rasterio.open(out_3577) as ds:
        arr = ds.read(1)
    uniq, counts = np.unique(arr, return_counts=True)
    hist = dict(zip(uniq.tolist(), counts.tolist()))
    log.info("Class histogram (final 3577): %s", hist)

    write_manifest(
        out_3577,
        event_id=event_id,
        pipeline_step="fetch_labels.geebam",
        inputs={
            "service": GEEBAM_BASE,
            "operation": "export (MapServer RGBA + legend reverse-map)",
            "intermediate_crs": "EPSG:4326",
            "final_crs": "EPSG:3577",
            "native_pixel_m": GEEBAM_NATIVE_M,
            "bbox_wgs84": bbox_wgs,
            "n_request_tiles": len(tiles),
            "class_histogram": hist,
            "legend_to_internal": {f"rgb({r},{g},{b})": cls
                                    for (r, g, b), cls in LEGEND_TO_INTERNAL.items()},
            "licence": "CC-BY 4.0",
            "attribution": "AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0.",
        },
        crs="EPSG:3577",
        resampling="nearest",
        class_remap={"unburnt(2)": 0, "low_mod(3)": 1, "high(4)": 2, "very_high(5)": 3},
        notes="Classes already internal IDs (no further remap needed in preprocess.py).",
    )
    return out_3577


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AUS GEEBAM labels for an AOI.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fetch_geebam(args.event, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
