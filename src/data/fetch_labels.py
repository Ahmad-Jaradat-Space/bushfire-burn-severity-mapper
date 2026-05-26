"""Fetch AUS GEEBAM fire-severity labels for a given AOI.

GEEBAM is served as an ArcGIS REST MapServer layer. We use the `exportImage`
operation to download the AOI bounding box as a single-band GeoTIFF in
EPSG:3577 (GDA94 Australian Albers — the native distribution CRS), then
reproject to the AOI's UTM zone in M4.

Endpoint:
  https://gis.environment.gov.au/gispubmap/rest/services/threats/AUS_GEEBAM_Fire_Severity/MapServer/0/exportImage

Implementation notes:
- ArcGIS exportImage caps a single request at ~4096×4096 px. Large AOIs
  (Gospers, East Gippsland) require request-tiling — see `_request_tiles`.
- The native pixel size is 40 m; we keep that on download and resample to 10 m
  on the S2 grid only in M4 (preprocess), with nearest-neighbour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84, load_aoi
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

GEEBAM_BASE = (
    "https://gis.environment.gov.au/gispubmap/rest/services/threats/"
    "AUS_GEEBAM_Fire_Severity/MapServer/0"
)
GEEBAM_EXPORT = f"{GEEBAM_BASE}/exportImage"
GEEBAM_REQUEST_CRS = "EPSG:3577"  # GDA94 Australian Albers — native
GEEBAM_NATIVE_M = 40
MAX_REQUEST_PX = 4096

log = get_logger(__name__)


def _request_tiles(bbox_3577: tuple[float, float, float, float], pixel_m: int = GEEBAM_NATIVE_M):
    """Yield sub-bbox tiles each <= MAX_REQUEST_PX in either dimension.

    bbox is (minx, miny, maxx, maxy) in EPSG:3577 metres.
    """
    minx, miny, maxx, maxy = bbox_3577
    width_m = maxx - minx
    height_m = maxy - miny
    max_width_m = MAX_REQUEST_PX * pixel_m
    n_x = max(1, int(-(-width_m // max_width_m)))
    n_y = max(1, int(-(-height_m // max_width_m)))
    step_x = width_m / n_x
    step_y = height_m / n_y
    for i in range(n_x):
        for j in range(n_y):
            yield (
                minx + i * step_x,
                miny + j * step_y,
                minx + (i + 1) * step_x,
                miny + (j + 1) * step_y,
            )


def fetch_geebam(event_id: str, out_dir: Path | None = None) -> Path:
    """Download GEEBAM for the AOI as a GeoTIFF in EPSG:3577.

    Skeleton (M1): validates the AOI exists, builds the request URL, writes a
    provenance manifest. The actual HTTP call + raster mosaic is implemented in M2.
    """
    feat = load_aoi(event_id)
    bbox_wgs = aoi_bbox_wgs84(event_id)
    log.info("AOI %s bbox (WGS84): %s", event_id, bbox_wgs)
    log.info("Properties: %s", feat["properties"])

    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "labels" / "aus_geebam" / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "label_native_3577.tif"

    # NOTE M2: implement the actual ArcGIS exportImage HTTP call here.
    # 1. Reproject bbox_wgs to EPSG:3577 via pyproj.
    # 2. For each tile in `_request_tiles(...)`, POST exportImage with params:
    #    bbox, bboxSR=3577, imageSR=3577, size=W,H, format=tiff, pixelType=U8,
    #    f=image. Save each tile, then rasterio.merge into out_path.
    # 3. Verify class histogram covers expected values {0,1,2,3,4,5}.
    log.warning("fetch_geebam SKELETON only — implement the HTTP call in M2.")

    write_manifest(
        out_path,
        event_id=event_id,
        pipeline_step="fetch_labels.geebam",
        inputs={
            "service": GEEBAM_BASE,
            "operation": "exportImage",
            "request_crs": GEEBAM_REQUEST_CRS,
            "native_pixel_m": GEEBAM_NATIVE_M,
            "bbox_wgs84": bbox_wgs,
            "licence": "CC-BY 4.0",
            "attribution": (
                "AUS GEEBAM © Commonwealth of Australia 2020, licensed CC-BY 4.0."
            ),
        },
        crs=GEEBAM_REQUEST_CRS,
        resampling=None,
        class_remap=None,
        notes="M1 skeleton — manifest only; actual download in M2.",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AUS GEEBAM labels for an AOI.")
    parser.add_argument("--event", required=True, help="Event ID matching configs/aois/<event>.geojson")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    path = fetch_geebam(args.event, args.out_dir)
    log.info("Wrote provenance for %s", path)


if __name__ == "__main__":
    main()
