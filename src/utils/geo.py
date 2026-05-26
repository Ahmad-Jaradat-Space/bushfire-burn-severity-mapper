"""Geo helpers: AOI loading, UTM picker, S2-grid snap."""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AOI_DIR = REPO_ROOT / "configs" / "aois"

# Sentinel-2 native grid is aligned to 10 m within each UTM zone, with tile origins
# at multiples of 60 m offsets from the UTM datum. For our work we snap to a 10 m grid
# inside each AOI's UTM zone — this is sufficient for label alignment within ±5 m.
S2_GRID_M = 10


def load_aoi(event_id: str) -> dict:
    """Load AOI feature dict (geometry + properties) for an event_id.

    Returns the single feature, not the FeatureCollection wrapper.
    """
    path = AOI_DIR / f"{event_id}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {path}")
    with path.open() as fh:
        fc = json.load(fh)
    if fc.get("type") != "FeatureCollection" or not fc.get("features"):
        raise ValueError(f"{path} is not a non-empty FeatureCollection")
    return fc["features"][0]


def aoi_bbox_wgs84(event_id: str) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in WGS84."""
    feat = load_aoi(event_id)
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def utm_zone_for_lonlat(lon: float, lat: float) -> tuple[int, str]:
    """Return (zone_number, hemisphere) for a single point."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    hemisphere = "N" if lat >= 0 else "S"
    return zone, hemisphere


def utm_epsg_for_aoi(event_id: str) -> int:
    """Return the EPSG code of the UTM zone covering the AOI centroid.

    Australian AOIs land in zones 50S–56S → EPSGs 32750–32756.
    """
    minx, miny, maxx, maxy = aoi_bbox_wgs84(event_id)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    zone, hemi = utm_zone_for_lonlat(cx, cy)
    base = 32700 if hemi == "S" else 32600
    return base + zone


def snap_to_s2_grid(value: float, grid: float = S2_GRID_M) -> float:
    """Snap a single coordinate (in projected metres) down to the nearest grid line."""
    return math.floor(value / grid) * grid
