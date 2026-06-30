"""Unit tests for fetch_labels (current MapServer/export + legend-decode API).

We don't hit the live GEEBAM endpoint here, those are integration tests. These
verify the pure-Python plumbing: WGS84 request-tile planning, the exact
RGBA→class legend reverse-map, and the dry-run path that writes a manifest with
no network.

(Updated from an earlier revision that referenced a removed ``_wgs_to_3577``
helper and a metres-based ``_request_tiles``; the fetcher now works in WGS84 and
reprojects the assembled raster to EPSG:3577 afterwards.)
"""

import json

import numpy as np

from src.data.fetch_labels import (
    GEEBAM_NATIVE_M,
    LEGEND_TO_INTERNAL,
    MAX_REQUEST_PX,
    _request_tiles,
    _rgb_to_class,
    fetch_geebam,
)


def test_request_tiles_single_when_small():
    # Kangaroo Island WGS84 bbox is well under one 4000 px @ 40 m request tile.
    bbox = (136.55, -36.10, 137.95, -35.50)
    tiles = list(_request_tiles(bbox, pixel_m=GEEBAM_NATIVE_M))
    assert len(tiles) == 1
    minx, miny, maxx, maxy, w, h = tiles[0]
    assert (minx, miny, maxx, maxy) == bbox
    assert 0 < w <= MAX_REQUEST_PX
    assert 0 < h <= MAX_REQUEST_PX


def test_request_tiles_splits_large_and_covers_bbox():
    # A 10° x 10° bbox is far larger than one request tile -> must split.
    bbox = (140.0, -38.0, 150.0, -28.0)
    tiles = list(_request_tiles(bbox, pixel_m=GEEBAM_NATIVE_M))
    assert len(tiles) > 1
    for minx, miny, maxx, maxy, w, h in tiles:
        assert w <= MAX_REQUEST_PX and h <= MAX_REQUEST_PX
        assert minx < maxx and miny < maxy
    # The tiles tile the full bbox.
    assert min(t[0] for t in tiles) == bbox[0]
    assert min(t[1] for t in tiles) == bbox[1]
    assert max(t[2] for t in tiles) == bbox[2]
    assert max(t[3] for t in tiles) == bbox[3]


def test_rgb_to_class_decodes_official_legend():
    # One pixel per legend colour, in class order, plus background + transparent.
    legend = list(LEGEND_TO_INTERNAL.items())
    h, w = 1, len(legend) + 2
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for i, ((r, g, b), _cls) in enumerate(legend):
        rgba[0, i] = (r, g, b, 255)
    rgba[0, len(legend)] = (247, 245, 239, 255)  # cream background -> ignore
    rgba[0, len(legend) + 1] = (0, 0, 0, 0)  # transparent -> ignore (alpha)
    out = _rgb_to_class(rgba)
    for i, (_rgb, cls) in enumerate(legend):
        assert out[0, i] == cls
    assert out[0, len(legend)] == 255
    assert out[0, len(legend) + 1] == 255


def test_rgb_to_class_tolerance():
    # A colour a few units off the "High" legend tuple still decodes to High.
    near = np.array([[[168 + 5, 56 - 4, 0 + 3, 255]]], dtype=np.uint8)
    assert _rgb_to_class(near)[0, 0] == LEGEND_TO_INTERNAL[(168, 56, 0)]
    # A colour far from every legend tuple is ignored.
    far = np.array([[[120, 120, 200, 255]]], dtype=np.uint8)
    assert _rgb_to_class(far)[0, 0] == 255


def test_dry_run_writes_manifest(tmp_path):
    out_path = fetch_geebam("kangaroo_island_2019_2020", out_dir=tmp_path, dry_run=True)
    manifest = out_path.with_suffix(out_path.suffix + ".provenance.json")
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["event_id"] == "kangaroo_island_2019_2020"
    assert payload["crs"] == "EPSG:4326"  # dry run records the request CRS
    assert payload["inputs"]["n_request_tiles"] >= 1
    assert "bbox_wgs84" in payload["inputs"]


def test_dry_run_all_aois(tmp_path):
    for aoi in [
        "kangaroo_island_2019_2020",
        "currowan_2019_2020",
        "gospers_mountain_2019_2020",
        "east_gippsland_2019_2020",
    ]:
        path = fetch_geebam(aoi, out_dir=tmp_path / aoi, dry_run=True)
        manifest = json.loads((path.with_suffix(path.suffix + ".provenance.json")).read_text())
        if aoi == "east_gippsland_2019_2020":  # largest AOI -> must tile
            assert manifest["inputs"]["n_request_tiles"] > 1
