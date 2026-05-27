"""Unit tests for fetch_labels.

We don't hit the live GEEBAM endpoint here — those are integration tests.
These verify pure-Python plumbing: bbox reprojection, request-tile planning,
and the dry-run code path that writes a manifest without network.
"""
import json

import pytest

from src.data.fetch_labels import (
    GEEBAM_NATIVE_M,
    MAX_REQUEST_PX,
    _request_tiles,
    _wgs_to_3577,
    fetch_geebam,
)


def test_wgs_to_3577_kangaroo():
    # Kangaroo Island bbox in WGS84 -> projected EPSG:3577 metres.
    # EPSG:3577 is centered at 132°E. KI is at ~137°E (5° east of centre)
    # so X is positive but small (~400-600 km); Y around -3.9M m.
    out = _wgs_to_3577(136.55, -36.10, 137.95, -35.50)
    assert 350_000 < out[0] < out[2] < 700_000
    assert -4_000_000 < out[1] < out[3] < -3_800_000
    # bbox must be ordered (minx < maxx and miny < maxy)
    assert out[0] < out[2] and out[1] < out[3]


def test_request_tiles_single_when_small():
    bbox = (0.0, 0.0, 100_000.0, 100_000.0)  # 100 km, 2500 px @ 40m
    tiles = list(_request_tiles(bbox, pixel_m=GEEBAM_NATIVE_M))
    assert len(tiles) == 1
    minx, miny, maxx, maxy, w, h = tiles[0]
    assert (minx, miny, maxx, maxy) == bbox
    assert w == 2500 and h == 2500


def test_request_tiles_splits_large():
    # 300 km square @ 40m = 7500 px per side → needs 2x2 tiles
    bbox = (0.0, 0.0, 300_000.0, 300_000.0)
    tiles = list(_request_tiles(bbox, pixel_m=GEEBAM_NATIVE_M))
    assert len(tiles) == 4
    for _minx, _miny, _maxx, _maxy, w, h in tiles:
        assert w <= MAX_REQUEST_PX
        assert h <= MAX_REQUEST_PX


def test_dry_run_writes_manifest(tmp_path):
    out_path = fetch_geebam("kangaroo_island_2019_2020", out_dir=tmp_path, dry_run=True)
    manifest = out_path.with_suffix(out_path.suffix + ".provenance.json")
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["event_id"] == "kangaroo_island_2019_2020"
    assert payload["crs"] == "EPSG:3577"
    assert payload["inputs"]["n_request_tiles"] >= 1
    assert "bbox_3577" in payload["inputs"]


def test_dry_run_all_aois(tmp_path):
    for aoi in [
        "kangaroo_island_2019_2020",
        "currowan_2019_2020",
        "gospers_mountain_2019_2020",
        "east_gippsland_2019_2020",
    ]:
        path = fetch_geebam(aoi, out_dir=tmp_path / aoi, dry_run=True)
        manifest = json.loads((path.with_suffix(path.suffix + ".provenance.json")).read_text())
        # East Gippsland is largest — sanity that it tiles
        if aoi == "east_gippsland_2019_2020":
            assert manifest["inputs"]["n_request_tiles"] > 1
