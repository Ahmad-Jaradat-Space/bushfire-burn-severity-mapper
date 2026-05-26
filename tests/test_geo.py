from src.utils.geo import (
    aoi_bbox_wgs84,
    load_aoi,
    snap_to_s2_grid,
    utm_epsg_for_aoi,
    utm_zone_for_lonlat,
)

AOI_IDS = [
    "kangaroo_island_2019_2020",
    "currowan_2019_2020",
    "gospers_mountain_2019_2020",
    "east_gippsland_2019_2020",
]


def test_all_aois_load():
    for aoi in AOI_IDS:
        feat = load_aoi(aoi)
        assert feat["properties"]["event_id"] == aoi
        assert feat["geometry"]["type"] == "Polygon"


def test_aoi_bboxes_inside_australia():
    for aoi in AOI_IDS:
        minx, miny, maxx, maxy = aoi_bbox_wgs84(aoi)
        # Continental Australia + Tasmania bbox sanity check
        assert 112.0 < minx < maxx < 154.0
        assert -45.0 < miny < maxy < -9.0


def test_utm_zones_australian_south():
    # Kangaroo Is ≈ lon 137 → zone 53S → 32753
    assert utm_epsg_for_aoi("kangaroo_island_2019_2020") == 32753
    # Currowan ≈ lon 150 → zone 56S → 32756
    assert utm_epsg_for_aoi("currowan_2019_2020") == 32756
    # Gospers Mountain ≈ lon 150.5 → zone 56S → 32756
    assert utm_epsg_for_aoi("gospers_mountain_2019_2020") == 32756
    # East Gippsland ≈ lon 148.5 → zone 55S → 32755
    assert utm_epsg_for_aoi("east_gippsland_2019_2020") == 32755


def test_utm_zone_number_canberra():
    zone, hemi = utm_zone_for_lonlat(149.13, -35.28)
    assert zone == 55
    assert hemi == "S"


def test_snap_to_s2_grid():
    assert snap_to_s2_grid(123.4) == 120
    assert snap_to_s2_grid(120.0) == 120
    assert snap_to_s2_grid(119.9999) == 110
