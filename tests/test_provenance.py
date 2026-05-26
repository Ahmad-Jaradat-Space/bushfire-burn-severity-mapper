import json

from src.utils.provenance import write_manifest


def test_write_manifest(tmp_path):
    raster = tmp_path / "demo.tif"
    manifest_path = write_manifest(
        raster,
        event_id="kangaroo_island_2019_2020",
        pipeline_step="unit_test",
        inputs={"stac_items": ["foo"], "source_url": "https://example"},
        crs="EPSG:32753",
        resampling="nearest",
        class_remap={"2": 0, "3": 1, "4": 2, "5": 3},
        notes="hello",
    )
    assert manifest_path.name == "demo.tif.provenance.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["event_id"] == "kangaroo_island_2019_2020"
    assert payload["pipeline_step"] == "unit_test"
    assert payload["crs"] == "EPSG:32753"
    assert payload["resampling"] == "nearest"
    assert payload["class_remap"] == {"2": 0, "3": 1, "4": 2, "5": 3}
    assert "generated_utc" in payload
    assert "git_sha" in payload
