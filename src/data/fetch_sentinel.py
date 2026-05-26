"""Query Microsoft Planetary Computer for Sentinel-2 L2A scenes covering an AOI.

Two windows are requested per event:
  - Pre-fire (event-specific window from AOI properties)
  - Post-fire (event-specific window from AOI properties)

The returned STAC item set is persisted to `data/raw/sentinel2/<event>/stac_items.json`
so downstream steps (cloud mask, compositing) operate on a frozen, reviewable manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84, load_aoi, utm_epsg_for_aoi
from src.utils.io import write_json
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
DEFAULT_CLOUD_LT = 30

log = get_logger(__name__)


def query_window(event_id: str, window: tuple[str, str], cloud_lt: int = DEFAULT_CLOUD_LT) -> list[dict]:
    """Query PC STAC for items intersecting the AOI inside `window`.

    Returns signed item dicts. M3 implements the actual `pystac_client` call.
    """
    bbox = aoi_bbox_wgs84(event_id)
    log.info("STAC query event=%s window=%s bbox=%s cloud<%d", event_id, window, bbox, cloud_lt)
    # NOTE M3: real implementation:
    #   client = pystac_client.Client.open(STAC_URL)
    #   search = client.search(collections=[COLLECTION], bbox=bbox,
    #                          datetime=f"{window[0]}/{window[1]}",
    #                          query={"eo:cloud_cover": {"lt": cloud_lt}})
    #   items = [planetary_computer.sign(i).to_dict() for i in search.items()]
    log.warning("query_window SKELETON only — real STAC call lands in M3.")
    return []


def fetch_event(event_id: str, out_dir: Path | None = None) -> Path:
    feat = load_aoi(event_id)
    props = feat["properties"]
    pre = tuple(props["pre_window"])
    post = tuple(props["post_window"])

    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "sentinel2" / event_id
    (out_dir / "pre").mkdir(parents=True, exist_ok=True)
    (out_dir / "post").mkdir(parents=True, exist_ok=True)

    pre_items = query_window(event_id, pre)
    post_items = query_window(event_id, post)

    manifest = {
        "event_id": event_id,
        "pre_window": list(pre),
        "post_window": list(post),
        "pre_items": pre_items,
        "post_items": post_items,
        "collection": COLLECTION,
        "stac_url": STAC_URL,
        "working_crs": f"EPSG:{utm_epsg_for_aoi(event_id)}",
    }
    items_path = out_dir / "stac_items.json"
    write_json(items_path, manifest)
    log.info("Wrote %d pre + %d post items to %s", len(pre_items), len(post_items), items_path)

    write_manifest(
        items_path,
        event_id=event_id,
        pipeline_step="fetch_sentinel.query",
        inputs={
            "stac_url": STAC_URL,
            "collection": COLLECTION,
            "pre_window": list(pre),
            "post_window": list(post),
            "cloud_lt": DEFAULT_CLOUD_LT,
            "attribution": (
                "Contains modified Copernicus Sentinel data [2018-2020] processed by ESA."
            ),
        },
        crs=f"EPSG:{utm_epsg_for_aoi(event_id)}",
        notes="M1 skeleton — STAC query lands in M3.",
    )
    return items_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Sentinel-2 L2A STAC for an AOI.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    fetch_event(args.event, args.out_dir)


if __name__ == "__main__":
    main()
