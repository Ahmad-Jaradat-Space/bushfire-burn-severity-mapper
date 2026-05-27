"""Query and download Sentinel-2 L2A scenes for an AOI's pre/post windows.

Two-stage flow:
  1. `fetch_event(event_id)` — queries Planetary Computer STAC and writes a
     frozen manifest (`stac_items.json`) under data/raw/sentinel2/<event>/.
     This is cheap, fast, and reviewable.
  2. `download_assets(event_id, kind)` — for each item in the manifest, signs
     the asset URLs and uses odc.stac.load to materialise a band stack to disk
     as a Cloud-Optimised GeoTIFF.

Splitting the two lets a reviewer (or Codex) inspect *what* will be downloaded
before bandwidth is spent.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import planetary_computer
import pystac_client
from odc.stac import load as odc_load

from src.utils.geo import REPO_ROOT, aoi_bbox_wgs84, load_aoi, utm_epsg_for_aoi
from src.utils.io import read_json, write_json
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL")
DEFAULT_CLOUD_LT = 30

log = get_logger(__name__)


def query_window(event_id: str, window: tuple[str, str],
                 cloud_lt: int = DEFAULT_CLOUD_LT) -> list[dict]:
    """Query Planetary Computer STAC for items intersecting the AOI inside `window`.

    Returns the list of (unsigned) STAC item dictionaries.
    """
    feat = load_aoi(event_id)
    geom = feat["geometry"]
    client = pystac_client.Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        intersects=geom,
        datetime=f"{window[0]}/{window[1]}",
        query={"eo:cloud_cover": {"lt": cloud_lt}},
    )
    items = [it.to_dict() for it in search.items()]
    log.info("STAC %s/%s @ cloud<%d → %d items", window[0], window[1], cloud_lt, len(items))
    return items


def fetch_event(event_id: str, out_dir: Path | None = None,
                cloud_lt: int = DEFAULT_CLOUD_LT) -> Path:
    """Persist the pre/post STAC manifests for an event."""
    feat = load_aoi(event_id)
    props = feat["properties"]
    pre = tuple(props["pre_window"])
    post = tuple(props["post_window"])

    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "sentinel2" / event_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_items = query_window(event_id, pre, cloud_lt)
    post_items = query_window(event_id, post, cloud_lt)

    manifest = {
        "event_id": event_id,
        "pre_window": list(pre),
        "post_window": list(post),
        "pre_items": pre_items,
        "post_items": post_items,
        "collection": COLLECTION,
        "stac_url": STAC_URL,
        "cloud_lt": cloud_lt,
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
            "cloud_lt": cloud_lt,
            "bbox_wgs84": aoi_bbox_wgs84(event_id),
            "n_pre_items": len(pre_items),
            "n_post_items": len(post_items),
            "attribution": "Contains modified Copernicus Sentinel data [2018-2020] processed by ESA.",
        },
        crs=f"EPSG:{utm_epsg_for_aoi(event_id)}",
        notes=None,
    )
    return items_path


def download_assets(event_id: str, kind: Literal["pre", "post"],
                    bands: tuple[str, ...] = DEFAULT_BANDS,
                    resolution: int = 10,
                    out_dir: Path | None = None) -> Path:
    """Sign + load + persist a multi-band stack of Sentinel-2 L2A scenes.

    Output: data/raw/sentinel2/<event>/<kind>/stack.nc (NetCDF, lossless).
    The compositing step (M3) consumes this stack.
    """
    out_dir = out_dir or REPO_ROOT / "data" / "raw" / "sentinel2" / event_id
    items_path = out_dir / "stac_items.json"
    if not items_path.exists():
        raise FileNotFoundError(f"Run fetch_event first; {items_path} missing")
    manifest = read_json(items_path)
    items = manifest[f"{kind}_items"]
    if not items:
        log.warning("No %s items in manifest — nothing to download.", kind)
        return out_dir / kind / "stack.nc"

    signed = [planetary_computer.sign(item) for item in items]
    epsg = utm_epsg_for_aoi(event_id)
    log.info("Loading %d %s items, bands=%s, crs=EPSG:%d, res=%dm",
             len(signed), kind, bands, epsg, resolution)

    ds = odc_load(
        signed,
        bands=list(bands),
        resolution=resolution,
        crs=f"EPSG:{epsg}",
        chunks={"x": 2048, "y": 2048, "time": 1},
        groupby="solar_day",
    )

    (out_dir / kind).mkdir(parents=True, exist_ok=True)
    stack_path = out_dir / kind / "stack.nc"
    ds.to_netcdf(stack_path)
    log.info("Wrote %s (%d time × %d bands × %d × %d)",
             stack_path, ds.sizes.get("time", 0), len(bands),
             ds.sizes.get("y", 0), ds.sizes.get("x", 0))

    write_manifest(
        stack_path,
        event_id=event_id,
        pipeline_step=f"fetch_sentinel.download.{kind}",
        inputs={
            "stac_items": [it["id"] for it in items],
            "stac_url": STAC_URL,
            "collection": COLLECTION,
            "bands": list(bands),
            "resolution_m": resolution,
            "attribution": "Contains modified Copernicus Sentinel data [2018-2020] processed by ESA.",
        },
        crs=f"EPSG:{epsg}",
        resampling="bilinear",
    )
    return stack_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel-2 L2A ingest.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--stage", choices=["query", "download", "all"], default="query")
    parser.add_argument("--kind", choices=["pre", "post"], default=None,
                        help="Only relevant for --stage download")
    parser.add_argument("--cloud-lt", type=int, default=DEFAULT_CLOUD_LT)
    args = parser.parse_args()

    if args.stage in ("query", "all"):
        fetch_event(args.event, cloud_lt=args.cloud_lt)
    if args.stage in ("download", "all"):
        if args.kind is None or args.stage == "all":
            for k in ("pre", "post"):
                download_assets(args.event, k)  # type: ignore[arg-type]
        else:
            download_assets(args.event, args.kind)


if __name__ == "__main__":
    main()
