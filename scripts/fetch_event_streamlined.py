"""Streamlined real-data fetch for a single AOI — memory-efficient version.

Goes from STAC query to per-AOI pre/post composite GeoTIFFs without holding
the full [T, C, H, W] stack in memory.

Per side (pre/post):
  1. Pick the N lowest-cloud STAC items from data/raw/sentinel2/<event>/stac_items.json
  2. Sign assets
  3. For each band:
       - odc.stac.load that one band + SCL across the chosen items
       - per-pixel nanmedian over cloud-masked observations (one band at a time)
       - free memory before moving to the next band
  4. Write 6-band float32 GeoTIFF + uint8 clear-pixel mask

This keeps peak memory at ~1-2 GB instead of ~10 GB, and emits progress
logs to stdout that flush in real time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer as pc
import pystac
import rasterio

from src.data.cloud_mask import scl_to_clear_mask
from src.utils.geo import REPO_ROOT, utm_epsg_for_aoi
from src.utils.io import read_json
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)

REFLECT = ("B02", "B03", "B04", "B08", "B11", "B12")


def _flush_print(msg: str) -> None:
    """Print + flush so logs appear in real time when stdout is piped."""
    print(msg, flush=True)


def _pick_best(items: list[dict], n: int) -> list[dict]:
    items = sorted(items, key=lambda it: it["properties"].get("eo:cloud_cover", 100))
    return items[:n]


def _sign_items(items: list[dict]) -> list[pystac.Item]:
    return [pc.sign(pystac.Item.from_dict(it)) for it in items]


def composite_one_side(signed_items, resolution: int, epsg: int, out_path: Path,
                       n_scenes: int, bbox_wgs: tuple[float, float, float, float] | None = None) -> dict:
    if not signed_items:
        raise ValueError("No items provided")

    # Common odc.stac.load kwargs — including bbox to clip to AOI (otherwise
    # the loaded grid covers the full extent of every intersecting S2 tile,
    # which is ~10x more pixels than we need).
    load_kwargs = dict(
        resolution=resolution, crs=f"EPSG:{epsg}",
        chunks={"x": 2048, "y": 2048, "time": 1}, groupby="solar_day",
    )
    if bbox_wgs is not None:
        # bbox is in WGS84 (lon/lat); odc.stac will project it to the target CRS
        load_kwargs["bbox"] = list(bbox_wgs)

    # First: load SCL across all items to build the per-time clear masks
    _flush_print(f"  loading SCL stack ({n_scenes} scenes) at {resolution}m...")
    ds_scl = odc.stac.load(signed_items, bands=["SCL"], **load_kwargs)
    scl = ds_scl["SCL"].astype("uint8").values    # [T, Y, X]
    T, H, W = scl.shape
    _flush_print(f"  SCL stack shape: T={T}, H={H}, W={W}")
    clear_per_t = np.stack([scl_to_clear_mask(scl[t]) for t in range(T)])
    del scl, ds_scl
    clear_any = clear_per_t.any(axis=0)
    _flush_print(f"  cloud mask built; clear pixel fraction = {100 * clear_any.mean():.1f}%")

    # Now: build composite band by band
    composite = np.zeros((len(REFLECT), H, W), dtype=np.float32)
    for ci, band in enumerate(REFLECT):
        _flush_print(f"  band {ci+1}/{len(REFLECT)} [{band}]: loading + reducing...")
        ds_b = odc.stac.load(signed_items, bands=[band], **load_kwargs)
        arr = ds_b[band].astype("float32").values    # [T, Y, X]
        arr /= 10000.0
        arr = np.where(clear_per_t, arr, np.nan)
        composite[ci] = np.nanmedian(arr, axis=0).astype("float32")
        del arr, ds_b
        _flush_print(f"  band {band} done (median range "
                     f"{np.nanmin(composite[ci]):.3f} - {np.nanmax(composite[ci]):.3f})")

    # Get the transform from a final small load (same load_kwargs to match grid)
    ds_ref = odc.stac.load(signed_items[:1], bands=["B02"], **load_kwargs)
    transform = ds_ref["B02"].isel(time=0).odc.geobox.transform
    crs = ds_ref["B02"].odc.geobox.crs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _flush_print(f"  writing {out_path.name}...")
    with rasterio.open(
        out_path, "w", driver="GTiff", width=W, height=H, count=6,
        dtype="float32", crs=str(crs), transform=transform,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        nodata=float("nan"),
    ) as dst:
        dst.write(composite)

    mask_path = out_path.parent / out_path.name.replace("_stack_", "_mask_")
    with rasterio.open(
        mask_path, "w", driver="GTiff", width=W, height=H, count=1,
        dtype="uint8", crs=str(crs), transform=transform,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        nodata=0,
    ) as dst:
        dst.write(clear_any.astype("uint8")[np.newaxis])

    return {"shape": [int(H), int(W)],
            "clear_fraction": float(clear_any.mean()),
            "n_scenes": int(T)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True)
    p.add_argument("--side", choices=["pre", "post", "both"], default="both")
    p.add_argument("--n-pre", type=int, default=4)
    p.add_argument("--n-post", type=int, default=4)
    p.add_argument("--resolution", type=int, default=30,
                   help="Working resolution in metres (default 30m for portfolio runs).")
    args = p.parse_args()

    from src.utils.geo import aoi_bbox_wgs84
    event_id = args.event
    epsg = utm_epsg_for_aoi(event_id)
    bbox_wgs = aoi_bbox_wgs84(event_id)
    raw_dir = REPO_ROOT / "data" / "raw" / "sentinel2" / event_id
    interim_dir = REPO_ROOT / "data" / "interim" / event_id
    interim_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(raw_dir / "stac_items.json")
    _flush_print(f"AOI bbox (WGS84): {bbox_wgs}; working CRS EPSG:{epsg}; "
                  f"resolution={args.resolution}m")

    if args.side in ("pre", "both"):
        _flush_print(f"=== PRE-fire composite for {event_id} ===")
        pre_items = _pick_best(manifest["pre_items"], args.n_pre)
        _flush_print(f"selected {len(pre_items)} pre items, clouds: " +
                     ", ".join(f"{it['properties'].get('eo:cloud_cover', 0):.1f}%"
                                for it in pre_items))
        pre_signed = _sign_items(pre_items)
        pre_meta = composite_one_side(pre_signed, args.resolution, epsg,
                                       interim_dir / "pre_stack_10m.tif",
                                       n_scenes=len(pre_signed), bbox_wgs=bbox_wgs)
        write_manifest(
            interim_dir / "pre_stack_10m.tif",
            event_id=event_id, pipeline_step="streamlined_fetch.pre",
            inputs={"stac_items": [it["id"] for it in pre_items],
                     "n_scenes": pre_meta["n_scenes"],
                     "bands": list(REFLECT), "resolution_m": args.resolution,
                     "composite": "nanmedian", "clear_fraction": pre_meta["clear_fraction"],
                     "attribution": "Contains modified Copernicus Sentinel data [2019] processed by ESA."},
            crs=f"EPSG:{epsg}", resampling="bilinear",
        )

    if args.side in ("post", "both"):
        _flush_print(f"=== POST-fire composite for {event_id} ===")
        post_items = _pick_best(manifest["post_items"], args.n_post)
        _flush_print(f"selected {len(post_items)} post items, clouds: " +
                     ", ".join(f"{it['properties'].get('eo:cloud_cover', 0):.1f}%"
                                for it in post_items))
        post_signed = _sign_items(post_items)
        post_meta = composite_one_side(post_signed, args.resolution, epsg,
                                        interim_dir / "post_stack_10m.tif",
                                        n_scenes=len(post_signed), bbox_wgs=bbox_wgs)
        write_manifest(
            interim_dir / "post_stack_10m.tif",
            event_id=event_id, pipeline_step="streamlined_fetch.post",
            inputs={"stac_items": [it["id"] for it in post_items],
                     "n_scenes": post_meta["n_scenes"],
                     "bands": list(REFLECT), "resolution_m": args.resolution,
                     "composite": "nanmedian", "clear_fraction": post_meta["clear_fraction"],
                     "attribution": "Contains modified Copernicus Sentinel data [2019-2020] processed by ESA."},
            crs=f"EPSG:{epsg}", resampling="bilinear",
        )

    _flush_print("=== done ===")


if __name__ == "__main__":
    main()
