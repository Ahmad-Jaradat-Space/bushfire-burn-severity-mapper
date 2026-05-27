"""Align an already-fetched GEEBAM raster to a per-AOI composite's CRS/grid.

Skips the redundant remap step in preprocess.py because the new fetch_labels.py
writes the label raster in internal class IDs (0/1/2/3, 255=ignore) directly.

Inputs:
  data/raw/labels/aus_geebam/<event>/label_native_3577.tif    (internal IDs)
  data/interim/<event>/post_stack_10m.tif                     (reference grid)
Output:
  data/interim/<event>/label_10m.tif                          (snapped)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)


def align(event_id: str) -> Path:
    interim = REPO_ROOT / "data" / "interim" / event_id
    label_src = REPO_ROOT / "data" / "raw" / "labels" / "aus_geebam" / event_id / "label_native_3577.tif"
    ref = interim / "post_stack_10m.tif"
    if not label_src.exists():
        raise FileNotFoundError(label_src)
    if not ref.exists():
        raise FileNotFoundError(ref)

    out = interim / "label_10m.tif"
    with rasterio.open(ref) as r:
        target_crs = r.crs
        target_transform = r.transform
        target_w, target_h = r.width, r.height

    with rasterio.open(label_src) as src:
        meta = {
            "driver": "GTiff", "height": target_h, "width": target_w,
            "count": 1, "dtype": "uint8",
            "crs": target_crs, "transform": target_transform,
            "nodata": 255, "compress": "deflate", "tiled": True,
            "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(out, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_crs=src.crs, dst_crs=target_crs,
                src_transform=src.transform, dst_transform=target_transform,
                src_nodata=255, dst_nodata=255,
                resampling=Resampling.nearest,
            )
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    uniq, counts = np.unique(arr, return_counts=True)
    hist = dict(zip(uniq.tolist(), counts.tolist()))
    log.info("Aligned %s → %s  histogram=%s", label_src.name, out.name, hist)

    write_manifest(
        out, event_id=event_id, pipeline_step="align_labels_to_composite",
        inputs={"source": str(label_src.relative_to(REPO_ROOT)),
                "reference": str(ref.relative_to(REPO_ROOT)),
                "class_histogram": hist},
        crs=str(target_crs), resampling="nearest",
        class_remap={"unburnt": 0, "low_mod": 1, "high": 2, "very_high": 3},
        notes="Labels already internal IDs (set by fetch_labels.py).",
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True)
    args = p.parse_args()
    align(args.event)


if __name__ == "__main__":
    main()
