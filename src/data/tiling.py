"""Tile the per-event interim stacks into model-ready patches.

Reads:  data/interim/<event>/{pre,post}_stack_10m.tif, label_10m.tif, mask_{pre,post}_10m.tif
Writes: data/processed/tiles/<event>/<split>/tile_<i>.npz
        data/processed/tile_index.parquet (global index)

Each .npz contains:
  pre   float32 [6, H, W]
  post  float32 [6, H, W]
  mask  uint8 [H, W]    1 where both pre & post clear
  label uint8 [H, W]    internal class IDs (255 = ignore)

Tiles are skipped if too cloud-occluded (>50% non-clear) or too label-empty
(>90% ignore). Tile splits assigned by the event-wise policy in config.yaml,
or by random tile split for the vertical-slice smoke-test mode.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import rasterio

from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

log = get_logger(__name__)


def _read_band_stack(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def iter_tile_windows(height: int, width: int, tile: int, stride: int) -> Iterator[tuple[int, int]]:
    for y in range(0, height - tile + 1, stride):
        for x in range(0, width - tile + 1, stride):
            yield y, x


def _vertical_slice_split(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.70:
        return "train"
    if r < 0.85:
        return "val"
    return "test"


def _event_split(event_id: str, cfg) -> str:
    if event_id in cfg.events.train:
        return "train"
    if event_id in cfg.events.val:
        return "val"
    if event_id in cfg.events.test:
        return "test"
    return "train"


def tile_event(event_id: str,
               cfg_path: str = "configs/config.yaml",
               max_cloud_frac: float = 0.5,
               max_ignore_frac: float = 0.9,
               split_mode: str = "event_wise",
               seed: int = 42) -> Path:
    cfg = load_config(cfg_path)
    interim = REPO_ROOT / "data" / "interim" / event_id
    out_root = REPO_ROOT / "data" / "processed" / "tiles" / event_id
    out_root.mkdir(parents=True, exist_ok=True)

    pre = _read_band_stack(interim / "pre_stack_10m.tif").astype(np.float32)    # [6, H, W]
    post = _read_band_stack(interim / "post_stack_10m.tif").astype(np.float32)
    with rasterio.open(interim / "mask_pre_10m.tif") as ds:
        mpre = ds.read(1)
    with rasterio.open(interim / "mask_post_10m.tif") as ds:
        mpost = ds.read(1)
    with rasterio.open(interim / "label_10m.tif") as ds:
        label = ds.read(1)

    mask = (mpre & mpost).astype(np.uint8)   # 1 where both clear
    _, H, W = pre.shape
    tile = int(cfg.data.tile_size)
    stride = int(cfg.data.tile_stride)

    rng = random.Random(seed)
    rows: list[dict] = []
    n_kept = 0
    n_drop_cloud = 0
    n_drop_ignore = 0
    n_total = 0
    for y, x in iter_tile_windows(H, W, tile, stride):
        n_total += 1
        m = mask[y:y + tile, x:x + tile]
        lab = label[y:y + tile, x:x + tile]
        if m.mean() < (1.0 - max_cloud_frac):
            n_drop_cloud += 1
            continue
        if (lab == 255).mean() > max_ignore_frac:
            n_drop_ignore += 1
            continue

        split = (_vertical_slice_split(rng) if split_mode == "random_tile"
                 else _event_split(event_id, cfg))
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        tile_path = out_dir / f"tile_{n_kept:05d}.npz"
        np.savez_compressed(
            tile_path,
            pre=pre[:, y:y + tile, x:x + tile],
            post=post[:, y:y + tile, x:x + tile],
            mask=m,
            label=lab,
        )
        rows.append({
            "event_id": event_id,
            "tile_path": str(tile_path.relative_to(REPO_ROOT)),
            "split": split,
            "y": y, "x": x,
            "clear_frac": float(m.mean()),
            "label_valid_frac": float((lab != 255).mean()),
        })
        n_kept += 1

    log.info("[%s] %d windows -> kept %d (cloud drop=%d, ignore drop=%d)",
             event_id, n_total, n_kept, n_drop_cloud, n_drop_ignore)

    index_path = REPO_ROOT / "data" / "processed" / f"tile_index_{event_id}.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(index_path, index=False)

    write_manifest(
        index_path,
        event_id=event_id,
        pipeline_step="tiling",
        inputs={
            "interim_dir": str(interim.relative_to(REPO_ROOT)),
            "tile_size": tile,
            "tile_stride": stride,
            "max_cloud_frac": max_cloud_frac,
            "max_ignore_frac": max_ignore_frac,
            "split_mode": split_mode,
            "n_windows": n_total,
            "n_kept": n_kept,
            "n_drop_cloud": n_drop_cloud,
            "n_drop_ignore": n_drop_ignore,
        },
        crs=None,
        resampling=None,
        notes=None,
    )
    return index_path


def main() -> None:
    p = argparse.ArgumentParser(description="Tile preprocessed event stacks.")
    p.add_argument("--event", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--split-mode", choices=["event_wise", "random_tile"], default="event_wise")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = tile_event(args.event, args.config, split_mode=args.split_mode, seed=args.seed)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
