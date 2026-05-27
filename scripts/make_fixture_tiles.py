"""Generate a tiny frozen fixture dataset for CI smoke training.

Writes:
  data/sample/tiles/synthetic_event/train/tile_00000..tile_00007.npz
  data/sample/tiles/synthetic_event/val/tile_00000..tile_00001.npz
  data/processed/tile_index_synthetic_event.parquet

No live network or imagery needed — purely synthetic so CI is hermetic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.geo import REPO_ROOT


def make_tile(seed: int, h: int = 64, w: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    pre = rng.uniform(0.05, 0.6, size=(6, h, w)).astype(np.float32)
    post = pre.copy()
    # Make a circular burn in the centre half the time so labels are balanced
    label = np.zeros((h, w), dtype=np.uint8)
    if seed % 2 == 0:
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        burn_high = r < h * 0.15
        burn_low = (r >= h * 0.15) & (r < h * 0.35)
        post[3][burn_high] *= 0.15; post[5][burn_high] *= 1.8
        post[3][burn_low] *= 0.6;   post[5][burn_low] *= 1.2
        label[burn_low] = 1
        label[burn_high] = 3
    mask = np.ones((h, w), dtype=np.uint8)
    return {"pre": pre, "post": post, "mask": mask, "label": label}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "sample")
    args = p.parse_args()

    tiles_root = args.out_root / "tiles" / "synthetic_event"
    rows = []
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        d = tiles_root / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            seed = (0 if split == "train" else 1000) + i
            tile = make_tile(seed)
            tp = d / f"tile_{i:05d}.npz"
            np.savez_compressed(tp, **tile)
            rows.append({
                "event_id": "synthetic_event",
                "tile_path": str(tp.relative_to(REPO_ROOT)),
                "split": split,
                "y": 0, "x": 0,
                "clear_frac": 1.0,
                "label_valid_frac": 1.0,
            })

    idx_path = REPO_ROOT / "data" / "processed" / "tile_index_synthetic_event.parquet"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(idx_path, index=False)
    print(f"Wrote {len(rows)} fixture tiles + index at {idx_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
