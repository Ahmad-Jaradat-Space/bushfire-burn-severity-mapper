"""PyTorch Dataset over the per-event tile index.

Each item returns (image, mask, label) tensors:
  image  float32 [18, H, W]   the 18-channel feature stack
  mask   bool    [H, W]       1 where pre & post are both clear
  label  int64   [H, W]       internal class IDs (255 = ignore)

Per-band z-score normalisation stats are computed from the TRAIN split only
and persisted to outputs/normalization/<run>.json so val/test use the same
statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.stack_features import DEFAULT_LAYOUT, build_stack
from src.utils.geo import REPO_ROOT


def _augment(image: np.ndarray, label: np.ndarray, mask: np.ndarray,
             cfg, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < float(cfg.augment.hflip):
        image = image[:, :, ::-1].copy()
        label = label[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    if rng.random() < float(cfg.augment.vflip):
        image = image[:, ::-1, :].copy()
        label = label[::-1, :].copy()
        mask = mask[::-1, :].copy()
    if rng.random() < float(cfg.augment.rotate90):
        k = int(rng.integers(1, 4))
        image = np.rot90(image, k, axes=(1, 2)).copy()
        label = np.rot90(label, k).copy()
        mask = np.rot90(mask, k).copy()
    jitter = float(cfg.augment.brightness_jitter)
    if jitter > 0:
        # Apply the same multiplicative jitter to all reflectance channels (first 12)
        scale = 1.0 + rng.uniform(-jitter, jitter)
        image[:12] = image[:12] * scale
    noise = float(cfg.augment.noise_sigma)
    if noise > 0:
        image[:12] = image[:12] + rng.normal(0, noise, size=image[:12].shape).astype(np.float32)
    return image, label, mask


class TileDataset(Dataset):
    """Reads .npz tiles, builds the 18-channel feature stack, optional augment."""

    def __init__(self, index_paths: list[Path], split: str, cfg=None,
                 max_tiles: int | None = None, augment: bool = False,
                 stats_path: Path | None = None, seed: int = 42):
        frames = []
        for p in index_paths:
            if not p.exists():
                continue
            frames.append(pd.read_parquet(p))
        if not frames:
            raise FileNotFoundError(f"No tile indices found: {index_paths}")
        df = pd.concat(frames, ignore_index=True)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_tiles is not None:
            df = df.iloc[:max_tiles].reset_index(drop=True)
        self.df = df
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.stats: dict | None = None
        if stats_path is not None and stats_path.exists():
            self.stats = json.loads(stats_path.read_text())

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        with np.load(REPO_ROOT / row["tile_path"]) as npz:
            pre = npz["pre"].astype(np.float32)
            post = npz["post"].astype(np.float32)
            mask = npz["mask"].astype(np.uint8)
            label = npz["label"].astype(np.int64)
        # Cloud-masked pixels in the composite are NaN from nanmedian. Replace
        # with 0 so they don't poison normalisation / autocast; the label is
        # already 255 (ignore) there and won't contribute to the loss.
        pre  = np.nan_to_num(pre,  nan=0.0, posinf=0.0, neginf=0.0)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)
        image = build_stack(pre, post)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image, label, mask

    def __getitem__(self, idx: int):
        image, label, mask = self._load(idx)
        if self.augment and self.cfg is not None:
            image, label, mask = _augment(image, label, mask, self.cfg, self.rng)
        if self.stats is not None:
            mean = np.asarray(self.stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
            std = np.asarray(self.stats["std"], dtype=np.float32).reshape(-1, 1, 1)
            image = (image - mean) / (std + 1e-6)
        return (
            torch.from_numpy(image.astype(np.float32)),
            torch.from_numpy(mask.astype(np.bool_)),
            torch.from_numpy(label.astype(np.int64)),
        )


def compute_normalisation(dataset: TileDataset, max_tiles: int = 200) -> dict:
    """Per-channel mean/std across up to `max_tiles` items. NaN-safe."""
    sums = np.zeros(len(DEFAULT_LAYOUT), dtype=np.float64)
    sqs = np.zeros(len(DEFAULT_LAYOUT), dtype=np.float64)
    n = 0
    for i in range(min(len(dataset), max_tiles)):
        image, _label, _mask = dataset._load(i)
        # _load already does nan_to_num; defensively keep nan-safe ops anyway
        flat = np.nan_to_num(image.reshape(image.shape[0], -1), nan=0.0)
        sums += flat.sum(axis=1)
        sqs += (flat ** 2).sum(axis=1)
        n += flat.shape[1]
    mean = sums / max(n, 1)
    var = (sqs / max(n, 1)) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "layout": list(DEFAULT_LAYOUT),
        "n_pixels": int(n),
    }
