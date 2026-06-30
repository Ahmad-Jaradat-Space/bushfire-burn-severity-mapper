"""Spatial sampling designs — the survey-statistics backbone.

Two designs the JD explicitly asks for:

* :func:`spatial_block_folds` — k-fold cross-validation where folds are split by
  *spatial super-blocks*, never by tile. Random k-fold over autocorrelated tiles
  leaks neighbours across the train/val boundary and inflates the score; blocking
  by super-block is the defensible design. (This generalises the event-wise
  hold-out to a within-region setting.)
* :func:`stratified_pixel_sample` — a probability sample of reference pixels
  stratified by the *mapped* class, the input an Olofsson area-adjusted accuracy
  estimate (see :mod:`src.evaluation.area_estimation`) actually needs. Equal or
  proportional allocation, with optional grid thinning to damp autocorrelation.

Both operate on the tile index produced by :mod:`src.data.tiling` (the ``y, x``
pixel offsets) or directly on a prediction raster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_spatial_blocks(tile_index: pd.DataFrame, super_block_px: int = 1024) -> pd.DataFrame:
    """Tag each tile with a spatial super-block id (per event, to avoid merging
    distinct fires into one block)."""
    df = tile_index.copy().reset_index(drop=True)
    br = (df["y"] // super_block_px).astype(int)
    bc = (df["x"] // super_block_px).astype(int)
    ev = df["event_id"].astype(str) if "event_id" in df.columns else "evt"
    df["block_id"] = ev + ":" + br.astype(str) + "_" + bc.astype(str)
    return df


def spatial_block_folds(
    tile_index: pd.DataFrame,
    k: int = 5,
    super_block_px: int = 1024,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Return ``[(train_idx, val_idx), ...]`` with blocks (not tiles) split across
    folds, so no super-block straddles the train/val boundary."""
    df = assign_spatial_blocks(tile_index, super_block_px)
    blocks = df["block_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    fold_of = {b: i % k for i, b in enumerate(blocks)}
    df["fold"] = df["block_id"].map(fold_of)
    folds = []
    for f in range(k):
        val = df.index[df["fold"] == f].to_numpy()
        train = df.index[df["fold"] != f].to_numpy()
        folds.append((train, val))
    return folds, df


def stratified_pixel_sample(
    map_classes: np.ndarray,
    n_per_class: int,
    ignore_index: int = 255,
    seed: int = 42,
    thin_px: int = 0,
) -> dict:
    """Draw a probability sample of pixels stratified by the mapped class.

    Returns ``{"rows","cols","classes","strata_sizes"}``. ``thin_px>0`` keeps at
    most one sample per ``thin_px×thin_px`` grid cell per stratum to reduce
    spatial autocorrelation in the sample.
    """
    rng = np.random.default_rng(seed)
    H, W = map_classes.shape
    rows, cols, classes = [], [], []
    strata_sizes: dict[int, int] = {}
    for cls in np.unique(map_classes):
        if cls == ignore_index:
            continue
        ys, xs = np.nonzero(map_classes == cls)
        strata_sizes[int(cls)] = int(ys.size)
        if ys.size == 0:
            continue
        if thin_px > 0:
            # one candidate per occupied grid cell
            cell = (ys // thin_px) * (W // thin_px + 1) + (xs // thin_px)
            order = rng.permutation(ys.size)
            seen = set()
            keep = []
            for idx in order:
                c = cell[idx]
                if c in seen:
                    continue
                seen.add(c)
                keep.append(idx)
            ys, xs = ys[keep], xs[keep]
        take = min(n_per_class, ys.size)
        sel = rng.choice(ys.size, size=take, replace=False)
        rows.extend(ys[sel].tolist())
        cols.extend(xs[sel].tolist())
        classes.extend([int(cls)] * take)
    return {
        "rows": np.array(rows),
        "cols": np.array(cols),
        "classes": np.array(classes),
        "strata_sizes": strata_sizes,
    }
