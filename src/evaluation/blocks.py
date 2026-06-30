"""Spatial blocking for autocorrelation-aware resampling.

Pixels in a burn-severity raster are strongly spatially autocorrelated:
neighbouring pixels almost always share a class. Treating each pixel as an
independent draw — the naive bootstrap — therefore badly *underestimates* the
sampling uncertainty of any map-accuracy metric. We instead resample contiguous
square blocks of pixels: correlation is preserved *inside* a block, and we only
assume independence *between* blocks (a spatial block bootstrap).

A block is a ``block_px x block_px`` square in raster (row, col) space. The
per-block confusion matrices returned here are the unit of resampling for
:mod:`src.evaluation.uncertainty` — summing all of them reproduces the global
confusion matrix exactly, so every estimator stays consistent with
:func:`src.evaluation.metrics.summary`.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import IGNORE_ID, confusion_matrix


def grid_blocks(shape: tuple[int, int], block_px: int) -> np.ndarray:
    """Return an int32 array ``[H, W]`` giving the block id of every pixel.

    Blocks tile the raster left-to-right, top-to-bottom; ids are contiguous
    integers ``0 .. n_blocks - 1``.
    """
    if block_px < 1:
        raise ValueError("block_px must be >= 1")
    h, w = shape
    rows = np.arange(h) // block_px
    cols = np.arange(w) // block_px
    n_block_cols = (w + block_px - 1) // block_px
    block = rows[:, None] * n_block_cols + cols[None, :]
    return block.astype(np.int32)


def block_confusions(
    pred: np.ndarray,
    true: np.ndarray,
    block_id: np.ndarray,
    num_classes: int,
    ignore_index: int = IGNORE_ID,
) -> dict[int, np.ndarray]:
    """Per-block confusion matrices.

    Returns ``{block_id: [num_classes, num_classes] int64}`` for every block
    holding at least one valid (non-ignore) pixel. ``cm[i, j]`` counts
    ``true == i`` predicted as ``j``, matching
    :func:`src.evaluation.metrics.confusion_matrix`. The sum over all returned
    blocks equals the global confusion matrix.
    """
    p = pred.ravel()
    t = true.ravel()
    b = block_id.ravel()
    valid = (t != ignore_index) & (p != ignore_index)
    p, t, b = p[valid], t[valid], b[valid]

    out: dict[int, np.ndarray] = {}
    if b.size == 0:
        return out
    order = np.argsort(b, kind="stable")
    b_s, p_s, t_s = b[order], p[order], t[order]
    uniq, starts = np.unique(b_s, return_index=True)
    bounds = list(starts) + [b_s.size]
    for k, blk in enumerate(uniq):
        s, e = bounds[k], bounds[k + 1]
        out[int(blk)] = confusion_matrix(p_s[s:e], t_s[s:e], num_classes, ignore_index)
    return out
