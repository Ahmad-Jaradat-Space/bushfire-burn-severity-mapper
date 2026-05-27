"""Build the feature stack consumed by classical ML and deep models.

Output is a [C, H, W] float32 array. Default channel layout (18 channels):

    pre:   B02, B03, B04, B08, B11, B12     (6)
    post:  B02, B03, B04, B08, B11, B12     (6)
    delta: dNBR, dNDVI, dNDMI, dNBR2, dBSI  (5)
    topo:  slope                            (1)
"""
from __future__ import annotations

import numpy as np

from src.features.indices import bsi, delta, nbr, nbr2, ndmi, ndvi

# Band positions in the [6, H, W] reflectance arrays produced by preprocess.py
B02, B03, B04, B08, B11, B12 = range(6)

DEFAULT_LAYOUT = (
    "pre_B02", "pre_B03", "pre_B04", "pre_B08", "pre_B11", "pre_B12",
    "post_B02", "post_B03", "post_B04", "post_B08", "post_B11", "post_B12",
    "dNBR", "dNDVI", "dNDMI", "dNBR2", "dBSI",
    "slope",
)


def indices_pre_post(pre: np.ndarray, post: np.ndarray) -> dict[str, np.ndarray]:
    """Return all 5 dIndices as a dict keyed by 'dNBR', 'dNDVI', ..."""
    out: dict[str, np.ndarray] = {}
    out["dNBR"]  = delta(nbr(pre[B08],  pre[B12]),  nbr(post[B08],  post[B12]))
    out["dNDVI"] = delta(ndvi(pre[B08], pre[B04]),  ndvi(post[B08], post[B04]))
    out["dNDMI"] = delta(ndmi(pre[B08], pre[B11]),  ndmi(post[B08], post[B11]))
    out["dNBR2"] = delta(nbr2(pre[B11], pre[B12]),  nbr2(post[B11], post[B12]))
    out["dBSI"]  = delta(
        bsi(pre[B02],  pre[B04],  pre[B08],  pre[B11]),
        bsi(post[B02], post[B04], post[B08], post[B11]),
    )
    return out


def build_stack(pre: np.ndarray, post: np.ndarray,
                slope: np.ndarray | None = None) -> np.ndarray:
    """Return [18, H, W] float32 stack matching DEFAULT_LAYOUT.

    Parameters
    ----------
    pre, post : [6, H, W] float32 reflectance in [0, 1]
    slope     : [H, W] float32 in degrees; if None, a zero plane is used.
    """
    deltas = indices_pre_post(pre, post)
    if slope is None:
        slope = np.zeros(pre.shape[1:], dtype=np.float32)
    stack = np.stack([
        pre[B02],  pre[B03],  pre[B04],  pre[B08],  pre[B11],  pre[B12],
        post[B02], post[B03], post[B04], post[B08], post[B11], post[B12],
        deltas["dNBR"], deltas["dNDVI"], deltas["dNDMI"], deltas["dNBR2"], deltas["dBSI"],
        slope.astype(np.float32),
    ]).astype(np.float32)
    assert stack.shape[0] == len(DEFAULT_LAYOUT), (stack.shape, DEFAULT_LAYOUT)
    return stack
