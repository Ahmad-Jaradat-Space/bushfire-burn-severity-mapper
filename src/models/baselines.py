"""dNBR threshold baselines (binary + multiclass).

Two converters:
  - `dnbr_binary(dnbr, threshold)`        → uint8 {0=unburnt, 1=burnt}
  - `dnbr_multiclass_usgs(dnbr)`          → uint8 {0..3, 255=ignore} mapped to
                                            our internal class layout
                                            (unburnt, low_mod, high, very_high).

USGS-style dNBR breakpoints (heuristic, not Aus-calibrated). The two USGS
moderate categories are collapsed into a single 'high' internal bin to match
the GEEBAM 4-class layout used elsewhere in the project.

    dNBR <= 0.10                  -> 0  unburnt
    0.10 < dNBR <= 0.27           -> 1  low_mod
    0.27 < dNBR <= 0.66           -> 2  high (USGS moderate-low + moderate-high)
    dNBR > 0.66                   -> 3  very_high
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import IGNORE_ID
from src.features.indices import nbr


def dnbr(pre: np.ndarray, post: np.ndarray, b08_idx: int = 3, b12_idx: int = 5) -> np.ndarray:
    """pre/post are [6, H, W] reflectance stacks following the B02..B12 order."""
    pre_nbr = nbr(pre[b08_idx], pre[b12_idx])
    post_nbr = nbr(post[b08_idx], post[b12_idx])
    return pre_nbr - post_nbr


def dnbr_binary(d: np.ndarray, threshold: float = 0.10,
                mask: np.ndarray | None = None) -> np.ndarray:
    out = (d > threshold).astype(np.uint8)
    if mask is not None:
        out[~mask.astype(bool)] = IGNORE_ID
    out[~np.isfinite(d)] = IGNORE_ID
    return out


def dnbr_multiclass_usgs(d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.full(d.shape, IGNORE_ID, dtype=np.uint8)
    finite = np.isfinite(d)
    out[finite & (d <= 0.10)] = 0
    out[finite & (d > 0.10) & (d <= 0.27)] = 1
    out[finite & (d > 0.27) & (d <= 0.66)] = 2
    out[finite & (d > 0.66)] = 3
    if mask is not None:
        out[~mask.astype(bool)] = IGNORE_ID
    return out
