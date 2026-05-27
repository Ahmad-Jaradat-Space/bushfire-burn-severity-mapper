"""GEEBAM → internal class remap utilities.

GEEBAM publishes 5 classes; we collapse `1` (unburnt-outside-extent) to an
ignore index and remap 2/3/4/5 to dense 0/1/2/3 for ML.
"""
from __future__ import annotations

import numpy as np

IGNORE_ID = 255

# GEEBAM source value → internal class ID (or IGNORE_ID)
GEEBAM_TO_INTERNAL: dict[int, int] = {
    0: IGNORE_ID,   # no-data
    1: IGNORE_ID,   # unburnt-outside-extent
    2: 0,           # unburnt
    3: 1,           # low-moderate
    4: 2,           # high
    5: 3,           # very-high
}

INTERNAL_NAMES = ["unburnt", "low_mod", "high", "very_high"]


def remap_geebam(arr: np.ndarray) -> np.ndarray:
    """Vectorised remap of a GEEBAM uint8 raster to dense internal class IDs."""
    out = np.full(arr.shape, IGNORE_ID, dtype=np.uint8)
    for src, dst in GEEBAM_TO_INTERNAL.items():
        out[arr == src] = dst
    return out


def class_histogram(arr: np.ndarray) -> dict[int, int]:
    """Return {class_id: count} including the ignore class."""
    uniq, counts = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}
