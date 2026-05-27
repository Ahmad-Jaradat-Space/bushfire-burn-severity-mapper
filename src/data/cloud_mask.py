"""Sentinel-2 L2A cloud / shadow / snow mask via the SCL band.

SCL class codes (per ESA L2A doc):
  0=no-data, 1=defective, 2=dark-area-pixel, 3=cloud-shadow,
  4=vegetation, 5=not-vegetated, 6=water,
  7=unclassified, 8=cloud-medium-prob, 9=cloud-high-prob,
  10=thin-cirrus, 11=snow

We mask {0, 1, 3, 8, 9, 10, 11} by default — i.e. drop everything that isn't
a clear surface observation. SCL class 7 (unclassified) is kept by default;
toggle via the `extra_mask_classes` argument if a particular AOI has unusual
noise.

After bitmap masking we dilate by `dilate_pixels` to absorb cloud-edge leakage.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

DEFAULT_SCL_MASK_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10, 11)


def scl_to_clear_mask(scl: np.ndarray,
                      mask_classes: tuple[int, ...] = DEFAULT_SCL_MASK_CLASSES,
                      dilate_pixels: int = 2) -> np.ndarray:
    """Return a bool array True where the pixel is a clear surface observation.

    `scl` is the L2A Scene Classification band (uint8).
    """
    blocked = np.isin(scl, list(mask_classes))
    if dilate_pixels > 0:
        struct = generate_binary_structure(2, 2)
        blocked = binary_dilation(blocked, structure=struct, iterations=dilate_pixels)
    return ~blocked


def clear_fraction(clear_mask: np.ndarray) -> float:
    """Fraction of pixels retained after masking."""
    return float(clear_mask.mean()) if clear_mask.size else 0.0
