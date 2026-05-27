"""High-fidelity synthetic Kangaroo Island scene for the notebook.

When real Sentinel-2 imagery and trained model predictions are not yet on disk,
the notebook falls back to this synthetic stand-in so the visual story still
renders. It is deliberately *labelled* as synthetic in every figure caption.

The scene is a deterministic, seeded fake that mimics:
  - the island geometry of Kangaroo Island
  - the ~210,000 ha burn footprint on the western two-thirds
  - radial severity gradient from "very high" core to "low/moderate" edge
  - per-band reflectance that produces realistic NBR / dNBR signals
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticScene:
    pre:       np.ndarray   # [6, H, W] reflectance
    post:      np.ndarray   # [6, H, W]
    severity:  np.ndarray   # uint8 [H, W] internal class IDs (255=ignore)
    land_mask: np.ndarray   # bool [H, W]
    burn_mask: np.ndarray   # bool [H, W]


def build(h: int = 512, w: int = 900, seed: int = 0) -> SyntheticScene:
    """Return a SyntheticScene that looks like Kangaroo Island after Black Summer."""
    rng = np.random.default_rng(seed)

    # --- geometry ----------------------------------------------------------
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    rx, ry = w * 0.40, h * 0.32
    land = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy) ** 2) / (ry ** 2) < 1.0

    burn_centre = (cy - 25, int(cx - rx * 0.40))
    rb_x = w * 0.22
    rb_y = h * 0.24
    radial = np.sqrt(((xx - burn_centre[1]) / rb_x) ** 2
                     + ((yy - burn_centre[0]) / rb_y) ** 2)
    burn = land & (radial < 1.0)

    # severity gradient
    severity = np.full((h, w), 255, dtype=np.uint8)
    severity[land] = 0
    severity[burn & (radial < 0.95)] = 1
    severity[burn & (radial < 0.70)] = 2
    severity[burn & (radial < 0.40)] = 3

    # --- pre-fire reflectance ---------------------------------------------
    # bands: B02 B03 B04 B08 B11 B12  (blue green red NIR SWIR1 SWIR2)
    pre = np.zeros((6, h, w), dtype=np.float32)

    # Land: healthy vegetation — green peak, strong NIR, low SWIR
    pre[0] = np.where(land, 0.040 + 0.010 * rng.random((h, w)), 0)   # blue
    pre[1] = np.where(land, 0.085 + 0.015 * rng.random((h, w)), 0)   # green
    pre[2] = np.where(land, 0.055 + 0.015 * rng.random((h, w)), 0)   # red (chlorophyll absorption)
    pre[3] = np.where(land, 0.320 + 0.050 * rng.random((h, w)), 0)   # NIR (cellular structure)
    pre[4] = np.where(land, 0.180 + 0.030 * rng.random((h, w)), 0)   # SWIR1
    pre[5] = np.where(land, 0.090 + 0.020 * rng.random((h, w)), 0)   # SWIR2

    # Water: low everywhere, slight blue/green peak
    sea = ~land
    pre[0] = np.where(sea, 0.085 + 0.010 * rng.random((h, w)), pre[0])
    pre[1] = np.where(sea, 0.075 + 0.010 * rng.random((h, w)), pre[1])
    pre[2] = np.where(sea, 0.060 + 0.008 * rng.random((h, w)), pre[2])
    pre[3] = np.where(sea, 0.030 + 0.010 * rng.random((h, w)), pre[3])
    pre[4] = np.where(sea, 0.015 + 0.005 * rng.random((h, w)), pre[4])
    pre[5] = np.where(sea, 0.010 + 0.005 * rng.random((h, w)), pre[5])

    # --- post-fire reflectance --------------------------------------------
    post = pre.copy()
    # severity controls how much NIR collapses and SWIR2 rises
    # use radial as severity proxy: 0 = unburned, 1 = very-high
    sev = np.where(burn, np.clip(1.0 - radial / 1.0, 0, 1), 0)
    # NIR collapse
    post[3] = np.where(burn, pre[3] * (1.0 - 0.80 * sev), pre[3])
    # SWIR2 spike: dry ash absorbs less of the SWIR2 region
    post[5] = np.where(burn, pre[5] + 0.35 * sev * (1 - pre[5]), pre[5])
    # SWIR1 rises moderately
    post[4] = np.where(burn, pre[4] + 0.20 * sev * (1 - pre[4]), pre[4])
    # Red rises slightly (less chlorophyll)
    post[2] = np.where(burn, pre[2] + 0.08 * sev, pre[2])
    # Green drops
    post[1] = np.where(burn, pre[1] * (1.0 - 0.50 * sev), pre[1])

    # add a hint of inland water in northern-east of the island
    pond_y = cy - int(0.18 * ry)
    pond_x = cx + int(0.20 * rx)
    pond = ((xx - pond_x) ** 2 / 22.0 ** 2 + (yy - pond_y) ** 2 / 16.0 ** 2) < 1.0
    for c, v in [(0, 0.085), (1, 0.075), (2, 0.060), (3, 0.030),
                 (4, 0.015), (5, 0.010)]:
        pre[c]  = np.where(pond, v, pre[c])
        post[c] = np.where(pond, v, post[c])

    pre  = np.clip(pre,  0, 1)
    post = np.clip(post, 0, 1)
    return SyntheticScene(pre=pre, post=post, severity=severity,
                          land_mask=land, burn_mask=burn)


def synthetic_model_predictions(severity: np.ndarray,
                                seed: int = 0) -> dict[str, np.ndarray]:
    """Five synthetic predictions for the same scene, deliberately differentiated
    so the comparison panel tells a story."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    # dNBR threshold — under-predicts very-high (collapses to high)
    p = severity.copy()
    p[p == 3] = 2
    out["baseline_dnbr"] = p
    # RF — sharp class edges with salt-and-pepper noise on the boundary
    p = severity.copy()
    edge = (severity != np.roll(severity, 1, axis=0)) | (severity != np.roll(severity, 1, axis=1))
    noise = (rng.random(severity.shape) < 0.05) & edge
    p[noise] = (p[noise] + rng.integers(-1, 2, size=noise.sum())).clip(0, 3).astype(np.uint8)
    p[severity == 255] = 255
    out["rf"] = p
    # XGB — similar to RF but slightly cleaner boundaries
    p = severity.copy()
    noise = (rng.random(severity.shape) < 0.03) & edge
    p[noise] = (p[noise] + rng.integers(-1, 2, size=noise.sum())).clip(0, 3).astype(np.uint8)
    p[severity == 255] = 255
    out["xgb"] = p
    # U-Net — smooth blob edges, occasionally bleeds high into low
    from scipy.ndimage import binary_dilation
    p = severity.copy()
    leak = binary_dilation(severity == 2, iterations=2) & (severity == 1)
    p[leak] = 2
    out["unet"] = p
    # SegFormer — preserves the gradient best
    out["segformer"] = severity.copy()
    return out
