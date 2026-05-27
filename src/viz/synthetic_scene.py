"""Plausible synthetic Kangaroo Island scene for the notebook.

When real Sentinel-2 imagery is not yet on disk, the notebook falls back
to this stand-in so the visual story still renders. It is deliberately
labelled as synthetic in every figure caption.

The scene mimics:
  - An irregular Kangaroo-shaped island (not a perfect ellipse)
  - Realistic temperate-forest reflectance values: olive-green canopy,
    moderate NIR, low SWIR
  - Deep-blue ocean with low overall reflectance
  - A burn footprint with a ragged perimeter (Perlin-like noise on a
    radial decay) following the real western-island distribution
  - Per-band reflectance values that produce realistic NBR / dNBR
    signals (NIR collapse + SWIR2 rise inside burn)
  - Subtle topographic shading on land
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def s2_truecolour(stack: np.ndarray, gamma: float = 0.85,
                  scale: float = 3.5) -> np.ndarray:
    """Render a 6-band reflectance stack [C, H, W] as a true-colour [H, W, 3] image.

    Uses a JOINT linear stretch — the same multiplier on every channel — so
    colour relationships (green > red in vegetation, etc.) are preserved.
    This is the standard Sentinel-2 visualisation: multiply RGB reflectance
    by ~3, apply a mild gamma to lift midtones, clip to [0, 1].

    Channel order in stack: B02 B03 B04 B08 B11 B12.
    """
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError(f"Expected [C>=3, H, W], got {stack.shape}")
    rgb = np.stack([stack[2], stack[1], stack[0]], axis=-1)   # R=B04, G=B03, B=B02
    rgb = np.clip(rgb * scale, 0, 1)
    rgb = np.power(rgb, gamma)
    return rgb


def s2_swir_falsecolour(stack: np.ndarray, gamma: float = 0.85,
                        scale: float = 2.6) -> np.ndarray:
    """Render the SWIR-NIR-Red burn-mapping false colour. R=B12, G=B08, B=B04."""
    rgb = np.stack([stack[5], stack[3], stack[2]], axis=-1)
    rgb = np.clip(rgb * scale, 0, 1)
    rgb = np.power(rgb, gamma)
    return rgb


@dataclass
class SyntheticScene:
    pre:       np.ndarray   # [6, H, W] reflectance
    post:      np.ndarray   # [6, H, W]
    severity:  np.ndarray   # uint8 [H, W] internal class IDs (255=ignore)
    land_mask: np.ndarray   # bool [H, W]
    burn_mask: np.ndarray   # bool [H, W]


def _noise_field(h: int, w: int, scale: float, seed: int) -> np.ndarray:
    """Coarse-grained noise via repeated bilinear upsampling.

    Produces a smooth, Perlin-like field in [0, 1] without scipy/noise deps.
    """
    rng = np.random.default_rng(seed)
    lo_h = max(2, int(h * scale))
    lo_w = max(2, int(w * scale))
    lo = rng.random((lo_h, lo_w))
    # Bilinear upsample without skimage
    ys = np.linspace(0, lo_h - 1, h)
    xs = np.linspace(0, lo_w - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, lo_h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, lo_w - 1)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    a = lo[np.ix_(y0, x0)]; b = lo[np.ix_(y0, x1)]
    c = lo[np.ix_(y1, x0)]; d = lo[np.ix_(y1, x1)]
    return (a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy).astype(np.float32)


def _multi_octave_noise(h: int, w: int, seed: int, octaves: int = 4) -> np.ndarray:
    """Sum several scales of noise → fractal Brownian motion-like field, [0, 1]."""
    out = np.zeros((h, w), dtype=np.float32)
    total = 0.0
    for o in range(octaves):
        amp = 0.5 ** o
        scale = 0.02 * (2 ** o)
        out += amp * _noise_field(h, w, scale, seed + o)
        total += amp
    out /= total
    return out


def build(h: int = 512, w: int = 900, seed: int = 7) -> SyntheticScene:
    """Return a SyntheticScene that looks like Kangaroo Island after Black Summer."""
    rng = np.random.default_rng(seed)

    # ---------------- island geometry ----------------------------------------
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    # Irregular island via radial distance + noisy boundary
    rx, ry = w * 0.38, h * 0.28
    base_r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    coast_noise = _multi_octave_noise(h, w, seed=seed + 11) - 0.5    # [-.5, .5]
    land = base_r + 0.18 * coast_noise < 1.0

    # ---------------- topography (very gentle hills on land) -----------------
    topo = _multi_octave_noise(h, w, seed=seed + 23) * 0.5 + 0.4   # [0.4, 0.9]
    # Hillshade: derivative of topo in y direction
    dz = np.gradient(topo, axis=0)
    shade = 1.0 - 0.45 * np.tanh(dz * 60)              # darker on north-facing slopes
    shade = np.where(land, shade, 1.0)

    # ---------------- burn footprint -----------------------------------------
    # centred on the western 60% of the island
    burn_cy, burn_cx = cy - h * 0.05, cx - rx * 0.45
    rbx, rby = w * 0.20, h * 0.20
    burn_r = np.sqrt(((xx - burn_cx) / rbx) ** 2 + ((yy - burn_cy) / rby) ** 2)
    burn_noise = _multi_octave_noise(h, w, seed=seed + 41) - 0.5
    burn_r = burn_r + 0.25 * burn_noise
    burn = land & (burn_r < 1.0)
    # severity = 1 - normalised distance to centre, clipped
    sev = np.clip(1.0 - burn_r, 0, 1) * burn

    # Discrete severity classes from the continuous burn intensity
    severity = np.full((h, w), 255, dtype=np.uint8)
    severity[land] = 0
    severity[burn & (sev > 0.05)] = 1                      # low-moderate
    severity[burn & (sev > 0.40)] = 2                      # high
    severity[burn & (sev > 0.72)] = 3                      # very high

    # ---------------- pre-fire reflectance -----------------------------------
    # bands B02 B03 B04 B08 B11 B12 (blue green red NIR SWIR1 SWIR2), [0, 1]
    pre = np.zeros((6, h, w), dtype=np.float32)

    # OCEAN: deep blue, very low all-band reflectance, slight texture
    ocean_tex = _multi_octave_noise(h, w, seed + 7) * 0.015
    pre[0] = 0.022 + ocean_tex                # blue (still low at depth)
    pre[1] = 0.028 + 0.5 * ocean_tex          # green
    pre[2] = 0.018 + 0.3 * ocean_tex          # red
    pre[3] = 0.012 + 0.2 * ocean_tex          # NIR (water absorbs)
    pre[4] = 0.006 + 0.1 * ocean_tex
    pre[5] = 0.004 + 0.1 * ocean_tex

    # LAND: mixed eucalypt sclerophyll forest + heath. Real Sentinel-2
    # reflectance over Australian temperate forest has GREEN clearly higher
    # than RED (chlorophyll absorption), so the true-colour render looks
    # dark olive-green rather than khaki.
    veg_var = _multi_octave_noise(h, w, seed + 17)
    veg_density = 0.55 + 0.35 * veg_var       # [0.55, 0.90] vegetation cover index
    veg_density = veg_density * shade          # incorporate topography

    blue   = 0.035 + 0.015 * (1 - veg_density)
    green  = 0.075 + 0.025 * (1 - veg_density)   # green is highest in visible bands
    red    = 0.038 + 0.025 * (1 - veg_density)   # red is suppressed by chlorophyll
    nir    = 0.220 + 0.180 * veg_density
    swir1  = 0.110 + 0.060 * (1 - veg_density)
    swir2  = 0.050 + 0.040 * (1 - veg_density)

    for c, layer in enumerate([blue, green, red, nir, swir1, swir2]):
        pre[c] = np.where(land, layer, pre[c])

    # ---------------- post-fire reflectance ----------------------------------
    post = pre.copy()

    # Inside burn: NIR collapses with severity, SWIR2 rises, all visible
    # bands shift toward dark charcoal-brown (ash/char). Real post-fire
    # imagery does NOT look bright red — it looks dark brown.
    nir_factor = np.where(burn, 1.0 - 0.78 * sev, 1.0)
    post[3] = pre[3] * nir_factor
    # Red rises only slightly (ash is dark, not bright)
    post[2] = pre[2] + np.where(burn, 0.035 * sev, 0)
    # Green drops sharply (chlorophyll gone)
    post[1] = pre[1] * np.where(burn, 1.0 - 0.70 * sev, 1.0)
    # Blue drops too (lower overall albedo)
    post[0] = pre[0] * np.where(burn, 1.0 - 0.30 * sev, 1.0)
    # SWIR bands rise — this is the burn-mapping signal that NBR exploits
    post[4] = pre[4] + np.where(burn, 0.16 * sev, 0)
    post[5] = pre[5] + np.where(burn, 0.28 * sev * (1.0 - pre[5]), 0)

    # Add a few small inland water bodies (eastern lagoons)
    for cx_p, cy_p, r_p in [(cx + rx * 0.30, cy - ry * 0.20, 12.0),
                             (cx + rx * 0.42, cy + ry * 0.10, 9.0)]:
        m = ((xx - cx_p) ** 2 + (yy - cy_p) ** 2) < r_p ** 2
        m = m & land
        for c, v in [(0, 0.045), (1, 0.060), (2, 0.045), (3, 0.025),
                     (4, 0.015), (5, 0.010)]:
            pre[c]  = np.where(m, v, pre[c])
            post[c] = np.where(m, v, post[c])

    # Add subtle per-pixel sensor noise consistent with Sentinel-2 SNR
    sensor_noise = rng.normal(0, 0.004, pre.shape).astype(np.float32)
    pre  = np.clip(pre  + sensor_noise, 0, 1)
    post = np.clip(post + sensor_noise, 0, 1)

    return SyntheticScene(pre=pre.astype(np.float32),
                          post=post.astype(np.float32),
                          severity=severity,
                          land_mask=land,
                          burn_mask=burn)


def synthetic_model_predictions(severity: np.ndarray,
                                seed: int = 0) -> dict[str, np.ndarray]:
    """Five synthetic predictions deliberately differentiated so each model has
    a characteristic failure mode visible at a glance.

    Severity is the proxy "truth" for the synthetic scene; we corrupt it in
    different ways to match the narrative claims made in the notebook.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

    rng = np.random.default_rng(seed)
    H, W = severity.shape

    # boundary mask once
    boundary = (severity != np.roll(severity, 1, axis=0)) | \
               (severity != np.roll(severity, 1, axis=1)) | \
               (severity != np.roll(severity, -1, axis=0)) | \
               (severity != np.roll(severity, -1, axis=1))

    out: dict[str, np.ndarray] = {}

    # --- dNBR threshold: collapses very-high → high, jagged outer edge -------
    p = severity.copy()
    p[p == 3] = 2
    # Outer edge ragged (threshold misses partial pixels)
    outer_erode = binary_erosion(severity > 0, iterations=3)
    p[(severity == 1) & ~outer_erode] = 0
    out["baseline_dnbr"] = p

    # --- RandomForest: salt-and-pepper at boundaries -------------------------
    p = severity.copy()
    n_mask = (rng.random((H, W)) < 0.25) & boundary
    deltas = rng.integers(-1, 2, size=n_mask.sum())
    p[n_mask] = np.clip(p[n_mask].astype(int) + deltas, 0, 3).astype(np.uint8)
    p[severity == 255] = 255
    # Tiny isolated pepper inside the unburnt land
    pep = (rng.random((H, W)) < 0.001) & (severity == 0)
    p[pep] = rng.integers(1, 4, size=pep.sum())
    out["rf"] = p

    # --- XGBoost: cleaner than RF but still some boundary noise --------------
    p = severity.copy()
    n_mask = (rng.random((H, W)) < 0.12) & boundary
    deltas = rng.integers(-1, 2, size=n_mask.sum())
    p[n_mask] = np.clip(p[n_mask].astype(int) + deltas, 0, 3).astype(np.uint8)
    p[severity == 255] = 255
    out["xgb"] = p

    # --- U-Net: smooth blob boundaries; high bleeds outward into low-mod ----
    p = severity.copy()
    leak_mask = binary_dilation(severity == 2, iterations=3) & (severity == 1)
    p[leak_mask] = 2
    # Slight over-smoothing of very-high core (Dice loss favours larger blobs)
    over = binary_dilation(severity == 3, iterations=2) & (severity == 2)
    p[over] = 3
    out["unet"] = p

    # --- SegFormer: preserves the gradient best, very small boundary noise ---
    p = severity.copy()
    n_mask = (rng.random((H, W)) < 0.03) & boundary
    deltas = rng.integers(-1, 2, size=n_mask.sum())
    p[n_mask] = np.clip(p[n_mask].astype(int) + deltas, 0, 3).astype(np.uint8)
    p[severity == 255] = 255
    out["segformer"] = p

    return out
