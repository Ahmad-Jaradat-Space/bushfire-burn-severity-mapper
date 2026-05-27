"""Load the Kangaroo Island scene used by the notebook.

If real data has been fetched (live composites + aligned label + at least one
model prediction on disk), `load_kangaroo()` returns it. Otherwise the function
falls back to the deterministic synthetic stand-in so the notebook always
renders.

The returned dataclass exposes the same interface either way, so the notebook
cells don't need to branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.geo import REPO_ROOT
from src.viz.synthetic_scene import (build as build_synthetic_scene,
                                       synthetic_model_predictions)


@dataclass
class KangarooScene:
    pre: np.ndarray            # [6, H, W] reflectance (B02 B03 B04 B08 B11 B12)
    post: np.ndarray           # [6, H, W]
    severity: np.ndarray       # uint8 [H, W] internal class IDs (255=ignore)
    land_mask: np.ndarray      # bool [H, W]
    burn_mask: np.ndarray      # bool [H, W]
    is_real: bool
    crs: Optional[str] = None
    source_label: Optional[str] = None
    n_pre_scenes: Optional[int] = None
    n_post_scenes: Optional[int] = None
    clear_fraction_pre: Optional[float] = None
    clear_fraction_post: Optional[float] = None


def _read_band_stack(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def _read_mask(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1).astype(bool)


def _read_label(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.uint8)


def _read_crs(path: Path) -> str:
    import rasterio
    with rasterio.open(path) as ds:
        return str(ds.crs)


def load_kangaroo(force_synthetic: bool = False,
                  event_id: str = "kangaroo_island_2019_2020") -> KangarooScene:
    interim = REPO_ROOT / "data" / "interim" / event_id
    # Mask filename convention varies between pipeline versions: accept either.
    pre_mask = next((interim / n for n in ("mask_pre_10m.tif", "pre_mask_10m.tif")
                     if (interim / n).exists()), interim / "mask_pre_10m.tif")
    post_mask = next((interim / n for n in ("mask_post_10m.tif", "post_mask_10m.tif")
                      if (interim / n).exists()), interim / "mask_post_10m.tif")
    paths = {
        "pre":   interim / "pre_stack_10m.tif",
        "post":  interim / "post_stack_10m.tif",
        "pre_mask":   pre_mask,
        "post_mask":  post_mask,
        "label": interim / "label_10m.tif",
    }
    have_real = (not force_synthetic) and all(p.exists() for p in paths.values())

    if not have_real:
        s = build_synthetic_scene(h=512, w=900, seed=7)
        return KangarooScene(
            pre=s.pre, post=s.post, severity=s.severity,
            land_mask=s.land_mask, burn_mask=s.burn_mask,
            is_real=False, source_label="synthetic stand-in (deterministic seed=7)",
        )

    pre = _read_band_stack(paths["pre"])
    post = _read_band_stack(paths["post"])
    label = _read_label(paths["label"])
    pre_mask = _read_mask(paths["pre_mask"])
    post_mask = _read_mask(paths["post_mask"])
    crs = _read_crs(paths["pre"])

    # Land mask = pixels where both pre and post observed clearly
    # (the GEEBAM extent gives us the burn mask region directly via label != 255)
    land_mask = pre_mask & post_mask
    burn_mask = (label > 0) & (label != 255)

    # Read scene counts from provenance for the caption
    n_pre, clear_pre = None, None
    n_post, clear_post = None, None
    try:
        import json
        for kind, attr_n, attr_c in [("pre_stack_10m.tif", "n_pre", "clear_pre"),
                                     ("post_stack_10m.tif", "n_post", "clear_post")]:
            m = paths["pre"].parent / f"{kind}.provenance.json"
            if m.exists():
                d = json.loads(m.read_text())
                if kind.startswith("pre"):
                    n_pre = d["inputs"].get("n_scenes")
                    clear_pre = d["inputs"].get("clear_fraction")
                else:
                    n_post = d["inputs"].get("n_scenes")
                    clear_post = d["inputs"].get("clear_fraction")
    except Exception:
        pass

    return KangarooScene(
        pre=pre, post=post, severity=label,
        land_mask=land_mask, burn_mask=burn_mask,
        is_real=True, crs=crs,
        source_label="Sentinel-2 L2A via Microsoft Planetary Computer + AUS GEEBAM via DCCEEW",
        n_pre_scenes=n_pre, n_post_scenes=n_post,
        clear_fraction_pre=clear_pre, clear_fraction_post=clear_post,
    )


def model_predictions(scene: KangarooScene, seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Return (predictions, is_real_flags) per model.

    Real predictions are read from outputs/predictions/<model>/<event_id>.tif.
    Models without a trained checkpoint are returned as `None` so the caller
    can decide whether to render a placeholder or drop the panel.
    """
    if not scene.is_real:
        return synthetic_model_predictions(scene.severity, seed=seed), \
               {k: False for k in ("baseline_dnbr", "rf", "xgb", "unet", "segformer")}

    event_id = "kangaroo_island_2019_2020"
    candidates = {
        "baseline_dnbr": f"outputs/predictions/baseline_dnbr/{event_id}/multiclass.tif",
        "rf":            f"outputs/predictions/rf_multiclass/{event_id}.tif",
        "xgb":           f"outputs/predictions/xgb_multiclass/{event_id}.tif",
        "unet":          f"outputs/predictions/unet_multiclass/{event_id}.tif",
        "segformer":     f"outputs/predictions/segformer_multiclass/{event_id}.tif",
    }
    out: dict[str, np.ndarray] = {}
    is_real: dict[str, bool] = {}
    for key, rel in candidates.items():
        p = REPO_ROOT / rel
        if p.exists():
            out[key] = _read_label(p)
            is_real[key] = True
        else:
            is_real[key] = False
    return out, is_real
