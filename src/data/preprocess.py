"""Pre/post composites + label alignment.

Inputs (from M3): data/raw/sentinel2/<event>/{pre,post}/stack.nc and
                 data/raw/labels/aus_geebam/<event>/label_native_3577.tif.

Outputs (under data/interim/<event>/):
  - pre_stack_10m.tif     6 reflectance bands × H × W, working UTM
  - post_stack_10m.tif    same shape
  - mask_pre_10m.tif      uint8 clear mask
  - mask_post_10m.tif     uint8 clear mask
  - label_10m.tif         remapped to internal class IDs (255 = ignore)

Reflectance composites: per-pixel temporal median over the cloud-masked stack.
Label resample: nearest-neighbour from EPSG:3577 native → working UTM at 10 m.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from rasterio.enums import Resampling

from src.data.class_map import remap_geebam
from src.data.cloud_mask import clear_fraction, scl_to_clear_mask
from src.utils.geo import REPO_ROOT, utm_epsg_for_aoi
from src.utils.logging_utils import get_logger
from src.utils.provenance import write_manifest

REFLECTANCE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

log = get_logger(__name__)


def _open_stack(stack_path: Path) -> xr.Dataset:
    return xr.open_dataset(stack_path, chunks={"time": 1})


def _build_composite(ds: xr.Dataset, bands: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (composite [C, H, W], clear_mask_any [H, W]) in float32 / bool."""
    scl = ds["SCL"].values  # [T, H, W] uint8
    clear_per_t = np.stack([scl_to_clear_mask(scl[t]) for t in range(scl.shape[0])])
    band_stack = np.stack([ds[b].values for b in bands])  # [C, T, H, W]
    band_stack = band_stack.astype(np.float32)
    band_stack /= 10000.0  # L2A reflectance scale

    # Mask non-clear pixels with NaN, then nanmedian over time → [C, H, W]
    bs = band_stack.transpose(1, 0, 2, 3)  # [T, C, H, W]
    clear_4d = np.broadcast_to(clear_per_t[:, None, :, :], bs.shape)
    bs = np.where(clear_4d, bs, np.nan)
    composite = np.nanmedian(bs, axis=0)  # [C, H, W]

    clear_any = clear_per_t.any(axis=0)  # [H, W]
    return composite, clear_any


def _write_geotiff(path: Path, data: np.ndarray, transform, crs, dtype=None,
                   nodata=None) -> None:
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    count, h, w = data.shape
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": dtype or data.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2 if (dtype or data.dtype) in (np.float32, np.float64, "float32", "float64") else 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype(meta["dtype"]))


def preprocess_event(event_id: str,
                     raw_dir: Path | None = None,
                     out_dir: Path | None = None) -> dict[str, Path]:
    raw_dir = raw_dir or REPO_ROOT / "data" / "raw"
    out_dir = out_dir or REPO_ROOT / "data" / "interim" / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    epsg = utm_epsg_for_aoi(event_id)
    working_crs = f"EPSG:{epsg}"

    outputs: dict[str, Path] = {}

    for kind in ("pre", "post"):
        stack_path = raw_dir / "sentinel2" / event_id / kind / "stack.nc"
        if not stack_path.exists():
            raise FileNotFoundError(f"Missing {kind} stack at {stack_path}")
        ds = _open_stack(stack_path)
        composite, clear_any = _build_composite(ds, REFLECTANCE_BANDS)

        # Reference transform/crs from the first reflectance DataArray
        ref = ds[REFLECTANCE_BANDS[0]].isel(time=0)
        transform = ref.rio.transform()
        crs = ref.rio.crs or working_crs

        out_stack = out_dir / f"{kind}_stack_10m.tif"
        _write_geotiff(out_stack, composite.astype(np.float32), transform, crs, dtype=np.float32,
                       nodata=float("nan"))
        out_mask = out_dir / f"mask_{kind}_10m.tif"
        _write_geotiff(out_mask, clear_any.astype(np.uint8), transform, crs, dtype=np.uint8, nodata=0)

        write_manifest(
            out_stack,
            event_id=event_id,
            pipeline_step=f"preprocess.composite.{kind}",
            inputs={
                "source_stack": str(stack_path.relative_to(REPO_ROOT)),
                "bands": list(REFLECTANCE_BANDS),
                "composite_method": "nanmedian",
                "scl_mask_classes": [0, 1, 3, 8, 9, 10, 11],
                "scl_dilate_pixels": 2,
                "clear_fraction": clear_fraction(clear_any),
            },
            crs=str(crs),
            resampling=None,
            notes=None,
        )
        outputs[f"{kind}_stack"] = out_stack
        outputs[f"{kind}_mask"] = out_mask
        log.info("[%s] composite -> %s  clear=%.1f%%", kind, out_stack.name,
                 100 * clear_fraction(clear_any))

    # Label alignment: reproject GEEBAM to working UTM at 10 m using nearest
    label_native = raw_dir / "labels" / "aus_geebam" / event_id / "label_native_3577.tif"
    if not label_native.exists():
        raise FileNotFoundError(f"Missing GEEBAM label: {label_native}")

    with rasterio.open(out_dir / "post_stack_10m.tif") as ref_ds:
        target_crs = ref_ds.crs
        target_transform = ref_ds.transform
        target_w = ref_ds.width
        target_h = ref_ds.height

    label = xr.open_dataarray(label_native, engine="rasterio")
    if "band" in label.dims:
        label = label.isel(band=0)
    label.rio.write_crs(label.rio.crs or "EPSG:3577", inplace=True)
    label_warped = label.rio.reproject(
        dst_crs=target_crs,
        shape=(target_h, target_w),
        transform=target_transform,
        resampling=Resampling.nearest,
    )

    remapped = remap_geebam(label_warped.values.astype(np.uint8))
    out_label = out_dir / "label_10m.tif"
    _write_geotiff(out_label, remapped.astype(np.uint8), target_transform, target_crs,
                   dtype=np.uint8, nodata=255)
    outputs["label"] = out_label

    write_manifest(
        out_label,
        event_id=event_id,
        pipeline_step="preprocess.label_align",
        inputs={
            "source_label": str(label_native.relative_to(REPO_ROOT)),
            "source_crs": "EPSG:3577",
            "source_resolution_m": 40,
            "target_resolution_m": 10,
            "remap_table": {"0/1": 255, "2": 0, "3": 1, "4": 2, "5": 3},
        },
        crs=str(target_crs),
        resampling="nearest",
        class_remap={"2": 0, "3": 1, "4": 2, "5": 3},
        notes="GEEBAM 40m → 10m nearest; class 1 (unburnt-outside-extent) and 0 (nodata) → ignore_id 255.",
    )

    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Build pre/post composites and aligned labels.")
    p.add_argument("--event", required=True)
    args = p.parse_args()
    outs = preprocess_event(args.event)
    for k, v in outs.items():
        log.info("%s -> %s", k, v)


if __name__ == "__main__":
    main()
