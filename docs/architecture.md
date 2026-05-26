# Architecture & Engineering Decisions

This document captures the **non-obvious decisions** that govern the pipeline. They are committed at M1 so downstream code doesn't drift.

## 1. Coordinate reference systems (CRS)

| Use case | CRS | Why |
|---|---|---|
| Sentinel-2 reflectance, composites, tiles, predictions | **Per-AOI UTM** (EPSG:32750–32756 for Australia) | Matches the native S2 tile projection within each zone; equal-area metric work; minimises resampling cost on download |
| GEEBAM label download (ArcGIS `exportImage`) | **EPSG:3577** (GDA94 Australian Albers) | Native distribution CRS for AUS GEEBAM and most DCCEEW raster products; requesting in 3577 avoids a server-side reproject on the way out |
| National-scale display maps (README, Folium overview) | **EPSG:3577** | Visually correct for whole-continent maps; Folium overlays auto-reproject to web Mercator for tile rendering |
| AOI bounding box JSON / STAC search | **EPSG:4326** (WGS84) | STAC `bbox` and `intersects` conventions; geojson default |

**Where reprojections happen:**
1. GEEBAM downloaded in EPSG:3577 → reprojected to the AOI's UTM zone with **nearest-neighbour** resampling (categorical data).
2. S2 scenes are kept in their native UTM zone. If an AOI straddles two UTM zones, scenes are reprojected to the dominant zone (>50% area coverage) using **bilinear** resampling — documented per AOI in `data/metadata/events.geojson`.
3. Land cover (DEA, 30 m) and DEM (GA SRTM, 30 m) are reprojected to the AOI UTM at 10 m using **nearest** for land cover and **bilinear** for elevation, then slope/aspect derived in UTM.

All reprojections are logged in the per-raster `*.provenance.json` sidecar.

## 2. Temporal-window policy

Every experiment runs in one of two modes (set in its experiment config under `experiment.temporal_mode`):

- **`event_specific`** — per-AOI Pre/Post windows from the AOI GeoJSON `properties.pre_window` / `post_window`. Tight windows give cleaner composites but make cross-AOI comparison harder.
- **`geebam_aligned`** — fixed southern-season windows used by AUS GEEBAM (`configs/config.yaml::temporal_windows.geebam_aligned`): pre `2018-04-15→2019-04-15`, post `2019-11-15→2020-02-15` and `2020-01-15→2020-05-15`. Use this mode when the experiment compares directly against GEEBAM class labels — it removes a confound that would otherwise be hidden in error analysis.

The baseline (M5) runs both modes and reports the delta. Subsequent models default to `event_specific` for the headline result, with `geebam_aligned` as an ablation in the model card.

## 3. Provenance manifest schema

Every raster under `data/interim/`, `data/processed/`, and `outputs/predictions/` has a sidecar `<file>.provenance.json` written by `src.utils.provenance.write_manifest()`. Schema:

```json
{
  "output_file": "post_stack_10m.tif",
  "event_id": "kangaroo_island_2019_2020",
  "pipeline_step": "preprocess",
  "git_sha": "abc1234",
  "generated_utc": "2026-05-27T09:00:00+00:00",
  "crs": "EPSG:32753",
  "resampling": "bilinear",
  "class_remap": null,
  "inputs": {
    "stac_items": ["S2B_MSIL2A_20200120T002659_N0213_R016_T54HXG_..."],
    "stac_endpoint": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "scl_mask_classes": [0, 1, 3, 8, 9, 10, 11]
  },
  "notes": null
}
```

This is what lets a reviewer (Codex or a hiring manager) re-derive any figure from its source assets.

## 4. Class map

GEEBAM publishes 5 classes; we collapse the `1=unburnt-outside-extent` value to an ignore index and remap the rest to dense 0–3 IDs for ML:

| GEEBAM raw | Meaning | Internal class ID | Internal name |
|---|---|---|---|
| 1 | Unburnt outside extent / no-data | `255` (ignore) | — |
| 2 | Unburnt | 0 | `unburnt` |
| 3 | Low to moderate | 1 | `low_mod` |
| 4 | High | 2 | `high` |
| 5 | Very high | 3 | `very_high` |

`ignore_index=255` is honoured by every model loss and every metric. **GEEBAM combines low+moderate into one class** — this is an upstream limitation we propagate into the model card.

## 5. Library anchors

- STAC search + signed assets — `pystac-client` + `planetary-computer` + `odc-stac`
- Raster IO + reproject — `rasterio` + `rioxarray`
- Classical ML — `scikit-learn` + `xgboost`
- U-Net — `segmentation_models_pytorch.Unet(encoder_name="resnet34", in_channels=18, classes=4)`
- SegFormer-B0 — HF `transformers.SegformerForSemanticSegmentation` from `nvidia/mit-b0`, first-conv weight inflation across 18 channels
- Metrics — `torchmetrics` (correct `ignore_index` handling)
- Config — `omegaconf` (no Hydra multirun overhead)
- Viz — `matplotlib`, `rasterio.plot`, `folium`, `contextily`
- Tracking — local TensorBoard

TorchGeo is **deliberately not** the anchor for v1 — its dataloaders assume a `RasterDataset` pattern that fights the event-wise pre/post compositing workflow. Revisit in v2 for the `ResNet50_Weights.SENTINEL2_ALL_MOCO` pretrained backbone.

## 6. MPS-specific decisions

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is exported in `scripts/*.sh` launchers, **not** set inside any Python module. Setting it after `torch` is imported is a no-op for the ops already registered.
- Mixed precision: `torch.autocast(device_type="mps", dtype=torch.bfloat16)`. Loss and final logits cast back to fp32. Gradient clipping `max_norm=1.0`.
- Step-time guard: every training script times the first 10 steps. If mean step time > 3× the recorded baseline for that model+batch+tile_size, abort with a CPU-fallback warning.
- Inference: sliding window at 256² stride 128, batch 8. M4 Pro's 64 GB unified memory holds full Kangaroo predictions in RAM — no tile-streaming write needed.
- `--fast-mode` toggle: 5 epochs, 200-tile subset, early stop. Used for first-time benchmarking and CI smoke runs.

## 7. Honest framing

- **Random tile split metrics are smoke-tests**, not generalisation numbers. README/figures published before M10 (event-wise holdout) carry that caption.
- **SegFormer may ship as "attempted benchmark with caveats"** if MPS training balloons past the 24 h budget. Honest framing beats forced completion.
- **AUS GEEBAM is proxy labels, not ground truth.** This caveat is visible in the README hero, every comparison panel, and `docs/model_card.md`.
