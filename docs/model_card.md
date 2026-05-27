# Model Card — Australian Bushfire Burn Severity Mapper

## Model details

- **Project**: Retrospective burn-severity mapping over four 2019–2020 Australian "Black Summer" fire events.
- **Versions evaluated**:
  - `baseline_dnbr` — threshold on Differenced Normalised Burn Ratio (binary at three thresholds + USGS-style multiclass)
  - `rf` — `sklearn.ensemble.RandomForestClassifier`, 500 trees, max_depth 30, on 18-channel per-pixel features
  - `xgb` — `xgboost.XGBClassifier`, `multi:softprob`, 800 trees, max_depth 8, on the same 18 features
  - `unet` — `segmentation_models_pytorch.Unet` with ResNet-34 encoder, 18 input channels, 4 output classes, trained from scratch
  - `segformer` — HuggingFace SegFormer-B0 (`nvidia/mit-b0`) with the first patch-embedding Conv2d **inflated from 3→18 channels** (averaged RGB kernel scaled by 3/18 to preserve output magnitude)
- **Task**: 4-class semantic segmentation `{unburnt, low_moderate, high, very_high}` at 10 m on Sentinel-2 imagery.
- **Hardware**: Mac Mini M4 Pro, 64 GB unified memory, Apple Silicon MPS only (no CUDA). `bfloat16` autocast on MPS; fp32 cast for loss + grad-norm clipped at 1.0.
- **Date**: 2026-05-27.

## Intended use

- **Research and education only.** Method comparison; reproducible illustration of an end-to-end remote-sensing ML pipeline.
- **Not for** emergency response, public warning, dispatch, evacuation planning, insurance claims, or any safety-of-life decision.

## Training data

| Source | Role | Licence |
|---|---|---|
| Sentinel-2 Level-2A surface reflectance via Microsoft Planetary Computer STAC | Imagery (6 reflectance bands per pre/post composite) | CC-BY-SA 3.0 IGO (Copernicus) |
| AUS GEEBAM Fire Severity (DCCEEW ArcGIS REST) | **Proxy** labels (4 collapsed classes; class 1 → ignore) | CC-BY 4.0 |
| NIAFED 2019–20 (planned M3 refinement) | Refine AOI polygons | CC-BY 4.0 |
| GA SRTM 1 second DEM | Slope feature | CC-BY 4.0 |
| DEA Land Cover | Per-land-cover stratified evaluation only | CC-BY 4.0 |

Areas of interest (final event-wise split):

| Event | Split | Region | Approx. extent |
|---|---|---|---|
| Currowan 2019–2020 | train | NSW South Coast | ~500,000 ha |
| Gospers Mountain 2019–2020 | train | NSW Blue Mountains / Wollemi | ~512,000 ha |
| Kangaroo Island 2019–2020 | val | SA | ~210,000 ha |
| East Gippsland 2019–2020 | test | VIC | ~1,500,000 ha |

Per-pixel sampling for the classical models is **stratified by class** at 50,000 pixels per class, drawn uniformly across the train tiles to prevent the majority `unburnt` class from drowning the minority severity classes.

Cloud-masking rule: Sentinel-2 SCL classes `{0, 1, 3, 8, 9, 10, 11}` are blocked, the mask is dilated by 2 pixels at 10 m, then a per-pixel temporal **median composite** is built over the cloud-cleared pre/post windows. Composites are written to disk in the per-AOI UTM zone; GEEBAM labels are nearest-neighbour reprojected from EPSG:3577 (40 m) to the same UTM grid at 10 m.

## Metrics

Metrics are computed under **two split protocols**:

1. **Vertical slice (smoke-test).** Kangaroo Island only, random tile split. *These numbers are spatially auto-correlated and inflate the true generalisation score by an unknown but non-trivial margin. They are reported solely to verify the pipeline runs end-to-end; never as a headline result.*
2. **Event-wise hold-out (M10, real numbers).** Train Currowan + Gospers Mountain, val Kangaroo Island, test East Gippsland.

Reported per (model, event):
- per-class IoU and F1 for `{unburnt, low_mod, high, very_high}`
- macro-IoU, macro-F1, accuracy
- 4×4 confusion matrix (row-normalised, saved as PNG)
- binary collapsed (burnt vs unburnt) F1 + IoU
- per-land-cover stratified macro-IoU (woody, shrubland, herbaceous, cropland, built_bare, water)
- per-slope stratified macro-IoU (5°, 15°, 30°, 60°, 90° bins)
- reliability diagram + Expected Calibration Error (ECE) + Brier score for the burnt-class probability

All metric tables and figures are regenerated from saved prediction GeoTIFFs by `python -m src.evaluation.evaluate --all-events`.

## Limitations

1. **AUS GEEBAM is a proxy, not ground truth.** The classes are derived from satellite indices without field calibration. Low and moderate severity are collapsed by the upstream dataset. AUS GEEBAM's own published comparisons against state products report only 48–82% overall agreement at four classes (72–92% at two classes), so any model claiming to "match GEEBAM perfectly" would actually be overfitting GEEBAM's specific algorithmic choices.
2. **Label resolution mismatch.** GEEBAM is published at 40 m; predictions are made at 10 m. Each 10 m prediction pixel is effectively supervised by a 40 m label cell — a 16× scale gap that flows through every metric. Per-class IoU is the right metric to read here; pixel accuracy is misleading.
3. **Severity is ecosystem-dependent.** GEEBAM itself moved from plain dNBR to vegetation-stratified RNBR specifically because dNBR underperforms in some Australian biomes. Any plain dNBR-only baseline will underperform especially in low-biomass and heterogeneous fire-edge contexts.
4. **Cloud / smoke gaps.** Pixels with no clear observation in either pre or post window are written as `ignore_id=255` and excluded from metrics. They are not silently filled.
5. **Topographic shadow can masquerade as burn.** Steep south-facing slopes can look dark in post-fire imagery even when unburnt. The slope-stratified metric is the diagnostic for this.
6. **Generalisation across fires is fragile.** Per-band normalisation statistics are computed from the train events only; whether the U-Net and SegFormer generalise to the test event (East Gippsland) is the *real* test of these models. Random-tile metrics from a single AOI flatter the result.

## Ethical considerations

- **Non-operational** boilerplate appears in the README, in `docs/demo/non_expert_panel.md`, in this model card, and in every public-facing figure.
- The proxy-label caveat is propagated to downstream users.
- No personally identifiable information is in the training data; the imagery covers public lands.
- The pipeline does **not** publish anything that could mislead a non-technical viewer into thinking the maps are operationally valid.

## Reproducibility

- Commit hash for any reported result is in `outputs/.../<run>/config_snapshot.yaml` and in the per-raster `*.provenance.json` sidecars.
- Random seed `42` is set across `random`, `numpy`, `torch`, and `PYTHONHASHSEED` by `src.utils.seed.set_seeds()`.
- `pyproject.toml` pins dependency lower bounds. A frozen `requirements.txt` snapshot is produced at the M12 commit; recompute with `pip freeze`.
- CI runs lint, unit tests, and a `--fast-mode` smoke training on frozen fixture tiles. No live STAC calls in CI.

## Codex review trail

Every milestone gate's Codex CLI review is committed under `docs/reviews/M<N>_codex_review.md` so that the *process* (not only the artefacts) is part of the portfolio.
