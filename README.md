# Australian Bushfire Burn Severity Mapper

> **Status — milestone M1 (scaffold).** The repository is being built in milestones; the README will fill in as results land. Latest hero figure and metrics will appear here at M6.5 (vertical-slice demo) and M12 (final).

Retrospective mapping of bushfire burn severity over four Black Summer 2019–2020 events from public Sentinel-2 imagery, comparing a threshold baseline, two classical ML models, and two deep semantic-segmentation models.

---

## ⚠️ Non-operational notice

This repository is **research and education only**. It is **not** for emergency response, public warning, dispatch, evacuation planning, or any safety-of-life decision. The labels driving the supervised models (AUS GEEBAM) are a **public proxy** derived from satellite indices — not field-validated ground truth.

See [`docs/data_dictionary.md`](docs/data_dictionary.md) and the (forthcoming) `docs/model_card.md` for the full caveat list.

---

## What this project does

1. **Ingests** public satellite imagery (Sentinel-2 Level-2A surface reflectance) via the Microsoft Planetary Computer STAC API for four Australian fire events.
2. **Aligns** publicly-published severity labels (AUS GEEBAM) onto the Sentinel-2 grid.
3. **Compares** five severity-mapping methods on the same data and the same event-wise hold-out split:
   - Threshold baseline on differenced Normalised Burn Ratio (dNBR)
   - RandomForest on engineered spectral + topographic features
   - XGBoost on the same features
   - U-Net (ResNet-34 encoder) semantic segmentation
   - SegFormer-B0 (HuggingFace `mit-b0`) semantic segmentation
4. **Reports** per-event IoU/F1, per-class precision/recall, per-land-cover stratification, calibration reliability, and confusion matrices.
5. **Publishes** a reproducible report card with figures regenerable from saved prediction GeoTIFFs.

## Areas of interest (AOIs)

| Event | Region | Date window |
|---|---|---|
| Kangaroo Island | SA | Dec 2019 – Feb 2020 |
| Currowan | NSW South Coast | Nov 2019 – Jan 2020 |
| Gospers Mountain | NSW Blue Mountains / Wollemi | Oct 2019 – Jan 2020 |
| East Gippsland | VIC | Nov 2019 – Mar 2020 |

## Quickstart (M1 — placeholder)

```bash
# Clone
git clone <repo> bushfire-burn-severity-mapper
cd bushfire-burn-severity-mapper

# Install (core deps; add [dl] extras for the deep models in M7–M8)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify the scaffold
pytest

# Source env (PYTORCH_ENABLE_MPS_FALLBACK=1 must be exported before `torch` imports)
source scripts/setup_env.sh

# Skeleton runs (no network calls yet — those land in M2/M3):
python -m src.data.fetch_labels --event kangaroo_island_2019_2020
python -m src.data.fetch_sentinel --event kangaroo_island_2019_2020
```

## Repository layout

See [`docs/architecture.md`](docs/architecture.md) for engineering decisions (CRS, temporal windows, provenance schema, MPS handling).

```
configs/        # root config + per-experiment overrides + AOI GeoJSONs
src/
  data/         # STAC ingest, labels, cloud mask, preprocess, tiling
  features/     # spectral indices and feature stacks
  models/       # datasets, RF / XGB / U-Net / SegFormer training
  evaluation/   # metrics, stratified reports, calibration
  viz/          # figure assembly for README and docs
  utils/        # seed, geo, io, logging, provenance
notebooks/      # exploratory + training + reporting notebooks
tests/          # unit tests (formulas, geo, provenance)
scripts/        # launchers that export PYTORCH_ENABLE_MPS_FALLBACK=1
docs/           # architecture, data dictionary, model card, reviews
LICENSES/       # upstream attribution notices per dataset
```

## Roadmap

| Milestone | Goal | Status |
|---|---|---|
| M1 | Scaffold + governance (CRS, temporal, licence, provenance) | ✅ |
| M2 | GEEBAM labels ingest | ☐ |
| M3 | Sentinel-2 ingest + cloud mask | ☐ |
| M4 | Preprocess + tiling + label alignment | ☐ |
| M5 | dNBR baseline on Kangaroo Island | ☐ |
| M6 | RandomForest + XGBoost | ☐ |
| M6.5 | Early non-technical demo (slider + GIF + plain-English panel) | ☐ |
| M7 | U-Net on MPS bf16 | ☐ |
| M8 | SegFormer-B0 on MPS | ☐ |
| M9 | Vertical-slice 5-method comparison | ☐ |
| M10 | Scale to all 4 AOIs (first valid generalisation metrics) | ☐ |
| M11 | Stratified evaluation + calibration | ☐ |
| M12 | Final portfolio polish + narrative | ☐ |

Each milestone is reviewed by [Codex CLI](https://github.com/openai/codex-cli); review transcripts are committed under `docs/reviews/M<N>_codex_review.md`.

## Licences and attribution

- Code: MIT (see [`LICENSE`](LICENSE)).
- Data: each upstream dataset has its own licence — see [`LICENSES/`](LICENSES/) and [`docs/data_dictionary.md`](docs/data_dictionary.md). Required attribution strings are listed there. **AUS GEEBAM is a public proxy label source, not ground truth** — downstream users must propagate that caveat.

## Citations

To be added at M12 alongside the model card.
