# What this project demonstrates for an ecological-monitoring platform

The case study is bushfire burn severity, but the capabilities are the ones an
AI-first ecological-monitoring platform (vegetation classification, rehabilitation
monitoring, weed mapping, biodiversity assessment) runs on. The mapping:

| Capability shown here | Where in the repo | Transfers directly to |
|---|---|---|
| **DL semantic segmentation** of high-res imagery (U-Net, SegFormer, **Prithvi-EO-2.0** fine-tune) | `src/models/`, `configs/experiments/` | vegetation-class & lifeform segmentation; rehab-stage mapping |
| **Foundation models** for scarce-label generalisation (frozen backbone + decoder beats from-scratch) | `src/models/prithvi_model.py`, fig 16 | weed / rare-species mapping where labels are few and sites vary |
| **Classical ML alongside DL** (RandomForest, XGBoost, feature importance) | `src/models/train_rf.py`, `train_xgb.py` | tabular/tree models where they beat or complement a net |
| **Geospatial engineering**: CRS/UTM, tiling, **mosaicking**, polygonise, **spatial joins/overlays**, **PostGIS** SQL, **QGIS** styling | `src/geospatial/`, `sql/`, `qgis/` | management-unit / tenure / bioregion reporting |
| **Spatial & survey sampling**: spatial block-CV, stratified pixel sampling, design-based (Olofsson) area estimation | `src/data/spatial_sampling.py`, `src/evaluation/area_estimation.py` | representative, defensible train/val/accuracy-assessment design |
| **Uncertainty quantification**: MC-dropout entropy, conformal prediction sets, calibration (ECE/Brier) | `src/evaluation/{uq_maps,conformal,calibration}.py` | trustworthy figures; flagging "I don't know" before a field check |
| **Applied statistics / econometrics**: spatial-block bootstrap CIs, McNemar, confounder-controlled GLM ("all else equal", cluster-robust SE) | `src/evaluation/uncertainty.py`, `src/stats/confounders.py` | honest effect sizes; controlling for terrain/vegetation confounds |
| **Evaluation for imbalanced / rare classes**: per-class precision/recall, rare-class focus | `src/evaluation/metrics.py`, fig 08b | rare species / rare lifeform reporting |
| **MLOps / scale**: Docker, MLflow tracking, tiled batch inference + throughput/memory/cost benchmark, **GCP** deployment design | `docker/`, `src/utils/tracking.py`, `src/inference/`, `docs/cloud_gcp.md` | running enterprise jobs reliably at landscape scale on cloud |
| **Reproducibility & engineering standards**: provenance sidecars, seeded runs, CI, 80+ tests, model card | `src/utils/provenance.py`, `.github/`, `tests/`, `docs/model_card.md` | maintainable production + R&D codebase |

The through-line is the same one the role asks for: turn imagery into
**decision-grade figures**, with confidence intervals, calibrated uncertainty,
and a vector roll-up a non-modeller can act on, and do it reproducibly at scale.
