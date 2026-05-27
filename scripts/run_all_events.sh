#!/usr/bin/env bash
# Run the full pipeline (fetch → preprocess → tile → train → eval) for every AOI.
# Used by M10 to fan out from the Kangaroo vertical slice to all four AOIs.
#
# Expects you've already run `source scripts/setup_env.sh` so MPS fallback is set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

EVENTS=("kangaroo_island_2019_2020" "currowan_2019_2020"
        "gospers_mountain_2019_2020" "east_gippsland_2019_2020")

echo "[run_all_events] Stage 1: data ingest"
for ev in "${EVENTS[@]}"; do
  python -m src.data.fetch_labels --event "${ev}"
  python -m src.data.fetch_sentinel --event "${ev}" --stage all
  python -m src.data.preprocess --event "${ev}"
  python -m src.data.tiling --event "${ev}" --split-mode event_wise
done

echo "[run_all_events] Stage 2: classical baselines"
for ev in "${EVENTS[@]}"; do
  python -m src.models.run_baseline --event "${ev}"
done

echo "[run_all_events] Stage 3: classical ML (event-wise split)"
python -m src.models.train_rf --config configs/experiments/rf_multiclass.yaml \
    experiment.split_mode=event_wise
python -m src.models.train_xgb --config configs/experiments/xgb_multiclass.yaml \
    experiment.split_mode=event_wise

echo "[run_all_events] Stage 4: deep models (event-wise split, fast-mode for first pass)"
python -m src.models.train_unet --config configs/experiments/unet_multiclass.yaml \
    --fast-mode experiment.split_mode=event_wise
python -m src.models.train_segformer --config configs/experiments/segformer_multiclass.yaml \
    --fast-mode experiment.split_mode=event_wise

echo "[run_all_events] Stage 5: full evaluation"
python -m src.evaluation.evaluate --all-events
python -m src.viz.readme_panels --mode overview

echo "[run_all_events] done."
