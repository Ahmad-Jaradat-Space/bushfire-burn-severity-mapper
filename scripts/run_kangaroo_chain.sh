#!/usr/bin/env bash
# Run the full Kangaroo Island pipeline once raw data is on disk.
# Idempotent: re-running is safe.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/setup_env.sh

EVENT=kangaroo_island_2019_2020
echo "=== 1) align labels to composite grid ==="
python scripts/align_labels_to_composite.py --event "$EVENT"

echo "=== 2) tile (vertical-slice random tile split) ==="
python -m src.data.tiling --event "$EVENT" --split-mode random_tile

echo "=== 3) dNBR baseline ==="
python -m src.models.run_baseline --event "$EVENT"

echo "=== 4) RandomForest (vertical-slice mode) ==="
python -m src.models.train_rf --config configs/experiments/rf_multiclass.yaml \
    "experiment.event=$EVENT" "experiment.split_mode=vertical_slice" \
    "sampling.pixels_per_class=20000" "rf.n_estimators=200"

echo "=== 5) XGBoost ==="
python -m src.models.train_xgb --config configs/experiments/xgb_multiclass.yaml \
    "experiment.event=$EVENT" "experiment.split_mode=vertical_slice" \
    "sampling.pixels_per_class=20000" "xgb.n_estimators=300"

echo "=== done ==="
ls -la outputs/predictions/baseline_dnbr/"$EVENT"/ outputs/predictions/rf_multiclass/ outputs/predictions/xgb_multiclass/ 2>&1 | head -20
