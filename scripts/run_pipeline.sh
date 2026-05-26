#!/usr/bin/env bash
# End-to-end pipeline for one AOI. Sources scripts/setup_env.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

EVENT="${1:-kangaroo_island_2019_2020}"
CONFIG="${2:-configs/experiments/baseline_dnbr.yaml}"

echo "[run_pipeline] event=${EVENT} config=${CONFIG}"

python -m src.data.fetch_labels   --event "${EVENT}"
python -m src.data.fetch_sentinel --event "${EVENT}"
# python -m src.data.preprocess    --event "${EVENT}" --config "${CONFIG}"
# python -m src.data.tiling        --event "${EVENT}" --config "${CONFIG}"
# python -m src.evaluation.evaluate --config "${CONFIG}" --model baseline_dnbr_usgs_multiclass
