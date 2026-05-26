#!/usr/bin/env bash
# Source this BEFORE running any training or fetch script.
# Reason: PYTORCH_ENABLE_MPS_FALLBACK=1 must be exported before `torch` is imported.

set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONHASHSEED=42

# Hint pyproj/rasterio to find PROJ data shipped with their wheels
unset PROJ_LIB 2>/dev/null || true
unset GDAL_DATA 2>/dev/null || true

echo "[setup_env] PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"
echo "[setup_env] PYTHONHASHSEED=$PYTHONHASHSEED"
