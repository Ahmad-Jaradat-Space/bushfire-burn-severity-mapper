#!/usr/bin/env bash
# Fetch composites + align labels + tile for the 3 non-Kangaroo AOIs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/setup_env.sh

EVENTS=("currowan_2019_2020" "gospers_mountain_2019_2020" "east_gippsland_2019_2020")

for ev in "${EVENTS[@]}"; do
  echo ""
  echo "##############################################"
  echo "## $ev"
  echo "##############################################"
  date

  echo "[1/3] composite Sentinel-2 (6 pre + 6 post @ 30m)"
  python -u scripts/fetch_event_streamlined.py --event "$ev" --side both --n-pre 6 --n-post 6 --resolution 30

  # Streamlined fetcher writes pre_mask_10m.tif / post_mask_10m.tif; rename
  # to match the convention expected by the rest of the pipeline.
  cd data/interim/"$ev"
  [ -f pre_mask_10m.tif ]  && mv -f pre_mask_10m.tif  mask_pre_10m.tif  || true
  [ -f post_mask_10m.tif ] && mv -f post_mask_10m.tif mask_post_10m.tif || true
  cd - > /dev/null

  echo "[2/3] align GEEBAM labels"
  python scripts/align_labels_to_composite.py --event "$ev"

  echo "[3/3] tile (event_wise split)"
  python -m src.data.tiling --event "$ev" --split-mode event_wise
done

echo ""
echo "=== all 3 AOIs done ==="
date
