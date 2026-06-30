"""Phase E CLI — scale benchmark + MLflow backfill, render figures 14/15.

python scripts/benchmark_inference.py            # mps + cpu throughput, cost projection
"""

from __future__ import annotations

import argparse

import torch

from src.inference.scale_report import render_all
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="e.g. mps cpu  (default: mps if available, plus cpu)",
    )
    args = p.parse_args()
    devices = args.devices
    if devices is None:
        devices = ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    res = render_all(REPO_ROOT / "docs" / "figures", devices=devices)
    log.info("scale benchmark:\n%s", res["benchmark_df"].to_string(index=False))
    log.info("wrote figures 14/15 to docs/figures")


if __name__ == "__main__":
    main()
