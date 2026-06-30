"""Phase A CLI, statistical rigor on the event-wise hold-out (Kangaroo Island).

Thin driver over :mod:`src.evaluation.uncertainty_report`. Computes spatial-block
bootstrap CIs, McNemar significance, paired bootstrap deltas, Olofsson
area-adjusted areas and rare-class precision/recall, then writes the JSON report
and figures 06/07/08/08b. The same library functions are called from the story
notebook, so the notebook regenerates exactly what it displays.

  python scripts/compute_uncertainty.py [--event ...] [--n-boot 2000] [--block-px 256]
"""

from __future__ import annotations

import argparse
import json

from src.evaluation.uncertainty_report import EVENT, compute_report, render_all
from src.utils.config import load_config
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default=EVENT)
    p.add_argument(
        "--block-px",
        type=int,
        default=None,
        help="spatial block size in pixels (default: config tile_size)",
    )
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    block_px = args.block_px
    if block_px is None:
        try:
            block_px = int(load_config("configs/config.yaml").data.tile_size)
        except Exception:
            block_px = 256

    report = compute_report(args.event, block_px=block_px, n_boot=args.n_boot, seed=args.seed)

    out_json = REPO_ROOT / "outputs" / "metrics" / "uncertainty" / f"{args.event}_eventwise.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out_json.relative_to(REPO_ROOT))

    render_all(report, REPO_ROOT / "docs" / "figures")
    log.info("wrote figures 06/07/08/08b to docs/figures")

    pref = ["baseline_dnbr", "rf", "xgb", "unet", "segformer"]
    for m in [m for m in pref if m in report["models"]]:
        b = report["models"][m]["bootstrap"]["metrics"]["macro_iou"]
        log.info(
            "  %-14s macro-IoU %.3f  [%.3f, %.3f] (%s)",
            report["models"][m]["label"],
            b["point"],
            b["ci_low"],
            b["ci_high"],
            b["ci_method"],
        )


if __name__ == "__main__":
    main()
