"""Phase B CLI — MC-dropout uncertainty + conformal sets on the event-wise U-Net.

Thin driver over :mod:`src.evaluation.uq_report`. Writes the scalar report and
figures 10/11.

  python scripts/compute_uq.py [--event ...] [--T 10] [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json

from src.evaluation.uq_report import EVENT, compute_uq, render_all
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default=EVENT)
    p.add_argument("--T", type=int, default=10, help="MC-dropout forward passes")
    p.add_argument("--alpha", type=float, default=0.1, help="1-alpha = conformal coverage")
    p.add_argument("--stride-frac", type=float, default=1.0)
    args = p.parse_args()

    report = compute_uq(args.event, T=args.T, alpha=args.alpha, stride_frac=args.stride_frac)
    render_all(report, REPO_ROOT / "docs" / "figures")

    scalar = {k: v for k, v in report.items() if k != "_arrays"}
    out_json = REPO_ROOT / "outputs" / "metrics" / "uncertainty" / f"{args.event}_uq.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(scalar, indent=2))
    log.info("wrote %s", out_json.relative_to(REPO_ROOT))
    log.info(
        "coverage %.3f (target %.3f) | mean set size %.2f | mean entropy %.3f | epistemic MI %.3f",
        report["empirical_coverage"],
        report["target_coverage"],
        report["mean_set_size"],
        report["mean_predictive_entropy"],
        report["mean_epistemic_mi"],
    )


if __name__ == "__main__":
    main()
