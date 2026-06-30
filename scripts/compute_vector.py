"""Phase D CLI, vectorise severity, roll up by management unit, render 12/13.

python scripts/compute_vector.py                 # figures + area table
POSTGIS_DSN=... python scripts/compute_vector.py --postgis   # also load + query PostGIS
"""

from __future__ import annotations

import argparse

from src.geospatial.report import EVENT, compute_vector_report, load_to_postgis, render_all
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default=EVENT)
    p.add_argument("--cell-m", type=float, default=5000.0)
    p.add_argument("--postgis", action="store_true", help="load layers + run sql/queries.sql")
    args = p.parse_args()

    rep = compute_vector_report(args.event, cell_m=args.cell_m)
    render_all(rep, REPO_ROOT / "docs" / "figures")
    log.info("severity area summary:\n%s", rep["summary"].round(1).to_string(index=False))
    log.info("wrote figures 12/13 to docs/figures")

    if args.postgis:
        try:
            results = load_to_postgis(rep)
            for name, df in results.items():
                log.info("PostGIS %s:\n%s", name, df.head(10).to_string(index=False))
        except Exception as exc:
            log.warning("PostGIS step skipped (%s)", exc)


if __name__ == "__main__":
    main()
