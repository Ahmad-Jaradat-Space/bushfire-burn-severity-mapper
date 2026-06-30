"""Smoke tests for the notebook's story figures.

The schematic + ribbon take no data, so they always render. The leaderboard and
methods card need the event-wise metrics JSON (under the git-ignored ``outputs/``
tree); those tests skip when the artifact has not been materialised, e.g. in CI.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from src.utils.geo import REPO_ROOT
from src.viz import story_figures as sf

EVENTWISE = REPO_ROOT / "outputs/metrics/uncertainty/kangaroo_island_2019_2020_eventwise.json"


def _png_ok(p) -> bool:
    return p.exists() and p.stat().st_size > 2000


def test_verdict_logic():
    report = {
        "paired_delta_vs_dnbr": {
            "prithvi": {"delta": 0.06, "p_value": 0.0},
            "unet": {"delta": 0.02, "p_value": 0.064},
            "rf": {"delta": -0.10, "p_value": 0.0},
        }
    }
    assert sf._verdict("baseline_dnbr", report)[1] == sf._MEH
    assert sf._verdict("prithvi", report)[1] == sf._GOOD
    assert "ties" in sf._verdict("unet", report)[0]
    assert sf._verdict("rf", report)[1] == sf._BAD


def test_journey_renders(tmp_path):
    assert _png_ok(sf.fig_journey(tmp_path / "journey.png"))


def test_split_schematic_renders(tmp_path):
    assert _png_ok(sf.fig_split_schematic(tmp_path / "split.png"))


@pytest.mark.skipif(not EVENTWISE.exists(), reason="event-wise metrics not materialised")
def test_leaderboard_and_methods_render(tmp_path):
    assert _png_ok(sf.fig_leaderboard(tmp_path / "lb.png"))
    assert _png_ok(sf.fig_methods_overview(tmp_path / "mo.png"))
