import numpy as np

from src.evaluation.calibration import reliability_data
from src.evaluation.stratified_reports import (
    LANDCOVER_GROUPS,
    stratify_by_landcover,
    stratify_by_slope,
)


def test_stratify_landcover_skips_empty():
    pred = np.full((10, 10), 0, dtype=np.uint8)
    true = pred.copy()
    lc = np.full((10, 10), 999, dtype=np.uint16)  # not in any group
    out = stratify_by_landcover(pred, true, lc)
    assert set(out.keys()) == set(LANDCOVER_GROUPS.keys())
    for v in out.values():
        assert v is None  # no group matched


def test_stratify_slope_bins_have_metrics():
    pred = np.full((4, 4), 1, dtype=np.uint8)
    true = pred.copy()
    slope = np.array([[0.0, 10.0, 20.0, 40.0]] * 4, dtype=np.float32)
    out = stratify_by_slope(pred, true, slope)
    # At least one bin should land
    assert any(v is not None for v in out.values())


def test_reliability_perfect_calibration():
    # Probabilities exactly match the empirical burn fraction
    rng = np.random.default_rng(0)
    n = 10_000
    p = rng.uniform(0, 1, n)
    t = (rng.uniform(0, 1, n) < p).astype(np.uint8)
    pred_proba = p.reshape(100, 100)
    true = t.reshape(100, 100)
    data = reliability_data(pred_proba, true, n_bins=10)
    assert data["ece"] < 0.05
    assert data["brier"] < 0.30
