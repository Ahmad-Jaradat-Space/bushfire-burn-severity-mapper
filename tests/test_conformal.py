import numpy as np

from src.evaluation.conformal import (
    aps_calibrate,
    aps_predict_sets,
    empirical_coverage,
    mean_set_size,
    set_size_map,
)


def _calibrated_data(n, c=4, seed=0):
    """Draw probs from a Dirichlet and labels FROM those probs, so the softmax
    is perfectly calibrated — conformal coverage must then hold."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(c) * 0.6, size=n)
    labels = np.array([rng.choice(c, p=probs[i]) for i in range(n)])
    return probs, labels


def test_aps_coverage_holds():
    probs, labels = _calibrated_data(20_000)
    cal, test = slice(0, 10_000), slice(10_000, None)
    alpha = 0.1
    qhat = aps_calibrate(probs[cal], labels[cal], alpha=alpha)
    sets = aps_predict_sets(probs[test], qhat)
    cov = empirical_coverage(sets, labels[test])
    assert cov >= (1 - alpha) - 0.02  # marginal guarantee (small slack)
    assert 1.0 <= mean_set_size(sets) <= 4.0


def test_aps_tighter_alpha_grows_sets():
    probs, labels = _calibrated_data(20_000, seed=1)
    cal, test = slice(0, 10_000), slice(10_000, None)
    q90 = aps_calibrate(probs[cal], labels[cal], alpha=0.10)
    q99 = aps_calibrate(probs[cal], labels[cal], alpha=0.01)
    s90 = mean_set_size(aps_predict_sets(probs[test], q90))
    s99 = mean_set_size(aps_predict_sets(probs[test], q99))
    assert q99 >= q90
    assert s99 >= s90  # higher coverage -> larger sets


def test_set_size_map_shape_and_range():
    rng = np.random.default_rng(2)
    cube = rng.dirichlet(np.ones(4), size=(8, 10)).astype(np.float32)  # [H,W,C]
    m = set_size_map(cube, qhat=0.8)
    assert m.shape == (8, 10)
    assert m.min() >= 1 and m.max() <= 4


def test_singleton_when_confident():
    # One near-certain class -> set of size 1 at moderate coverage.
    probs = np.array([[0.97, 0.01, 0.01, 0.01]])
    sets = aps_predict_sets(probs, qhat=0.9)
    assert sets.sum() == 1 and sets[0, 0]
