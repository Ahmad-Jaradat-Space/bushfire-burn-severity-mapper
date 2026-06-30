"""Split-conformal prediction sets (APS) for severity classification.

A softmax number is not a probability you can bet on — it is uncalibrated and
gives no coverage guarantee. Conformal prediction fixes that distribution-free:
given a held-out calibration set, it returns per-pixel *sets* of severity classes
that contain the truth with a user-chosen probability (e.g. 90%), no matter how
miscalibrated the underlying model is. Where the model is confident the set is a
singleton; where it is unsure the set grows — an honest, operator-readable
uncertainty.

We use Adaptive Prediction Sets (APS, Romano, Sesia & Candès 2020): the
nonconformity score is the cumulative softmax mass from the most-likely class
down to and including the true class.
"""

from __future__ import annotations

import numpy as np


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """APS nonconformity score per sample: cumulative prob up to the true class.

    probs : [N, C] softmax rows; labels : [N] int class ids.
    """
    order = np.argsort(-probs, axis=1)  # classes, most→least likely
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    ranks = np.argmax(order == labels[:, None], axis=1)  # position of true class
    return cum[np.arange(len(labels)), ranks]


def aps_calibrate(probs_cal: np.ndarray, labels_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Conformal threshold q̂ giving ≥ (1-alpha) marginal coverage."""
    s = aps_scores(probs_cal, labels_cal)
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def aps_predict_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean [N, C] membership: smallest top-k whose cumulative mass ≥ q̂."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # number of classes to include: first rank where cum ≥ q̂, inclusive (≥1).
    k = (cum < qhat).sum(axis=1) + 1
    k = np.clip(k, 1, probs.shape[1])
    sets = np.zeros_like(probs, dtype=bool)
    idx = np.arange(len(probs))
    for rank in range(probs.shape[1]):
        take = rank < k
        sets[idx[take], order[take, rank]] = True
    return sets


def empirical_coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples whose true class is in the predicted set."""
    return float(sets[np.arange(len(labels)), labels].mean())


def mean_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


def set_size_map(probs_hw_c: np.ndarray, qhat: float) -> np.ndarray:
    """Per-pixel set size from an [H, W, C] probability cube."""
    h, w, c = probs_hw_c.shape
    sets = aps_predict_sets(probs_hw_c.reshape(-1, c), qhat)
    return sets.sum(axis=1).reshape(h, w).astype(np.uint8)
