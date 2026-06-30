"""Uncertainty and significance for map-accuracy metrics.

Three tools, all built on top of confusion matrices so they stay numerically
identical to :func:`src.evaluation.metrics.summary`:

* :func:`bootstrap_macro_metrics`, a **spatial block bootstrap** confidence
  interval on macro-IoU / macro-F1. Resamples the per-block confusion matrices
  from :mod:`src.evaluation.blocks` (not pixels), so the CI respects spatial
  autocorrelation instead of pretending every pixel is independent. Reports a
  bias-corrected-and-accelerated (BCa) interval, falling back to the percentile
  interval at the degenerate edges.
* :func:`paired_bootstrap_delta`, a CI on the *difference* between two models'
  macro metric, resampling the **same** blocks for both (a paired design).
* :func:`mcnemar_test`, McNemar's paired test on per-pixel correctness, the
  standard way to ask "is model A significantly more accurate than model B on
  the same scene?"

These let the notebook state effect sizes with error bars and p-values rather
than bare point estimates, the difference between "U-Net beats dNBR" and
"U-Net beats dNBR by 0.12 macro-IoU (95% CI 0.09-0.15, McNemar p < 1e-10)".
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.evaluation.metrics import IGNORE_ID, per_class_f1, per_class_iou

_METRIC_FNS = {
    "macro_iou": lambda cm: float(per_class_iou(cm).mean()),
    "macro_f1": lambda cm: float(per_class_f1(cm).mean()),
}


def _macro(cm: np.ndarray, metric: str) -> float:
    try:
        return _METRIC_FNS[metric](cm)
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(f"unknown metric {metric!r}; choose from {list(_METRIC_FNS)}") from exc


def _stack(block_cms: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    keys = sorted(block_cms)
    mats = (
        np.stack([block_cms[k].astype(np.float64) for k in keys]) if keys else np.empty((0, 0, 0))
    )
    return keys, mats


def _bca_interval(
    reps: np.ndarray, point: float, mats: np.ndarray, metric: str, alpha: float
) -> tuple[float, float, str]:
    """Bias-corrected-and-accelerated CI; degrade to percentile when unstable."""
    lo_pct, hi_pct = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    prop = float(np.mean(reps < point))
    if prop <= 0.0 or prop >= 1.0:
        return float(lo_pct), float(hi_pct), "percentile"
    z0 = stats.norm.ppf(prop)

    # Acceleration via jackknife over blocks (leave-one-block-out).
    total = mats.sum(axis=0)
    n = mats.shape[0]
    jack = np.array([_macro(total - mats[i], metric) for i in range(n)])
    diff = jack.mean() - jack
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    a = 0.0 if denom == 0 else float(np.sum(diff**3) / denom)

    def adjust(z: float) -> float:
        num = z0 + z
        return float(stats.norm.cdf(z0 + num / (1.0 - a * num)))

    al = adjust(stats.norm.ppf(alpha / 2))
    au = adjust(stats.norm.ppf(1 - alpha / 2))
    if not (0.0 < al < au < 1.0):
        return float(lo_pct), float(hi_pct), "percentile"
    lo = float(np.percentile(reps, 100 * al))
    hi = float(np.percentile(reps, 100 * au))
    return lo, hi, "bca"


def bootstrap_macro_metrics(
    block_cms: dict[int, np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    metrics: tuple[str, ...] = ("macro_iou", "macro_f1"),
) -> dict:
    """Spatial-block bootstrap CIs for macro metrics.

    Parameters
    ----------
    block_cms : dict
        Per-block confusion matrices from
        :func:`src.evaluation.blocks.block_confusions`.
    n_boot : int
        Bootstrap replicates.
    alpha : float
        ``1 - alpha`` is the coverage (0.05 -> 95% CI).
    """
    keys, mats = _stack(block_cms)
    n_blocks = len(keys)
    if n_blocks == 0:
        raise ValueError("block_cms is empty, nothing to bootstrap")
    total = mats.sum(axis=0)
    rng = np.random.default_rng(seed)

    out: dict = {"n_blocks": n_blocks, "n_boot": n_boot, "alpha": alpha, "metrics": {}}
    for metric in metrics:
        point = _macro(total, metric)
        reps = np.empty(n_boot)
        for bi in range(n_boot):
            idx = rng.integers(0, n_blocks, size=n_blocks)
            reps[bi] = _macro(mats[idx].sum(axis=0), metric)
        lo_pct, hi_pct = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci_lo, ci_hi, method = _bca_interval(reps, point, mats, metric, alpha)
        out["metrics"][metric] = {
            "point": point,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "ci_method": method,
            "percentile_low": float(lo_pct),
            "percentile_high": float(hi_pct),
            "se": float(reps.std(ddof=1)) if n_boot > 1 else 0.0,
        }
    return out


def paired_bootstrap_delta(
    block_cms_a: dict[int, np.ndarray],
    block_cms_b: dict[int, np.ndarray],
    metric: str = "macro_iou",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI on ``metric(A) - metric(B)`` over shared spatial blocks."""
    keys = sorted(set(block_cms_a) & set(block_cms_b))
    if not keys:
        raise ValueError("models share no spatial blocks")
    A = np.stack([block_cms_a[k].astype(np.float64) for k in keys])
    B = np.stack([block_cms_b[k].astype(np.float64) for k in keys])
    n = len(keys)
    point = _macro(A.sum(axis=0), metric) - _macro(B.sum(axis=0), metric)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[i] = _macro(A[idx].sum(axis=0), metric) - _macro(B[idx].sum(axis=0), metric)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two = 2.0 * min(float(np.mean(reps <= 0)), float(np.mean(reps >= 0)))
    return {
        "metric": metric,
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p_two)),
        "n_blocks": n,
        "n_boot": n_boot,
    }


def mcnemar_test(
    pred_a: np.ndarray, pred_b: np.ndarray, true: np.ndarray, ignore_index: int = IGNORE_ID
) -> dict:
    """McNemar's paired test on per-pixel correctness of two predictions.

    ``b`` = pixels where A is right and B is wrong; ``c`` = the reverse. Only
    the discordant pixels carry information. Uses the exact two-sided binomial
    test for small discordant counts and the continuity-corrected chi-square
    otherwise.
    """
    pa, pb, t = pred_a.ravel(), pred_b.ravel(), true.ravel()
    valid = (t != ignore_index) & (pa != ignore_index) & (pb != ignore_index)
    pa, pb, t = pa[valid], pb[valid], t[valid]
    ca = pa == t
    cb = pb == t
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    base = {
        "b": b,
        "c": c,
        "n_discordant": n,
        "acc_a": float(ca.mean()) if ca.size else 0.0,
        "acc_b": float(cb.mean()) if cb.size else 0.0,
    }
    if n == 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "method": "no-discordant"}
    if n < 25:
        k = min(b, c)
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))
        return {**base, "statistic": float(k), "p_value": p, "method": "exact-binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, 1))
    return {**base, "statistic": float(stat), "p_value": p, "method": "chi2-cc"}
