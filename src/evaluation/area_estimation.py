"""Design-based, area-adjusted accuracy and area estimates (Olofsson et al.).

A wall-to-wall map's per-pixel accuracy is a *biased* estimator of the true
class areas whenever the map mislabels — over-mapped classes inflate, and the
sampling variance is invisible. Olofsson et al. 2014 ("Good practices for
estimating area and assessing accuracy of land change", *RSE* 148:42-57) give
the design-based correction: weight the confusion matrix by each map class's
mapped area, then report user's / producer's / overall accuracy and unbiased
class areas, each with a standard error and 95% confidence interval.

This is the estimator an ecology consultancy needs before quoting "X hectares
burnt at high severity" — it turns a map into a *figure with error bars*.

The confusion matrix convention matches :func:`src.evaluation.metrics.summary`:
``cm[i, j]`` counts reference (truth) class ``i`` predicted (mapped) as ``j``.

Note
----
The variance formulas assume the rows of the confusion matrix come from a
*probability sample* stratified by the map class. Applied wall-to-wall (every
pixel as its own sample), the point estimates are exact and the SEs are a
conservative within-stratum-binomial approximation; for a defensible operational
number, build ``cm`` from a stratified random sample (see
:mod:`src.data.spatial_sampling`).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def olofsson_area_accuracy(
    cm: np.ndarray,
    mapped_area_px: np.ndarray | None = None,
    pixel_area_m2: float = 100.0,
    alpha: float = 0.05,
) -> dict:
    """Area-adjusted accuracy and area estimates with confidence intervals.

    Parameters
    ----------
    cm : ndarray ``[q, q]``
        Confusion counts, ``cm[i, j]`` = reference ``i`` mapped as ``j``
        (the output of :func:`src.evaluation.metrics.confusion_matrix`).
    mapped_area_px : ndarray ``[q]`` or None
        Total pixels the *full map* assigns to each class. If None, derived from
        ``cm`` (wall-to-wall case): the column sums.
    pixel_area_m2 : float
        Ground area of one pixel (10 m Sentinel-2 -> 100 m²). Areas are returned
        in both m² and hectares.
    alpha : float
        ``1 - alpha`` coverage for the CIs (0.05 -> 95%).
    """
    cm = np.asarray(cm, dtype=np.float64)
    q = cm.shape[0]
    if cm.shape != (q, q):
        raise ValueError("cm must be square")
    z = float(stats.norm.ppf(1 - alpha / 2))

    # Work in map-rows convention: m[i, j] = mapped i, reference j.
    m = cm.T
    if mapped_area_px is None:
        mapped_area_px = m.sum(axis=1)
    n_i = m.sum(axis=1)  # sample size per map class (row)
    big_n = float(np.asarray(mapped_area_px, dtype=np.float64).sum())
    if big_n <= 0:
        raise ValueError("mapped_area_px sums to zero")
    w = np.asarray(mapped_area_px, dtype=np.float64) / big_n  # area weights

    safe_n = np.where(n_i > 0, n_i, 1.0)
    # Estimated area proportions p_ij = W_i * m_ij / n_i.
    p = (w[:, None] * m) / safe_n[:, None]
    p[n_i == 0] = 0.0
    p_dot_j = p.sum(axis=0)  # estimated reference proportions

    overall = float(np.trace(p))
    users = np.divide(np.diag(m), safe_n, out=np.zeros(q), where=n_i > 0)
    producers = np.divide(np.diag(p), p_dot_j, out=np.zeros(q), where=p_dot_j > 0)

    # --- variances (Olofsson 2014 eqs 5-7, 10) ---
    denom = np.where(n_i > 1, n_i - 1, np.nan)

    # Overall accuracy SE (eq 5).
    var_overall = float(np.nansum(w**2 * users * (1 - users) / denom))
    se_overall = float(np.sqrt(max(var_overall, 0.0)))

    # User's accuracy SE (eq 6): simple binomial within stratum.
    var_users = np.where(n_i > 1, users * (1 - users) / denom, 0.0)
    se_users = np.sqrt(np.clip(var_users, 0.0, None))

    # Reference-proportion SE (eq 10) -> area SE.
    ratio = np.divide(m, safe_n[:, None], out=np.zeros_like(m), where=n_i[:, None] > 0)
    var_pj = np.nansum((w[:, None] ** 2) * ratio * (1 - ratio) / denom[:, None], axis=0)
    se_pj = np.sqrt(np.clip(var_pj, 0.0, None))

    # Producer's accuracy SE (eq 7).
    n_hat_dot_j = (w / safe_n) @ m * big_n  # estimated reference pixel count per class
    se_producers = np.zeros(q)
    for j in range(q):
        if p_dot_j[j] <= 0 or n_hat_dot_j[j] <= 0:
            continue
        n_dot_j_j = mapped_area_px[j]
        term1 = (n_dot_j_j**2) * ((1 - producers[j]) ** 2) * users[j] * (1 - users[j])
        term1 = term1 / denom[j] if n_i[j] > 1 else 0.0
        term2 = 0.0
        for i in range(q):
            if i == j or n_i[i] <= 1:
                continue
            r = m[i, j] / n_i[i]
            term2 += (mapped_area_px[i] ** 2) * r * (1 - r) / (n_i[i] - 1)
        var_pj_prod = (producers[j] ** 2) * term2
        se_producers[j] = float(np.sqrt(max(term1 + var_pj_prod, 0.0)) / (n_hat_dot_j[j]))

    total_area_m2 = big_n * pixel_area_m2
    area_m2 = p_dot_j * total_area_m2
    se_area_m2 = se_pj * total_area_m2

    def _ci(point, se):
        return float(point - z * se), float(point + z * se)

    classes = []
    for j in range(q):
        u_lo, u_hi = _ci(users[j], se_users[j])
        pr_lo, pr_hi = _ci(producers[j], se_producers[j])
        a_lo, a_hi = _ci(area_m2[j], se_area_m2[j])
        classes.append(
            {
                "class": j,
                "users_accuracy": float(users[j]),
                "users_ci": [u_lo, u_hi],
                "producers_accuracy": float(producers[j]),
                "producers_ci": [pr_lo, pr_hi],
                "area_adjusted_m2": float(area_m2[j]),
                "area_adjusted_ha": float(area_m2[j] / 1e4),
                "area_ci_ha": [a_lo / 1e4, a_hi / 1e4],
                "area_mapped_ha": float(mapped_area_px[j] * pixel_area_m2 / 1e4),
            }
        )

    o_lo, o_hi = _ci(overall, se_overall)
    return {
        "alpha": alpha,
        "n_classes": q,
        "n_sample": int(cm.sum()),
        "pixel_area_m2": pixel_area_m2,
        "overall_accuracy": overall,
        "overall_accuracy_se": se_overall,
        "overall_accuracy_ci": [o_lo, o_hi],
        "total_area_ha": total_area_m2 / 1e4,
        "per_class": classes,
    }
