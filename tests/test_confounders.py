import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.stats.confounders import fit_error_glm, odds_ratio_table


def _synthetic_frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    dnbr_z = rng.standard_normal(n)
    # error probability decreases with dNBR (strong burn signal -> fewer errors)
    p = 1.0 / (1.0 + np.exp(-(-1.4 * dnbr_z)))
    err = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "error": err,
            "true_class": rng.integers(0, 4, n),
            "pre_ndvi_z": rng.standard_normal(n),
            "dnbr_z": dnbr_z,
            "slope_z": rng.standard_normal(n),
            "block_id": rng.integers(0, 25, n),
        }
    )


def test_glm_recovers_confounder_sign_and_shape():
    df = _synthetic_frame()
    res = fit_error_glm(df, cluster_robust=True)
    tab = odds_ratio_table(res)
    assert {"coef", "odds_ratio", "or_low", "or_high", "p_value"} <= set(tab.columns)
    # higher dNBR -> lower odds of error (OR < 1), and it should be significant
    assert tab.loc["dnbr_z", "odds_ratio"] < 1.0
    assert tab.loc["dnbr_z", "p_value"] < 0.05
    # an unrelated covariate should not be strongly signalled
    assert 0.8 < tab.loc["pre_ndvi_z", "odds_ratio"] < 1.25


def test_glm_runs_without_slope_column():
    df = _synthetic_frame().drop(columns=["slope_z"])
    res = fit_error_glm(df, cluster_robust=False)
    assert "slope_z" not in odds_ratio_table(res).index
