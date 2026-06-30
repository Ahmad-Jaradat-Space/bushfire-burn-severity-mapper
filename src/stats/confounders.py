"""Confounder-controlled error analysis, "where does the model err, all else equal?"

A raw error map conflates everything: maybe the model looks worst on steep
ground only because steep ground is also where the rare severe class lives. The
econometric answer is a regression that holds the other factors fixed. We fit a
logistic GLM of per-pixel error on the candidate confounders,

    error ~ pre-fire NDVI + slope + dNBR + C(true severity class)

, and read each odds ratio *controlling for the rest*. Crucially the standard
errors are **cluster-robust by spatial block**: 9 million pixels are not 9
million independent observations, and naive SEs would make every coefficient
spuriously significant (the same pseudo-replication trap the spatial-block
bootstrap fixes for the headline metric).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.features.indices import nbr, ndvi
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.theme import ACCENT, INK_LIGHT, apply_theme, thin_axes

log = get_logger(__name__)

EVENT = "kangaroo_island_2019_2020"
DEFAULT_PRED = "outputs/predictions/unet_eventwise/{event}.tif"
# Band positions in the [6, H, W] composite (B02,B03,B04,B08,B11,B12)
B04, B08, B12 = 2, 3, 5
CLASS_NAMES = {0: "Unburnt", 1: "Low–Mod", 2: "High", 3: "Very High"}


def _read(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(band)


def _read_stack(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def assemble_error_frame(
    event: str = EVENT,
    pred_path: str | None = None,
    n_sample: int = 200_000,
    block_px: int = 256,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample per-pixel (error, confounders) for the event-wise model."""
    interim = REPO_ROOT / "data" / "interim" / event
    pre = _read_stack(interim / "pre_stack_10m.tif")
    post = _read_stack(interim / "post_stack_10m.tif")
    label = _read(interim / "label_10m.tif")
    pred = _read(REPO_ROOT / (pred_path or DEFAULT_PRED.format(event=event)))

    pre_ndvi = ndvi(pre[B08], pre[B04])
    dnbr = nbr(pre[B08], pre[B12]) - nbr(post[B08], post[B12])
    slope_path = interim / "slope_10m.tif"
    slope = _read(slope_path) if slope_path.exists() else None

    valid = (label != 255) & (pred != 255)
    rows, cols = np.nonzero(valid)
    rng = np.random.default_rng(seed)
    if rows.size > n_sample:
        sel = rng.choice(rows.size, n_sample, replace=False)
        rows, cols = rows[sel], cols[sel]

    _, W = label.shape
    df = pd.DataFrame(
        {
            "error": (pred[rows, cols] != label[rows, cols]).astype(int),
            "true_class": label[rows, cols].astype(int),
            "pre_ndvi": pre_ndvi[rows, cols].astype(np.float32),
            "dnbr": dnbr[rows, cols].astype(np.float32),
            "block_id": (rows // block_px) * (W // block_px + 1) + (cols // block_px),
        }
    )
    if slope is not None:
        df["slope"] = slope[rows, cols].astype(np.float32)
    # Drop non-finite confounders (NDVI/NBR divide-by-zero) so statsmodels does
    # not silently drop rows and desync the cluster-group vector.
    check = ["pre_ndvi", "dnbr"] + (["slope"] if "slope" in df else [])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=check).reset_index(drop=True)
    for col in ["pre_ndvi", "dnbr", "slope"]:
        if col in df:
            sd = df[col].std() or 1.0
            df[f"{col}_z"] = (df[col] - df[col].mean()) / sd
    log.info(
        "error frame: n=%d  error rate=%.3f  slope=%s",
        len(df),
        df["error"].mean(),
        "yes" if slope is not None else "MISSING",
    )
    return df


def fit_error_glm(df: pd.DataFrame, cluster_robust: bool = True):
    """Logistic GLM of error on confounders, cluster-robust SE by spatial block."""
    import statsmodels.formula.api as smf

    terms = ["pre_ndvi_z", "dnbr_z"] + (["slope_z"] if "slope_z" in df else [])
    formula = "error ~ " + " + ".join(terms) + " + C(true_class)"
    kw = {"disp": 0}
    if cluster_robust:
        kw.update(cov_type="cluster", cov_kwds={"groups": df["block_id"]})
    try:
        return smf.logit(formula, data=df).fit(**kw)
    except Exception as exc:  # perfect separation etc.
        log.warning("MLE failed (%s); falling back to L2-regularised fit.", exc)
        return smf.logit(formula, data=df).fit_regularized(alpha=1.0, disp=0)


def odds_ratio_table(res) -> pd.DataFrame:
    conf = res.conf_int()
    return pd.DataFrame(
        {
            "coef": res.params,
            "odds_ratio": np.exp(res.params),
            "or_low": np.exp(conf[0]),
            "or_high": np.exp(conf[1]),
            "p_value": res.pvalues,
        }
    )


_PRETTY = {
    "pre_ndvi_z": "Pre-fire NDVI (+1 SD)",
    "dnbr_z": "ΔNBR (+1 SD)",
    "slope_z": "Slope (+1 SD)",
    "C(true_class)[T.1]": "True: Low–Mod (vs Unburnt)",
    "C(true_class)[T.2]": "True: High (vs Unburnt)",
    "C(true_class)[T.3]": "True: Very High (vs Unburnt)",
}


def fig_odds_ratios(res, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    tab = odds_ratio_table(res).drop(index="Intercept", errors="ignore")
    tab = tab.loc[[i for i in _PRETTY if i in tab.index]]
    labels = [_PRETTY[i] for i in tab.index]
    y = np.arange(len(tab))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(tab) + 1.8))
    ax.axvline(1.0, color=INK_LIGHT, lw=1, ls="--")
    for yi, (_, r) in zip(y, tab.iterrows(), strict=False):
        sig = r["p_value"] < 0.05
        ax.plot([r["or_low"], r["or_high"]], [yi, yi], color=ACCENT if sig else "#9A9A9A", lw=2)
        ax.plot(r["odds_ratio"], yi, "o", color=ACCENT if sig else "#9A9A9A", ms=8)
        ax.text(
            r["or_high"] * 1.02,
            yi,
            f"OR={r['odds_ratio']:.2f}" + ("" if sig else " (ns)"),
            va="center",
            fontsize=9,
            color=INK_LIGHT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for the model being wrong  (>1 = more error, all else equal)")
    ax.set_title("Where the event-wise U-Net fails, controlling for confounders", loc="left")
    thin_axes(ax)
    ax.grid(True, axis="x", alpha=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default=EVENT)
    p.add_argument("--n-sample", type=int, default=200_000)
    args = p.parse_args()
    df = assemble_error_frame(args.event, n_sample=args.n_sample)
    res = fit_error_glm(df)
    log.info("\n%s", odds_ratio_table(res).round(3).to_string())
    fig_odds_ratios(res, REPO_ROOT / "docs" / "figures" / "09_confounder_glm.png")
    log.info("wrote docs/figures/09_confounder_glm.png")


if __name__ == "__main__":
    main()
