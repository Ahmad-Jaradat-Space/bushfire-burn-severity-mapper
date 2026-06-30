"""Phase E reporting: scale benchmark (fig 14) + tracked-experiments view (fig 15)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.inference.batch_infer import EVENT, benchmark
from src.utils.geo import REPO_ROOT
from src.utils.logging_utils import get_logger
from src.viz.theme import apply_theme, thin_axes

log = get_logger(__name__)

# model dir -> display label, for the tracked-runs comparison
MODEL_DIRS = {
    "unet": "U-Net (in-domain)",
    "segformer": "SegFormer-B0 (in-domain)",
    "unet_eventwise": "U-Net (event-wise)",
}


def run_benchmark(devices: list[str] | None = None) -> pd.DataFrame:
    df = benchmark(events=[EVENT], devices=devices)
    out = REPO_ROOT / "outputs" / "metrics" / "scale_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def fig_benchmark(df: pd.DataFrame, out: Path) -> Path:
    apply_theme()
    out = Path(out)
    x = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(top=0.82, wspace=0.22)
    a1.bar(x, df["px_per_s"] / 1e6, color="#3A5F76", width=0.6)
    a1.set_xticks(x)
    a1.set_xticklabels(df["device"].str.upper())
    a1.set_ylabel("Megapixels / second")
    a1.set_title("Inference throughput", loc="left")
    for i, v in enumerate(df["px_per_s"]):
        a1.text(
            i,
            v / 1e6,
            f"{v/1e6:.2f} MP/s\n{df['peak_rss_mb'].iloc[i]:.0f} MB",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    thin_axes(a1)

    a2.bar(x, df["proj_hours_state"], color="#B8553A", width=0.6)
    a2.set_xticks(x)
    a2.set_xticklabels(df["device"].str.upper())
    a2.set_ylabel("Projected wall-clock hours")
    a2.set_title("One worker · 80 Mha state · 0.5 m aerial", loc="left")
    for i, (h, u) in enumerate(zip(df["proj_hours_state"], df["proj_usd_state"], strict=False)):
        a2.text(i, h, f"{h:.0f} h\n≈${u:,.0f}", ha="center", va="bottom", fontsize=8)
    thin_axes(a2)

    fig.suptitle(
        "From a laptop to a continent: throughput, memory and cost",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=13.5,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def log_runs_to_mlflow() -> list[str]:
    """Backfill MLflow with the runs already on disk so the tracking store and
    `mlflow ui` reflect every trained model."""
    from src.utils import tracking

    logged = []
    for d, name in MODEL_DIRS.items():
        hp = REPO_ROOT / "outputs" / "models" / d / "history.json"
        if not hp.exists():
            continue
        history = json.loads(hp.read_text())
        with tracking.start_run(run_name=name):
            tracking.log_params({"model_dir": d})
            for h in history:
                if h.get("val"):
                    tracking.log_metrics(
                        {
                            "val_macro_iou": h["val"]["macro_iou"],
                            "val_macro_f1": h["val"]["macro_f1"],
                        },
                        step=h["epoch"],
                    )
        logged.append(name)
    log.info("logged %d runs to MLflow store at outputs/mlruns", len(logged))
    return logged


def fig_tracking(out: Path) -> Path:
    apply_theme()
    out = Path(out)
    fig, ax = plt.subplots(figsize=(9, 5))
    for d, name in MODEL_DIRS.items():
        hp = REPO_ROOT / "outputs" / "models" / d / "history.json"
        if not hp.exists():
            continue
        history = json.loads(hp.read_text())
        ep = [h["epoch"] for h in history if h.get("val")]
        iou = [h["val"]["macro_iou"] for h in history if h.get("val")]
        if ep:
            ax.plot(ep, iou, marker="o", ms=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro IoU")
    ax.set_title("Tracked experiments: validation curve per run (MLflow-logged)", loc="left")
    ax.legend(fontsize=9)
    thin_axes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(figdir: Path, devices: list[str] | None = None) -> dict:
    figdir = Path(figdir)
    df = run_benchmark(devices=devices)
    log_runs_to_mlflow()
    return {
        "benchmark_df": df,
        "benchmark": fig_benchmark(df, figdir / "14_scale_benchmark.png"),
        "tracking": fig_tracking(figdir / "15_mlflow_runs.png"),
    }
