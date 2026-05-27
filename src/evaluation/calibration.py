"""Reliability diagram + Brier score for the burnt-class probability.

Inputs:
  pred_proba  float32 [H, W]   prob(burnt) — comes from RF.predict_proba or
                                the softmax of U-Net/SegFormer logits
  true        uint8   [H, W]   internal class IDs (0=unburnt; 1+=burnt; 255 ignore)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import IGNORE_ID


def reliability_data(pred_proba: np.ndarray, true: np.ndarray,
                     n_bins: int = 10) -> dict:
    valid = true != IGNORE_ID
    p = pred_proba.ravel()[valid.ravel()]
    t = (true.ravel()[valid.ravel()] > 0).astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    bin_mean_p = np.zeros(n_bins)
    bin_mean_t = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        bin_mean_p[b] = p[m].mean()
        bin_mean_t[b] = t[m].mean()
        bin_n[b] = m.sum()
    ece = float(np.sum(bin_n * np.abs(bin_mean_t - bin_mean_p)) / max(bin_n.sum(), 1))
    brier = float(((p - t) ** 2).mean())
    return {
        "bin_mean_p": bin_mean_p.tolist(),
        "bin_mean_t": bin_mean_t.tolist(),
        "bin_n": bin_n.tolist(),
        "ece": ece,
        "brier": brier,
        "n_total": int(bin_n.sum()),
    }


def plot_reliability(data: dict, out_path: Path, title: str = "") -> Path:
    bin_mean_p = np.array(data["bin_mean_p"])
    bin_mean_t = np.array(data["bin_mean_t"])
    bin_n = np.array(data["bin_n"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect calibration")
    # Bin centres weighted by count
    sizes = 50 + 250 * (bin_n / max(bin_n.max(), 1))
    ax.scatter(bin_mean_p[bin_n > 0], bin_mean_t[bin_n > 0],
               s=sizes[bin_n > 0], alpha=0.7)
    ax.set_xlabel("Predicted P(burnt)")
    ax.set_ylabel("Empirical fraction burnt")
    ax.set_title(f"{title}\nECE={data['ece']:.3f}  Brier={data['brier']:.3f}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
