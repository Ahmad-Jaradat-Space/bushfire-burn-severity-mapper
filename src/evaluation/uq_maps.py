"""Per-pixel predictive uncertainty via Monte-Carlo dropout.

The event-wise U-Net carries a ``Dropout2d`` before its segmentation head
(:func:`src.models.unet_model.build_unet`). Keeping that dropout *on* at
inference and averaging ``T`` stochastic forward passes is Monte-Carlo dropout
(Gal & Ghahramani 2016): a cheap Bayesian approximation that yields a posterior
over the softmax. From it we read three maps:

* **predictive entropy** of the mean prediction, total uncertainty,
* **expected entropy** (mean of per-pass entropies), aleatoric (data) noise,
* **mutual information** = predictive − expected, epistemic (model) uncertainty,
  i.e. *what the model does not know* and more data could fix.

All passes are accumulated online, so memory stays at a couple of [C,H,W] cubes
regardless of ``T``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> int:
    """Put every dropout layer into train mode (leave the rest in eval). Returns
    the number of dropout modules activated, 0 means MC-dropout is a no-op."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


def _softmax_np(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def predictive_entropy(probs_chw: np.ndarray) -> np.ndarray:
    """Shannon entropy across the class axis of a [C, H, W] probability cube."""
    return -(probs_chw * np.log(probs_chw + 1e-12)).sum(axis=0)


def _sliding_softmax_pass(
    model, image_norm: np.ndarray, tile: int, stride: int, device, num_classes: int
) -> np.ndarray:
    """One forward pass over the whole image → softmax probs [C, H, W]."""
    _, H, W = image_norm.shape
    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end, x_end = min(y + tile, H), min(x + tile, W)
                y0, x0 = max(0, y_end - tile), max(0, x_end - tile)
                patch = image_norm[:, y0:y_end, x0:x_end]
                pt = torch.from_numpy(patch.astype(np.float32))[None].to(device)
                logits = model(pt).float().cpu().numpy()[0]
                prob_sum[:, y0:y_end, x0:x_end] += _softmax_np(logits, axis=0)
                counts[y0:y_end, x0:x_end] += 1
    return prob_sum / np.maximum(counts[None], 1)


def mc_predict(
    model,
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tile: int,
    stride: int,
    device,
    num_classes: int = 4,
    T: int = 10,
    mc_dropout: bool = True,
) -> dict:
    """Monte-Carlo-dropout predictive distribution over a full H×W image.

    Returns mean_prob [C,H,W], predictive/expected entropy and mutual-information
    maps [H,W], and the argmax prediction [H,W] uint8. With ``T=1`` and
    ``mc_dropout=False`` this is an ordinary deterministic prediction.
    """
    image_norm = (image - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    model.eval()
    n_drop = enable_mc_dropout(model) if mc_dropout else 0

    _, H, W = image_norm.shape
    sum_prob = np.zeros((num_classes, H, W), dtype=np.float64)
    sum_entropy = np.zeros((H, W), dtype=np.float64)
    for _ in range(T):
        p = _sliding_softmax_pass(model, image_norm, tile, stride, device, num_classes)
        sum_prob += p
        sum_entropy += predictive_entropy(p)

    mean_prob = (sum_prob / T).astype(np.float32)
    pred_ent = predictive_entropy(mean_prob)  # total
    exp_ent = (sum_entropy / T).astype(np.float32)  # aleatoric
    mutual_info = np.clip(pred_ent - exp_ent, 0, None)  # epistemic
    return {
        "mean_prob": mean_prob,
        "pred": mean_prob.argmax(axis=0).astype(np.uint8),
        "predictive_entropy": pred_ent.astype(np.float32),
        "expected_entropy": exp_ent,
        "mutual_information": mutual_info.astype(np.float32),
        "T": T,
        "n_dropout_modules": n_drop,
    }
