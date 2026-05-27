"""Combined CE + Dice loss with ignore_index support, MPS-safe.

The Dice term operates on softmax probabilities and ignores `ignore_index`
pixels in both the prediction and the target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255,
              eps: float = 1e-6) -> torch.Tensor:
    """logits: [B, C, H, W] float, target: [B, H, W] int64."""
    num_classes = logits.shape[1]
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index)                          # [B, H, W]
    target_safe = target.clone()
    target_safe[~valid] = 0
    onehot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()
    valid_4d = valid.unsqueeze(1).float()
    probs = probs * valid_4d
    onehot = onehot * valid_4d
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
    per_class = 1.0 - (2 * inter + eps) / (denom + eps)
    return per_class.mean()


def combo_loss(logits: torch.Tensor, target: torch.Tensor,
               ce_weight: float = 0.5, dice_weight: float = 0.5,
               ignore_index: int = 255) -> tuple[torch.Tensor, dict]:
    ce = F.cross_entropy(logits.float(), target, ignore_index=ignore_index)
    dl = dice_loss(logits, target, ignore_index)
    total = ce_weight * ce + dice_weight * dl
    return total, {"ce": float(ce.detach()), "dice": float(dl.detach()),
                   "total": float(total.detach())}
