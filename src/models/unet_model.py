"""Thin wrapper around segmentation_models_pytorch.Unet."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    in_channels: int = 18,
    num_classes: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    dropout: float = 0.1,
) -> torch.nn.Module:
    """Build an SMP U-Net with 18 input channels and 4 output classes.

    Pretrained ImageNet weights only cover 3 channels, for our 18-channel
    stack the simplest path is to train from scratch (encoder_weights=None).
    Once we have working baselines, M11+ can revisit ImageNet pretrain by
    averaging the RGB conv weights across our 18 input bands.

    ``dropout`` inserts a ``Dropout2d`` on the decoder's final feature map
    (just before the segmentation head). This both regularises training and,
    crucially, gives us a genuine Monte-Carlo-dropout path for per-pixel
    epistemic uncertainty at inference (see ``src.evaluation.uq_maps``). smp's
    ``Unet`` has no dropout argument, so we wire it in here; ``dropout=0``
    reproduces the plain architecture.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=list(decoder_channels),
        activation=None,
    )
    if dropout and dropout > 0:
        # Prepend Dropout2d to the segmentation head so MC-dropout perturbs the
        # final decoder features. (Shifts head state-dict keys by one index,
        # checkpoints must be trained and loaded with the same dropout setting.)
        head = model.segmentation_head
        model.segmentation_head = nn.Sequential(nn.Dropout2d(p=float(dropout)), *head)
    return model
