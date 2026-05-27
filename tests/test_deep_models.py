"""Forward-pass tests for U-Net + SegFormer adapted to 18-channel input.

These run on CPU so the test suite stays portable. MPS device tests are gated
on torch.backends.mps.is_available().
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

torch = pytest.importorskip("torch")


def test_unet_forward_18ch_cpu():
    from src.models.unet_model import build_unet
    m = build_unet(in_channels=18, num_classes=4, encoder_name="resnet34",
                   encoder_weights=None)
    m.eval()
    x = torch.randn(1, 18, 256, 256)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 4, 256, 256)


def test_unet_smaller_encoder():
    from src.models.unet_model import build_unet
    m = build_unet(in_channels=18, num_classes=4, encoder_name="mobilenet_v2",
                   encoder_weights=None)
    m.eval()
    x = torch.randn(1, 18, 128, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 4, 128, 128)


def test_combo_loss_with_ignore():
    from src.models.losses import combo_loss
    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (2, 32, 32))
    target[:, :8, :] = 255    # ignore band
    loss, comps = combo_loss(logits, target, ce_weight=0.5, dice_weight=0.5,
                             ignore_index=255)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert "ce" in comps and "dice" in comps


def test_segformer_first_conv_inflation():
    """The inflated first-conv should accept 18 channels and produce sane output."""
    pytest.importorskip("transformers")
    from src.models.segformer_model import SegformerWrapper
    m = SegformerWrapper(in_channels=18, num_classes=4, backbone="nvidia/mit-b0")
    m.eval()
    x = torch.randn(1, 18, 128, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 4, 128, 128)
    # First-conv weight inflation: walk the model to find the first Conv2d and
    # confirm its in_channels is now 18, not 3. The exact attribute path differs
    # across transformers versions, so we search by structure.
    found = [mod for mod in m.modules() if mod.__class__.__name__ == "Conv2d"]
    assert found, "Expected at least one Conv2d module"
    # First Conv2d in iteration order is the patch embedding
    assert found[0].in_channels == 18, f"First conv still has {found[0].in_channels} channels"
