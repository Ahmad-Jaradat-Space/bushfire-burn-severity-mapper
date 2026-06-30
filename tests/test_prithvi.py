"""Prithvi-EO-2.0 wrapper smoke test.

Skipped wherever terratorch is absent (e.g. CI installs only [dev,dl], not
[fm]); run locally where the foundation-model stack and cached weights exist.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")


def test_prithvi_forward_shape_and_frozen_backbone():
    from src.models.prithvi_model import build_prithvi

    m = build_prithvi(num_classes=4, frozen=True).eval()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # frozen 300M backbone -> only a tiny decoder trains
    assert trainable < 0.05 * total
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 4, 128, 128)
