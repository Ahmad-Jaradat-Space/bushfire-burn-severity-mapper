import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.evaluation.uq_maps import (
    enable_mc_dropout,
    mc_predict,
    predictive_entropy,
)


def test_predictive_entropy_bounds():
    C, H, W = 4, 5, 6
    uniform = np.full((C, H, W), 1 / C, dtype=np.float32)
    onehot = np.zeros((C, H, W), dtype=np.float32)
    onehot[0] = 1.0
    np.testing.assert_allclose(predictive_entropy(uniform), np.log(C), rtol=1e-5)
    assert predictive_entropy(onehot).max() < 1e-5


def test_enable_mc_dropout_counts_and_toggles():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Dropout2d(0.2), nn.Conv2d(4, 4, 1))
    model.eval()
    n = enable_mc_dropout(model)
    assert n == 1
    # the dropout module is now training, the conv layers are not
    assert model[1].training is True
    assert model[0].training is False


def test_mc_predict_shapes_and_epistemic_nonneg():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(18, 8, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.3), nn.Conv2d(8, 4, 1)
    )
    image = np.random.default_rng(0).standard_normal((18, 40, 40)).astype(np.float32)
    mean = np.zeros(18, dtype=np.float32)
    std = np.ones(18, dtype=np.float32)
    out = mc_predict(
        model,
        image,
        mean,
        std,
        tile=32,
        stride=32,
        device=torch.device("cpu"),
        num_classes=4,
        T=5,
        mc_dropout=True,
    )
    assert out["mean_prob"].shape == (4, 40, 40)
    assert out["pred"].shape == (40, 40)
    assert out["n_dropout_modules"] == 1
    # mutual information (epistemic) is clipped to be non-negative
    assert out["mutual_information"].min() >= 0.0
    # probabilities sum to 1 across classes
    np.testing.assert_allclose(out["mean_prob"].sum(axis=0), 1.0, rtol=1e-4)
