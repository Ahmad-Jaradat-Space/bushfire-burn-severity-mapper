import numpy as np

from src.features.stack_features import DEFAULT_LAYOUT, build_stack, indices_pre_post


def _synthetic_pre_post(h=8, w=8):
    rng = np.random.default_rng(0)
    pre = rng.uniform(0.05, 0.6, size=(6, h, w)).astype(np.float32)
    post = pre.copy()
    # Simulate a fire by tanking NIR (B08, idx 3) and boosting SWIR2 (B12, idx 5)
    post[3] *= 0.3
    post[5] *= 1.5
    return pre, post


def test_indices_keys():
    pre, post = _synthetic_pre_post()
    deltas = indices_pre_post(pre, post)
    assert set(deltas.keys()) == {"dNBR", "dNDVI", "dNDMI", "dNBR2", "dBSI"}
    # dNBR should be positive on average where we simulated burn
    assert deltas["dNBR"].mean() > 0


def test_build_stack_shape():
    pre, post = _synthetic_pre_post(16, 24)
    stack = build_stack(pre, post)
    assert stack.shape == (18, 16, 24)
    assert stack.dtype == np.float32
    assert len(DEFAULT_LAYOUT) == 18


def test_build_stack_with_slope():
    pre, post = _synthetic_pre_post()
    slope = np.full(pre.shape[1:], 12.0, dtype=np.float32)
    stack = build_stack(pre, post, slope=slope)
    np.testing.assert_allclose(stack[-1], 12.0)
