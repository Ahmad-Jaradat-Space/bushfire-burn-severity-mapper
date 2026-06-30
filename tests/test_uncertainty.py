import numpy as np

from src.evaluation.blocks import block_confusions, grid_blocks
from src.evaluation.metrics import confusion_matrix, per_class_iou
from src.evaluation.uncertainty import (
    bootstrap_macro_metrics,
    mcnemar_test,
    paired_bootstrap_delta,
)


def _toy_scene(seed=0):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 2, size=(20, 20)).astype(np.uint8)
    pred = true.copy()
    # Flip ~15% of pixels so macro-IoU sits strictly between 0 and 1.
    flip = rng.random((20, 20)) < 0.15
    pred[flip] = 1 - pred[flip]
    return pred, true


def test_grid_blocks_shapes_and_count():
    blk = grid_blocks((20, 20), 5)
    assert blk.shape == (20, 20)
    assert blk.min() == 0
    assert blk.max() == 15  # 4x4 grid of 5px blocks


def test_block_confusions_sum_to_global():
    pred, true = _toy_scene()
    blk = grid_blocks(pred.shape, 5)
    cms = block_confusions(pred, true, blk, num_classes=2)
    summed = sum(cms.values())
    np.testing.assert_array_equal(summed, confusion_matrix(pred, true, 2))


def test_bootstrap_ci_brackets_point_estimate():
    pred, true = _toy_scene()
    blk = grid_blocks(pred.shape, 5)
    cms = block_confusions(pred, true, blk, num_classes=2)
    res = bootstrap_macro_metrics(cms, n_boot=500, seed=1)
    global_iou = float(per_class_iou(confusion_matrix(pred, true, 2)).mean())
    m = res["metrics"]["macro_iou"]
    assert abs(m["point"] - global_iou) < 1e-9
    assert m["ci_low"] <= m["point"] <= m["ci_high"]
    assert m["se"] > 0
    assert res["n_blocks"] == 16


def test_mcnemar_identical_predictions_is_insignificant():
    _, true = _toy_scene()
    out = mcnemar_test(true.copy(), true.copy(), true)
    assert out["n_discordant"] == 0
    assert out["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    true = np.ones((10, 20), dtype=np.uint8)
    pred_a = true.copy()  # always correct
    pred_b = true.copy()
    pred_b.ravel()[:100] = 0  # 100 pixels wrong
    out = mcnemar_test(pred_a, pred_b, true)
    assert out["b"] == 100 and out["c"] == 0
    assert out["method"] == "chi2-cc"
    assert out["p_value"] < 1e-6
    assert out["acc_a"] > out["acc_b"]


def test_paired_bootstrap_delta_sign():
    true = np.tile(np.array([0, 1], dtype=np.uint8), (20, 10))
    pred_good = true.copy()
    pred_bad = true.copy()
    pred_bad.ravel()[: pred_bad.size // 3] = 1 - pred_bad.ravel()[: pred_bad.size // 3]
    blk = grid_blocks(true.shape, 5)
    cms_a = block_confusions(pred_good, true, blk, 2)
    cms_b = block_confusions(pred_bad, true, blk, 2)
    out = paired_bootstrap_delta(cms_a, cms_b, n_boot=500, seed=2)
    assert out["delta"] > 0
    assert out["ci_low"] <= out["delta"] <= out["ci_high"]
