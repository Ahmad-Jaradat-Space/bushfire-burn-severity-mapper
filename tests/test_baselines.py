import numpy as np

from src.evaluation.metrics import binary_summary, summary
from src.models.baselines import dnbr, dnbr_binary, dnbr_multiclass_usgs


def _synth_pre_post(h=32, w=32, burn_ratio=0.5, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.uniform(0.05, 0.6, size=(6, h, w)).astype(np.float32)
    # Index order: B02 B03 B04 B08 B11 B12 — NIR is idx 3, SWIR2 is idx 5
    post = pre.copy()
    burn = np.zeros((h, w), dtype=bool)
    burn[:, : int(w * burn_ratio)] = True
    post[3][burn] *= 0.2     # NIR collapse
    post[5][burn] *= 1.6     # SWIR2 spike
    return pre, post, burn


def test_dnbr_positive_where_burnt():
    pre, post, burn = _synth_pre_post()
    d = dnbr(pre, post)
    assert d[burn].mean() > d[~burn].mean()


def test_binary_threshold_recovers_burn():
    pre, post, burn = _synth_pre_post()
    d = dnbr(pre, post)
    pred = dnbr_binary(d, threshold=0.10)
    # Should mostly recover burn extent on synthetic data
    accuracy = (pred == burn.astype(np.uint8)).mean()
    assert accuracy > 0.9


def test_multiclass_assigns_severity_inside_burnt():
    pre, post, burn = _synth_pre_post()
    d = dnbr(pre, post)
    mc = dnbr_multiclass_usgs(d)
    # Burnt half should not all be class 0 (unburnt)
    inside = mc[burn]
    assert (inside != 0).mean() > 0.5


def test_metrics_summary_keys():
    pred = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    true = pred.copy()
    s = summary(pred, true, num_classes=4)
    assert s["macro_iou"] == 1.0
    assert s["accuracy"] == 1.0


def test_binary_summary_perfect():
    pred = np.array([0, 1, 1, 0], dtype=np.uint8)
    true = pred.copy()
    b = binary_summary(pred, true)
    assert b["macro_iou"] == 1.0
    assert b["f1_burnt"] == 1.0


def test_metrics_ignore_index_excluded():
    pred = np.array([0, 1, 255, 2], dtype=np.uint8)
    true = np.array([0, 1, 0, 2], dtype=np.uint8)
    s = summary(pred, true, num_classes=4)
    # 255 in pred is dropped, so we have 3 pixels all correct
    assert s["n_pixels"] == 3
    assert s["accuracy"] == 1.0
