"""End-to-end smoke test using synthetic in-memory pre/post + labels.

Verifies that:
- the 18-channel feature stack composes correctly,
- the dNBR multiclass baseline produces a non-trivial prediction,
- RandomForest training + prediction wires up without errors and beats random,
- the metrics summary returns the expected keys with sensible values.

Does NOT require any downloaded Sentinel-2 imagery or GEEBAM labels.
"""
import numpy as np

from src.evaluation.metrics import summary
from src.features.stack_features import build_stack
from src.models.baselines import dnbr, dnbr_multiclass_usgs


def _synth_aoi(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.uniform(0.05, 0.6, size=(6, h, w)).astype(np.float32)
    post = pre.copy()
    # Make a circular high-severity patch in the centre
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    burn_high = r < 15
    burn_low = (r >= 15) & (r < 30)
    post[3][burn_high] *= 0.15
    post[5][burn_high] *= 1.8
    post[3][burn_low] *= 0.6
    post[5][burn_low] *= 1.2
    label = np.zeros((h, w), dtype=np.uint8)        # 0 = unburnt
    label[burn_low] = 1                              # low_mod
    label[burn_high] = 3                             # very_high
    return pre, post, label


def test_synthetic_dnbr_baseline_beats_random():
    pre, post, label = _synth_aoi()
    d = dnbr(pre, post)
    pred = dnbr_multiclass_usgs(d)
    s = summary(pred, label, num_classes=4)
    # Should be much better than random across 4 classes (chance acc=0.25)
    assert s["accuracy"] > 0.6
    assert s["macro_iou"] > 0.1


def test_synthetic_rf_beats_random(tmp_path, monkeypatch):
    """Run a tiny RF on synthetic data to prove the wiring is correct."""
    from sklearn.ensemble import RandomForestClassifier

    pre, post, label = _synth_aoi(h=96, w=96)
    stack = build_stack(pre, post).reshape(18, -1).T   # [H*W, 18]
    y = label.ravel()
    rng = np.random.default_rng(0)
    n = stack.shape[0]
    idx = rng.permutation(n)
    split = n // 2
    Xtr, Xte = stack[idx[:split]], stack[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]

    clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte).astype(np.uint8)
    s = summary(pred, yte, num_classes=4)
    assert s["accuracy"] > 0.85, f"RF accuracy too low: {s['accuracy']}"
    # dNBR-related features should dominate
    feat_imp = clf.feature_importances_
    top_idx = np.argsort(feat_imp)[-3:]
    # In our 18-channel layout indices 12..16 are dNBR/dNDVI/dNDMI/dNBR2/dBSI
    assert any(i >= 12 for i in top_idx)
