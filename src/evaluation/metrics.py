"""Multi-class segmentation metrics with `ignore_index` support.

All functions operate on flattened uint8 arrays and treat `ignore_index`
pixels as not-present in both prediction and truth.
"""
from __future__ import annotations

import numpy as np

IGNORE_ID = 255


def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int,
                     ignore_index: int = IGNORE_ID) -> np.ndarray:
    p = pred.ravel()
    t = true.ravel()
    valid = (t != ignore_index) & (p != ignore_index)
    p = p[valid]
    t = t[valid]
    idx = t * num_classes + p
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return _safe_div(tp, tp + fp + fn)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    return _safe_div(2 * prec * rec, prec + rec)


def precision_recall(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def summary(pred: np.ndarray, true: np.ndarray, num_classes: int,
            ignore_index: int = IGNORE_ID) -> dict:
    cm = confusion_matrix(pred, true, num_classes, ignore_index)
    iou = per_class_iou(cm)
    f1 = per_class_f1(cm)
    prec, rec = precision_recall(cm)
    acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    return {
        "num_classes": num_classes,
        "confusion_matrix": cm.tolist(),
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "macro_iou": float(iou.mean()),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "accuracy": acc,
        "n_pixels": int(cm.sum()),
    }


def binary_summary(pred: np.ndarray, true: np.ndarray,
                   ignore_index: int = IGNORE_ID) -> dict:
    """Burnt vs unburnt: class 0 = unburnt, classes 1+ = burnt."""
    p = pred.copy()
    t = true.copy()
    valid = (t != ignore_index) & (p != ignore_index)
    p = (p[valid] > 0).astype(np.uint8)
    t = (t[valid] > 0).astype(np.uint8)
    cm = confusion_matrix(p, t, 2, ignore_index=255)
    iou = per_class_iou(cm)
    f1 = per_class_f1(cm)
    return {
        "iou_unburnt": float(iou[0]),
        "iou_burnt": float(iou[1]),
        "f1_unburnt": float(f1[0]),
        "f1_burnt": float(f1[1]),
        "macro_iou": float(iou.mean()),
        "macro_f1": float(f1.mean()),
        "n_pixels": int(cm.sum()),
    }
