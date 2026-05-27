"""Tabular per-pixel dataset for classical ML.

Reads the per-event tile npz files produced by `src.data.tiling`, recomputes
the 18-channel feature stack on the fly, drops cloud-occluded and ignore-label
pixels, then **stratified-samples** N pixels per class per tile so that minority
severity classes (high, very_high) aren't drowned by unburnt pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import IGNORE_ID
from src.features.stack_features import DEFAULT_LAYOUT, build_stack
from src.utils.geo import REPO_ROOT


def _flat_features(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    stack = build_stack(pre, post)                 # [18, H, W]
    return stack.reshape(stack.shape[0], -1).T     # [H*W, 18]


def sample_event_pixels(event_id: str,
                        splits: tuple[str, ...] = ("train",),
                        pixels_per_class: int = 50_000,
                        seed: int = 42,
                        num_classes: int = 4) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) sampled from this event's tiles.

    Per-class stratification is performed within each (event, split) — we draw
    up to `pixels_per_class` pixels uniformly across all tiles in that split.
    """
    rng = np.random.default_rng(seed)
    idx_path = REPO_ROOT / "data" / "processed" / f"tile_index_{event_id}.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    df = pd.read_parquet(idx_path)
    df = df[df["split"].isin(splits)]
    if df.empty:
        raise ValueError(f"No tiles for event={event_id} in splits={splits}")

    bucket_X: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}
    bucket_y: dict[int, list[np.ndarray]] = {c: [] for c in range(num_classes)}
    quota = pixels_per_class

    for _, row in df.iterrows():
        with np.load(REPO_ROOT / row["tile_path"]) as npz:
            pre = npz["pre"].astype(np.float32)
            post = npz["post"].astype(np.float32)
            mask = npz["mask"].astype(bool)
            label = npz["label"]

        valid = mask & (label != IGNORE_ID)
        if not valid.any():
            continue
        feat = _flat_features(pre, post)         # [H*W, 18]
        lab_flat = label.ravel()
        valid_flat = valid.ravel()

        for c in range(num_classes):
            class_idx = np.where(valid_flat & (lab_flat == c))[0]
            if len(class_idx) == 0:
                continue
            taken = sum(a.shape[0] for a in bucket_X[c])
            remaining = quota - taken
            if remaining <= 0:
                continue
            take = min(len(class_idx), remaining)
            chosen = rng.choice(class_idx, size=take, replace=False)
            bucket_X[c].append(feat[chosen])
            bucket_y[c].append(np.full(take, c, dtype=np.uint8))

    X_all = np.concatenate([np.concatenate(v) for v in bucket_X.values() if v], axis=0) \
        if any(bucket_X.values()) else np.zeros((0, 18), dtype=np.float32)
    y_all = np.concatenate([np.concatenate(v) for v in bucket_y.values() if v], axis=0) \
        if any(bucket_y.values()) else np.zeros((0,), dtype=np.uint8)
    return X_all, y_all, list(DEFAULT_LAYOUT)


def stack_events(event_ids: list[str], splits: tuple[str, ...],
                 pixels_per_class: int = 50_000,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    Xs, ys = [], []
    feature_names: list[str] = []
    for ev in event_ids:
        X, y, names = sample_event_pixels(ev, splits, pixels_per_class, seed)
        if X.size > 0:
            Xs.append(X); ys.append(y); feature_names = names
    if not Xs:
        raise ValueError(f"No pixels sampled from events={event_ids} splits={splits}")
    return np.concatenate(Xs), np.concatenate(ys), feature_names


def predict_full_event(model, event_id: str) -> tuple[np.ndarray, object, object]:
    """Predict the full interim composite for an event using a fitted sklearn-like model."""
    import rasterio
    interim = REPO_ROOT / "data" / "interim" / event_id
    with rasterio.open(interim / "pre_stack_10m.tif") as ds:
        pre = ds.read().astype(np.float32)
        transform = ds.transform
        crs = ds.crs
    with rasterio.open(interim / "post_stack_10m.tif") as ds:
        post = ds.read().astype(np.float32)
    with rasterio.open(interim / "mask_pre_10m.tif") as ds:
        mpre = ds.read(1)
    with rasterio.open(interim / "mask_post_10m.tif") as ds:
        mpost = ds.read(1)
    mask = (mpre & mpost).astype(bool)

    feat = _flat_features(pre, post)              # [H*W, 18]
    pred = np.full(feat.shape[0], IGNORE_ID, dtype=np.uint8)
    valid_flat = mask.ravel()
    if valid_flat.any():
        pred[valid_flat] = model.predict(feat[valid_flat]).astype(np.uint8)
    pred = pred.reshape(pre.shape[1], pre.shape[2])
    return pred, transform, crs
