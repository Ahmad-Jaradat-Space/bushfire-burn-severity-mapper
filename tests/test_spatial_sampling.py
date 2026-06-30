import numpy as np
import pandas as pd

from src.data.spatial_sampling import (
    assign_spatial_blocks,
    spatial_block_folds,
    stratified_pixel_sample,
)


def _toy_index(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "event_id": ["e"] * n,
            "y": rng.integers(0, 4096, n),
            "x": rng.integers(0, 4096, n),
        }
    )


def test_blocks_are_fold_disjoint():
    idx = _toy_index()
    folds, df = spatial_block_folds(idx, k=5, super_block_px=1024, seed=1)
    for train, val in folds:
        train_blocks = set(df.loc[train, "block_id"])
        val_blocks = set(df.loc[val, "block_id"])
        assert train_blocks.isdisjoint(val_blocks)  # no block straddles the split


def test_folds_partition_all_tiles():
    idx = _toy_index()
    folds, df = spatial_block_folds(idx, k=5, seed=2)
    covered = set()
    for _train, val in folds:
        covered |= set(val.tolist())
    assert covered == set(range(len(df)))


def test_assign_blocks_quantises_position():
    idx = pd.DataFrame({"event_id": ["e", "e"], "y": [10, 1100], "x": [10, 10]})
    df = assign_spatial_blocks(idx, super_block_px=1024)
    assert df["block_id"].iloc[0] != df["block_id"].iloc[1]  # different super-block rows


def test_stratified_pixel_sample_allocation():
    m = np.zeros((50, 50), dtype=np.uint8)
    m[:25] = 0
    m[25:] = 1
    m[0, 0] = 255  # ignore
    out = stratified_pixel_sample(m, n_per_class=30, seed=3)
    cls = out["classes"]
    assert (cls == 0).sum() == 30 and (cls == 1).sum() == 30
    assert 255 not in cls
    # sampled coordinates really carry the claimed class
    assert all(m[r, c] == k for r, c, k in zip(out["rows"], out["cols"], cls, strict=False))
