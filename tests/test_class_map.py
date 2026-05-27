import numpy as np

from src.data.class_map import (
    GEEBAM_TO_INTERNAL,
    IGNORE_ID,
    INTERNAL_NAMES,
    class_histogram,
    remap_geebam,
)


def test_remap_basic():
    src = np.array([[0, 1, 2, 3], [4, 5, 1, 2]], dtype=np.uint8)
    out = remap_geebam(src)
    expected = np.array([[IGNORE_ID, IGNORE_ID, 0, 1], [2, 3, IGNORE_ID, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_internal_names_length():
    assert len(INTERNAL_NAMES) == 4


def test_remap_preserves_shape():
    src = np.zeros((10, 20), dtype=np.uint8)
    src[5, 5] = 4
    out = remap_geebam(src)
    assert out.shape == src.shape
    assert out[5, 5] == 2


def test_class_histogram():
    src = np.array([0, 0, 2, 2, 2, 4], dtype=np.uint8)
    hist = class_histogram(src)
    assert hist == {0: 2, 2: 3, 4: 1}


def test_geebam_remap_table_complete():
    # All GEEBAM values 0-5 are mapped
    assert set(GEEBAM_TO_INTERNAL.keys()) == {0, 1, 2, 3, 4, 5}
