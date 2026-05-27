import numpy as np

from src.data.cloud_mask import DEFAULT_SCL_MASK_CLASSES, clear_fraction, scl_to_clear_mask


def test_clear_classes_kept():
    # SCL=4 (vegetation), 5 (not-veg), 6 (water) should all be kept
    scl = np.array([[4, 5, 6], [4, 5, 6]], dtype=np.uint8)
    out = scl_to_clear_mask(scl, dilate_pixels=0)
    assert out.all()


def test_cloud_classes_masked():
    scl = np.array([[4, 9, 4], [4, 4, 4]], dtype=np.uint8)
    out = scl_to_clear_mask(scl, dilate_pixels=0)
    assert not out[0, 1]
    # Without dilation, neighbours stay clear
    assert out[0, 0] and out[0, 2]


def test_dilation_expands_mask():
    scl = np.full((7, 7), 4, dtype=np.uint8)
    scl[3, 3] = 9  # one cloud pixel
    out_no_dilate = scl_to_clear_mask(scl, dilate_pixels=0)
    out_dilate = scl_to_clear_mask(scl, dilate_pixels=2)
    blocked_no = (~out_no_dilate).sum()
    blocked_d = (~out_dilate).sum()
    assert blocked_d > blocked_no


def test_default_classes_cover_clouds():
    assert {8, 9, 10} <= set(DEFAULT_SCL_MASK_CLASSES)


def test_clear_fraction():
    arr = np.array([[True, False], [True, True]])
    assert abs(clear_fraction(arr) - 0.75) < 1e-6
