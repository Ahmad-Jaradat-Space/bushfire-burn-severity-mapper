import numpy as np

from src.features.indices import bsi, delta, nbr, nbr2, ndmi, ndvi


def test_nbr_extremes():
    # Pure NIR returns +1; pure SWIR2 returns -1
    assert np.isclose(nbr(1.0, 0.0), 1.0, atol=1e-6)
    assert np.isclose(nbr(0.0, 1.0), -1.0, atol=1e-6)


def test_ndvi_zero_vegetation():
    # Red = NIR → no vegetation contrast → 0
    assert np.isclose(ndvi(0.3, 0.3), 0.0, atol=1e-6)


def test_ndmi_returns_finite():
    assert np.isfinite(ndmi(0.3, 0.4))


def test_nbr2_array():
    swir1 = np.array([0.2, 0.4])
    swir2 = np.array([0.1, 0.5])
    out = nbr2(swir1, swir2)
    np.testing.assert_allclose(out, [(0.2 - 0.1) / (0.3), (0.4 - 0.5) / (0.9)], atol=1e-6)


def test_bsi_array():
    out = bsi(np.array([0.1]), np.array([0.2]),
              np.array([0.3]), np.array([0.4]))
    expected = ((0.4 + 0.2) - (0.3 + 0.1)) / ((0.4 + 0.2) + (0.3 + 0.1))
    np.testing.assert_allclose(out, [expected], atol=1e-6)


def test_delta_sign():
    # Post-fire NBR drops → dNBR positive in burnt pixels
    pre = nbr(0.4, 0.1)   # vegetated
    post = nbr(0.2, 0.3)  # burnt
    assert delta(pre, post) > 0
