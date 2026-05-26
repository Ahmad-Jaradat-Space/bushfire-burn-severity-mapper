"""Spectral indices used as features.

All formulas operate on band reflectance in [0, 1] (scaled from Sentinel-2 L2A
DN by 1/10000). Numpy ndarray or xarray DataArray inputs are both supported.
"""
from __future__ import annotations


def _safe_div(num, den):
    # Both numpy and xarray handle scalar 1e-10 cleanly.
    return num / (den + 1e-10)


def nbr(b08, b12):
    """Normalised Burn Ratio = (NIR - SWIR2) / (NIR + SWIR2)."""
    return _safe_div(b08 - b12, b08 + b12)


def ndvi(b08, b04):
    """Normalised Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""
    return _safe_div(b08 - b04, b08 + b04)


def ndmi(b08, b11):
    """Normalised Difference Moisture Index = (NIR - SWIR1) / (NIR + SWIR1)."""
    return _safe_div(b08 - b11, b08 + b11)


def nbr2(b11, b12):
    """NBR2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)."""
    return _safe_div(b11 - b12, b11 + b12)


def bsi(b02, b04, b08, b11):
    """Bare Soil Index = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))."""
    num = (b11 + b04) - (b08 + b02)
    den = (b11 + b04) + (b08 + b02)
    return _safe_div(num, den)


def delta(pre, post):
    """dIndex = pre - post (positive delta indicates loss/burn for NBR, NDVI, NDMI)."""
    return pre - post
