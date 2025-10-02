"""Robust per-tile z-score normalization and simple fusion utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np


def robust_z(x: np.ndarray, clamp: float = 6.0) -> np.ndarray:
    """Compute robust z-score using median and MAD.

    z = (x - median) / (1.4826 * MAD)

    Behavior:
    - Ignores NaNs in statistics.
    - If MAD == 0: return zeros where input is finite, NaN where input is NaN.
    - Clamp final z to [-clamp, clamp].
    - Output dtype is float32; NaN nodata preserved.

    Parameters
    ----------
    x : np.ndarray
        Input array (2D recommended) with floats, may contain NaNs.
    clamp : float, optional
        Absolute clamp value for z-scores, by default 6.0.

    Returns
    -------
    np.ndarray
        Robust z-scores with dtype float32 and NaNs for nodata.
    """
    x = np.asarray(x, dtype="float32")
    # Compute robust center and scale ignoring NaNs
    med = np.nanmedian(x)
    # If all-NaN, return all-NaN
    if np.isnan(med):
        return np.full_like(x, np.nan, dtype="float32")

    mad = np.nanmedian(np.abs(x - med))

    # Handle zero MAD (constant field over valid pixels)
    if mad == 0 or np.isnan(mad):
        z = np.zeros_like(x, dtype="float32")
        # Preserve NaNs from x
        z[np.isnan(x)] = np.nan
        return z

    scale = np.float32(1.4826) * np.float32(mad)
    z = (x - np.float32(med)) / scale
    # Clamp while preserving NaNs
    z = np.clip(z, -np.float32(clamp), np.float32(clamp), out=z)
    z[np.isnan(x)] = np.nan
    return z.astype("float32", copy=False)


def fuse_layers(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Fuse two z-score layers by mean ignoring NaNs, passing through when only one is available.

    - If both z1 and z2 have values at a pixel, output mean(z1, z2).
    - If only one has value, pass it through.
    - If both are NaN, output NaN.

    Parameters
    ----------
    z1 : np.ndarray
        First z-score layer.
    z2 : np.ndarray
        Second z-score layer.

    Returns
    -------
    np.ndarray
        Fused array, dtype float32, with NaNs for nodata.
    """
    z1 = np.asarray(z1, dtype="float32")
    z2 = np.asarray(z2, dtype="float32")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}")

    stack = np.stack([z1, z2], axis=0)
    fused = np.nanmean(stack, axis=0)
    # Where both were NaN, nanmean returns NaN already; ensure dtype float32
    return fused.astype("float32", copy=False)


__all__ = ["robust_z", "fuse_layers"]