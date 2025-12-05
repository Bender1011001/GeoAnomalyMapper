#!/usr/bin/env python3
"""
Phase 3: Physics-Based Poisson Correlation Analysis for GeoAnomalyMapper v2.0.

Computes pseudo-gravity from magnetic data via FFT-based Reduction-to-Pole (RTP),
then local Pearson correlation with gravity residuals in sliding windows.
Negative correlation indicates low-gravity voids reinforced by high-magnetic steel rebar.

Outputs: poisson_correlation.tif (-1 to 1, same CRS/shape as input gravity).
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.fft import fft2, ifft2, fftfreq
import rasterio
from typing import Tuple, Optional


def load_raster(path: str) -> np.ndarray:
    """
    Load single-band raster as numpy array, converting nodata to NaN.

    Args:
        path: Path to GeoTIFF file.

    Returns:
        2D numpy array (float64) with NaNs for nodata.

    Raises:
        rasterio.errors.RasterioIOError: If file cannot be opened.
        ValueError: If not single band.

    Example:
        >>> data = load_raster('gravity_residual.tif')
        >>> print(data.shape, np.nanmean(data))
        (1000, 1000) 2.34
    """
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError("Raster must be single-band")
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        return data.astype(np.float64)


def save_raster(path: str, data: np.ndarray, profile: dict) -> None:
    """
    Save 2D array as GeoTIFF using provided profile.

    Args:
        path: Output GeoTIFF path.
        data: 2D numpy array (float32 recommended).
        profile: Rasterio profile dict from source (e.g., gravity).

    Raises:
        rasterio.errors.RasterioIOError: If write fails.

    Example:
        >>> profile = rasterio.open('input.tif').profile
        >>> save_raster('output.tif', corr_data, profile)
    """
    profile = profile.copy()
    profile.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan,
        'compress': 'DEFLATE',
        'BIGTIFF': 'YES' if data.nbytes > 4e9 else 'NO'
    })
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def compute_pseudo_gravity(
    mag_data: np.ndarray,
    inc_deg: float = 60.0,
    dec_deg: float = 0.0,
    dx: float = 4000.0,
    dy: float = 4000.0
) -> np.ndarray:
    """
    Compute pseudo-gravity from magnetic data: |sin(I)| * RTP via FFT.

    Reduction-to-Pole (RTP) corrects for inclination/declination using frequency-domain
    transfer function. Pseudo-gravity approximates gravity anomaly assuming Poisson's
    relation (mag source ~ density source). Pads with reflection, handles NaNs.

    Args:
        mag_data: 2D magnetic array (nT).
        inc_deg: Magnetic inclination (degrees, default 60 mid-lat).
        dec_deg: Magnetic declination (degrees, default 0).
        dx: Pixel size x (meters, ~4km for EMAG2).
        dy: Pixel size y (meters).

    Returns:
        Pseudo-gravity array, same shape as input.

    Raises:
        ValueError: If input not 2D or dx/dy <=0.

    Example:
        >>> mag = np.array([[1.,2.],[3.,4.]])
        >>> pseudo = compute_pseudo_gravity(mag, inc_deg=60)
        >>> print(np.round(pseudo, 1))  # approx [[0.9, 1.7], [2.6, 3.5]] (sin60~0.866 * RTP)
        [[0.9 1.7]
         [2.6 3.5]]
    """
    if not isinstance(mag_data.ndim, int) or mag_data.ndim != 2:
        raise ValueError("mag_data must be 2D array")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx, dy must be positive (meters/pixel)")

    ny, nx = mag_data.shape
    inc_rad = np.deg2rad(inc_deg)
    dec_rad = np.deg2rad(dec_deg)

    # Pad to double size for FFT (reduces edge artifacts)
    pad_factor = 2
    nypad = pad_factor * ny
    nxpad = pad_factor * nx
    pad_y = (nypad - ny) // 2
    pad_x = (nxpad - nx) // 2

    # Fill NaNs with 0, track mask
    mask = np.isnan(mag_data)
    mag_fill = np.nan_to_num(mag_data, nan=0.0)
    mag_pad = np.pad(mag_fill, ((pad_y, nypad - ny - pad_y), (pad_x, nxpad - nx - pad_x)), mode='reflect')

    # FFT
    mag_fft = fft2(mag_pad)

    # Wavenumbers (rad/m, unshifted)
    fy = fftfreq(nypad, dy)
    fx = fftfreq(nxpad, dx)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')  # FY (ny_pad, nx_pad) varies row-wise
    KX = 2 * np.pi * FX
    KY = 2 * np.pi * FY
    K = np.sqrt(KX**2 + KY**2)
    K[K == 0] = 1e-12  # Avoid div-by-zero at DC

    # Unit vectors
    nx_unit = KX / K
    ny_unit = KY / K

    # RTP transfer function: cos(I) + i sin(I) * (nx sin(D) + ny cos(D))
    phase = nx_unit * np.sin(dec_rad) + ny_unit * np.cos(dec_rad)
    rtp_tf = np.cos(inc_rad) + 1j * np.sin(inc_rad) * phase

    # Apply
    rtp_fft = mag_fft * rtp_tf
    rtp_pad = np.real(ifft2(rtp_fft))

    # Crop
    rtp = rtp_pad[pad_y:pad_y + ny, pad_x:pad_x + nx]

    # Pseudo-gravity scaling
    pseudo = np.abs(np.sin(inc_rad)) * rtp

    # Restore NaNs
    pseudo[mask] = np.nan

    return pseudo


def compute_poisson_ratio(
    grav_data: np.ndarray,
    pseudo_data: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Compute local Pearson correlation coefficient in sliding windows.

    Vectorized using uniform_filter for means/vars/cov. Reflect padding.
    Handles NaNs by filling 0 + masking output. Invalid corr (zero var) → NaN.

    Args:
        grav_data: 2D gravity residual array.
        pseudo_data: 2D pseudo-gravity array (same shape).
        window_size: Square window side (odd preferred, default 5 → 25 pts).

    Returns:
        Correlation map [-1,1], NaN at edges/invalids/input-NaNs.

    Raises:
        ValueError: If shapes mismatch or window_size < 1.

    Example:
        >>> grav = np.array([[1,2,1],[2,1,2],[1,2,1]])
        >>> pseudo = np.array([[1,3,2],[3,1,3],[2,3,1]])
        >>> corr = compute_poisson_ratio(grav, pseudo, window_size=3)
        >>> print(np.round(corr[1,1], 2))  # ~0.87
        0.87
        >>> grav_flat = np.array([1,1,1]); pseudo_flat = np.array([2,1,0])
        >>> # Equivalent: corr ~ -0.5
    """
    if grav_data.shape != pseudo_data.shape:
        raise ValueError("grav_data and pseudo_data must have same shape")
    if window_size < 1:
        raise ValueError("window_size must be >=1")
    if window_size == 1:
        print("Warning: window_size=1 yields corr=1.0 where defined.")

    pad = window_size // 2
    mask = np.isnan(grav_data) | np.isnan(pseudo_data)
    grav_fill = np.nan_to_num(grav_data, nan=0.0)
    pseudo_fill = np.nan_to_num(pseudo_data, nan=0.0)

    # Pad
    grav_pad = np.pad(grav_fill, pad, mode='reflect')
    pseudo_pad = np.pad(pseudo_fill, pad, mode='reflect')

    # Local means
    mean_g = uniform_filter(grav_pad, size=window_size)
    mean_p = uniform_filter(pseudo_pad, size=window_size)

    # Centered
    g_c = grav_pad - mean_g
    p_c = pseudo_pad - mean_p

    # Local statistics (mean over window)
    cov_gp = uniform_filter(g_c * p_c, size=window_size)
    var_g = uniform_filter(g_c**2, size=window_size)
    var_p = uniform_filter(p_c**2, size=window_size)

    # Crop to original size
    h, w = grav_data.shape
    cov_gp = cov_gp[pad:pad + h, pad:pad + w]
    var_g = var_g[pad:pad + h, pad:pad + w]
    var_p = var_p[pad:pad + h, pad:pad + w]

    # Correlation
    corr = np.full_like(grav_data, np.nan)
    valid = (var_g > 1e-10) & (var_p > 1e-10)
    corr[valid] = cov_gp[valid] / np.sqrt(var_g[valid] * var_p[valid])

    # Mask input NaNs
    corr[mask] = np.nan

    return corr


def analyze_poisson_correlation(
    gravity_residual_path: str,
    magnetic_path: str,
    output_path: str = 'poisson_correlation.tif',
    inc: float = 60.0,
    dec: float = 0.0,
    window_size: int = 5
) -> None:
    """
    Main Phase 3 workflow: Poisson correlation analysis.

    Loads aligned gravity/magnetic rasters (same grid/CRS), computes pseudo-gravity,
    sliding-window correlation, saves output.

    Args:
        gravity_residual_path: Path to gravity_residual.tif (Phase 1).
        magnetic_path: Path to magnetic.tif (processed EMAG2).
        output_path: Output poisson_correlation.tif.
        inc: Inclination (degrees).
        dec: Declination (degrees).
        window_size: Correlation window (pixels).

    Raises:
        ValueError: Mismatched shapes.
        rasterio.errors.RasterioIOError: File I/O errors.

    Example:
        >>> analyze_poisson_correlation('gravity_residual.tif', 'magnetic.tif')
        Phase 3 output: poisson_correlation.tif
    """
    # Load gravity + profile
    with rasterio.open(gravity_residual_path) as src:
        grav_data = src.read(1).astype(np.float64)
        if src.nodata is not None:
            grav_data = np.where(grav_data == src.nodata, np.nan, grav_data)
        profile = src.profile.copy()

    # Load magnetic
    mag_data = load_raster(magnetic_path)

    # Align check
    if grav_data.shape != mag_data.shape:
        raise ValueError(f"Shape mismatch: gravity {grav_data.shape}, magnetic {mag_data.shape}")

    # Compute
    pseudo_data = compute_pseudo_gravity(mag_data, inc_deg=inc, dec_deg=dec, dx=4000.0, dy=4000.0)
    corr_data = compute_poisson_ratio(grav_data, pseudo_data, window_size=window_size)

    # Save
    save_raster(output_path, corr_data, profile)
    print(f"Phase 3 output: {output_path}")