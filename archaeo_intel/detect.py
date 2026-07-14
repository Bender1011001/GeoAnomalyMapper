"""Detection analytics for archaeological surface proxies.

Every function here has a synthetic unit test (tests/test_archaeo_intel.py);
the mound channel additionally reproduces the Tell Brak positive control and
is externally validated against the Menze-Ur catalog (4/4 confident hits).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def robust_z(a: np.ndarray) -> np.ndarray:
    """Median/MAD z-score (outlier-resistant)."""
    med = np.nanmedian(a)
    mad = 1.4826 * np.nanmedian(np.abs(a - med)) + 1e-9
    return (a - med) / mad


def prominence(dem: np.ndarray, background_sigma_px: float = 15) -> np.ndarray:
    """Topographic prominence: DEM minus large-scale surface. Positive bumps
    = mounds. The single most reliable free-data tell signal for LARGE mounds."""
    filled = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    return dem - gaussian_filter(filled, background_sigma_px)


def regional_roughness(dem: np.ndarray, window_px: int = 15) -> float:
    """Scalar regional-flatness decision (median local relief, meters).

    Real tells sit on flat alluvial plains. If this exceeds ~8 m the AOI is
    dunes/hills/mountains and mound detection mass-false-positives — suppress
    wholesale. MUST be a scalar: a per-pixel gate lets an isolated mound make
    its own neighborhood 'rough' and gate itself out (bug found the hard way —
    it silently deleted Tell Brak)."""
    var = uniform_filter(dem.astype("float64") ** 2, window_px) - \
        uniform_filter(dem.astype("float64"), window_px) ** 2
    return float(np.sqrt(np.nanmedian(np.clip(var, 0, None))))


def contrast_snr(anom: np.ndarray, grid, width: int, height: int,
                 lat: float, lon: float, r_core_px: int = 4,
                 r_ring=(10, 20)) -> float:
    """Core-vs-ring contrast in MAD units at a geographic point."""
    x0, y0, x1, y1 = grid
    c = int((lon - x0) / (x1 - x0) * width)
    r = int((y1 - lat) / (y1 - y0) * height)
    yy, xx = np.ogrid[:height, :width]
    d = np.hypot(yy - r, xx - c)
    core = anom[d <= r_core_px]
    ring = anom[(d >= r_ring[0]) & (d <= r_ring[1])]
    bg = np.nanmedian(ring)
    mad = 1.4826 * np.nanmedian(np.abs(ring - bg)) + 1e-9
    return float((np.nanmedian(core) - bg) / mad)


# ---------------------------------------------------------------- hollow-ways

def ridge_response(anom: np.ndarray, sigmas=(1.5, 2.5, 4.0)) -> np.ndarray:
    """Multi-scale bright-ridge (sato vesselness) response — linear features."""
    from skimage.filters import sato
    a = np.nan_to_num(anom, nan=0.0)
    z = np.clip(robust_z(a), -6, 12)
    return sato(z, sigmas=sigmas, black_ridges=False, mode="reflect")


def structure_orientation(img: np.ndarray, sigma: float = 3) -> np.ndarray:
    """Per-pixel line orientation (radians mod pi, IMAGE coords y-down) from
    the structure tensor. For a linear feature the dominant eigenvector is
    across the line; +pi/2 gives the along-line orientation."""
    gy, gx = np.gradient(gaussian_filter(img, 1))
    jxx = gaussian_filter(gx * gx, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    theta = 0.5 * np.arctan2(2 * jxy, (jxx - jyy))
    return theta + np.pi / 2


def radial_alignment(ridge: np.ndarray, theta: np.ndarray, grid,
                     width: int, height: int, lat_c: float, lon_c: float,
                     rmin_km: float = 1.0, rmax_km: float = 5.0,
                     top_frac: float = 0.05) -> float:
    """Do the strongest linears in an annulus point AT the center?

    Circular statistic on axial data: ridge-weighted mean of cos(2*(theta -
    bearing)); 0 under random orientation, 1 under perfect radial alignment.
    CRITICAL: theta is in image coords (y down) while geographic dy is y-up,
    so the bearing uses -dy (validated on synthetic radial patterns — the
    unflipped version scores ~0 on a perfect radial network)."""
    x0, y0, x1, y1 = grid
    lats = y1 - (np.arange(height) + 0.5) / height * (y1 - y0)
    lons = x0 + (np.arange(width) + 0.5) / width * (x1 - x0)
    lon_g, lat_g = np.meshgrid(lons, lats)
    dx = (lon_g - lon_c) * np.cos(np.radians(lat_c)) * 111.0
    dy = (lat_g - lat_c) * 111.0
    r = np.hypot(dx, dy)
    ann = (r >= rmin_km) & (r <= rmax_km) & np.isfinite(ridge)
    if ann.sum() < 100:
        return float("nan")
    thr = np.quantile(ridge[ann], 1 - top_frac)
    sel = ann & (ridge >= thr)
    bearing = np.arctan2(-dy, dx)
    dth = theta - bearing
    w = ridge[sel]
    return float(np.sum(np.cos(2 * dth[sel]) * w) / (np.sum(w) + 1e-9))
