"""Physical source inversion from surface deformation bowls.

A localized subsidence bowl is the surface expression of a subsurface volume
loss. The Mogi point-source model (Mogi 1958) relates a volume change dV at
depth d to the surface displacement field. Inverting an observed bowl for
(depth, volume-rate) is what turns "the ground is sinking" into "a source at
~D meters is losing ~V m^3/yr" — the physical discriminator between a shallow
collapsing void and deep, broad aquifer compaction.

Mogi vertical surface displacement at radial distance r from the epicenter:

    u_z(r) = C * d / (d^2 + r^2)^(3/2),   C = (1 - nu) * dV / pi

The bowl's half-width at half-maximum fixes the depth:
    r_half = d * sqrt(2^(2/3) - 1)  ->  d = r_half / 0.7664
and the peak amplitude then fixes dV. We provide a robust nonlinear fit that
estimates (d, dV, x0, y0) from a sampled bowl, plus closed-form initial
guesses. Units: meters throughout; dV in m^3 (negative for volume loss).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

POISSON_DEFAULT = 0.25
_HALF_WIDTH_FACTOR = np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)  # ~0.7664


def mogi_uz(r: np.ndarray, depth: float, dV: float, nu: float = POISSON_DEFAULT) -> np.ndarray:
    """Vertical surface displacement (m) of a Mogi source. r, depth in meters."""
    r = np.asarray(r, dtype=np.float64)
    C = (1.0 - nu) * dV / np.pi
    return C * depth / np.power(depth * depth + r * r, 1.5)


def depth_from_bowl_width(r_half_m: float) -> float:
    """Source depth (m) from the bowl's half-width at half-maximum (m)."""
    return float(r_half_m) / _HALF_WIDTH_FACTOR


def volume_from_peak(peak_uz_m: float, depth_m: float, nu: float = POISSON_DEFAULT) -> float:
    """Volume change dV (m^3) from peak vertical displacement and depth.

    At r=0, u_z = (1-nu) dV / (pi d^2)  ->  dV = peak * pi * d^2 / (1-nu).
    """
    return float(peak_uz_m) * np.pi * depth_m * depth_m / (1.0 - nu)


def _typical_spacing(x: np.ndarray, y: np.ndarray) -> float:
    """Median nearest-neighbour spacing of scattered sample points (meters)."""
    vals = []
    for arr in (np.unique(x), np.unique(y)):
        if arr.size >= 2:
            d = np.diff(np.sort(arr))
            d = d[d > 0]
            if d.size:
                vals.append(float(np.median(d)))
    return max(vals) if vals else 1.0


@dataclass
class MogiFit:
    depth_m: float = np.nan
    volume_m3: float = np.nan
    x0_m: float = np.nan
    y0_m: float = np.nan
    peak_uz_m: float = np.nan
    rmse_m: float = np.nan
    r2: float = np.nan
    n_points: int = 0
    converged: bool = False


def fit_mogi(
    x: np.ndarray,
    y: np.ndarray,
    uz: np.ndarray,
    *,
    nu: float = POISSON_DEFAULT,
    depth_bounds: Tuple[float, float] = (20.0, 5000.0),
) -> MogiFit:
    """Fit a Mogi source (depth, dV, epicenter) to a sampled vertical-displacement
    bowl. x, y, uz are 1-D arrays in meters (map coords and LOS-vertical disp).

    Uses a closed-form initial guess (peak location, bowl half-width) then a
    bounded Levenberg-Marquardt refine. Returns NaN fields if degenerate.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    uz = np.asarray(uz, np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(uz)
    x, y, uz = x[m], y[m], uz[m]
    n = x.size
    if n < 8:
        return MogiFit(n_points=n)

    # Epicenter guess: median location of the strongest decile of |uz|.
    # A single-pixel peak (unwrapping spike, corner reflector) would otherwise
    # lock the initial epicenter and skew the whole radius field; the top-|uz|
    # median centroid is robust to isolated outliers while still centering on
    # a genuine bowl.
    sign = -1.0 if np.nanmin(uz) < -abs(np.nanmax(uz)) else 1.0
    strong = np.abs(uz) >= np.nanpercentile(np.abs(uz), 90)
    if strong.sum() >= 3:
        x0 = float(np.median(x[strong]))
        y0 = float(np.median(y[strong]))
    else:
        peak_idx = int(np.argmin(uz) if sign < 0 else np.argmax(uz))
        x0, y0 = float(x[peak_idx]), float(y[peak_idx])
    # Peak amplitude estimate, robust to lone spikes: extreme of the strong set.
    peak = float(np.nanmin(uz[strong]) if sign < 0 else np.nanmax(uz[strong])) \
        if strong.any() else float(np.nanmin(uz) if sign < 0 else np.nanmax(uz))

    # Half-width guess from radial profile.
    r = np.hypot(x - x0, y - y0)
    half_level = peak / 2.0
    if sign < 0:
        within = r[uz <= half_level]
    else:
        within = r[uz >= half_level]
    r_half = float(np.nanmax(within)) if within.size else float(np.nanmedian(r))
    # Floor r_half at the data's own sampling scale (median nearest-neighbour
    # spacing), NOT the window extent — a window-sized floor would inflate the
    # initial depth for compact features inside broad AOIs.
    spacing = _typical_spacing(x, y)
    r_half = max(r_half, spacing)
    d0 = np.clip(depth_from_bowl_width(r_half), *depth_bounds)
    dV0 = volume_from_peak(peak, d0, nu)

    try:
        from scipy.optimize import least_squares
    except Exception:
        # Closed-form only.
        pred = mogi_uz(r, d0, dV0, nu)
        resid = uz - pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        return MogiFit(depth_m=d0, volume_m3=dV0, x0_m=x0, y0_m=y0,
                       peak_uz_m=peak, rmse_m=rmse, n_points=n, converged=False)

    span = (x.max() - x.min()) + (y.max() - y.min())

    def residuals(p):
        d, dV, cx, cy = p
        rr = np.hypot(x - cx, y - cy)
        return mogi_uz(rr, d, dV, nu) - uz

    p0 = [d0, dV0, x0, y0]
    lb = [depth_bounds[0], -np.inf, x0 - span, y0 - span]
    ub = [depth_bounds[1], np.inf, x0 + span, y0 + span]
    try:
        sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf", max_nfev=2000)
    except Exception:
        return MogiFit(depth_m=d0, volume_m3=dV0, x0_m=x0, y0_m=y0,
                       peak_uz_m=peak, n_points=n, converged=False)

    d, dV, cx, cy = sol.x
    pred = mogi_uz(np.hypot(x - cx, y - cy), d, dV, nu)
    resid = uz - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((uz - uz.mean()) ** 2)) or np.nan
    return MogiFit(
        depth_m=float(d),
        volume_m3=float(dV),
        x0_m=float(cx),
        y0_m=float(cy),
        peak_uz_m=float(mogi_uz(0.0, d, dV, nu)),
        rmse_m=float(np.sqrt(ss_res / n)),
        r2=float(1.0 - ss_res / ss_tot) if np.isfinite(ss_tot) else np.nan,
        n_points=n,
        converged=bool(sol.success),
    )
