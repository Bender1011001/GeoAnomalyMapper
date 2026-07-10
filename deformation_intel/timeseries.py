"""Per-pixel displacement time-series analytics.

All functions operate on a displacement cube ``disp`` of shape (T, H, W) in
meters (line-of-sight, negative = motion away from radar / subsidence) sampled
at decimal-year times ``t`` of shape (T,). They are vectorized over pixels and
NaN-aware. No I/O — so they are cheap to unit-test against synthetic signals.

Outputs are physical and interpretable:
- velocity_m_yr: robust linear rate (Theil-Sen-like via least squares on
  a decorrelated design; sign convention preserved).
- accel_m_yr2: quadratic curvature (change in rate per year). Strongly negative
  = accelerating subsidence — the collapse-precursor signal.
- seasonal_amp_m: amplitude of the fitted annual sinusoid (groundwater breathing).
- residual_rmse_m: fit quality after removing trend + accel + seasonal.
- r2: coefficient of determination of the full model.
- breakpoint_year / pre_rate / post_rate: single-knot regime change via a grid
  search that maximizes explained variance; only reported when it beats the
  linear model by a margin.
- time_to_threshold_yr: extrapolated years until cumulative displacement crosses
  a caller-supplied threshold, using the most recent rate (linear) or the
  accelerating quadratic — whichever the model selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def _valid_mask(disp: np.ndarray) -> np.ndarray:
    return np.isfinite(disp)


def stitch_reference_eras(
    disp: np.ndarray,
    sec_dates: Sequence,
    ref_dates: Sequence,
) -> np.ndarray:
    """Stitch OPERA mini-stack eras into one continuous cumulative series.

    OPERA DISP-S1 resets displacement to ~0 at each reference epoch. Each
    granule's ``displacement`` is relative to its era reference. To build a
    continuous per-pixel cumulative series we add, to each era, the cumulative
    displacement carried over from the prior era at the boundary epoch.

    disp: (T,H,W) per-epoch displacement (meters), each relative to its era ref.
    sec_dates: length-T secondary acquisition dates (sortable, e.g. 'YYYYMMDD').
    ref_dates: length-T era reference dates aligned with each epoch.

    Returns a (T,H,W) cube of continuous cumulative displacement, sorted by
    secondary date. Boundary offset uses the prior era's value at the epoch
    nearest the new era's reference date (exact when the boundary epoch is
    shared, which is the common OPERA overlap case).
    """
    disp = np.asarray(disp, dtype=np.float64)
    T = disp.shape[0]
    if len(sec_dates) != T or len(ref_dates) != T:
        raise ValueError("sec_dates/ref_dates length must match disp T")
    order = sorted(range(T), key=lambda i: str(sec_dates[i]))
    disp = disp[order]
    sec = [str(sec_dates[i]) for i in order]
    ref = [str(ref_dates[i]) for i in order]

    # Group consecutive epochs into eras keyed by reference date, preserving
    # chronological era order (a reference date defines one contiguous era).
    era_order: List[str] = []
    for r in ref:
        if not era_order or era_order[-1] != r:
            era_order.append(r)

    out = np.empty_like(disp)
    running_offset = np.zeros(disp.shape[1:], dtype=np.float64)
    prev_last_idx: Optional[int] = None
    prev_out_by_date: dict = {}

    for era in era_order:
        idxs = [i for i in range(T) if ref[i] == era]
        # offset to make this era continuous with the previous stitched era.
        if prev_last_idx is not None:
            # nearest prior-era stitched epoch to this era's reference date.
            if era in prev_out_by_date:
                offset = prev_out_by_date[era]
            else:
                # Nearest by TRUE calendar distance. Naive int(YYYYMMDD)
                # subtraction is non-linear across month/year boundaries
                # (|20200101-20191231| = 8870 for a 1-day gap) and can pick
                # the wrong bridge epoch when the exact boundary is missing
                # (e.g. under epoch subsampling).
                from datetime import datetime as _dt
                era_dt = _dt.strptime(era, "%Y%m%d")
                nearest = min(
                    prev_out_by_date,
                    key=lambda d: abs((_dt.strptime(d, "%Y%m%d") - era_dt).days),
                )
                offset = prev_out_by_date[nearest]
            running_offset = offset
        for i in idxs:
            out[i] = disp[i] + running_offset
        prev_out_by_date = {sec[i]: out[i] for i in idxs}
        prev_last_idx = idxs[-1]
    return out


def design_matrix(t: np.ndarray, *, seasonal: bool, quadratic: bool) -> np.ndarray:
    """Build a design matrix for [intercept, slope, (quad), (sin, cos)]."""
    t = np.asarray(t, dtype=np.float64)
    tc = t - t.mean()
    cols = [np.ones_like(tc), tc]
    if quadratic:
        cols.append(tc * tc)
    if seasonal:
        cols.append(np.sin(2 * np.pi * t))
        cols.append(np.cos(2 * np.pi * t))
    return np.column_stack(cols)


@dataclass
class PixelFit:
    velocity_m_yr: float = np.nan
    accel_m_yr2: float = np.nan
    seasonal_amp_m: float = np.nan
    residual_rmse_m: float = np.nan
    r2: float = np.nan
    n_obs: int = 0
    breakpoint_year: float = np.nan
    pre_rate_m_yr: float = np.nan
    post_rate_m_yr: float = np.nan
    breakpoint_gain: float = 0.0


def fit_pixel(
    t: np.ndarray,
    y: np.ndarray,
    *,
    seasonal: bool = True,
    min_obs: int = 12,
) -> PixelFit:
    """Fit trend+acceleration+seasonal to a single pixel series.

    Robust to NaNs. Returns a PixelFit; fields are NaN if under-determined.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(t)
    n = int(m.sum())
    if n < min_obs:
        return PixelFit(n_obs=n)
    tt, yy = t[m], y[m]

    A = design_matrix(tt, seasonal=seasonal, quadratic=True)
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    fitted = A @ coef
    resid = yy - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2)) or np.nan
    r2 = 1.0 - ss_res / ss_tot if np.isfinite(ss_tot) else np.nan
    rmse = float(np.sqrt(ss_res / n))

    # coef layout: [intercept, slope, quad, (sin, cos)]
    slope = float(coef[1])
    quad = float(coef[2])
    seasonal_amp = np.nan
    if seasonal:
        s, c = float(coef[3]), float(coef[4])
        seasonal_amp = float(np.hypot(s, c))

    # velocity = robust MEAN rate over the record (the linear slope). Reporting
    # the quadratic value at the last epoch amplifies end-point noise and, on
    # sparse/noisy series, produces spurious extreme rates. Acceleration (the
    # quadratic curvature) separately captures "getting worse over time".
    velocity_mean = slope
    accel = 2.0 * quad  # m/yr^2

    fit = PixelFit(
        velocity_m_yr=velocity_mean,
        accel_m_yr2=accel,
        seasonal_amp_m=seasonal_amp,
        residual_rmse_m=rmse,
        r2=r2,
        n_obs=n,
    )

    # Single-knot regime change (piecewise-linear) — only if it clearly helps.
    bp = _best_breakpoint(tt, yy, base_ss_res=_linear_ss_res(tt, yy), ss_tot=ss_tot)
    if bp is not None:
        fit.breakpoint_year = bp["year"]
        fit.pre_rate_m_yr = bp["pre_rate"]
        fit.post_rate_m_yr = bp["post_rate"]
        fit.breakpoint_gain = bp["gain"]
    return fit


def _linear_ss_res(t: np.ndarray, y: np.ndarray) -> float:
    A = np.column_stack([np.ones_like(t), t - t.mean()])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sum((y - A @ coef) ** 2))


def _best_breakpoint(t: np.ndarray, y: np.ndarray, base_ss_res: float, ss_tot: float,
                     min_seg: int = 6, gain_threshold: float = 0.25,
                     min_unexplained_frac: float = 0.02) -> Optional[dict]:
    """Grid-search a continuous piecewise-linear knot; report only if it clearly
    helps. Guards against spurious knots on near-perfect linear fits by first
    requiring the linear model to leave meaningful unexplained variance."""
    order = np.argsort(t)
    t, y = t[order], y[order]
    n = len(t)
    if n < 2 * min_seg:
        return None
    # If the linear model already explains essentially everything, there is no
    # regime change to find (dividing tiny residuals would fabricate a knot).
    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return None
    if base_ss_res / ss_tot < min_unexplained_frac:
        return None
    best = None
    tmean = t.mean()
    for k in range(min_seg, n - min_seg):
        knot = t[k]
        # continuous piecewise-linear basis: [1, (t-tmean), max(0, t-knot)]
        hinge = np.maximum(0.0, t - knot)
        A = np.column_stack([np.ones_like(t), t - tmean, hinge])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = float(np.sum((y - A @ coef) ** 2))
        if best is None or ss < best[0]:
            pre = float(coef[1])
            post = float(coef[1] + coef[2])
            best = (ss, knot, pre, post)
    if best is None or base_ss_res <= 0:
        return None
    gain = (base_ss_res - best[0]) / base_ss_res
    if gain < gain_threshold:
        return None
    return {"year": float(best[1]), "pre_rate": best[2], "post_rate": best[3], "gain": float(gain)}


def fit_cube(
    t: np.ndarray,
    disp: np.ndarray,
    *,
    seasonal: bool = True,
    min_obs: int = 12,
) -> dict:
    """Vectorized trend+accel+seasonal fit over a (T,H,W) cube.

    Breakpoint/regime-change search is deliberately NOT part of the cube pass
    (it is per-pixel and slow); run `fit_pixel` on flagged candidate series —
    detect.py does exactly this for each clustered anomaly.
    """
    t = np.asarray(t, dtype=np.float64)
    T, H, W = disp.shape
    Y = disp.reshape(T, -1).astype(np.float64)
    valid = np.isfinite(Y)
    nobs = valid.sum(axis=0)
    keep = nobs >= min_obs

    A_full = design_matrix(t, seasonal=seasonal, quadratic=True)
    ncoef = A_full.shape[1]
    coefs = np.full((ncoef, Y.shape[1]), np.nan)
    rmse = np.full(Y.shape[1], np.nan)
    r2 = np.full(Y.shape[1], np.nan)

    # Group columns by identical validity pattern is overkill; instead solve
    # per-pixel only where the NaN pattern differs. Fast path: no NaNs.
    if valid.all():
        coef, *_ = np.linalg.lstsq(A_full, Y, rcond=None)
        coefs = coef
        resid = Y - A_full @ coef
        rmse = np.sqrt(np.mean(resid**2, axis=0))
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = 1.0 - np.sum(resid**2, axis=0) / ss_tot
    else:
        idxs = np.where(keep)[0]
        for j in idxs:
            col = Y[:, j]
            mm = np.isfinite(col)
            A = A_full[mm]
            yv = col[mm]
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            coefs[:, j] = coef
            res = yv - A @ coef
            rmse[j] = np.sqrt(np.mean(res**2))
            sst = np.sum((yv - yv.mean()) ** 2)
            r2[j] = 1.0 - np.sum(res**2) / sst if sst > 0 else np.nan

    slope = coefs[1]
    quad = coefs[2]
    velocity_mean = slope  # robust mean rate over the record (see fit_pixel)
    accel = 2.0 * quad
    if seasonal:
        seasonal_amp = np.hypot(coefs[3], coefs[4])
    else:
        seasonal_amp = np.full(Y.shape[1], np.nan)

    def rs(a):
        return a.reshape(H, W)

    return {
        "velocity_m_yr": rs(velocity_mean),
        "accel_m_yr2": rs(accel),
        "seasonal_amp_m": rs(seasonal_amp),
        "residual_rmse_m": rs(rmse),
        "r2": rs(r2),
        "n_obs": rs(nobs.astype(np.float64)),
        "mean_slope_m_yr": rs(slope),
    }


def time_to_threshold(
    last_cumulative_m: float,
    velocity_m_yr: float,
    accel_m_yr2: float,
    threshold_m: float,
) -> float:
    """Years until cumulative |displacement| reaches threshold_m, extrapolating
    the local quadratic. Returns np.inf if never (moving away from threshold),
    np.nan if inputs non-finite. Sign convention: threshold and displacement are
    compared on the same signed axis (use negative threshold for subsidence)."""
    if not all(np.isfinite(x) for x in (last_cumulative_m, velocity_m_yr, accel_m_yr2, threshold_m)):
        return np.nan
    remaining = threshold_m - last_cumulative_m
    # Solve 0.5*a*x^2 + v*x - remaining = 0 for smallest positive x.
    a = 0.5 * accel_m_yr2
    b = velocity_m_yr
    c = -remaining
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return np.inf
        x = c / -b  # -c/b
        return x if x > 0 else np.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    sq = np.sqrt(disc)
    roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    pos = [r for r in roots if r > 1e-9]
    return float(min(pos)) if pos else np.inf
