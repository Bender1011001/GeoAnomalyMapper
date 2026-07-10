"""Unified deformation-anomaly detector.

Turns a displacement cube into a ranked list of classified, explainable
anomalies by layering the channels:

  motion   -> per-pixel velocity + acceleration + seasonal (timeseries.py)
  cluster  -> connected subsidence/uplift bowls above a robust threshold
  physical -> Mogi source inversion per bowl (depth + volume-rate) (sources.py)
  temporal -> regime-change / breakpoint on the bowl's mean series
  classify -> label {accelerating_subsidence, steady_subsidence,
              seasonal_dominated, uplift, ambiguous} with a confidence and a
              plain-language "why"
  forecast -> time-to-threshold from the local velocity+acceleration

Context/rejection layers (groundwater, lithology, land-use) attach as optional
per-anomaly annotations via `context_samplers` — callables (lat, lon) -> value
— so the detector stays dependency-light and testable while remaining the one
place fusion happens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .sources import fit_mogi
from .timeseries import fit_cube, fit_pixel, time_to_threshold


@dataclass
class Anomaly:
    rank: int
    lat: float
    lon: float
    kind: str                      # "subsidence" | "uplift"
    classification: str
    confidence: float
    peak_velocity_cm_yr: float
    mean_velocity_cm_yr: float
    accel_cm_yr2: float
    seasonal_amp_cm: float
    area_km2: float
    n_pixels: int
    sigma: float
    cumulative_cm: float
    breakpoint_year: Optional[float]
    pre_rate_cm_yr: Optional[float]
    post_rate_cm_yr: Optional[float]
    source_depth_m: Optional[float]
    source_depth_range_m: Optional[List[float]]
    source_volume_rate_m3_yr: Optional[float]
    source_fit_r2: Optional[float]
    source_growth: Optional[str]
    time_to_threshold_yr: Optional[float]
    is_localized: bool
    regional_correlation: Optional[float]
    rate_reliable: bool
    void_likelihood: float
    why: str
    context: Dict[str, float] = field(default_factory=dict)


def _label(mask: np.ndarray):
    from scipy.ndimage import label
    return label(mask)


def detect_anomalies(
    cube: dict,
    *,
    velocity_threshold_cm_yr: float = 3.0,
    min_pixels: int = 6,
    min_sigma: float = 4.0,
    adaptive_threshold: bool = True,
    threshold_floor_cm_yr: float = 0.8,
    seasonal_dominated_ratio: float = 1.5,
    accel_flag_cm_yr2: float = 0.5,
    forecast_threshold_cm: float = 30.0,
    context_samplers: Optional[Dict[str, Callable[[float, float], float]]] = None,
    pixel_size_m: float = 30.0,
    min_valid_fraction: float = 0.5,
) -> List[Anomaly]:
    """Detect and classify deformation anomalies in an OPERA/HyP3 cube dict.

    cube must provide: 'cube' (T,H,W) meters, 't' (decimal yr), 'x','y' (meters,
    projected grid), 'crs_wkt'. Velocity is referenced to the AOI robust median
    so a broad regional signal doesn't swamp local bowls.
    """
    disp = np.asarray(cube["cube"], dtype=np.float64)
    t = np.asarray(cube["t"], dtype=np.float64)
    x = np.asarray(cube["x"], dtype=np.float64)
    y = np.asarray(cube["y"], dtype=np.float64)

    fields = fit_cube(t, disp, seasonal=True, min_obs=max(12, len(t) // 3))
    vel = fields["velocity_m_yr"]
    acc = fields["accel_m_yr2"]
    seas = fields["seasonal_amp_m"]

    # Under-observed pixels produce artifact velocities: a pixel masked in
    # most eras has its valid epochs clumped in time, and a seasonal+trend fit
    # on that clump yields slopes several x the true rate (measured on the
    # full-archive Hutchinson cube: gappy pixels fit ~3 cm/yr where
    # well-observed neighbors fit ~0.8). Require coverage across the record,
    # not just a raw observation count.
    n_valid = np.isfinite(disp).sum(axis=0)
    vel = np.where(n_valid >= min_valid_fraction * disp.shape[0], vel, np.nan)

    # AOI reference series: regional water-table "breathing", measured on QUIET
    # ground only. Using the plain AOI median would let a large anomaly leak its
    # own signal into the reference and self-correlate. Aquifer pumping cones
    # co-move with this reference; an independent void collapse does not.
    T = disp.shape[0]
    med0 = np.nanmedian(vel[np.isfinite(vel)]) if np.isfinite(vel).any() else 0.0
    quiet = np.isfinite(vel) & (np.abs(vel - med0) < 0.5 * (velocity_threshold_cm_yr / 100.0))
    if quiet.sum() >= 200:
        ref_series = np.nanmedian(disp[:, quiet], axis=1)
    else:
        ref_series = np.nanmedian(disp.reshape(T, -1), axis=1)
    ref_resid = _detrended(t, ref_series)
    # If the quiet ground barely moves (no regional breathing), correlation
    # against it is noise — disable the aquifer fingerprint in that case.
    ref_ok = np.isfinite(ref_resid).sum() >= 8 and np.nanstd(ref_resid) >= 0.002

    med = np.nanmedian(vel)
    vel_rel = vel - med
    sigma = 1.4826 * np.nanmedian(np.abs(vel_rel[np.isfinite(vel_rel)]))
    sigma = float(sigma) if sigma > 1e-9 else 1e-9

    if adaptive_threshold:
        # Noise-calibrated threshold: in quiet terrain (e.g. OPERA desert
        # frames with sigma ~0.2 cm/yr) a fixed 2-3 cm/yr floor is >10 sigma
        # and blinds us to small weak features; in noisy agricultural frames
        # it under-thresholds. Scale with the measured field noise, bounded
        # below by an absolute floor against atmospheric residue.
        thr = max(threshold_floor_cm_yr / 100.0,
                  min(velocity_threshold_cm_yr / 100.0, min_sigma * sigma))
    else:
        thr = velocity_threshold_cm_yr / 100.0
    from pyproj import Transformer
    inv = Transformer.from_crs(cube["crs_wkt"], "EPSG:4326", always_xy=True)
    px_km2 = (pixel_size_m ** 2) / 1e6

    anomalies: List[Anomaly] = []
    for kind, mask in (("subsidence", (vel_rel <= -thr)),
                       ("uplift", (vel_rel >= thr))):
        mask = mask & np.isfinite(vel_rel)
        labeled, n = _label(mask)
        for k in range(1, n + 1):
            pmask = labeled == k
            npx = int(pmask.sum())
            if npx < min_pixels:
                continue
            rows, cols = np.where(pmask)
            vals = vel_rel[rows, cols]
            pi = int(np.argmax(np.abs(vals)))
            pr, pc = int(rows[pi]), int(cols[pi])
            pk_sigma = abs(float(vals[pi])) / sigma
            if pk_sigma < min_sigma:
                continue

            # bowl mean series (cluster-averaged for stability) + temporal fit
            series = np.nanmean(disp[:, rows, cols], axis=1)
            pfit = fit_pixel(t, series, seasonal=True, min_obs=max(12, len(t) // 3))
            finite = series[np.isfinite(series)]
            cumulative = float(finite[-1] - finite[0]) if finite.size > 2 else np.nan

            # Regional co-movement (aquifer fingerprint): correlation of the
            # bowl's excess motion with the quiet-ground reference residual.
            regional_r = _regional_correlation(t, series, ref_series, ref_resid) if ref_ok else None

            # physical source inversion (local meters relative to peak)
            xr_ = x[cols] - x[pc]
            yr_ = y[rows] - y[pr]
            mogi = fit_mogi(xr_, yr_, vals)  # vals are LOS ~ vertical here
            # volume RATE: convert bowl volume change to per-year via velocity/uz scale
            vol_rate = None
            if np.isfinite(mogi.volume_m3) and np.isfinite(mogi.peak_uz_m) and abs(mogi.peak_uz_m) > 1e-6:
                # mogi fit was on velocity field (m/yr), so volume_m3 is already m^3/yr
                vol_rate = float(mogi.volume_m3)

            lon_p, lat_p = inv.transform(float(x[pc]), float(y[pr]))

            seasonal_amp = float(np.nanmean(seas[rows, cols])) if np.isfinite(seas[rows, cols]).any() else np.nan
            peak_v = float(vals[pi])
            mean_v = float(np.nanmean(vals))
            accel = float(acc[pr, pc])

            classification, why, conf = _classify(
                kind, peak_v, accel, seasonal_amp, pfit,
                seasonal_dominated_ratio, accel_flag_cm_yr2, pk_sigma,
            )

            # Localized (point-source, possible void) vs regional (aquifer/
            # tectonic sheet). A real void gives a clean, compact Mogi bowl; a
            # broad compaction field makes the Mogi fit fail and rail its depth.
            area_km2 = npx * px_km2
            depth_railed = (not np.isfinite(mogi.depth_m)) or mogi.depth_m <= 25.0 or mogi.depth_m >= 4500.0
            good_mogi = (mogi.r2 is not None) and np.isfinite(mogi.r2) and mogi.r2 > 0.4
            is_localized = bool(good_mogi and (not depth_railed) and area_km2 < 3.0)

            # Broad, non-localized subsidence is regional -> not a void.
            if kind == "subsidence" and classification != "seasonal_dominated" and not is_localized:
                classification = "regional_subsidence"
                why = (f"broad subsidence {peak_v*100:.1f} cm/yr over {area_km2:.2f} km^2; "
                       f"Mogi point-source fit poor (r2={mogi.r2 if mogi.r2 is not None else float('nan'):.2f}, "
                       f"depth {'railed' if depth_railed else 'ok'}) — regional aquifer/tectonic, not a void")
                conf = min(conf, 0.6)

            # Self-consistency: the claimed peak rate must roughly agree with
            # the bowl's observed cumulative displacement over the record.
            # Wild disagreement (or opposite sign) marks a fit artifact
            # (sparse-era stitching drift, gappy-pixel overfit) — measured on
            # the 36-epoch national cubes where this exact failure appeared.
            rate_reliable = True
            span_yr = float(t.max() - t.min()) if t.size > 1 else np.nan
            if np.isfinite(cumulative) and np.isfinite(span_yr) and span_yr > 1:
                cum_rate = cumulative / span_yr
                if abs(peak_v) > 0.005:
                    same_sign = np.sign(cum_rate) == np.sign(peak_v)
                    ratio = abs(cum_rate) / abs(peak_v)
                    rate_reliable = bool(same_sign and 0.1 <= ratio)
            void_likelihood = _void_likelihood(
                kind, classification, is_localized, good_mogi, mogi.depth_m, seasonal_amp, peak_v,
                regional_r=regional_r,
            )
            if not rate_reliable:
                void_likelihood = round(void_likelihood * 0.3, 2)
                why += " [RATE UNRELIABLE: claimed rate inconsistent with observed cumulative displacement]"

            # Deeper characterization only for the candidates that matter
            # (bootstrap + growth history are ~100 extra fits per bowl).
            depth_range = None
            growth = None
            if is_localized and void_likelihood >= 0.5:
                from .sources import fit_mogi_bootstrap
                boot = fit_mogi_bootstrap(xr_, yr_, vals, n_boot=30)
                if np.isfinite(boot["depth_lo_m"]):
                    depth_range = [round(boot["depth_lo_m"], 0), round(boot["depth_hi_m"], 0)]
                growth = _source_growth(t, disp, rows, cols, xr_, yr_)

            last_cum = float(np.nanmean(series[-3:])) if finite.size >= 3 else np.nan
            target = last_cum - forecast_threshold_cm / 100.0 if kind == "subsidence" else last_cum + forecast_threshold_cm / 100.0
            ttt = time_to_threshold(last_cum, pfit.velocity_m_yr, pfit.accel_m_yr2, target)

            ctx: Dict[str, float] = {}
            if context_samplers:
                for name, fn in context_samplers.items():
                    try:
                        ctx[name] = float(fn(lat_p, lon_p))
                    except Exception:
                        ctx[name] = float("nan")

            anomalies.append(Anomaly(
                rank=0, lat=round(lat_p, 5), lon=round(lon_p, 5), kind=kind,
                classification=classification, confidence=round(conf, 2),
                peak_velocity_cm_yr=round(peak_v * 100, 1),
                mean_velocity_cm_yr=round(mean_v * 100, 1),
                accel_cm_yr2=round(accel * 100, 2),
                seasonal_amp_cm=round(seasonal_amp * 100, 1) if np.isfinite(seasonal_amp) else float("nan"),
                area_km2=round(npx * px_km2, 3), n_pixels=npx, sigma=round(pk_sigma, 1),
                cumulative_cm=round(cumulative * 100, 1) if np.isfinite(cumulative) else float("nan"),
                breakpoint_year=None if not np.isfinite(pfit.breakpoint_year) else round(pfit.breakpoint_year, 1),
                pre_rate_cm_yr=None if not np.isfinite(pfit.pre_rate_m_yr) else round(pfit.pre_rate_m_yr * 100, 1),
                post_rate_cm_yr=None if not np.isfinite(pfit.post_rate_m_yr) else round(pfit.post_rate_m_yr * 100, 1),
                source_depth_m=None if not np.isfinite(mogi.depth_m) else round(mogi.depth_m, 0),
                source_depth_range_m=depth_range,
                source_volume_rate_m3_yr=None if vol_rate is None else round(vol_rate, 0),
                source_fit_r2=None if not np.isfinite(mogi.r2) else round(mogi.r2, 2),
                source_growth=growth,
                time_to_threshold_yr=None if not np.isfinite(ttt) else round(float(ttt), 1),
                is_localized=is_localized,
                regional_correlation=None if regional_r is None else round(regional_r, 2),
                rate_reliable=rate_reliable,
                void_likelihood=round(void_likelihood, 2),
                why=why, context=ctx,
            ))

    # rank: void-likelihood first (localized, void-plausible), then signal.
    def _priority(a: Anomaly):
        return -(2.0 * a.void_likelihood + a.sigma * abs(a.peak_velocity_cm_yr) / 20.0)

    anomalies.sort(key=_priority)
    for i, a in enumerate(anomalies, 1):
        a.rank = i
    return anomalies


def _detrended(t: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Residual after removing a linear trend (NaN-aware); NaN where invalid."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    m = np.isfinite(series)
    if m.sum() < 6:
        return out
    A = np.column_stack([np.ones(m.sum()), t[m] - t[m].mean()])
    coef, *_ = np.linalg.lstsq(A, series[m], rcond=None)
    out[m] = series[m] - A @ coef
    return out


def _regional_correlation(t: np.ndarray, series: np.ndarray,
                          ref_series: np.ndarray, ref_resid: np.ndarray):
    """Aquifer fingerprint: Pearson r between the bowl's EXCESS motion
    (series minus quiet-ground reference — common-mode removed, detrended)
    and the regional residual. All ground shares the regional breathing, so
    correlating raw series would tag everything; only a pumping cone's excess
    still tracks the water table."""
    excess = series - ref_series
    e_resid = _detrended(t, excess)
    m = np.isfinite(e_resid) & np.isfinite(ref_resid)
    if m.sum() < 8:
        return None
    a, b = e_resid[m], ref_resid[m]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _source_growth(t, disp, rows, cols, xr_, yr_):
    """Compare Mogi volume-rate for the early vs late half of the record.

    Answers 'is the underlying source growing?' — a growing cavity shows a
    larger volume-loss rate in the late half.
    """
    from .sources import fit_mogi
    mid = np.median(t)
    early, late = t <= mid, t > mid
    if early.sum() < 8 or late.sum() < 8:
        return None
    rates = []
    for m in (early, late):
        sub = disp[m][:, rows, cols]
        # per-pixel linear rate over the half-record
        tt = t[m]
        A = np.column_stack([np.ones(tt.size), tt - tt.mean()])
        vals = np.full(sub.shape[1], np.nan)
        for j in range(sub.shape[1]):
            col = sub[:, j]
            mm = np.isfinite(col)
            if mm.sum() >= 6:
                coef, *_ = np.linalg.lstsq(A[mm], col[mm], rcond=None)
                vals[j] = coef[1]
        fit = fit_mogi(xr_, yr_, vals)
        rates.append(fit.volume_m3 if np.isfinite(fit.volume_m3) else None)
    if rates[0] is None or rates[1] is None:
        return None
    early_rate, late_rate = abs(rates[0]), abs(rates[1])
    if late_rate > 1.5 * early_rate:
        return "growing"
    if late_rate < 0.67 * early_rate:
        return "slowing"
    return "steady"


def _void_likelihood(kind, classification, is_localized, good_mogi, depth_m,
                     seasonal_amp_m, peak_v_m_yr, regional_r=None) -> float:
    """0..1 heuristic that a subsidence anomaly reflects a subsurface VOID
    (localized collapse) rather than aquifer/tectonic/seasonal processes."""
    if kind == "uplift" or classification in ("seasonal_dominated", "regional_subsidence"):
        return 0.05 if classification == "regional_subsidence" else 0.1
    score = 0.0
    if is_localized:
        score += 0.4
    if classification == "accelerating_subsidence":
        score += 0.3
    if good_mogi and np.isfinite(depth_m) and 30.0 <= depth_m <= 800.0:
        score += 0.2   # void-plausible source depth with a real point-source fit
    if abs(peak_v_m_yr) >= 0.03:
        score += 0.1
    # Aquifer fingerprint: a bowl that co-breathes with the regional signal is
    # very likely a pumping cone, not an independent collapse.
    if regional_r is not None:
        if regional_r >= 0.85:
            score *= 0.3
        elif regional_r >= 0.7:
            score *= 0.6
    return float(min(score, 1.0))


def _classify(kind, peak_v_m_yr, accel_m_yr2, seasonal_amp_m, pfit,
              seasonal_ratio, accel_flag_cm_yr2, pk_sigma):
    """Return (classification, why, confidence in [0,1]).

    accel_flag_cm_yr2 is a threshold already in cm/yr^2.
    """
    peak_cm = peak_v_m_yr * 100
    accel_cm = accel_m_yr2 * 100
    seas_cm = (seasonal_amp_m or 0) * 100
    net_cm = abs(peak_cm)

    # Seasonal dominance: annual swing dwarfs the net trend -> likely groundwater.
    if np.isfinite(seas_cm) and net_cm > 0 and seas_cm > seasonal_ratio * net_cm:
        return ("seasonal_dominated",
                f"annual swing {seas_cm:.1f} cm exceeds {seasonal_ratio:g}x the net "
                f"{peak_cm:+.1f} cm/yr trend — likely groundwater/aquifer, not a void",
                min(0.6, 0.3 + pk_sigma / 20))

    if kind == "uplift":
        return ("uplift", f"surface rising {peak_cm:+.1f} cm/yr (injection/rebound/heave)",
                min(0.7, 0.3 + pk_sigma / 20))

    # subsidence branch (acceleration threshold already in cm/yr^2)
    accelerating = accel_cm <= -accel_flag_cm_yr2 or (
        pfit.breakpoint_gain > 0.25 and np.isfinite(pfit.post_rate_m_yr)
        and pfit.post_rate_m_yr < pfit.pre_rate_m_yr - 0.02
    )
    if accelerating:
        why = f"subsiding {peak_cm:.1f} cm/yr and ACCELERATING ({accel_cm:+.1f} cm/yr^2)"
        if pfit.breakpoint_gain > 0.25 and np.isfinite(pfit.breakpoint_year):
            why += (f"; regime change ~{pfit.breakpoint_year:.0f} "
                    f"({pfit.pre_rate_m_yr*100:+.1f} -> {pfit.post_rate_m_yr*100:+.1f} cm/yr)")
        why += " — collapse-precursor pattern"
        return ("accelerating_subsidence", why, min(0.9, 0.5 + pk_sigma / 15))

    return ("steady_subsidence",
            f"steady subsidence {peak_cm:.1f} cm/yr, no significant acceleration",
            min(0.75, 0.4 + pk_sigma / 20))
