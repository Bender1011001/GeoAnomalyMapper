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
    source_volume_rate_m3_yr: Optional[float]
    source_fit_r2: Optional[float]
    time_to_threshold_yr: Optional[float]
    is_localized: bool
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
    seasonal_dominated_ratio: float = 1.5,
    accel_flag_cm_yr2: float = 0.5,
    forecast_threshold_cm: float = 30.0,
    context_samplers: Optional[Dict[str, Callable[[float, float], float]]] = None,
    pixel_size_m: float = 30.0,
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

    med = np.nanmedian(vel)
    vel_rel = vel - med
    sigma = 1.4826 * np.nanmedian(np.abs(vel_rel[np.isfinite(vel_rel)]))
    sigma = float(sigma) if sigma > 1e-9 else 1e-9

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

            void_likelihood = _void_likelihood(
                kind, classification, is_localized, good_mogi, mogi.depth_m, seasonal_amp, peak_v,
            )

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
                source_volume_rate_m3_yr=None if vol_rate is None else round(vol_rate, 0),
                source_fit_r2=None if not np.isfinite(mogi.r2) else round(mogi.r2, 2),
                time_to_threshold_yr=None if not np.isfinite(ttt) else round(float(ttt), 1),
                is_localized=is_localized,
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


def _void_likelihood(kind, classification, is_localized, good_mogi, depth_m,
                     seasonal_amp_m, peak_v_m_yr) -> float:
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
