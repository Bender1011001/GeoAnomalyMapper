"""Candidate generation — Stage 1 of the funnel (cheap numeric priors).

Candidates come from MULTIPLE INDEPENDENT priors so the funnel is not blind to
static features (the existing InSAR sweep only sees motion):

    grid       unbiased spatial sampling (no signal, coverage floor)
    dem        local-relief outliers + closed depressions (Copernicus GLO-30)
    spectral   Sentinel-2 brightness / NDVI pixels that are outliers vs their
               own neighbourhood
    change     Sentinel-2 early-years vs recent-years difference — things that
               appeared or vanished are inherently interesting

Each prior fetches ONE regional raster mosaic (not one request per candidate),
so Stage 1 costs a handful of COG range-reads for an entire region. Peaks are
extracted with `features.local_outlier` — novelty relative to surroundings,
never absolute magnitude (mountains would otherwise drown everything; that
failure is documented in archaeo_intel.detect.regional_roughness).

A candidate is a plain dict: {lat, lon, priors: {name: peak_z}, score1}.
`merge_candidates` deduplicates across priors; multi-prior agreement raises
score1 but NOTHING is vetoed here — Stage 1 only decides who gets imagery.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from interesting_intel.features import local_outlier

logger = logging.getLogger(__name__)

M_PER_DEG_LAT = 111_320.0


def _bbox_shape(bbox, px_m: float, max_px: int = 4096):
    """(width, height) pixels for a lon/lat bbox at ~px_m metres per pixel."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mid = 0.5 * (lat_min + lat_max)
    w = int((lon_max - lon_min) * M_PER_DEG_LAT * math.cos(math.radians(mid)) / px_m)
    h = int((lat_max - lat_min) * M_PER_DEG_LAT / px_m)
    return max(min(w, max_px), 16), max(min(h, max_px), 16)


def rc_to_latlon(row: float, col: float, grid, width: int, height: int):
    lon_min, lat_min, lon_max, lat_max = grid
    lon = lon_min + (col + 0.5) / width * (lon_max - lon_min)
    lat = lat_max - (row + 0.5) / height * (lat_max - lat_min)
    return lat, lon


# ------------------------------------------------------------ pure functions

def grid_candidates(bbox, spacing_m: float = 500.0) -> List[dict]:
    """Unbiased grid sample — the coverage floor when no prior fires."""
    lon_min, lat_min, lon_max, lat_max = bbox
    dlat = spacing_m / M_PER_DEG_LAT
    mid = 0.5 * (lat_min + lat_max)
    dlon = dlat / max(math.cos(math.radians(mid)), 1e-6)
    out = []
    lat = lat_min + dlat / 2
    while lat < lat_max:
        lon = lon_min + dlon / 2
        while lon < lon_max:
            out.append({"lat": round(lat, 6), "lon": round(lon, 6),
                        "priors": {"grid": 0.0}, "score1": 0.0})
            lon += dlon
        lat += dlat
    return out


def anomaly_peaks(z_img: np.ndarray, grid, prior: str, min_z: float = 3.5,
                  min_sep_px: int = 6, max_peaks: int = 1500) -> List[dict]:
    """Local maxima of an outlier image -> candidate dicts.

    Peak extraction happens on the z-image of ONE prior at a time; any
    cross-prior logic (clustering, agreement) is applied later among the
    merged survivors — never among raw per-pixel detections (the
    neighbours-among-raw-hits bug class).
    """
    from scipy.ndimage import maximum_filter

    z = np.asarray(z_img, dtype="float32")
    zi = np.nan_to_num(z, nan=-np.inf)
    h, w = zi.shape
    footprint = 2 * min_sep_px + 1
    is_peak = (zi == maximum_filter(zi, size=footprint)) & (zi >= min_z)
    rows, cols = np.nonzero(is_peak)
    if len(rows) == 0:
        return []
    order = np.argsort(zi[rows, cols])[::-1][:max_peaks]
    out = []
    for i in order:
        r, c = int(rows[i]), int(cols[i])
        lat, lon = rc_to_latlon(r, c, grid, w, h)
        zval = float(zi[r, c])
        out.append({"lat": round(lat, 6), "lon": round(lon, 6),
                    "priors": {prior: round(zval, 2)},
                    "score1": round(min(zval, 10.0), 2)})
    return out


def merge_candidates(cands: Sequence[dict], radius_m: float = 300.0) -> List[dict]:
    """Deduplicate candidates across priors within radius_m.

    Strongest candidate wins the location; contributions from other priors are
    folded into its `priors` dict. Multi-prior agreement adds +1 to score1 per
    extra independent prior (mild — agreement raises rank, absence never
    vetoes).
    """
    dlat = radius_m / M_PER_DEG_LAT
    cells: Dict[tuple, int] = {}
    kept: List[dict] = []
    for c in sorted(cands, key=lambda c: -c["score1"]):
        mid_cos = max(math.cos(math.radians(c["lat"])), 1e-6)
        key = (int(c["lat"] / dlat), int(c["lon"] * mid_cos / dlat))
        hit = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                idx = cells.get((key[0] + dr, key[1] + dc))
                if idx is None:
                    continue
                k = kept[idx]
                dm = math.hypot((k["lat"] - c["lat"]) * M_PER_DEG_LAT,
                                (k["lon"] - c["lon"]) * M_PER_DEG_LAT * mid_cos)
                if dm <= radius_m:
                    hit = idx
                    break
            if hit is not None:
                break
        if hit is None:
            cells[key] = len(kept)
            kept.append({**c, "priors": dict(c["priors"])})
        else:
            k = kept[hit]
            for name, v in c["priors"].items():
                if name not in k["priors"] or v > k["priors"][name]:
                    k["priors"][name] = v
    for k in kept:
        extra = sum(1 for p in k["priors"] if p != "grid") - 1
        k["score1"] = round(k["score1"] + max(extra, 0) * 1.0, 2)
    return kept


# ------------------------------------------------------- networked fetchers

def fetch_dem_mosaic(bbox, px_m: float = 30.0, max_px: int = 4096,
                     log=logger.info) -> Optional[np.ndarray]:
    """Copernicus GLO-30 mosaic over bbox (first-valid-pixel fill)."""
    from archaeo_intel.data_access import read_grid, stac_search

    w, h = _bbox_shape(bbox, px_m, max_px)
    feats = stac_search("cop-dem-glo-30", list(bbox), limit=20)
    if not feats:
        log("dem: no GLO-30 tiles found")
        return None
    acc = np.full((h, w), np.nan, "float32")
    for f in feats:
        try:
            g = read_grid(f["assets"]["data"]["href"], bbox, w, h)
        except Exception as exc:
            log(f"dem: skip tile ({type(exc).__name__})")
            continue
        acc = np.where(np.isfinite(acc), acc, g)
        if np.isfinite(acc).mean() > 0.99:
            break
    if not np.isfinite(acc).any():
        return None
    return acc


def fetch_s2_mosaic(bbox, bands=("red", "nir"), px_m: float = 20.0,
                    years=(2024, 2025, 2026), max_px: int = 4096,
                    max_scenes: int = 8, log=logger.info
                    ) -> Optional[Dict[str, np.ndarray]]:
    """Cloud-masked first-valid-pixel Sentinel-2 mosaic of the given bands.

    Uses composite.find_scenes (low-cloud, sorted) + the SCL keep-mask; cheaper
    than a temporal median and good enough for outlier priors.
    """
    from rasterio.enums import Resampling

    from archaeo_intel.composite import find_scenes, scl_mask
    from archaeo_intel.data_access import read_grid

    w, h = _bbox_shape(bbox, px_m, max_px)
    feats = find_scenes(list(bbox), season=("01-01", "12-31"), years=years,
                        cloud_lt=15, max_scenes=30)
    if not feats:
        log("s2: no low-cloud scenes found")
        return None
    acc = {b: np.full((h, w), np.nan, "float32") for b in bands}
    used = 0
    for f in feats:
        assets = f.get("assets", {})
        if not all(b in assets for b in bands) or "scl" not in assets:
            continue
        try:
            scl = read_grid(assets["scl"]["href"], bbox, w, h,
                            resampling=Resampling.nearest)
            good = scl_mask(scl)
            if good.mean() < 0.2:
                continue
            for b in bands:
                g = read_grid(assets[b]["href"], bbox, w, h)
                g[~good] = np.nan
                acc[b] = np.where(np.isfinite(acc[b]), acc[b], g)
            used += 1
        except Exception as exc:
            log(f"s2: skip {f.get('id')} ({type(exc).__name__})")
            continue
        if np.isfinite(acc[bands[0]]).mean() > 0.98 or used >= max_scenes:
            break
    log(f"s2: mosaic from {used} scenes, "
        f"{np.isfinite(acc[bands[0]]).mean():.0%} coverage")
    if used == 0 or not np.isfinite(acc[bands[0]]).any():
        return None
    return acc


# ------------------------------------------------------------ prior drivers

def dem_priors(bbox, px_m: float = 30.0, bg_px: int = 33, min_z: float = 4.0,
               max_peaks: int = 800, dem: Optional[np.ndarray] = None,
               log=logger.info) -> List[dict]:
    """Local-relief outliers + closed depressions from the DEM.

    Relief = DEM minus its large-scale surface (the archaeo_intel.prominence
    construction); scored via local_outlier so flat-plain bumps beat
    mountain noise. Depressions get their own prior name — closed pits
    (craters, sinkholes, quarries) are a distinct kind of interesting.
    """
    from scipy.ndimage import gaussian_filter

    if dem is None:
        dem = fetch_dem_mosaic(bbox, px_m=px_m, log=log)
    if dem is None:
        return []
    filled = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    relief = dem - gaussian_filter(filled, max(500.0 / px_m, 3.0))
    z = local_outlier(np.abs(relief), bg_px=bg_px)
    peaks = anomaly_peaks(z, bbox, "dem_relief", min_z=min_z,
                          max_peaks=max_peaks)
    zpit = local_outlier(np.clip(-relief, 0, None), bg_px=bg_px)
    peaks += anomaly_peaks(zpit, bbox, "dem_depression", min_z=min_z,
                           max_peaks=max_peaks // 2)
    log(f"dem_priors: {len(peaks)} peaks")
    return peaks


def spectral_priors(bbox, px_m: float = 20.0, bg_px: int = 33,
                    min_z: float = 4.0, max_peaks: int = 800,
                    mosaic: Optional[Dict[str, np.ndarray]] = None,
                    log=logger.info) -> List[dict]:
    """Sentinel-2 pixels whose brightness or NDVI is an outlier vs their
    neighbourhood."""
    if mosaic is None:
        mosaic = fetch_s2_mosaic(bbox, bands=("red", "nir"), px_m=px_m, log=log)
    if mosaic is None:
        return []
    red, nir = mosaic["red"], mosaic["nir"]
    ndvi = (nir - red) / (nir + red + 1e-6)
    z = np.fmax(np.abs(local_outlier(red, bg_px)),
                np.abs(local_outlier(ndvi, bg_px)))
    peaks = anomaly_peaks(z, bbox, "spectral", min_z=min_z, max_peaks=max_peaks)
    log(f"spectral_priors: {len(peaks)} peaks")
    return peaks


def change_priors(bbox, px_m: float = 20.0, bg_px: int = 33,
                  min_z: float = 4.5, max_peaks: int = 800,
                  early_years=(2017, 2018, 2019), late_years=(2024, 2025, 2026),
                  log=logger.info) -> List[dict]:
    """Things that appeared or vanished between the S2 early and late epochs.

    Each epoch mosaic is robust-normalised before differencing so a global
    illumination/processing shift between years doesn't read as change
    everywhere (a plausibility gate, not a nicety).
    """
    early = fetch_s2_mosaic(bbox, bands=("red",), px_m=px_m,
                            years=early_years, log=log)
    late = fetch_s2_mosaic(bbox, bands=("red",), px_m=px_m,
                           years=late_years, log=log)
    if early is None or late is None:
        return []

    def norm(a):
        med = np.nanmedian(a)
        mad = 1.4826 * np.nanmedian(np.abs(a - med)) + 1e-9
        return (a - med) / mad

    diff = np.abs(norm(late["red"]) - norm(early["red"]))
    z = local_outlier(diff, bg_px)
    peaks = anomaly_peaks(z, bbox, "change", min_z=min_z, max_peaks=max_peaks)
    log(f"change_priors: {len(peaks)} peaks")
    return peaks


def generate_candidates(bbox, *, use_grid: bool = False,
                        grid_spacing_m: float = 500.0,
                        use_change: bool = True, merge_radius_m: float = 300.0,
                        dem: Optional[np.ndarray] = None,
                        log=logger.info) -> List[dict]:
    """Run all priors over a bbox and return the merged candidate list.

    Priors that fail (no data, network trouble) contribute nothing rather than
    aborting the run — but at least one prior must produce candidates, else
    this raises (a run that "completed" with zero candidates is the
    counter-going-up failure mode).
    """
    cands: List[dict] = []
    cands += dem_priors(bbox, dem=dem, log=log)
    cands += spectral_priors(bbox, log=log)
    if use_change:
        cands += change_priors(bbox, log=log)
    if use_grid:
        cands += grid_candidates(bbox, spacing_m=grid_spacing_m)
    if not cands:
        raise RuntimeError("no prior produced any candidate — refusing to "
                           "report an empty region as swept")
    merged = merge_candidates(cands, radius_m=merge_radius_m)
    log(f"generate_candidates: {len(cands)} raw -> {len(merged)} merged")
    return merged
