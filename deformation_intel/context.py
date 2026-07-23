"""Context samplers for detect_anomalies — imagery-based confound checks.

Motivation (the Gila Bend lesson, 2026-07-22): during the North-America
desert sweep, an accelerating subsidence bowl near Gila Bend AZ passed an
OSM-only auto-explain (OSM reported zero farmland within 1.5 km) but NAIP
imagery showed it sitting on the rim of a center-pivot irrigation complex —
classic agricultural groundwater drawdown, not a void. OSM landuse tags are
routinely absent for desert agriculture, so a settlement/void screen that
trusts OSM alone will mis-promote irrigation bowls.

This module adds an IMAGERY-based agriculture check to complement OSM:
- `center_pivot_score(gray)`  : pure, network-free circle detector (Hough).
- `field_regularity_score(gray)`: pure straight-field-edge detector.
- `agriculture_score(gray)`   : max of the two -> [0,1].
- `naip_agriculture_sampler`  : fetches a NAIP chip and scores it.
- `osm_infrastructure_sampler`: Overpass count of mines/wells/plants/farmland.
- `make_default_samplers`     : dict ready for detect_anomalies(context_samplers=...).

The pure scorers are unit-tested on synthetic imagery (no network).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional

import numpy as np

# Decision threshold: agriculture_score >= this indicates cultivated land.
# Real center-pivot circles score ~0.6-0.8 and cultivated field-blocks ~0.5-1.0,
# while natural desert texture floors around ~0.3 on the circle detector; 0.4
# separates them with margin. Callers should compare against this, not 0.
AGRICULTURE_THRESHOLD = 0.4


def _norm(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype="float32")
    finite = np.isfinite(g)
    if finite.sum() < g.size * 0.5:
        return np.zeros_like(g)
    lo, hi = np.nanpercentile(g[finite], [2, 98])
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(g, nan=0.5)


def center_pivot_score(gray: np.ndarray, px_per_m: float = 0.1) -> float:
    """Detect center-pivot irrigation circles. Returns [0,1] (Hough peak).

    px_per_m: pixels per metre of the input chip (default 0.1 = 10 m/px).
    Pivots are ~400-800 m diameter -> 200-400 m radius. Score is the strongest
    Hough-circle accumulator peak in that radius band; >~0.3 indicates a pivot.
    """
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    g = _norm(gray)
    if g.shape[0] < 40 or g.shape[1] < 40:
        return 0.0
    edges = canny(g, sigma=2.5)
    r_lo = max(int(180 * px_per_m), 8)
    r_hi = max(int(420 * px_per_m), r_lo + 4)
    radii = np.arange(r_lo, r_hi, max((r_hi - r_lo) // 12, 1))
    if len(radii) == 0:
        return 0.0
    h = hough_circle(edges, radii)
    if h.size == 0:
        return 0.0
    accums, *_ = hough_circle_peaks(h, radii, total_num_peaks=3)
    return float(np.clip(max(accums) if len(accums) else 0.0, 0, 1))


def field_regularity_score(gray: np.ndarray) -> float:
    """Detect cultivated fields by block-tone contrast. Returns [0,1].

    Irrigated fields form large blocks of extreme tone (dark wet/vegetated
    or bright bare) that survive field-scale blurring; natural desert blurs
    to a near-uniform mid-tone. Score = fraction of the blurred image whose
    tone is far from the median. Robust to high-frequency desert texture
    (which averages out) and to irregular wash edges (thin, small area)."""
    from scipy.ndimage import uniform_filter

    g = _norm(gray)
    if g.shape[0] < 40 or g.shape[1] < 40:
        return 0.0
    k = max(int(0.06 * min(g.shape)), 3)     # ~field-scale smoothing window
    blur = uniform_filter(g, size=k)
    med = np.median(blur)
    extreme = np.abs(blur - med) > 0.22      # clearly dark/bright blocks
    frac = extreme.mean()
    return float(np.clip(frac / 0.20, 0, 1))  # ~20% extreme area -> score 1


def agriculture_score(gray: np.ndarray, px_per_m: float = 0.1) -> float:
    """Combined imagery agriculture score [0,1] = max(pivot, field)."""
    return max(center_pivot_score(gray, px_per_m),
               field_regularity_score(gray))


def naip_agriculture_sampler(read_grid_fn: Callable, stac_search_fn: Callable,
                             half_km: float = 1.25, px: int = 250):
    """Build a (lat, lon) -> agriculture_score sampler backed by NAIP.

    read_grid_fn(url, grid, w, h) and stac_search_fn(collection, bbox, ...)
    match archaeo_intel.data_access signatures. Fails safe (returns nan).
    """
    MPC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    def sampler(lat: float, lon: float) -> float:
        dlat = half_km / 111.0
        dlon = dlat / np.cos(np.radians(lat))
        grid = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        feats = stac_search_fn("naip", list(grid),
                               datetime="2019-01-01T00:00:00Z/2025-12-31T23:59:59Z",
                               limit=8, endpoint=MPC)
        acc = np.full((px, px), np.nan, "float32")
        for f in feats:
            try:
                g = read_grid_fn(f["assets"]["image"]["href"], grid, px, px,
                                 signed=True)
            except Exception:
                continue
            acc = np.where(np.isfinite(acc), acc, g)
            if np.isfinite(acc).mean() > 0.95:
                break
        if np.isfinite(acc).mean() < 0.5:
            return float("nan")
        m_per_px = (2 * half_km * 1000) / px
        return agriculture_score(acc, px_per_m=1.0 / m_per_px)

    return sampler


def osm_infrastructure_sampler(radius_m: int = 1500, retries: int = 3):
    """Build a (lat, lon) -> count sampler for man-made subsidence drivers
    (quarries, wells, pumping, power plants, industrial, farmland) via
    Overpass. Complements the imagery check; fails safe (nan)."""
    def sampler(lat: float, lon: float) -> float:
        q = (f"[out:json][timeout:25];("
             f'way(around:{radius_m},{lat},{lon})[landuse~"quarry|industrial|farmland"];'
             f'node(around:{radius_m},{lat},{lon})[man_made~"petroleum_well|water_well|pumping_station"];'
             f'node(around:{radius_m},{lat},{lon})[power~"plant|generator"];'
             f'way(around:{radius_m},{lat},{lon})[power~"plant"];'
             f");out count;")
        url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(q)
        for _ in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "deformation_intel/1.0"})
                d = json.loads(urllib.request.urlopen(req, timeout=40).read())
                for el in d.get("elements", []):
                    if el.get("type") == "count":
                        return float(el.get("tags", {}).get("total", 0))
                return 0.0
            except Exception:
                continue
        return float("nan")

    return sampler


def make_default_samplers(read_grid_fn: Optional[Callable] = None,
                          stac_search_fn: Optional[Callable] = None
                          ) -> Dict[str, Callable[[float, float], float]]:
    """Ready-to-use context_samplers dict for detect_anomalies.

    Always includes 'osm_infra'. If read_grid_fn+stac_search_fn are given (or
    archaeo_intel is importable), also includes 'naip_agriculture' — the
    imagery check that OSM alone misses (the Gila Bend lesson).
    """
    samplers: Dict[str, Callable[[float, float], float]] = {
        "osm_infra": osm_infrastructure_sampler(),
    }
    if read_grid_fn is None or stac_search_fn is None:
        try:
            from archaeo_intel.data_access import read_grid, stac_search
            read_grid_fn, stac_search_fn = read_grid, stac_search
        except Exception:
            read_grid_fn = stac_search_fn = None
    if read_grid_fn is not None and stac_search_fn is not None:
        samplers["naip_agriculture"] = naip_agriculture_sampler(read_grid_fn,
                                                                stac_search_fn)
    return samplers
