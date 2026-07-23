"""Context samplers for detect_anomalies — imagery-based confound checks.

Motivation (the Gila Bend lesson, 2026-07-22): during the North-America
desert sweep, an accelerating subsidence bowl near Gila Bend AZ passed an
OSM-only auto-explain (OSM reported zero farmland within 1.5 km) but NAIP
imagery showed it sitting on the rim of a center-pivot irrigation complex —
classic agricultural groundwater drawdown, not a void. OSM landuse tags are
routinely absent for desert agriculture, so a settlement/void screen that
trusts OSM alone will mis-promote irrigation bowls.

This module adds an IMAGERY-based agriculture check to complement OSM:
- `field_regularity_score(gray)`: straight-field-edge count (primary, real-
  data validated). `center_pivot_score` (Hough circles) is EXPERIMENTAL only.
- `agriculture_score(gray)`   : imagery cultivation score -> [0,1].
- `naip_agriculture_sampler`  : fetches a NAIP chip and scores it.
- `slope_sampler`             : Copernicus-DEM mean slope (deg).
- `osm_infrastructure_sampler`: Overpass count of mines/wells/plants/farmland.
- `is_cultivated_confound(ag, slope)`: the DECISION rule.
- `make_default_samplers`     : dict ready for detect_anomalies(context_samplers=...).

IMPORTANT limitation (real-data validation, 2026-07-22): the straight-line
agriculture detector also fires on steep PARALLEL MOUNTAIN LINEATIONS
(drainage gullies / ridges). Two barren mountain sites (Mojave Preserve,
Cabeza Prieta) scored as cultivated on lines alone. Agriculture is flat, so
the correct confound test pairs a high agriculture score with LOW terrain
slope from the DEM (image texture cannot tell fields from mountains; slope
can). Use `is_cultivated_confound(agriculture, slope_deg)`, not the
agriculture score by itself. Validated separation on FLAT sites:
barren Mojave 0.07 / Dixie Valley 0.13 vs irrigation Gila Bend / Imperial /
Yuma / Central Valley / Coachella all 1.00.

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
    """EXPERIMENTAL — Hough-circle response for center-pivot irrigation.

    Returns [0,1]. NOT reliable on real NAIP at <~5 m/px and NOT part of
    agriculture_score: real-data validation (2026-07-22) showed it scores
    barren mountain terrain (arc-like ridges -> false circles) HIGHER than
    actual pivot fields (whose crop patterns fragment the circle edge). Kept
    for high-resolution / synthetic use; use field_regularity_score for real
    imagery. px_per_m: pixels per metre (default 0.1 = 10 m/px).
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
    """Detect cultivated land by counting long straight edges. Returns [0,1].

    This is the primary, real-data-validated agriculture signal. Cultivated
    land — gridded fields AND center-pivot complexes (which carry access
    roads, canal lines and rectangular remnant plots) — produces many long
    STRAIGHT boundaries. Natural desert is curved/fractal (washes meander,
    ridges arc) and yields few straight lines, even where terrain has strong
    tonal contrast. Validated 2026-07-22 on real NAIP: barren Mojave bajada
    = 2 lines, Gila Bend irrigation = 16 lines (block-tone contrast, tried
    first, FAILED here — it fired on mountains/playa; straight-line count
    does not)."""
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    g = _norm(gray)
    if g.shape[0] < 40 or g.shape[1] < 40:
        return 0.0
    edges = canny(g, sigma=2.0)
    minlen = int(0.12 * min(g.shape))
    lines = probabilistic_hough_line(edges, threshold=10, line_length=minlen,
                                     line_gap=3)
    n = len(lines)
    # Floor of 5 discards the handful of chance-straight segments that even
    # natural texture yields; /15 puts the threshold (0.4 -> 11 lines) inside
    # the real-data gap (barren Mojave 2 lines, Gila Bend irrigation 16).
    base = np.clip((n - 5) / 15.0, 0, 1)
    if base <= 0 or n < 4:
        return float(base)
    # GRID gate: cultivated land is a grid (two ~perpendicular line families);
    # parallel lineations (alluvial-fan channels, mountain drainage) are one
    # family. Demote parallel-only line fields — they are NOT agriculture even
    # when flat (real-data: Mojave Preserve fan, 15 lines, 0.0 perpendicular).
    angs = np.array([np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
                     for (x0, y0), (x1, y1) in lines])
    hist, edg = np.histogram(angs, bins=18, range=(0, 180))
    dom = edg[np.argmax(hist)] + 5.0
    perp = (dom + 90.0) % 180.0
    d_perp = np.minimum(np.abs(angs - perp), 180 - np.abs(angs - perp))
    perp_frac = (d_perp < 20).mean()
    grid_factor = 1.0 if perp_frac >= 0.12 else 0.3
    return float(np.clip(base * grid_factor, 0, 1))


def agriculture_score(gray: np.ndarray, px_per_m: float = 0.1) -> float:
    """Imagery agriculture score [0,1]. Uses the real-data-validated
    straight-line field detector (field_regularity_score). The Hough-circle
    center_pivot_score is intentionally NOT combined in: it is unreliable on
    real NAIP (see its docstring). px_per_m kept for signature compatibility."""
    return field_regularity_score(gray)


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


def slope_sampler(read_grid_fn: Callable, stac_search_fn: Callable,
                  half_deg: float = 0.006, px: int = 44):
    """Build a (lat, lon) -> mean terrain slope (degrees) sampler from the
    Copernicus 30 m DEM. Agriculture is flat; this is the reliable
    flat-vs-mountain discriminator (NAIP texture is NOT — parallel mountain
    drainage lineations mimic field edges; see agriculture_score notes).
    Fails safe (nan)."""
    def sampler(lat: float, lon: float) -> float:
        grid = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
        try:
            f = stac_search_fn("cop-dem-glo-30", list(grid), limit=1)
            dem = read_grid_fn(f[0]["assets"]["data"]["href"], grid, px, px)
            d = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
            gy, gx = np.gradient(d, 30.0)
            return float(np.degrees(np.arctan(np.hypot(gx, gy))).mean())
        except Exception:
            return float("nan")

    return sampler


def is_cultivated_confound(agriculture: float, slope_deg: float,
                           slope_max: float = 3.0) -> bool:
    """Decision rule for 'this subsidence is agricultural pumping, not a void'.
    Requires BOTH a high imagery agriculture score AND flat terrain, because
    the straight-line agriculture detector also fires on steep parallel
    mountain lineations (real-data validation 2026-07-22: barren Mojave
    Preserve / Cabeza Prieta mountains scored high on lines but are excluded
    here by slope)."""
    if not np.isfinite(agriculture):
        return False
    if agriculture < AGRICULTURE_THRESHOLD:
        return False
    # slope unknown -> do not veto on agriculture alone (mountain risk)
    if not np.isfinite(slope_deg):
        return False
    return slope_deg <= slope_max


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
        samplers["slope_deg"] = slope_sampler(read_grid_fn, stac_search_fn)
    return samplers
