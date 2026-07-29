"""Pure computer-vision scorers for imagery chips — Stage 2 of the funnel.

Everything here is offline (numpy/scipy/skimage only, no network) so it can be
unit-tested on synthetic AND real cached chips. The scorers are RANKING
features, not classifiers: per the project's hard-won rules, nothing in this
module vetoes a candidate — a weak signal adjusts rank, it does not delete.

Design principle: rank by NOVELTY RELATIVE TO SURROUNDINGS, not absolute
score. `texture_vector` + `novelty_score` implement that — a chip is compared
against a background sample drawn from the same region, so a circle in the
desert outranks the same circle in a center-pivot field.

Real-imagery calibration (2026-07-27, NAIP/S2 chips in
tests/data/interesting_chips/): geometry scorers were tuned on the positive
controls (Bingham Canyon, Minuteman silo field) against natural terrain
(Racetrack Playa surroundings) — see tests/test_interesting_features.py.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# Number of dimensions in texture_vector — background matrices must match.
TEXTURE_DIMS = 10


def normalize01(gray: np.ndarray) -> np.ndarray:
    """Robust [0,1] normalisation (2-98 percentile clip, nan -> median)."""
    g = np.asarray(gray, dtype="float32")
    finite = np.isfinite(g)
    if finite.sum() < max(g.size // 2, 1):
        return np.zeros_like(g)
    lo, hi = np.nanpercentile(g[finite], [2, 98])
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(g, nan=0.5)


def chip_ok(gray: np.ndarray, min_finite: float = 0.6) -> bool:
    """Plausibility gate: is this chip real imagery worth scoring?

    Catches the classic silent-failure modes (all-nan reads, nodata borders,
    constant fill values) BEFORE they enter the ranking — the project rule is
    that plausibility gates come before ranking, or the shortlist fills with
    artifacts.
    """
    g = np.asarray(gray, dtype="float32")
    if g.ndim != 2 or g.size < 32 * 32:
        return False
    finite = np.isfinite(g)
    if finite.mean() < min_finite:
        return False
    vals = g[finite]
    lo, hi = np.percentile(vals, [2, 98])
    return bool(hi - lo > 1e-6 and np.std(vals) > 1e-6)


# ------------------------------------------------------------------ geometry

def line_stats(gray: np.ndarray) -> Dict[str, float]:
    """Straight-line census of a chip: count + grid-ness.

    Same probabilistic-Hough machinery that survived real-data validation in
    deformation_intel.context (straight edges = man-made; natural desert is
    curved/fractal), but returned as raw statistics so callers can rank
    instead of threshold.
    """
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    g = normalize01(gray)
    if g.shape[0] < 40 or g.shape[1] < 40:
        return {"n_lines": 0.0, "perp_frac": 0.0}
    edges = canny(g, sigma=2.0)
    # minlen 0.2 x chip with a tight gap: measured 2026-07-27, chance
    # alignments in dense noise edges (13k edge px) produce ~30 lines at
    # 0.12x/gap=3 but ZERO at 0.2x/gap=2, while a real grid keeps all 22.
    minlen = int(0.2 * min(g.shape))
    lines = probabilistic_hough_line(edges, threshold=10, line_length=minlen,
                                     line_gap=2)
    n = len(lines)
    if n < 2:
        return {"n_lines": float(n), "perp_frac": 0.0}
    angs = np.array([np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
                     for (x0, y0), (x1, y1) in lines])
    hist, edg = np.histogram(angs, bins=18, range=(0, 180))
    dom = edg[np.argmax(hist)] + 5.0
    perp = (dom + 90.0) % 180.0
    d_perp = np.minimum(np.abs(angs - perp), 180 - np.abs(angs - perp))
    return {"n_lines": float(n), "perp_frac": float((d_perp < 20).mean())}


def straightedge_score(gray: np.ndarray) -> float:
    """Man-made-ness from straight-edge density. [0,1], ranking feature."""
    return float(np.clip((line_stats(gray)["n_lines"] - 3) / 25.0, 0, 1))


def circle_score(gray: np.ndarray) -> float:
    """Hough-circle response over radii scaled to the chip. [0,1].

    Known limitation (real-data lesson from center_pivot_score): arc-like
    natural ridges can fire this. That is why it is a RANKING feature weighted
    below straight edges, never a veto or a standalone detector.

    Scored as PEAK OVER BACKGROUND, not raw accumulator: measured 2026-07-27,
    dense chaotic edges give raw Hough maxima ~0.6 (HIGHER than a true
    circle's 0.39 at sigma 2), but the best-radius / median-radius ratio
    separates cleanly (true circle ~3.6x, noise ~1.8x).
    """
    from skimage.feature import canny
    from skimage.transform import hough_circle

    g = normalize01(gray)
    if g.shape[0] < 64 or g.shape[1] < 64:
        return 0.0
    edges = canny(g, sigma=2.5)
    r_hi = min(g.shape) // 3
    radii = np.arange(8, max(r_hi, 12), max((r_hi - 8) // 10, 2))
    h = hough_circle(edges, radii)
    if h.size == 0:
        return 0.0
    per_radius = h.max(axis=(1, 2))
    peak = float(per_radius.max())
    med = float(np.median(per_radius))
    ratio = peak / (med + 1e-6)
    return float(np.clip((ratio - 1.5) / 3.0, 0, 1))


def symmetry_score(gray: np.ndarray) -> float:
    """Mirror/rotational symmetry of the HIGH-PASSED chip. [0,1].

    High-pass first: raw chips correlate with their flips through smooth
    background gradients alone, which made every chip look 'symmetric'.
    """
    g = normalize01(gray)
    if g.shape[0] < 48 or g.shape[1] < 48:
        return 0.0
    hp = g - gaussian_filter(g, min(g.shape) / 8.0)
    s = float(np.std(hp))
    if s < 1e-6:
        return 0.0
    best = 0.0
    for flipped in (hp[::-1, :], hp[:, ::-1], hp[::-1, ::-1]):
        c = float(np.corrcoef(hp.ravel(), flipped.ravel())[0, 1])
        best = max(best, c)
    return float(np.clip(best, 0, 1))


def radiality_score(gray: np.ndarray) -> float:
    """Concentric/radial structure around the chip centre. [0,1].

    Gradient radiality (fast-radial-symmetry style): in a concentric
    structure the strong gradients point at/away from the centre, so the
    mean |cos| between gradient direction and the radial direction rises
    above the isotropic-texture expectation of 2/pi ~ 0.64.

    Chosen over two rejected designs (real-fixture measurements 2026-07-27):
    Hough peak-over-median erases structures with circles at EVERY radius
    (Richat scored 0.13, BELOW its background), and polar ring-coherence is
    destroyed by regional tone gradients and small centre offsets (Richat
    0.27 vs background 0.26). Gradient radiality: Richat 0.36 and Bingham's
    terraced pit 0.33 vs their backgrounds' <= 0.22, with flat/linear terrain
    at ~0.0.
    """
    g = normalize01(gray)
    h, w = g.shape
    if min(h, w) < 64:
        return 0.0
    g = gaussian_filter(g, 2.0)
    gy, gx = np.gradient(g)
    yy, xx = np.mgrid[0:h, 0:w]
    ry, rx = yy - (h - 1) / 2.0, xx - (w - 1) / 2.0
    rn = np.hypot(ry, rx) + 1e-6
    mag = np.hypot(gx, gy)
    keep = (mag > np.percentile(mag, 75)) & (rn > 0.05 * min(h, w)) \
        & (rn < 0.48 * min(h, w))
    if keep.sum() < 50:
        return 0.0
    cosang = np.abs(gx * rx / rn + gy * ry / rn)[keep] / (mag[keep] + 1e-9)
    m = float(np.mean(np.clip(cosang, 0, 1)))
    return float(np.clip((m - 0.64) / 0.36, 0, 1))


def faint_line_score(gray: np.ndarray) -> float:
    """Long straight lines too faint for canny (geoglyphs, old roads,
    hollow-ways). [0,1].

    Radon transform of the CONTRAST-CLIPPED high-passed chip: clipping the
    robust z at +-1.5 sigma saturates the high-contrast structures (washes,
    field blocks) that otherwise dominate, while a long faint straight line
    integrates coherently along exactly one (angle, offset) bin and stands
    out against the per-angle median.

    Rejected design (real-fixture measurement 2026-07-27): sato-ridge top-3%
    mask + Hough scored the Nazca lines chip 0.0 — the bright dendritic
    washes own the ridge-response tail, so the geoglyph lines never enter
    the mask. Radon-clipped scores Nazca above its whole background ring.
    """
    from skimage.transform import radon, resize

    from archaeo_intel.detect import robust_z

    g = normalize01(gray)
    if g.shape[0] < 64 or g.shape[1] < 64:
        return 0.0
    if min(g.shape) > 256:
        s = 256.0 / min(g.shape)
        g = resize(g, (int(g.shape[0] * s), int(g.shape[1] * s)),
                   anti_aliasing=True).astype("float32")
    z = np.clip(robust_z(g), -1.5, 1.5)
    h, w = z.shape
    n = min(h, w)
    z = z[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n]
    hp = z - gaussian_filter(z, 4.0)
    yy, xx = np.mgrid[0:n, 0:n]
    hp = hp * (np.hypot(yy - n / 2, xx - n / 2) < n / 2 - 2)
    theta = np.linspace(0., 180., 120, endpoint=False)
    sino = radon(hp, theta=theta, circle=True)
    med = np.median(sino, axis=0)
    mad = 1.4826 * np.median(np.abs(sino - med), axis=0) + 1e-6
    zz = np.abs(sino - med) / mad
    # top-5-bin mean, NOT a percentile: a single line occupies only ~3 of
    # the ~30k sinogram bins, so even the 99.9th percentile misses it
    # (measured: one 1.5-sigma synthetic line scored 0.0 by percentile but
    # 11.9 by top-5 against a ~4.8 noise floor).
    peak = float(np.sort(zz.ravel())[-5:].mean())
    return float(np.clip((peak - 5.0) / 10.0, 0, 1))


def geometry_score(gray: np.ndarray) -> float:
    """Combined man-made/organized-geometry rank feature. [0,1].

    Straight edges lead (the only geometry signal that survived real-data
    validation in this project); faint-line and radiality cover the
    organized-but-low-contrast structures the controls exposed; circles and
    symmetry contribute but cannot carry the score alone.
    """
    return chip_features(gray)["geometry"]


# ------------------------------------------------------------------- texture

def edge_density(gray: np.ndarray) -> float:
    """Fraction of canny edge pixels. Cheap man-made/texture-busyness cue."""
    from skimage.feature import canny

    g = normalize01(gray)
    if g.shape[0] < 40 or g.shape[1] < 40:
        return 0.0
    return float(canny(g, sigma=2.0).mean())


def sharpness(gray: np.ndarray) -> float:
    """90th-percentile gradient magnitude — crisp edges vs mush."""
    g = normalize01(gray)
    gy, gx = np.gradient(g)
    return float(np.percentile(np.hypot(gx, gy), 90))


def tone_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the tone histogram (bits, 0..6)."""
    g = normalize01(gray)
    hist, _ = np.histogram(g, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def orientation_coherence(gray: np.ndarray) -> float:
    """Mean structure-tensor coherence over strong-gradient pixels. [0,1].

    Striped/aligned texture (plough lines, lineations, runways) scores high;
    isotropic natural texture scores low.
    """
    g = normalize01(gray)
    gy, gx = np.gradient(gaussian_filter(g, 1))
    jxx = gaussian_filter(gx * gx, 3)
    jyy = gaussian_filter(gy * gy, 3)
    jxy = gaussian_filter(gx * gy, 3)
    trace = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (trace + 1e-9)
    mag = np.hypot(gx, gy)
    strong = mag > np.percentile(mag, 75)
    if strong.sum() < 16:
        return 0.0
    return float(np.clip(coh[strong].mean(), 0, 1))


def center_contrast(gray: np.ndarray) -> float:
    """Signed centre-disc vs annulus contrast in surround-sigma units.

    Positive = bright centre (pads, spoil, structures), negative = dark centre
    (pits, ponds, craters). Same disc-vs-annulus construction that fixed the
    well-pad detector (connected components fragment; annuli don't).
    """
    g = normalize01(gray)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2.0, xx - w / 2.0) / (0.5 * min(h, w))
    core = r < 0.25
    ring = (r > 0.55) & (r < 0.95)
    if core.sum() < 16 or ring.sum() < 16:
        return 0.0
    spread = float(np.std(g[ring])) + 1e-6
    return float((np.mean(g[core]) - np.mean(g[ring])) / spread)


def chip_features(gray: np.ndarray) -> dict:
    """All per-chip CV features, each primitive computed exactly once.

    Returns {"vector": TEXTURE_DIMS array, "geometry": float} plus the named
    primitives. The funnel's Stage 2 calls this once per chip — calling
    texture_vector() and geometry_score() separately doubles the radon/Hough
    work for nothing.
    """
    g = normalize01(gray)
    st = straightedge_score(gray)
    fl = faint_line_score(gray)
    rad = radiality_score(gray)
    circ = circle_score(gray)
    sym = symmetry_score(gray)
    vec = np.array([
        float(np.mean(g)),
        float(np.std(g)),
        edge_density(gray),
        sharpness(gray),
        tone_entropy(gray),
        orientation_coherence(gray),
        st,
        abs(center_contrast(gray)) / 5.0,
        rad,
        fl,
    ], dtype="float32")
    geometry = float(np.clip(0.35 * st + 0.25 * fl + 0.25 * rad
                             + 0.075 * circ + 0.075 * sym, 0, 1))
    return {"vector": vec, "geometry": geometry, "straightedge": st,
            "faint_line": fl, "radiality": rad, "circle": circ,
            "symmetry": sym}


def texture_vector(gray: np.ndarray) -> np.ndarray:
    """TEXTURE_DIMS-element descriptor used for out-of-placeness ranking."""
    return chip_features(gray)["vector"]


def novelty_score(vec: np.ndarray, background: Sequence[np.ndarray]) -> float:
    """How out-of-place is this chip vs a sample of chips from the SAME region?

    Robust per-dimension z (median/MAD over the background sample), clipped at
    10 sigma, RMS-combined. Requires >= 8 background chips; returns nan below
    that so callers cannot silently rank on a meaningless baseline.
    """
    bg = np.asarray(list(background), dtype="float32")
    if bg.ndim != 2 or bg.shape[0] < 8 or bg.shape[1] != len(vec):
        return float("nan")
    med = np.median(bg, axis=0)
    mad = 1.4826 * np.median(np.abs(bg - med), axis=0)
    # Floor each dimension's MAD at a fraction of its median spread so one
    # near-constant background dimension can't blow the score up.
    mad = np.maximum(mad, 0.05 * (np.std(bg, axis=0) + 1e-6) + 1e-6)
    z = np.clip(np.abs((np.asarray(vec, "float32") - med) / mad), 0, 10)
    return float(np.sqrt(np.mean(z ** 2)))


def local_outlier(img: np.ndarray, bg_px: int = 33) -> np.ndarray:
    """Per-pixel out-of-placeness: (img - local mean) / local std.

    The workhorse of Stage-1 priors: a 10 m bump on a flat plain scores huge,
    the same bump in mountains scores ~1 — which is exactly the
    novelty-relative-to-surroundings behaviour the funnel wants (and avoids
    the mass-false-positive mountains failure documented in
    archaeo_intel.detect.regional_roughness).
    """
    a = np.asarray(img, dtype="float64")
    med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
    x = np.nan_to_num(a, nan=med)
    m = uniform_filter(x, bg_px)
    v = uniform_filter(x * x, bg_px) - m * m
    sd = np.sqrt(np.clip(v, 0, None))
    # floor at a fraction of the global robust spread so ultra-flat patches
    # (playas) don't divide by ~zero and explode
    glob = 1.4826 * np.nanmedian(np.abs(a - med)) + 1e-9
    sd = np.maximum(sd, 0.1 * glob)
    out = (x - m) / sd
    out[~np.isfinite(a)] = np.nan
    return out.astype("float32")
