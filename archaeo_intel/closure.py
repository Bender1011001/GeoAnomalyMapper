"""Closure-phase analysis — the discarded InSAR signal, repurposed.

Every standard InSAR pipeline throws away *closure phase* as calibration
error: for three SAR acquisitions 1,2,3 the sum of the unwrapped
interferometric phases around the loop, C = phi_12 + phi_23 - phi_13, should
be ~0 for an ideal scatterer. Non-zero closure encodes soil-moisture /
dielectric heterogeneity and physical change on the ground.

Validated in this project on free HyP3 data:
- Construction/disturbance detector: closure rises at a documented site-work
  onset and LEADS the coherence signal by ~6 weeks (Track 2, 2026-07-16).
- ARCHAEOLOGY channel (2026-07-21): mean |closure| separates known tell
  sites from steppe controls at AUC 0.619 (141 sites), REPLICATED at 0.603
  on an independent AOI 90 km away (733 sites), both clearing a pre-
  registered 0.60 bar. Ranking-grade; mechanism (subsurface-moisture
  heterogeneity over buried architecture) inferred, not proven.

This module provides the pure, unit-tested core of that analysis. Fetching /
HyP3 orchestration stays in scripts; the math and the separability metric
live here so they are importable and verified.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def triplet_closure(phi_12: np.ndarray, phi_23: np.ndarray,
                    phi_13: np.ndarray, reference: str = "median") -> np.ndarray:
    """Closure phase of one triplet: C = phi_12 + phi_23 - phi_13.

    phi_ij are unwrapped interferometric phase arrays (radians) for pairs
    (1->2), (2->3), (1->3), same shape. A constant scene-wide offset (orbital
    / reference-pixel ambiguity) is removed by subtracting the scene
    ``reference`` ('median' (default) or 'mean'); pass reference='none' to
    keep the raw sum. NaNs propagate (invalid pixels stay invalid).
    """
    c = phi_12 + phi_23 - phi_13
    if reference == "none":
        return c
    finite = np.isfinite(c)
    if not finite.any():
        return c
    off = np.nanmedian(c) if reference == "median" else np.nanmean(c)
    return c - off


def closure_magnitude_stack(pairs: Dict[Tuple[str, str], np.ndarray],
                            dates: Sequence[str]) -> np.ndarray:
    """Mean |closure| over every consecutive date triplet.

    pairs maps (date_i, date_j) -> unwrapped phase array. `dates` is the
    sorted acquisition list. For each i, if the three legs (i,i+1),(i+1,i+2),
    (i,i+2) are all present, accumulate |triplet_closure|. Returns the
    per-pixel mean magnitude (radians); pixels with no valid triplet are NaN.
    Raises ValueError if fewer than one complete triplet exists.
    """
    d = list(dates)
    mags: List[np.ndarray] = []
    for i in range(len(d) - 2):
        k12, k23, k13 = (d[i], d[i + 1]), (d[i + 1], d[i + 2]), (d[i], d[i + 2])
        if k12 in pairs and k23 in pairs and k13 in pairs:
            mags.append(np.abs(triplet_closure(pairs[k12], pairs[k23], pairs[k13])))
    if not mags:
        raise ValueError("no complete date triplet in `pairs`")
    stack = np.stack(mags)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def auc_ci(auc: float, n_pos: int, n_neg: int,
           z: float = 1.96) -> Tuple[float, float, float]:
    """Hanley-McNeil standard error and CI for a rank AUC.

    Returns (se, lo, hi). Reporting an AUC without this is how the project
    ended up describing 0.619 [0.554, 0.684] as "clears the 0.60 bar" — a
    point estimate whose interval comfortably includes 0.55. Always report it.
    """
    if n_pos <= 0 or n_neg <= 0 or not np.isfinite(auc):
        return (float("nan"), float("nan"), float("nan"))
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    return (se, float(a - z * se), float(a + z * se))


def _sample(stat: np.ndarray, box: Tuple[float, float, float, float],
            lat: float, lon: float, rad: int = 2) -> float:
    """Median of `stat` in a (2*rad+1) window at (lat, lon). box = (lon_min,
    lat_min, lon_max, lat_max); stat rows go north->south (row 0 = lat_max)."""
    h, w = stat.shape
    lon0, lat0, lon1, lat1 = box
    c = int((lon - lon0) / (lon1 - lon0) * w)
    r = int((lat1 - lat) / (lat1 - lat0) * h)
    sl = stat[max(r - rad, 0):r + rad + 1, max(c - rad, 0):c + rad + 1]
    return float(np.nanmedian(sl)) if np.isfinite(sl).any() else float("nan")


def separability(stat: np.ndarray, box: Tuple[float, float, float, float],
                 sites: Sequence[Tuple[float, float]],
                 controls: Sequence[Tuple[float, float]], rad: int = 2) -> dict:
    """Rank-AUC separability of `stat` between sites and controls.

    Returns {'auc', 'separability' (=max(auc,1-auc)), 'se', 'ci_low',
    'ci_high', 'n_sites', 'n_controls', 'site_median', 'control_median'}.
    Separability is polarity-agnostic (as used for the archaeology channels);
    >=0.60 was the pre-registered validation bar.

    ALWAYS read the CI, not just the point estimate: at n~141 the 95% interval
    on an AUC is about +-0.065, so 0.60-bar pass/fail language is not
    statistically meaningful at that sample size (see docs/CRITIQUE.md 1.1).
    """
    s = np.array([_sample(stat, box, la, lo, rad) for la, lo in sites])
    c = np.array([_sample(stat, box, la, lo, rad) for la, lo in controls])
    s = s[np.isfinite(s)]
    c = c[np.isfinite(c)]
    if len(s) == 0 or len(c) == 0:
        return {"auc": float("nan"), "separability": float("nan"),
                "se": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_sites": int(len(s)), "n_controls": int(len(c)),
                "site_median": float("nan"), "control_median": float("nan")}
    allv = np.concatenate([s, c])
    rk = allv.argsort().argsort().astype(float)
    auc = (rk[:len(s)].sum() - len(s) * (len(s) - 1) / 2) / (len(s) * len(c))
    sep = float(max(auc, 1 - auc))
    se, lo, hi = auc_ci(sep, len(s), len(c))
    return {"auc": float(auc), "separability": sep,
            "se": se, "ci_low": lo, "ci_high": hi,
            "n_sites": int(len(s)), "n_controls": int(len(c)),
            "site_median": float(np.median(s)),
            "control_median": float(np.median(c))}
