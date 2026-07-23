"""Unit tests for archaeo_intel.closure (pure, no network).

Locks in the validated closure-phase math: a phase-consistent interferogram
stack closes to ~0, an injected disturbance raises |closure| locally, and the
separability metric recovers sites-vs-controls.
"""
import numpy as np
import pytest

from archaeo_intel.closure import (
    closure_magnitude_stack,
    separability,
    triplet_closure,
)


def _consistent_phases(shape=(60, 60), seed=0):
    """Build phi_ij from an underlying per-date phase field so the loop closes
    exactly: phi_ij = p_j - p_i => phi_12 + phi_23 - phi_13 == 0."""
    rng = np.random.default_rng(seed)
    p = [rng.normal(0, 2.0, shape).astype("float32") for _ in range(3)]
    phi_12 = p[1] - p[0]
    phi_23 = p[2] - p[1]
    phi_13 = p[2] - p[0]
    return phi_12, phi_23, phi_13


def test_consistent_triplet_closes_to_zero():
    phi_12, phi_23, phi_13 = _consistent_phases()
    c = triplet_closure(phi_12, phi_23, phi_13)
    assert np.nanmax(np.abs(c)) < 1e-3


def test_injected_disturbance_raises_closure():
    phi_12, phi_23, phi_13 = _consistent_phases(seed=1)
    # a non-closing anomaly on one leg (physical change between 2 and 3)
    phi_23 = phi_23.copy()
    phi_23[25:35, 25:35] += 1.5
    c = triplet_closure(phi_12, phi_23, phi_13)
    site = np.abs(c[25:35, 25:35]).mean()
    background = np.abs(c[:10, :10]).mean()
    assert site > 1.0
    assert background < 1e-2


def test_scene_offset_is_removed():
    phi_12, phi_23, phi_13 = _consistent_phases(seed=2)
    # add a constant scene-wide offset -> median referencing must cancel it
    c = triplet_closure(phi_12 + 5.0, phi_23, phi_13, reference="median")
    assert np.nanmax(np.abs(c)) < 1e-3
    raw = triplet_closure(phi_12 + 5.0, phi_23, phi_13, reference="none")
    assert abs(np.nanmedian(raw) - 5.0) < 1e-3


def test_magnitude_stack_needs_a_triplet():
    dates = ["20230101", "20230113"]
    with pytest.raises(ValueError):
        closure_magnitude_stack({}, dates)


def test_magnitude_stack_averages_triplets():
    shape = (40, 40)
    dates = ["d0", "d1", "d2", "d3"]
    # build consistent stacks per pair from per-date fields, inject a site
    rng = np.random.default_rng(3)
    p = {d: rng.normal(0, 1.0, shape).astype("float32") for d in dates}
    pairs = {}
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            pairs[(dates[i], dates[j])] = p[dates[j]] - p[dates[i]]
    # inject persistent disturbance into every leg touching d2/d3 boundary
    for (a, b), arr in pairs.items():
        if a == "d2" and b == "d3":
            arr[10:15, 10:15] += 2.0
    stat = closure_magnitude_stack(pairs, dates)
    assert stat.shape == shape
    assert np.isfinite(stat).all()
    assert stat[10:15, 10:15].mean() > stat[30:, 30:].mean()


def test_separability_recovers_sites():
    # a stat field high at 'site' coords, low elsewhere
    h, w = 100, 100
    box = (40.0, 36.0, 40.5, 36.5)   # lon0,lat0,lon1,lat1
    stat = np.full((h, w), 0.1, "float32")
    sites, controls = [], []
    rng = np.random.default_rng(4)
    for _ in range(20):
        r, c = rng.integers(5, h - 5), rng.integers(5, w - 5)
        stat[r - 2:r + 3, c - 2:c + 3] = 0.9
        lat = box[3] - (r + 0.5) / h * (box[3] - box[1])
        lon = box[0] + (c + 0.5) / w * (box[2] - box[0])
        sites.append((lat, lon))
    for _ in range(20):
        r, c = rng.integers(5, h - 5), rng.integers(5, w - 5)
        lat = box[3] - (r + 0.5) / h * (box[3] - box[1])
        lon = box[0] + (c + 0.5) / w * (box[2] - box[0])
        controls.append((lat, lon))
    res = separability(stat, box, sites, controls)
    assert res["n_sites"] == 20 and res["n_controls"] == 20
    assert res["separability"] > 0.8
    assert res["site_median"] > res["control_median"]


def test_separability_safe_when_empty():
    box = (40.0, 36.0, 40.5, 36.5)
    stat = np.full((10, 10), np.nan, "float32")
    res = separability(stat, box, [(36.2, 40.2)], [(36.3, 40.3)])
    assert np.isnan(res["separability"])
