"""Unit tests for deformation_intel.context imagery scorers (no network).

These lock in the Gila Bend lesson: a center-pivot irrigation pattern must
score high on the imagery agriculture check, while natural-desert-like noise
must score low — so the pipeline never again mis-promotes an agricultural
groundwater bowl as a void on the strength of an empty OSM query.
"""
import numpy as np
import pytest

from deformation_intel.context import (
    AGRICULTURE_THRESHOLD,
    agriculture_score,
    center_pivot_score,
    field_regularity_score,
)


def _draw_circle(size=250, radius=30, cx=None, cy=None):
    cx = size // 2 if cx is None else cx
    cy = size // 2 if cy is None else cy
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - cy, xx - cx)
    img = np.full((size, size), 0.5, "float32")
    ring = np.abs(r - radius) < 1.5
    img[r < radius] = 0.15          # dark irrigated interior
    img[ring] = 0.95                # bright pivot boundary
    return img


def _draw_field_grid(size=250, step=45):
    img = np.full((size, size), 0.5, "float32")
    for k in range(0, size, step):
        img[k:k + 3, :] = 0.9
        img[:, k:k + 3] = 0.9
    # alternate block tones
    for i, r0 in enumerate(range(0, size, step)):
        for j, c0 in enumerate(range(0, size, step)):
            img[r0 + 3:r0 + step, c0 + 3:c0 + step] = 0.25 if (i + j) % 2 else 0.78
    return img


def _desert_noise(size=250, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.06, (size, size)).astype("float32")
    # a realistic dendritic wash: a random-walk channel (natural desert
    # drainage meanders irregularly — it is NOT a smooth arc or straight
    # line, so it must not trigger the circle or field detectors).
    x = size / 2
    for y in range(size):
        x = np.clip(x + rng.normal(0, 1.6), 3, size - 4)
        base[y, int(x) - 1:int(x) + 2] = 0.2
    return np.clip(base, 0, 1)


def test_center_pivot_detected():
    # a clean pivot circle scores well above the agriculture threshold
    img = _draw_circle(radius=30)
    assert center_pivot_score(img, px_per_m=0.1) > 0.50


def test_center_pivot_below_threshold_in_noise():
    # natural desert texture stays below the agriculture decision threshold
    for seed in range(6):
        assert center_pivot_score(_desert_noise(seed=seed), px_per_m=0.1) < AGRICULTURE_THRESHOLD


def test_field_grid_scores_higher_than_noise():
    grid = field_regularity_score(_draw_field_grid())
    noise = field_regularity_score(_desert_noise(seed=2))
    assert grid > noise
    assert grid > 0.5


def test_agriculture_score_separates_ag_from_desert():
    # both agriculture morphologies (pivot circle, field grid) clear the
    # threshold; desert stays below it — the Gila Bend guarantee.
    pivot = agriculture_score(_draw_circle(radius=32), px_per_m=0.1)
    fields = agriculture_score(_draw_field_grid(), px_per_m=0.1)
    assert pivot >= AGRICULTURE_THRESHOLD
    assert fields >= AGRICULTURE_THRESHOLD
    for seed in range(6):
        assert agriculture_score(_desert_noise(seed=seed), px_per_m=0.1) < AGRICULTURE_THRESHOLD


def test_scorers_safe_on_tiny_or_nan_input():
    assert center_pivot_score(np.zeros((10, 10))) == 0.0
    assert field_regularity_score(np.zeros((10, 10))) == 0.0
    nan_img = np.full((250, 250), np.nan, "float32")
    assert agriculture_score(nan_img) == 0.0


def test_make_default_samplers_has_osm():
    from deformation_intel.context import make_default_samplers
    s = make_default_samplers(read_grid_fn=None, stac_search_fn=None)
    assert "osm_infra" in s
    assert callable(s["osm_infra"])
