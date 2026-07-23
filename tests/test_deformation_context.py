"""Unit tests for deformation_intel.context imagery scorers (no network).

Locks in the Gila Bend lesson AND the real-data correction of 2026-07-22:
- Cultivated land (gridded fields, pivot complexes) has many long STRAIGHT
  edges and must score high.
- Natural desert must score low — INCLUDING terrain with strong tonal blocks
  (mountains, playa). An earlier block-tone detector passed synthetic tests
  but false-fired on real barren Mojave bajada; the straight-line detector
  does not. The `test_terrain_blocks_score_low` case captures that failure.
"""
import numpy as np

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
    img[r < radius] = 0.15
    img[np.abs(r - radius) < 1.5] = 0.95
    return img


def _draw_field_grid(size=250, step=45):
    img = np.full((size, size), 0.5, "float32")
    for k in range(0, size, step):
        img[k:k + 3, :] = 0.9
        img[:, k:k + 3] = 0.9
    for i, r0 in enumerate(range(0, size, step)):
        for j, c0 in enumerate(range(0, size, step)):
            img[r0 + 3:r0 + step, c0 + 3:c0 + step] = 0.25 if (i + j) % 2 else 0.78
    return img


def _desert_noise(size=250, seed=0):
    """Dendritic random-walk wash in fine texture — natural desert drainage."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.06, (size, size)).astype("float32")
    x = size / 2
    for y in range(size):
        x = np.clip(x + rng.normal(0, 2.4), 3, size - 4)
        base[y, int(x) - 1:int(x) + 2] = 0.2
    return np.clip(base, 0, 1)


def _terrain_blocks(size=250, seed=0):
    """Strong tonal blocks with NO straight edges — mountains + playa. This is
    the real-data failure mode: high tonal contrast but not agriculture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype("float32")
    # smooth low-frequency relief (a couple of gaussian 'mountains' + a 'playa')
    img = np.full((size, size), 0.5, "float32")
    for _ in range(4):
        cy, cx = rng.uniform(0, size, 2)
        s = rng.uniform(35, 70)
        amp = rng.uniform(-0.35, 0.35)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
    img += rng.normal(0, 0.03, (size, size)).astype("float32")
    return np.clip(img, 0, 1)


def test_center_pivot_runs_and_bounded():
    # experimental detector: just ensure it returns a valid [0,1] score
    s = center_pivot_score(_draw_circle(radius=30), px_per_m=0.1)
    assert 0.0 <= s <= 1.0


def test_field_grid_high():
    assert field_regularity_score(_draw_field_grid()) >= AGRICULTURE_THRESHOLD


def test_dendritic_wash_low():
    for seed in range(6):
        assert field_regularity_score(_desert_noise(seed=seed)) < AGRICULTURE_THRESHOLD


def test_terrain_blocks_score_low():
    # THE real-data regression: tonal blocks (mountains/playa) must NOT read as
    # agriculture. Straight-line count stays low because relief edges are curved.
    for seed in range(6):
        assert agriculture_score(_terrain_blocks(seed=seed)) < AGRICULTURE_THRESHOLD


def test_agriculture_separates_fields_from_desert():
    fields = agriculture_score(_draw_field_grid())
    assert fields >= AGRICULTURE_THRESHOLD
    for seed in range(6):
        assert agriculture_score(_desert_noise(seed=seed)) < AGRICULTURE_THRESHOLD
        assert agriculture_score(_terrain_blocks(seed=seed)) < AGRICULTURE_THRESHOLD


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
