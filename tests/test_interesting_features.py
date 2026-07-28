"""Synthetic unit tests for interesting_intel.features (no network).

Real-imagery counterparts live in test_interesting_real_chips.py — per the
project rule that synthetic tests alone are NOT validation.
"""
import numpy as np
import pytest

from interesting_intel import features as F


def _grid_img(size=250, step=45):
    img = np.full((size, size), 0.5, "float32")
    for k in range(0, size, step):
        img[k:k + 3, :] = 0.9
        img[:, k:k + 3] = 0.9
    return img


def _noise(size=250, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.08, (size, size)).astype("float32")
    from scipy.ndimage import gaussian_filter
    return np.clip(gaussian_filter(base, 1.5), 0, 1)


def _circle_img(size=250, radius=40):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = np.full((size, size), 0.5, "float32")
    img[np.abs(r - radius) < 2] = 0.95
    return img


def test_chip_ok_gates():
    assert F.chip_ok(_noise())
    assert not F.chip_ok(np.full((100, 100), np.nan, "float32"))
    assert not F.chip_ok(np.full((100, 100), 0.7, "float32"))   # constant
    assert not F.chip_ok(np.zeros((10, 10), "float32"))          # too small
    half_nan = _noise()
    half_nan[:, :60] = np.nan
    assert F.chip_ok(half_nan, min_finite=0.6)
    mostly_nan = _noise()
    mostly_nan[:, :200] = np.nan
    assert not F.chip_ok(mostly_nan)


def test_straightedge_grid_vs_noise():
    assert F.straightedge_score(_grid_img()) > 0.3
    assert F.straightedge_score(_noise()) < 0.1


def test_circle_score_circle_vs_noise():
    assert F.circle_score(_circle_img()) > F.circle_score(_noise()) + 0.1


def test_symmetry_mirror_vs_noise():
    half = _noise(seed=3)[:, :125]
    mirrored = np.concatenate([half, half[:, ::-1]], axis=1)
    assert F.symmetry_score(mirrored) > 0.5
    assert F.symmetry_score(_noise(seed=4)) < 0.3


def test_center_contrast_sign():
    bright = _noise(seed=5).copy()
    yy, xx = np.mgrid[0:250, 0:250]
    core = np.hypot(yy - 125, xx - 125) < 25
    bright[core] += 0.5
    dark = _noise(seed=5).copy()
    dark[core] -= 0.5
    assert F.center_contrast(bright) > 1.0
    assert F.center_contrast(dark) < -1.0


def test_texture_vector_shape_and_finite():
    v = F.texture_vector(_noise())
    assert v.shape == (F.TEXTURE_DIMS,)
    assert np.isfinite(v).all()


def test_novelty_outlier_vs_member():
    rng = np.random.default_rng(0)
    bg = [rng.normal(0, 0.05, F.TEXTURE_DIMS).astype("float32")
          for _ in range(20)]
    member = bg[0] + 0.01
    outlier = bg[0] + 3.0
    assert F.novelty_score(outlier, bg[1:]) > F.novelty_score(member, bg[1:]) + 2
    assert np.isnan(F.novelty_score(member, bg[:5]))   # too few backgrounds


def test_local_outlier_flat_plain_beats_mountains():
    """The same bump must score far higher on a flat plain than amid rough
    terrain — novelty relative to surroundings, the module's core contract."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:200, 0:200]
    bump = 5.0 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 4.0 ** 2))
    flat = bump + rng.normal(0, 0.05, (200, 200))
    rough = bump + rng.normal(0, 3.0, (200, 200))
    z_flat = F.local_outlier(flat)[100, 100]
    z_rough = F.local_outlier(rough)[100, 100]
    assert z_flat > 3 * max(z_rough, 0.1)


def test_local_outlier_preserves_nan():
    img = np.ones((100, 100), "float32")
    img[10, 10] = np.nan
    out = F.local_outlier(img)
    assert np.isnan(out[10, 10])
    assert np.isfinite(out[50, 50])


def test_geometry_score_ordering():
    assert F.geometry_score(_grid_img()) > F.geometry_score(_noise()) + 0.15


def _rings_img(size=250, period=18):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = 0.5 + 0.25 * np.sin(2 * np.pi * r / period)
    rng = np.random.default_rng(2)
    return (img + rng.normal(0, 0.03, (size, size))).astype("float32")


def test_radiality_rings_vs_noise_and_stripes():
    assert F.radiality_score(_rings_img()) > 0.4
    assert F.radiality_score(_noise(seed=6)) < 0.15
    stripes = np.tile(np.sin(np.arange(250) / 6), (250, 1)).astype("float32")
    assert F.radiality_score(stripes) < 0.3


def _faint_line_img(size=250, amp=0.06, seed=8):
    """A single long straight line at ~1.5x the noise sigma — invisible to
    canny, findable by the radon integral."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.5, 0.04, (size, size)).astype("float32")
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.abs((yy - size / 2) * 0.4 - (xx - size / 2) * 0.9) / np.hypot(0.4, 0.9)
    img[d < 1.2] += amp
    return img


def test_faint_line_vs_noise():
    assert F.faint_line_score(_faint_line_img()) > \
        F.faint_line_score(_noise(seed=9)) + 0.2
    assert F.straightedge_score(_faint_line_img()) == 0.0   # canny misses it
