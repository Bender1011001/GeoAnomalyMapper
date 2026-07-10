"""Synthetic round-trip tests for Mogi source inversion.

Plant a Mogi source with known (depth, volume, epicenter), sample its surface
bowl, and assert the inversion recovers the parameters within tolerance.
"""
import numpy as np
import pytest

from deformation_intel.sources import (
    mogi_uz,
    depth_from_bowl_width,
    volume_from_peak,
    fit_mogi,
)


def _grid(n=41, extent_m=2000.0):
    xs = np.linspace(-extent_m, extent_m, n)
    X, Y = np.meshgrid(xs, xs)
    return X.ravel(), Y.ravel()


def test_forward_peak_and_falloff():
    d, dV = 300.0, -1.0e5
    peak = mogi_uz(0.0, d, dV)
    far = mogi_uz(3000.0, d, dV)
    assert peak < 0  # volume loss -> subsidence at center
    assert abs(far) < abs(peak) / 10  # decays with distance


def test_depth_from_bowl_width_roundtrip():
    d = 250.0
    # radius where uz is half the peak
    r_half = d * np.sqrt(2 ** (2 / 3) - 1)
    assert depth_from_bowl_width(r_half) == pytest.approx(d, rel=1e-6)


def test_volume_from_peak_roundtrip():
    d, dV = 400.0, -2.5e5
    peak = mogi_uz(0.0, d, dV)
    assert volume_from_peak(peak, d) == pytest.approx(dV, rel=1e-6)


@pytest.mark.parametrize("d,dV,x0,y0", [
    (300.0, -1.0e5, 0.0, 0.0),
    (500.0, -3.0e5, 400.0, -250.0),
    (150.0, -4.0e4, -600.0, 300.0),
])
def test_fit_recovers_parameters(d, dV, x0, y0):
    X, Y = _grid()
    r = np.hypot(X - x0, Y - y0)
    uz = mogi_uz(r, d, dV)
    fit = fit_mogi(X, Y, uz)
    assert fit.converged
    assert fit.depth_m == pytest.approx(d, rel=0.12)
    assert fit.volume_m3 == pytest.approx(dV, rel=0.15)
    assert fit.x0_m == pytest.approx(x0, abs=80.0)
    assert fit.y0_m == pytest.approx(y0, abs=80.0)
    assert fit.r2 > 0.98


def test_fit_robust_to_noise():
    d, dV = 350.0, -2.0e5
    X, Y = _grid(n=51)
    r = np.hypot(X, Y)
    rng = np.random.default_rng(3)
    uz = mogi_uz(r, d, dV) + 0.002 * rng.standard_normal(r.size)  # 2 mm noise
    fit = fit_mogi(X, Y, uz)
    assert fit.depth_m == pytest.approx(d, rel=0.25)
    assert np.sign(fit.volume_m3) == np.sign(dV)


def test_fit_robust_to_single_pixel_outlier_spike():
    """A lone corrupted pixel far from the bowl must not hijack the epicenter
    initialization (median-of-strong-decile init)."""
    d, dV, x0, y0 = 300.0, -2.0e5, 0.0, 0.0
    X, Y = _grid(n=41)
    r = np.hypot(X - x0, Y - y0)
    uz = mogi_uz(r, d, dV)
    # inject a spike 3x the true peak amplitude at a far corner
    spike_idx = int(np.argmax(np.hypot(X - 1800, Y - 1800) < 100))
    uz = uz.copy()
    uz[spike_idx] = 3.0 * uz.min()
    fit = fit_mogi(X, Y, uz)
    # epicenter must stay near the true bowl, not the corner spike
    assert np.hypot(fit.x0_m - x0, fit.y0_m - y0) < 400.0
    assert fit.depth_m == pytest.approx(d, rel=0.5)


def test_fit_underdetermined_returns_nan():
    fit = fit_mogi(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([-0.01, -0.02]))
    assert np.isnan(fit.depth_m)
    assert fit.n_points == 2
