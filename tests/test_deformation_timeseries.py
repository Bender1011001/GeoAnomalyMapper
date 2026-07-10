"""Synthetic-signal unit tests for deformation_intel.timeseries.

Each test plants a signal with known parameters and asserts the analytics
recover them within tolerance. This is the trust anchor for the detector.
"""
import numpy as np
import pytest

from deformation_intel.timeseries import (
    fit_pixel,
    fit_cube,
    time_to_threshold,
    stitch_reference_eras,
)


def _times(n=120, years=9.5, start=2016.5):
    return start + np.linspace(0, years, n)


def test_recovers_linear_velocity():
    t = _times()
    true_v = -0.08  # 8 cm/yr subsidence
    y = true_v * (t - t[0])
    fit = fit_pixel(t, y, seasonal=False)
    assert fit.velocity_m_yr == pytest.approx(true_v, abs=2e-3)
    assert abs(fit.accel_m_yr2) < 1e-3
    assert fit.r2 > 0.999


def test_recovers_acceleration():
    t = _times()
    tc = t - t.mean()
    true_accel = -0.02  # rate worsens 2 cm/yr each year
    y = 0.5 * true_accel * tc**2  # pure quadratic
    fit = fit_pixel(t, y, seasonal=False)
    assert fit.accel_m_yr2 == pytest.approx(true_accel, abs=3e-3)


def test_recovers_seasonal_amplitude_without_biasing_trend():
    t = _times()
    true_v = -0.05
    amp = 0.03
    y = true_v * (t - t[0]) + amp * np.sin(2 * np.pi * t)
    fit = fit_pixel(t, y, seasonal=True)
    assert fit.velocity_m_yr == pytest.approx(true_v, abs=5e-3)
    assert fit.seasonal_amp_m == pytest.approx(amp, abs=5e-3)


def test_seasonal_signal_not_mistaken_for_trend_when_seasonal_enabled():
    t = _times()
    y = 0.04 * np.sin(2 * np.pi * t)  # purely seasonal, zero net trend
    fit = fit_pixel(t, y, seasonal=True)
    assert abs(fit.velocity_m_yr) < 8e-3
    assert fit.seasonal_amp_m == pytest.approx(0.04, abs=6e-3)


def test_nan_robustness():
    t = _times()
    y = -0.06 * (t - t[0])
    y[::7] = np.nan  # drop ~14% of epochs
    fit = fit_pixel(t, y, seasonal=False)
    assert fit.velocity_m_yr == pytest.approx(-0.06, abs=3e-3)
    assert fit.n_obs < len(t)


def test_underdetermined_returns_nan():
    t = _times(n=5)
    y = -0.06 * (t - t[0])
    fit = fit_pixel(t, y, min_obs=12)
    assert np.isnan(fit.velocity_m_yr)
    assert fit.n_obs == 5


def test_breakpoint_detects_regime_change():
    t = _times(n=140)
    knot = 2021.0
    pre, post = -0.01, -0.15  # slow then fast subsidence
    y = np.where(t < knot, pre * (t - t[0]),
                 pre * (knot - t[0]) + post * (t - knot))
    fit = fit_pixel(t, y, seasonal=False)
    assert fit.breakpoint_gain > 0.25
    assert fit.breakpoint_year == pytest.approx(knot, abs=0.6)
    assert fit.post_rate_m_yr < fit.pre_rate_m_yr  # accelerated subsidence


def test_no_breakpoint_on_clean_linear():
    t = _times(n=140)
    y = -0.05 * (t - t[0])
    fit = fit_pixel(t, y, seasonal=False)
    assert np.isnan(fit.breakpoint_year)  # no spurious regime change


def test_time_to_threshold_linear():
    # at -0.10 m/yr, from 0, reach -0.30 m in 3 years
    ttt = time_to_threshold(0.0, -0.10, 0.0, -0.30)
    assert ttt == pytest.approx(3.0, abs=1e-6)


def test_time_to_threshold_accelerating_is_sooner_than_linear():
    lin = time_to_threshold(0.0, -0.10, 0.0, -0.50)
    acc = time_to_threshold(0.0, -0.10, -0.05, -0.50)
    assert acc < lin  # acceleration reaches threshold sooner


def test_time_to_threshold_never_when_moving_away():
    assert time_to_threshold(0.0, +0.10, 0.0, -0.30) == np.inf


def test_fit_cube_matches_fit_pixel_and_shapes():
    t = _times(n=80)
    H, W = 4, 5
    rng = np.random.default_rng(0)
    disp = np.empty((len(t), H, W))
    truth_v = np.linspace(-0.12, 0.02, H * W).reshape(H, W)
    for i in range(H):
        for j in range(W):
            disp[:, i, j] = truth_v[i, j] * (t - t[0]) + 0.002 * rng.standard_normal(len(t))
    out = fit_cube(t, disp, seasonal=False)
    assert out["velocity_m_yr"].shape == (H, W)
    assert np.allclose(out["velocity_m_yr"], truth_v, atol=5e-3)


def test_fit_cube_handles_nan_pixels():
    t = _times(n=60)
    disp = np.full((len(t), 3, 3), np.nan)
    disp[:, 0, 0] = -0.07 * (t - t[0])
    out = fit_cube(t, disp, seasonal=False, min_obs=12)
    assert out["velocity_m_yr"][0, 0] == pytest.approx(-0.07, abs=3e-3)
    assert np.isnan(out["velocity_m_yr"][1, 1])


# ---- OPERA era stitching ----

def _make_two_era_series():
    """Constant -0.10 m/yr subsidence split into two OPERA-style eras that each
    reset to zero at their reference date, sharing the boundary epoch."""
    rate = -0.10
    era1_dates = ["20200101", "20200201", "20200301", "20200401"]  # ref 20200101
    era2_dates = ["20200401", "20200501", "20200601", "20200701"]  # ref 20200401 (shared boundary)
    all_dates = era1_dates + era2_dates
    refs = ["20200101"] * 4 + ["20200401"] * 4

    def frac(d):
        # crude month fraction of a year for the synthetic
        return (int(d[4:6]) - 1) / 12.0

    H = W = 2
    disp = np.zeros((len(all_dates), H, W))
    for i, d in enumerate(all_dates):
        era_ref = refs[i]
        dt_years = frac(d) - frac(era_ref)
        disp[i] = rate * dt_years  # relative to each era's own reference
    return disp, all_dates, refs, rate


def test_stitch_makes_eras_continuous_no_fabricated_jump():
    disp, dates, refs, rate = _make_two_era_series()
    stitched = stitch_reference_eras(disp, dates, refs)
    # The stitched cumulative at each date should equal rate*(months since 20200101).
    def frac(d):
        return (int(d[4:6]) - 1) / 12.0
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    expected = np.array([rate * frac(dates[i]) for i in order])
    got = np.array([stitched[i, 0, 0] for i in order])
    # No discontinuity at the era boundary (2020-04): monotone, matches expected.
    assert np.allclose(got, expected, atol=1e-9)


def test_stitch_preserves_single_era():
    t_dates = ["20200101", "20200201", "20200301"]
    refs = ["20200101"] * 3
    disp = np.zeros((3, 1, 1))
    disp[:, 0, 0] = [0.0, -0.01, -0.02]
    stitched = stitch_reference_eras(disp, t_dates, refs)
    assert np.allclose(stitched[:, 0, 0], [0.0, -0.01, -0.02])
