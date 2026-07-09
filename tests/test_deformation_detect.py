"""End-to-end synthetic test for the unified detector.

Plant a Mogi subsidence source with a known accelerating time history into a
synthetic OPERA-like cube (UTM grid, decimal-year axis) and assert the detector
finds it, classifies it as accelerating subsidence, recovers a plausible source
depth, and produces a forecast.
"""
import numpy as np
import pytest

from deformation_intel.detect import detect_anomalies
from deformation_intel.sources import mogi_uz


def _synthetic_cube(n_epochs=60, grid=60, pixel_m=30.0):
    # UTM-like grid centered at a plausible California easting/northing.
    x0, y0 = 600000.0, 4245000.0
    x = x0 + (np.arange(grid) - grid / 2) * pixel_m
    y = y0 + (np.arange(grid) - grid / 2) * pixel_m
    X, Y = np.meshgrid(x, y)

    # Mogi source under grid center: depth 300 m, accelerating volume loss.
    depth, x_src, y_src = 300.0, x0, y0
    r = np.hypot(X - x_src, Y - y_src)
    spatial = mogi_uz(r, depth, -1.0e5)         # unit spatial bowl shape (peak ~ -X m)
    spatial = spatial / abs(spatial.min())      # normalize peak to -1

    t = 2016.5 + np.linspace(0, 9.0, n_epochs)
    tc = t - t[0]
    # accelerating cumulative amplitude: quadratic in time (cm at peak)
    amp_m = -(0.02 * tc + 0.004 * tc**2)        # up to ~ -0.5 m at end, accelerating
    rng = np.random.default_rng(1)
    cube = np.empty((n_epochs, grid, grid))
    for i in range(n_epochs):
        cube[i] = spatial * (-amp_m[i]) * -1.0 + 0.002 * rng.standard_normal((grid, grid))
        # (spatial is negative bowl; scale by cumulative amplitude)
    # simpler: cube[i] = spatial_bowl_normalized * cumulative_amplitude
    for i in range(n_epochs):
        cube[i] = spatial * abs(amp_m[i]) + 0.002 * rng.standard_normal((grid, grid))

    return {
        "cube": cube, "t": t, "x": x, "y": y,
        "crs_wkt": "EPSG:32610",  # UTM 10N (California)
        "frame": "TEST",
    }


def test_detects_and_classifies_accelerating_bowl():
    cube = _synthetic_cube()
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                 min_sigma=3.0)
    assert len(anomalies) >= 1
    top = anomalies[0]
    assert top.kind == "subsidence"
    assert top.classification == "accelerating_subsidence"
    assert top.peak_velocity_cm_yr < 0
    assert top.accel_cm_yr2 < 0            # accelerating
    assert top.confidence >= 0.5
    assert "collapse-precursor" in top.why
    # source depth recovered in a plausible band around the planted 300 m
    assert top.source_depth_m is not None
    assert 120 <= top.source_depth_m <= 700
    # forecast produced
    assert top.time_to_threshold_yr is not None


def test_quiet_cube_yields_no_anomalies():
    cube = _synthetic_cube()
    # replace signal with pure small noise -> no bowls above threshold
    rng = np.random.default_rng(2)
    cube["cube"] = 0.001 * rng.standard_normal(cube["cube"].shape)
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=3.0, min_pixels=6,
                                 min_sigma=5.0)
    assert anomalies == []


def test_localized_bowl_has_high_void_likelihood():
    cube = _synthetic_cube()
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                 min_sigma=3.0)
    top = anomalies[0]
    assert top.is_localized is True
    assert top.void_likelihood >= 0.6      # clean Mogi bowl -> void-plausible
    assert top.classification == "accelerating_subsidence"


def test_broad_regional_field_is_not_a_void():
    # A broad, monotonic subsidence RAMP across the whole AOI (aquifer sheet):
    # accelerating in time but no localized point source -> regional, low void.
    grid, n = 60, 50
    x0, y0 = 600000.0, 4245000.0
    x = x0 + (np.arange(grid)) * 30.0
    y = y0 + (np.arange(grid)) * 30.0
    t = 2016.5 + np.linspace(0, 9.0, n)
    tc = t - t[0]
    ramp = np.linspace(0, 1, grid)[None, :] * np.ones((grid, 1))  # smooth W->E ramp
    rng = np.random.default_rng(4)
    cube_arr = np.empty((n, grid, grid))
    for i in range(n):
        amp = -(0.03 * tc[i] + 0.004 * tc[i] ** 2)  # accelerating basin-wide
        cube_arr[i] = ramp * amp + 0.002 * rng.standard_normal((grid, grid))
    cube = {"cube": cube_arr, "t": t, "x": x, "y": y,
            "crs_wkt": "EPSG:32610", "frame": "TEST"}
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=6,
                                 min_sigma=3.0)
    # Whatever is flagged should be regional (poor point-source fit), not void.
    for a in anomalies:
        if a.kind == "subsidence":
            assert a.void_likelihood <= 0.4
            assert a.classification in ("regional_subsidence", "seasonal_dominated")


def test_context_sampler_attaches_annotations():
    cube = _synthetic_cube()
    called = {}

    def fake_groundwater(lat, lon):
        called["hit"] = (lat, lon)
        return 0.42

    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                 min_sigma=3.0,
                                 context_samplers={"groundwater_index": fake_groundwater})
    assert anomalies
    assert anomalies[0].context.get("groundwater_index") == pytest.approx(0.42)
    assert "hit" in called
