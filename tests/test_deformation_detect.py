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
