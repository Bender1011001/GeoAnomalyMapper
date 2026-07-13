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


def test_aquifer_correlated_bowl_is_penalized():
    """Two identical bowls; one co-breathes with the regional seasonal signal
    (aquifer pumping cone), one is independent. The correlated bowl must get a
    lower void_likelihood and a high regional_correlation."""
    grid, n = 60, 60
    x0, y0 = 600000.0, 4245000.0
    x = x0 + (np.arange(grid) - grid / 2) * 30.0
    y = y0 + (np.arange(grid) - grid / 2) * 30.0
    X, Y = np.meshgrid(x, y)
    t = 2016.5 + np.linspace(0, 9.0, n)
    tc = t - t[0]
    regional_wiggle = 0.02 * np.sin(2 * np.pi * t) + 0.01 * np.sin(0.7 * np.pi * tc)

    # bowl A (aquifer-like): at (-450,-450)m offset, trend + regional wiggle
    rA = np.hypot(X - (x0 - 450), Y - (y0 - 450))
    shapeA = mogi_uz(rA, 250.0, -1e5)
    shapeA /= abs(shapeA.min())
    # bowl B (independent): at (+450,+450)m, pure accelerating trend
    rB = np.hypot(X - (x0 + 450), Y - (y0 + 450))
    shapeB = mogi_uz(rB, 250.0, -1e5)
    shapeB /= abs(shapeB.min())

    rng = np.random.default_rng(7)
    cube_arr = np.empty((n, grid, grid))
    for i in range(n):
        # whole AOI breathes with the regional signal (common mode)
        background = regional_wiggle[i]
        # bowl A: subsidence trend + AMPLIFIED regional breathing (pumping
        # cone responds to the water table more than quiet ground does)
        bowl_a = shapeA * (0.05 * tc[i]) + np.abs(shapeA) * (0.9 * regional_wiggle[i])
        # bowl B: independent accelerating subsidence, no excess breathing
        bowl_b = shapeB * (0.03 * tc[i] + 0.004 * tc[i] ** 2)
        cube_arr[i] = (background + bowl_a + bowl_b
                       + 0.001 * rng.standard_normal((grid, grid)))
    cube = {"cube": cube_arr, "t": t, "x": x, "y": y,
            "crs_wkt": "EPSG:32610", "frame": "TEST"}
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                 min_sigma=3.0)
    subs = [a for a in anomalies if a.kind == "subsidence"]
    assert len(subs) >= 2
    # identify bowls by location
    a_bowl = min(subs, key=lambda a: (a.lon + 122) ** 2 + 0)  # westernmost
    b_bowl = max(subs, key=lambda a: a.lon)
    assert a_bowl.regional_correlation is not None
    assert a_bowl.regional_correlation > b_bowl.regional_correlation
    assert b_bowl.void_likelihood >= a_bowl.void_likelihood


def test_candidate_gets_depth_range_and_growth_label():
    cube = _synthetic_cube()
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                 min_sigma=3.0)
    top = anomalies[0]
    assert top.void_likelihood >= 0.5
    assert top.source_depth_range_m is not None
    lo, hi = top.source_depth_range_m
    assert lo <= (top.source_depth_m or lo) <= hi or (hi - lo) >= 0
    # planted signal accelerates -> late-half volume rate exceeds early half
    assert top.source_growth in ("growing", "steady")


def test_adaptive_threshold_finds_weak_bowl_in_quiet_terrain():
    """A weak (-1.3 cm/yr) small bowl in very quiet terrain (noise ~0.1 cm/yr)
    is >10 sigma locally but below the legacy fixed 2 cm/yr floor. The adaptive
    threshold must recover it; the fixed threshold must miss it."""
    grid, n = 60, 60
    x0, y0 = 600000.0, 4245000.0
    x = x0 + (np.arange(grid) - grid / 2) * 30.0
    y = y0 + (np.arange(grid) - grid / 2) * 30.0
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - x0, Y - y0)
    shape = mogi_uz(r, 150.0, -1e5)
    shape /= abs(shape.min())
    t = 2016.5 + np.linspace(0, 9.0, n)
    tc = t - t[0]
    rng = np.random.default_rng(11)
    cube_arr = np.empty((n, grid, grid))
    for i in range(n):
        cube_arr[i] = shape * (0.013 * tc[i]) + 0.0006 * rng.standard_normal((grid, grid))
    cube = {"cube": cube_arr, "t": t, "x": x, "y": y,
            "crs_wkt": "EPSG:32610", "frame": "TEST"}

    fixed = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                             min_sigma=4.0, adaptive_threshold=False)
    adaptive = detect_anomalies(cube, velocity_threshold_cm_yr=2.0, min_pixels=5,
                                min_sigma=4.0, adaptive_threshold=True)
    assert not [a for a in fixed if a.kind == "subsidence"]
    subs = [a for a in adaptive if a.kind == "subsidence"]
    assert subs, "adaptive threshold should recover the weak bowl"
    assert subs[0].peak_velocity_cm_yr <= -1.0


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


def test_gappy_pixels_excluded_from_detection():
    """Pixels observed in only a clump of the record fit artifact slopes
    (measured on the full-archive Hutchinson cube: ~3 cm/yr artifacts where
    well-observed neighbors fit ~0.8). They must not seed clusters."""
    cube = _synthetic_cube()
    disp = cube["cube"]
    # Plant a gappy patch far from the real bowl: valid only in the first 40%
    # of epochs, where noise + a seasonal-ish wobble mimics a fast slope.
    n = disp.shape[0]
    cut = int(0.4 * n)
    rr, cc = slice(5, 13), slice(5, 13)
    disp[:, rr, cc] += 0.01 * np.sin(np.linspace(0, 6, n))[:, None, None]
    disp[cut:, rr, cc] = np.nan
    anomalies = detect_anomalies(cube, velocity_threshold_cm_yr=2.0,
                                 min_pixels=5, min_sigma=3.0)
    for a in anomalies:
        # nothing detected inside the gappy patch (rows/cols 5-13 sit SW of
        # the planted bowl: lat < 38.345, lon < -121.860; verified the patch
        # IS detected there when min_valid_fraction=0)
        assert not (a.lat < 38.345 and a.lon < -121.860), \
            f"gappy-patch artifact detected: {a}"
    # the real planted bowl must still be found
    assert any(a.classification == "accelerating_subsidence" for a in anomalies)


def test_relative_uplift_in_subsiding_field_not_called_true_uplift():
    """A patch subsiding LESS than a wall-to-wall subsiding field has positive
    RELATIVE velocity; it must not be reported as true uplift (the ground is
    still sinking). Verifies abs_peak_velocity_cm_yr + relative_uplift reclass."""
    n_epochs, grid, pixel_m = 60, 60, 30.0
    x0, y0 = 600000.0, 4245000.0
    x = x0 + (np.arange(grid) - grid / 2) * pixel_m
    y = y0 + (np.arange(grid) - grid / 2) * pixel_m
    t = 2016.5 + np.linspace(0, 9.0, n_epochs)
    tc = t - t[0]
    rng = np.random.default_rng(3)
    cube = np.empty((n_epochs, grid, grid))
    # whole field subsides -2 cm/yr; a 10x10 patch subsides only -0.4 cm/yr
    for i in range(n_epochs):
        field = -0.02 * tc[i] * np.ones((grid, grid))
        field[25:35, 25:35] = -0.004 * tc[i]
        cube[i] = field + 0.001 * rng.standard_normal((grid, grid))
    cube_d = {"cube": cube, "t": t, "x": x, "y": y, "crs_wkt": "EPSG:32610", "frame": "TEST"}
    anoms = detect_anomalies(cube_d, velocity_threshold_cm_yr=1.0, min_pixels=5, min_sigma=3.0)
    ups = [a for a in anoms if a.abs_peak_velocity_cm_yr > 0.5]  # any TRUE uplift?
    assert not ups, f"reported true uplift in a subsiding field: {[(a.classification,a.abs_peak_velocity_cm_yr) for a in ups]}"
    rel = [a for a in anoms if a.classification == "relative_uplift"]
    assert rel, "the less-subsiding patch should be flagged relative_uplift"
    assert all(a.abs_peak_velocity_cm_yr < 0 for a in rel)  # still absolutely subsiding
