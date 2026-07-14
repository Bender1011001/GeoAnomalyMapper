"""Offline unit tests for archaeo_intel (no network).

The radial_alignment tests encode two real bugs found during development:
(1) image-coord y-flip vs geographic bearing scrambled orientations, and
(2) the |cos| statistic had almost no dynamic range (uniform-random mean is
2/pi ~ 0.64). A metric change MUST keep the synthetic radial network detectable.
"""
import numpy as np
import pytest

from archaeo_intel.composite import bsi, local_anomaly, ndmi, scl_mask
from archaeo_intel.detect import (contrast_snr, prominence, radial_alignment,
                                  regional_roughness, ridge_response, robust_z,
                                  structure_orientation)
from archaeo_intel.catalog import classify_hit, nearest_site_km


GRID = (41.0, 36.61, 41.11, 36.72)   # ~12 km box, matches field usage


def synthetic_radial(size=700, n_rays=8, amp=2.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (size, size)).astype("float32")
    yy, xx = np.mgrid[:size, :size]
    cy = cx = size // 2
    for ang in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        t = (xx - cx) * dx + (yy - cy) * dy
        d = np.abs(-(xx - cx) * dy + (yy - cy) * dx)
        img[(t > 20) & (t < size * 0.47) & (d < 3)] += amp
    return img


class TestDetect:
    def test_prominence_finds_synthetic_mound(self):
        dem = np.zeros((200, 200), "float32")
        yy, xx = np.mgrid[:200, :200]
        dem += 10 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 8 ** 2))
        prom = prominence(dem, background_sigma_px=15)
        r, c = np.unravel_index(np.nanargmax(prom), prom.shape)
        assert abs(r - 100) <= 2 and abs(c - 100) <= 2
        assert prom[100, 100] > 5

    def test_regional_roughness_flat_vs_dunes(self):
        rng = np.random.default_rng(1)
        flat = rng.normal(0, 0.5, (300, 300))
        yy, xx = np.mgrid[:300, :300]
        # dune wavelength ~25 px, comparable to the 15 px roughness window
        dunes = 20 * np.sin(xx / 4.0) * np.cos(yy / 5.5)
        assert regional_roughness(flat) < 2
        assert regional_roughness(dunes) > 6

    def test_contrast_snr_sign_and_magnitude(self):
        anom = np.zeros((400, 400), "float32")
        rng = np.random.default_rng(2)
        anom += rng.normal(0, 0.1, anom.shape)
        anom[196:204, 196:204] += 3.0                 # hot core at grid center
        lat_c = (GRID[1] + GRID[3]) / 2
        lon_c = (GRID[0] + GRID[2]) / 2
        snr = contrast_snr(anom, GRID, 400, 400, lat_c, lon_c)
        assert snr > 5
        far = contrast_snr(anom, GRID, 400, 400, lat_c + 0.03, lon_c + 0.03)
        assert abs(far) < 3

    def test_robust_z_ignores_outliers(self):
        a = np.zeros(1000)
        a[:10] = 1e6
        z = robust_z(a)
        assert abs(np.median(z)) < 1e-6

    def test_radial_alignment_detects_synthetic_network(self):
        img = synthetic_radial()
        ridge = ridge_response(img)
        theta = structure_orientation(img)
        lat_c = (GRID[1] + GRID[3]) / 2
        lon_c = (GRID[0] + GRID[2]) / 2
        s_true = radial_alignment(ridge, theta, GRID, *img.shape[::-1],
                                  lat_c, lon_c, rmin_km=1.0, rmax_km=5.0)
        # far off-center scores must be materially lower
        rng = np.random.default_rng(3)
        nulls = []
        for _ in range(15):
            la = rng.uniform(GRID[1] + 0.02, GRID[3] - 0.02)
            lo = rng.uniform(GRID[0] + 0.02, GRID[2] - 0.02)
            if 111 * np.hypot(la - lat_c, (lo - lon_c) * 0.8) < 2.5:
                continue
            v = radial_alignment(ridge, theta, GRID, *img.shape[::-1], la, lo)
            if np.isfinite(v):
                nulls.append(v)
        assert s_true > 0.3, "true center must show strong radial alignment"
        assert s_true > np.max(nulls), "true center must beat all off-center nulls"

    def test_radial_alignment_null_on_random_noise(self):
        rng = np.random.default_rng(4)
        img = rng.normal(0, 1, (700, 700)).astype("float32")
        ridge = ridge_response(img)
        theta = structure_orientation(img)
        lat_c = (GRID[1] + GRID[3]) / 2
        lon_c = (GRID[0] + GRID[2]) / 2
        s = radial_alignment(ridge, theta, GRID, 700, 700, lat_c, lon_c)
        assert abs(s) < 0.3


class TestComposite:
    def test_indices_ranges(self):
        red = np.full((5, 5), 0.3); green = np.full((5, 5), 0.25)
        blue = np.full((5, 5), 0.2); nir = np.full((5, 5), 0.4)
        swir = np.full((5, 5), 0.35)
        b = bsi(red, green, blue, nir, swir)
        m = ndmi(nir, swir)
        assert np.all(np.abs(b) <= 1) and np.all(np.abs(m) <= 1)

    def test_scl_mask_keeps_soil_drops_cloud(self):
        scl = np.array([[5.0, 8.0], [4.0, 9.0]])
        good = scl_mask(scl)
        assert good.tolist() == [[True, False], [True, False]]

    def test_local_anomaly_removes_gradient(self):
        yy, xx = np.mgrid[:200, :200]
        img = (xx / 20.0).astype("float64")           # pure large-scale ramp
        img[100, 100] += 5.0                          # plus one local spike
        an = local_anomaly(img, background_sigma_px=25)
        assert an[100, 100] > 3
        assert abs(np.median(an)) < 0.3


class TestCatalog:
    def test_nearest_and_classify(self):
        cat_lat = np.array([36.6675, 36.70])
        cat_lon = np.array([41.0575, 41.10])
        assert nearest_site_km(36.6675, 41.0575, cat_lat, cat_lon) < 0.01
        label, d = classify_hit(36.6675, 41.0580, cat_lat, cat_lon)
        assert label == "in-catalog"
        label, d = classify_hit(36.60, 41.00, cat_lat, cat_lon)
        assert label == "novel-candidate" and d > 1.5


@pytest.mark.skipif(
    not (pytest.importorskip("pathlib").Path(__file__).parent.parent /
         "data" / "archaeo" / "menze_ur_catalog.npz").exists(),
    reason="catalog npz not present")
def test_menze_ur_tell_brak_sanity():
    from archaeo_intel.catalog import load_menze_ur
    lat, lon = load_menze_ur()
    assert len(lat) > 14000
    assert nearest_site_km(36.6675, 41.0575, lat, lon) < 0.2   # Tell Brak
