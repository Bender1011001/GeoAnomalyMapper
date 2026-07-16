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


class TestCorona:
    def test_fit_affine_exact_recovery(self):
        from archaeo_intel.corona import apply_affine, fit_affine
        # synthetic truth: lon = 39 + 1e-3*col - 2e-4*row ; lat = 36 - 9e-4*row
        cols_rows = [(0, 0), (500, 0), (0, 500), (500, 500), (250, 100)]
        geo = [(39 + 1e-3*c - 2e-4*r, 36 - 9e-4*r) for c, r in cols_rows]
        M = fit_affine(cols_rows, geo)
        lon, lat = apply_affine(M, 123, 456)
        assert abs(lon - (39 + 0.123 - 0.0912)) < 1e-9
        assert abs(lat - (36 - 0.4104)) < 1e-9

    def test_fit_affine_needs_three_points(self):
        from archaeo_intel.corona import fit_affine
        with pytest.raises(ValueError):
            fit_affine([(0, 0), (1, 1)], [(39, 36), (39.1, 36.1)])

    @staticmethod
    def _truth_ll(c, r):
        # gently nonlinear truth (panoramic-like): quadratic terms present
        lon = 39.0 + 1e-3*c - 2e-4*r + 3e-8*c*c
        lat = 36.0 - 9e-4*r + 2e-8*c*r
        return lon, lat

    @staticmethod
    def _truth_ll_affine(c, r):
        # exact affine truth: inverse is exact, so geometry tests are clean
        lon = 39.0 + 1e-3*c - 2e-4*r
        lat = 36.0 - 9e-4*r + 1e-4*c
        return lon, lat

    def _gcps(self, pts, truth=None):
        from archaeo_intel.corona import add_gcp
        truth = truth or self._truth_ll
        g = None
        for c, r in pts:
            lon, lat = truth(c, r)
            g = add_gcp(g, c, r, lon, lat)
        return g

    def test_order2_fit_beats_affine_on_panoramic_truth(self):
        from archaeo_intel.corona import fit_report
        pts = [(c, r) for c in (0, 400, 800, 1200) for r in (0, 300, 600)]
        gcps = self._gcps(pts)
        rep1 = fit_report(gcps, order=1)
        rep2 = fit_report(gcps, order=2)
        assert rep2["rms_m"] < rep1["rms_m"] / 5
        assert rep2["rms_m"] < 1.0            # exact family -> ~0 residual

    def test_fit_report_flags_bad_gcp(self):
        from archaeo_intel.corona import fit_report
        # enough well-spread GCPs that one outlier's leverage is small
        pts = [(c, r) for c in (0, 300, 600, 900, 1200)
               for r in (0, 250, 500)]
        gcps = self._gcps(pts, truth=self._truth_ll_affine)
        gcps[7]["lon"] += 0.01                # ~900 m mis-click
        rep = fit_report(gcps, order=1)
        worst = int(np.argmax(rep["residuals_m"]))
        assert worst == 7
        assert rep["residuals_m"][7] > 3 * np.median(rep["residuals_m"])

    def test_gcp_roundtrip_persistence(self, tmp_path):
        from archaeo_intel.corona import load_gcps, save_gcps
        gcps = self._gcps([(0, 0), (10, 20), (30, 5)])
        save_gcps(tmp_path / "g.json", gcps, "http://example/strip.ntf")
        back, url = load_gcps(tmp_path / "g.json")
        assert back == gcps and url == "http://example/strip.ntf"

    def test_warp_to_grid_synthetic_roundtrip(self):
        from archaeo_intel.corona import fit_transform, ll_to_px, warp_to_grid
        # synthetic 'strip': value encodes an affine ground pattern with a
        # bright square landmark at a known lon/lat
        H, W = 600, 900
        yy, xx = np.mgrid[:H, :W]
        strip = (0.1 * xx + 0.05 * yy).astype("float32")
        pts = [(0, 0), (800, 0), (0, 500), (800, 500), (400, 250),
               (200, 400), (700, 100)]
        gcps = self._gcps(pts, truth=self._truth_ll_affine)
        model = fit_transform(gcps, order=1)
        lm_lon, lm_lat = self._truth_ll_affine(450.0, 300.0)   # landmark truth
        c, r = ll_to_px(model, lm_lon, lm_lat)
        strip[int(r[0])-4:int(r[0])+5, int(c[0])-4:int(c[0])+5] = 999.0
        img, bbox = warp_to_grid(strip, gcps, res_m=30.0, order=1)
        # the bright landmark must appear within ~3 output px of its lon/lat;
        # use the centroid of the bright blob (argmax picks a corner cell)
        ys, xs = np.where(img > 900)
        assert len(ys) > 0
        iy, ix = float(ys.mean()), float(xs.mean())
        lon0, lat0, lon1, lat1 = bbox
        lon_hit = lon0 + (ix + 0.5) / img.shape[1] * (lon1 - lon0)
        lat_hit = lat1 - (iy + 0.5) / img.shape[0] * (lat1 - lat0)
        assert img[int(round(iy)), int(round(ix))] > 900
        assert abs(lon_hit - lm_lon) * 111e3 < 90       # < 3 px at 30 m
        assert abs(lat_hit - lm_lat) * 111e3 < 90
        assert np.isfinite(img[int(round(iy)), int(round(ix))])

    def test_warp_rejects_insufficient_gcps_for_order2(self):
        from archaeo_intel.corona import fit_transform
        gcps = self._gcps([(0, 0), (10, 0), (0, 10), (10, 10)])
        with pytest.raises(ValueError):
            fit_transform(gcps, order=2)


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
