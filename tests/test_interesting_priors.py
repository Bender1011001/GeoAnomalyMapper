"""Unit tests for interesting_intel.priors pure functions (no network)."""
import numpy as np

from interesting_intel import priors as P


BBOX = (-100.2, 38.0, -100.0, 38.2)     # ~17x22 km


def test_grid_candidates_spacing():
    cands = P.grid_candidates(BBOX, spacing_m=2000.0)
    assert 50 <= len(cands) <= 130
    lats = sorted({c["lat"] for c in cands})
    assert abs((lats[1] - lats[0]) * P.M_PER_DEG_LAT - 2000.0) < 50
    for c in cands[:5]:
        assert BBOX[1] < c["lat"] < BBOX[3]
        assert BBOX[0] < c["lon"] < BBOX[2]


def test_rc_to_latlon_corners():
    lat, lon = P.rc_to_latlon(0, 0, BBOX, 100, 100)
    assert abs(lat - BBOX[3]) < 0.01 and abs(lon - BBOX[0]) < 0.01
    lat, lon = P.rc_to_latlon(99, 99, BBOX, 100, 100)
    assert abs(lat - BBOX[1]) < 0.01 and abs(lon - BBOX[2]) < 0.01


def test_anomaly_peaks_finds_planted_peaks():
    z = np.zeros((200, 200), "float32")
    z[50, 60] = 8.0
    z[150, 140] = 6.0
    z[52, 62] = 5.0          # shoulder of peak 1 — must be suppressed
    peaks = P.anomaly_peaks(z, BBOX, "test", min_z=4.0, min_sep_px=6)
    assert len(peaks) == 2
    assert peaks[0]["priors"]["test"] == 8.0
    lat, lon = P.rc_to_latlon(50, 60, BBOX, 200, 200)
    assert abs(peaks[0]["lat"] - lat) < 1e-4
    assert abs(peaks[0]["lon"] - lon) < 1e-4


def test_anomaly_peaks_threshold_and_nan():
    z = np.full((100, 100), np.nan, "float32")
    z[10, 10] = 3.0
    assert P.anomaly_peaks(z, BBOX, "t", min_z=3.5) == []
    z[20, 20] = 4.0
    assert len(P.anomaly_peaks(z, BBOX, "t", min_z=3.5)) == 1


def test_merge_candidates_folds_priors():
    a = {"lat": 38.1, "lon": -100.1, "priors": {"dem_relief": 6.0},
         "score1": 6.0}
    b = {"lat": 38.1001, "lon": -100.1001, "priors": {"spectral": 4.0},
         "score1": 4.0}
    far = {"lat": 38.15, "lon": -100.15, "priors": {"spectral": 5.0},
           "score1": 5.0}
    merged = P.merge_candidates([a, b, far], radius_m=300.0)
    assert len(merged) == 2
    top = max(merged, key=lambda c: c["score1"])
    assert set(top["priors"]) == {"dem_relief", "spectral"}
    assert top["score1"] == 7.0          # 6.0 + 1.0 multi-prior bonus
    assert max(c["score1"] for c in merged if c is not top) == 5.0


def test_merge_candidates_grid_gets_no_bonus():
    g = {"lat": 38.1, "lon": -100.1, "priors": {"grid": 0.0}, "score1": 0.0}
    merged = P.merge_candidates([g], radius_m=300.0)
    assert merged[0]["score1"] == 0.0
