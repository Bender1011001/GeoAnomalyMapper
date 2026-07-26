"""Unit tests for the frozen candidate-screening rule."""
from deformation_intel.screen import (
    cluster_sizes,
    min_plant_distance_km,
    screen_candidates,
)


def _c(lat, lon, **kw):
    d = {"lat": lat, "lon": lon, "is_localized": True, "rate_reliable": True,
         "classification": "accelerating_subsidence", "accel_cm_yr2": -1.5,
         "area_km2": 0.008, "void_likelihood": 1.0,
         "peak_velocity_cm_yr": -4.0}
    d.update(kw)
    return d


def test_cluster_sizes_counts_neighbors():
    # 3 bowls within ~5 km + 1 isolated 200 km away
    cands = [_c(32.00, -103.00), _c(32.02, -103.01), _c(32.01, -103.03),
             _c(34.00, -101.00)]
    sizes = cluster_sizes(cands, radius_km=15.0)
    assert sizes[:3] == [3, 3, 3]
    assert sizes[3] == 1


def test_min_plant_distance():
    d = min_plant_distance_km(32.0, -103.0, [(32.05, -103.0), (40.0, -100.0)])
    assert 5.0 < d < 6.0     # ~5.5 km
    assert min_plant_distance_km(0, 0, []) == float("inf")


def test_isolated_clean_candidate_survives():
    cands = [_c(34.5591, -116.7685)]   # Mojave-lead-like, isolated
    res = screen_candidates(cands)
    assert len(res["survivors"]) == 1
    assert not res["rejected"]


def test_cluster_is_vetoed():
    # 6 bowls in a tight cluster -> all rejected as in_cluster (regional process)
    cands = [_c(39.6 + 0.01 * i, -118.5) for i in range(6)]
    res = screen_candidates(cands, cluster_max=5, cluster_radius_km=15.0)
    assert not res["survivors"]
    assert all(c["reject_reason"] == "in_cluster" for c in res["rejected"])


def test_geothermal_proximity_vetoes():
    cands = [_c(33.27, -115.60)]        # Salton-like
    res = screen_candidates(cands, plants=[(33.30, -115.62)])  # ~4 km away
    assert not res["survivors"]
    assert res["rejected"][0]["reject_reason"] == "near_power_plant"


def test_cultivated_vetoes():
    cands = [_c(33.0, -113.3)]          # Gila-Bend-like
    res = screen_candidates(cands, cultivated_fn=lambda c: True)
    assert res["rejected"][0]["reject_reason"] == "cultivated"


def test_regional_and_large_and_lowvl_rejected():
    reg = _c(39.0, -117.0, classification="regional_subsidence")
    big = _c(35.0, -110.0, area_km2=0.5)
    low = _c(36.0, -111.0, void_likelihood=0.4)
    res = screen_candidates([reg, big, low])
    reasons = {c["reject_reason"] for c in res["rejected"]}
    assert reasons == {"regional", "too_large", "low_void_likelihood"}
    assert not res["survivors"]


def test_survivors_ranked_by_severity():
    strong = _c(34.0, -116.0, accel_cm_yr2=-2.0, peak_velocity_cm_yr=-6.0)
    weak = _c(31.0, -103.0, accel_cm_yr2=-1.0, peak_velocity_cm_yr=-1.5)
    res = screen_candidates([weak, strong])
    assert res["survivors"][0]["lat"] == 34.0   # strongest first


def test_screen_annotations_present():
    res = screen_candidates([_c(34.5, -116.7)])
    s = res["survivors"][0]["screen"]
    assert "cluster_size" in s and "nearest_plant_km" in s and "cultivated" in s
