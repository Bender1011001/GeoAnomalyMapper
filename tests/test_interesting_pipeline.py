"""Offline tests for interesting_intel.pipeline: caching, ranking rules,
resumability. All network entry points are monkeypatched — the end-to-end
test runs the whole funnel on synthetic rasters and chips.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import interesting_intel.pipeline as pl
import interesting_intel.priors as P


# ------------------------------------------------------------------ helpers

def _synthetic_chip(lat, lon, half_m, px, interesting=False):
    rng = np.random.default_rng(abs(int(lat * 1e4)) + abs(int(lon * 1e4)))
    gray = rng.normal(100, 8, (px, px)).astype("float32")
    if interesting:                       # a bright rectangular compound
        gray[px // 4: px // 2, px // 4: px // 2] = 220.0
        gray[px // 4 + 10: px // 2 - 10, px // 4 + 10: px // 2 - 10] = 40.0
    rgb = np.clip(np.stack([gray] * 3, -1), 0, 255).astype("uint8")
    return {"gray": gray, "rgb": rgb, "source": "s2",
            "res_m": 2 * half_m / px}


# ------------------------------------------------------------------- ranking

def test_pct_rank_handles_nan():
    r = pl.pct_rank([1.0, float("nan"), 3.0, 2.0])
    assert r[1] == 0.5
    assert r[0] == 0.0 and r[2] == 1.0


def test_rank_candidates_demotes_but_never_deletes():
    base = {"score1": 5.0, "novelty": 4.0, "geometry": 0.5,
            "pad": 0.0, "agriculture": 0.0, "slope_deg": 1.0}
    clean = {**base, "lat": 1.0, "lon": 1.0}
    farmed = {**base, "lat": 2.0, "lon": 2.0, "agriculture": 0.9}
    padded = {**base, "lat": 3.0, "lon": 3.0, "pad": 0.8}
    ranked = pl.rank_candidates([clean, farmed, padded])
    assert len(ranked) == 3                       # nothing vetoed
    assert ranked[0]["lat"] == 1.0                # clean wins
    assert "cultivated" in [d for r in ranked for d in r["demoted_for"]]
    assert "industrial_pad" in [d for r in ranked for d in r["demoted_for"]]


def test_rank_candidates_no_slope_no_cultivated_demotion():
    """Agriculture score alone must NOT demote — mountains mimic field lines;
    the confound needs flat terrain confirmed (the Mojave lesson)."""
    a = {"lat": 1.0, "lon": 1.0, "score1": 5.0, "novelty": 4.0,
         "geometry": 0.5, "pad": 0.0, "agriculture": 0.9,
         "slope_deg": float("nan")}
    ranked = pl.rank_candidates([a])
    assert ranked[0]["demoted_for"] == []


def test_rank_candidates_osm_adjusts_mildly():
    base = {"score1": 5.0, "novelty": 4.0, "geometry": 0.5,
            "pad": 0.0, "agriculture": 0.0, "slope_deg": 1.0}
    unmapped = {**base, "lat": 1.0, "lon": 1.0, "osm_infra": 0.0}
    mapped = {**base, "lat": 2.0, "lon": 2.0, "osm_infra": 25.0}
    unknown = {**base, "lat": 3.0, "lon": 3.0}
    ranked = pl.rank_candidates([unmapped, mapped, unknown])
    assert [r["lat"] for r in ranked].index(1.0) < \
           [r["lat"] for r in ranked].index(2.0)
    assert len(ranked) == 3


def test_parse_focus_interest():
    assert pl.parse_focus_interest("... 4. interest: 3 because ...") == 3
    assert pl.parse_focus_interest("Interest rating (0-3): 2") == 2
    assert pl.parse_focus_interest("4. Interest 0–3\n- 0. ordinary") == 0
    assert pl.parse_focus_interest("nothing to see") == 0
    assert pl.parse_focus_interest("") == 0


def test_mpc_expiry_parsed_as_utc():
    """time.mktime parsed the MPC token expiry as LOCAL time, so on a
    UTC-negative machine tokens looked valid hours after they expired and
    long runs drowned in 403s (2026-07-27). The parse must be timezone-proof."""
    from archaeo_intel.data_access import parse_expiry_utc

    assert parse_expiry_utc("1970-01-01T00:02:00Z") == 120.0
    assert parse_expiry_utc("2026-07-27T12:00:00Z") == 1785153600.0


def test_parse_worth_glance():
    assert pl.parse_worth_glance("6. worth_a_glance 0-3: 3 — striking") == 3
    assert pl.parse_worth_glance("worth a glance: 2") == 2
    assert pl.parse_worth_glance("6. Worth_a_glance 0–3\n- 1, mildly") == 1
    assert pl.parse_worth_glance("no rating here") == 0


def test_slope_at_synthetic():
    dem = np.tile(np.arange(100, dtype="float32") * 30.0, (100, 1))  # 45 deg
    bbox = (0.0, 0.0, 0.03, 0.03)
    s = pl.slope_at(dem, bbox, 0.015, 0.015, px_m=30.0)
    assert 40.0 < s < 50.0
    assert np.isnan(pl.slope_at(None, bbox, 0.015, 0.015, 30.0))


# ------------------------------------------------------------------- caching

def test_chip_cache_fetch_once(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(lat, lon, half_m, px):
        calls.append((lat, lon))
        return _synthetic_chip(lat, lon, half_m, px)

    monkeypatch.setattr(pl, "fetch_chip_s2", fake_fetch)
    cache = pl.ChipCache(tmp_path / "cache")
    c1 = cache.get(10.0, 30.0, 500, 128, "s2")
    c2 = cache.get(10.0, 30.0, 500, 128, "s2")
    assert c1 is not None and c2 is not None
    assert len(calls) == 1                       # second hit came from disk
    assert cache.hits == 1
    np.testing.assert_allclose(c1["gray"], c2["gray"], rtol=1e-6)


def test_chip_cache_failure_marker(tmp_path, monkeypatch):
    calls = []

    def failing(lat, lon, half_m, px):
        calls.append(1)
        return None

    monkeypatch.setattr(pl, "fetch_chip_s2", failing)
    cache = pl.ChipCache(tmp_path / "cache")
    assert cache.get(10.0, 30.0, 500, 128, "s2") is None
    assert cache.get(10.0, 30.0, 500, 128, "s2") is None
    assert len(calls) == 1                       # .fail marker stopped retry
    assert cache.get(10.0, 30.0, 500, 128, "s2", retry_failed=True) is None
    assert len(calls) == 2


def test_fetch_many_aborts_on_total_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "fetch_chip_s2", lambda *a: None)
    cache = pl.ChipCache(tmp_path / "cache")
    pts = [{"lat": 10.0 + i * 0.01, "lon": 30.0} for i in range(30)]
    with pytest.raises(RuntimeError, match="all failed"):
        pl.fetch_many(cache, pts, 500, 128, "s2", workers=4,
                      log=lambda m: None)


# -------------------------------------------------------------- end to end

BBOX = (30.0, 10.0, 30.06, 10.06)      # outside CONUS -> s2 source


def _patch_world(monkeypatch):
    """Synthetic world: flat DEM with one bump, flat spectra with one spot;
    the chip at the DEM bump contains man-made-looking geometry."""
    h = w = 200
    dem = np.zeros((h, w), "float32")
    dem[60, 80] = 12.0                                    # the bump
    monkeypatch.setattr(P, "fetch_dem_mosaic", lambda *a, **k: dem)

    def fake_s2(bbox, bands=("red", "nir"), **k):
        red = np.full((h, w), 1000.0, "float32")
        nir = np.full((h, w), 2000.0, "float32")
        red[140, 40] = 4000.0                             # spectral outlier
        out = {"red": red, "nir": nir}
        return {b: out[b] for b in bands}

    monkeypatch.setattr(P, "fetch_s2_mosaic", fake_s2)
    bump_lat, bump_lon = P.rc_to_latlon(60, 80, BBOX, w, h)

    def fake_chip(lat, lon, half_m, px):
        hot = abs(lat - bump_lat) < 3e-4 and abs(lon - bump_lon) < 3e-4
        return _synthetic_chip(lat, lon, half_m, px, interesting=hot)

    monkeypatch.setattr(pl, "fetch_chip_s2", fake_chip)
    return bump_lat, bump_lon


def test_funnel_end_to_end_offline(tmp_path, monkeypatch):
    bump_lat, bump_lon = _patch_world(monkeypatch)
    out = tmp_path / "run"
    report = pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128,
                           workers=4, max_stage2=50, n_background=20,
                           use_vlm=False, use_change=False,
                           filter_sheets=False, log=lambda m: None)
    art = out / "artifacts"
    ranked = json.loads((art / "stage2_ranked.json").read_text())
    assert (art / "stage1_candidates.json").exists()
    assert len(ranked) > 0
    assert (out / "report.md").exists()
    assert (out / "final_queue.png").exists()
    assert list((out / "sheets").glob("sheet_*.png"))
    # the seeded "interesting" chip at the DEM bump must rank at the top
    top = ranked[0]
    assert abs(top["lat"] - bump_lat) < 3e-4
    assert abs(top["lon"] - bump_lon) < 3e-4
    assert report["stages"]["stage2"]["scored"] == len(ranked)


def test_funnel_resumes_without_refetching(tmp_path, monkeypatch):
    _patch_world(monkeypatch)
    out = tmp_path / "run"
    pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128, workers=4,
                  max_stage2=50, n_background=20, use_vlm=False,
                  use_change=False, filter_sheets=False, log=lambda m: None)

    def boom(*a):
        raise AssertionError("resume must not refetch chips")

    monkeypatch.setattr(pl, "fetch_chip_s2", boom)
    report = pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128,
                           workers=4, max_stage2=50, n_background=20,
                           use_vlm=False, use_change=False,
                           filter_sheets=False, log=lambda m: None)
    assert (out / "report.md").exists()
    assert report["stages"]["stage2"]["scored"] > 0


def test_compose_panel_offline():
    """Pure panel composition: handles present, missing, and all-missing
    tiles without network or errors."""
    from interesting_intel.filters import compose_panel

    rng = np.random.default_rng(0)
    good = rng.uniform(0, 1, (64, 64, 3)).astype("float32")
    img = compose_panel([("a", good), ("b", None), ("c", good)],
                        tile_px=100, cols=2, title="t")
    assert img.width == 200 and img.height > 200
    img2 = compose_panel([("only", None)], tile_px=80)
    assert img2.width == 80


def test_funnel_filter_sheets_offline(tmp_path, monkeypatch):
    """Stage 5 writes one dossier per queue entry via the (mocked) gatherer,
    and a resumed run does not regenerate existing sheets."""
    import interesting_intel.filters as fl

    _patch_world(monkeypatch)
    calls = []

    def fake_gather(lat, lon, **k):
        calls.append((lat, lon))
        rng = np.random.default_rng(1)
        return {k2: rng.uniform(0, 1, (32, 32, 3)).astype("float32")
                for k2, _ in fl.TILE_ORDER}

    monkeypatch.setattr(fl, "gather_views", fake_gather)
    out = tmp_path / "run"
    report = pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128,
                           workers=4, max_stage2=30, n_background=20,
                           use_vlm=False, use_change=False,
                           filter_sheets=True, log=lambda m: None)
    sheets = list((out / "filters").glob("rank_*_filters.png"))
    assert len(sheets) == report["stages"]["stage5"]["queue"]
    assert len(calls) == len(sheets)
    n_first = len(calls)
    monkeypatch.setattr(pl, "fetch_chip_s2",
                        lambda *a: (_ for _ in ()).throw(AssertionError()))
    pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128, workers=4,
                  max_stage2=30, n_background=20, use_vlm=False,
                  use_change=False, filter_sheets=True, log=lambda m: None)
    assert len(calls) == n_first          # resume reused existing sheets


def test_funnel_seeds_enter_pool(tmp_path, monkeypatch):
    _patch_world(monkeypatch)
    out = tmp_path / "run"
    seed = {"lat": 10.03, "lon": 30.03, "name": "control"}
    pl.run_funnel(BBOX, out, source="s2", half_m=300, px=128, workers=4,
                  max_stage2=50, n_background=20, use_vlm=False,
                  use_change=False, filter_sheets=False, seeds=[seed],
                  log=lambda m: None)
    cands = json.loads(
        (out / "artifacts" / "stage1_candidates.json").read_text())
    assert any(c.get("seed_name") == "control" for c in cands)
