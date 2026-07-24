"""Unit tests for the pure parts of the region sweep driver."""
import math

import pytest

from deformation_intel.sweep import rank_candidates, tile_grid, _neighborhood


def test_tile_grid_covers_bbox():
    bbox = (-103.6, 31.6, -102.6, 32.4)   # ~1 deg lon x 0.8 deg lat
    tiles = tile_grid(bbox, tile_km=24.0)
    assert len(tiles) > 0
    for lat, lon in tiles.values():
        assert 31.6 <= lat <= 32.4
        assert -103.6 <= lon <= -102.6


def test_tile_grid_spacing_scales_with_tile_km():
    bbox = (-103.6, 31.6, -102.6, 32.4)
    coarse = tile_grid(bbox, tile_km=48.0)
    fine = tile_grid(bbox, tile_km=12.0)
    assert len(fine) > len(coarse)


def test_tile_grid_keys_are_stable():
    bbox = (-103.6, 31.6, -102.6, 32.4)
    a = tile_grid(bbox, 24.0)
    b = tile_grid(bbox, 24.0)
    assert a == b   # deterministic -> resume works


def test_tile_grid_rejects_bad_bbox():
    with pytest.raises(ValueError):
        tile_grid((-102.6, 31.6, -103.6, 32.4))   # lon_max < lon_min


def test_neighborhood_limits_span():
    tiles = {"a": (32.0, -103.0), "b": (32.5, -103.2), "far": (40.0, -100.0)}
    nb = _neighborhood(tiles, "a", span_deg=3.0)
    assert "a" in nb and "b" in nb and "far" not in nb


def test_rank_candidates_localized_first():
    cands = [
        {"is_localized": False, "void_likelihood": 1.0, "accel_cm_yr2": -9,
         "peak_velocity_cm_yr": -9},
        {"is_localized": True, "void_likelihood": 0.5, "accel_cm_yr2": -0.1,
         "peak_velocity_cm_yr": -1},
    ]
    ranked = rank_candidates(cands)
    assert ranked[0]["is_localized"] is True   # any localized beats regional


def test_rank_candidates_orders_by_severity():
    a = {"is_localized": True, "void_likelihood": 1.0, "accel_cm_yr2": -2.0,
         "peak_velocity_cm_yr": -6.0}
    b = {"is_localized": True, "void_likelihood": 1.0, "accel_cm_yr2": -0.2,
         "peak_velocity_cm_yr": -1.0}
    ranked = rank_candidates([b, a])
    assert ranked[0] is a


def test_rank_candidates_handles_missing_fields():
    ranked = rank_candidates([{"is_localized": True}, {}])
    assert len(ranked) == 2   # no crash on absent keys
