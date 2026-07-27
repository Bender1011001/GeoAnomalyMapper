"""Tests for the review funnel (ranking, contact sheets, industrial veto)."""
import numpy as np

from deformation_intel.review import (
    build_review_package,
    contact_sheet,
    rank_for_review,
)


def _c(lat, lon, vel=-5.0, acc=-1.0, pad=0.0, ag=0.0, cluster=1):
    return {"lat": lat, "lon": lon, "peak_velocity_cm_yr": vel,
            "accel_cm_yr2": acc, "cumulative_cm": vel * 5,
            "context": {"industrial_pad": pad, "naip_agriculture": ag},
            "screen": {"cluster_size": cluster}}


def test_unexplained_outranks_explained():
    # same signal strength, but one sits on a well pad -> the clean one wins
    clean = _c(32.0, -103.0, vel=-5, acc=-1.0, pad=0.0)
    on_pad = _c(33.0, -104.0, vel=-6, acc=-1.2, pad=0.9)
    assert rank_for_review([on_pad, clean])[0]["lat"] == 32.0


def test_isolated_outranks_clustered():
    lone = _c(32.0, -103.0, cluster=1)
    crowd = _c(33.0, -104.0, cluster=8)
    assert rank_for_review([crowd, lone])[0]["lat"] == 32.0


def test_stronger_signal_ranks_higher_all_else_equal():
    weak = _c(32.0, -103.0, vel=-1, acc=-0.1)
    strong = _c(33.0, -104.0, vel=-8, acc=-2.0)
    assert rank_for_review([weak, strong])[0]["lat"] == 33.0


def test_contact_sheet_written(tmp_path):
    chips = [np.random.default_rng(i).normal(0.5, 0.1, (60, 60)) for i in range(7)]
    labels = [f"site {i}" for i in range(7)]
    p = contact_sheet(chips, labels, tmp_path / "s.png", cols=4, thumb=60,
                      title="test")
    assert p.exists() and p.stat().st_size > 0


def test_contact_sheet_rejects_empty(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        contact_sheet([], [], tmp_path / "x.png")


def test_build_package_vetoes_industrial(tmp_path):
    cands = [_c(32.0 + i * 0.1, -103.0) for i in range(6)]

    def chip_fn(lat, lon):
        return np.full((50, 50), 0.5, "float32")

    # scorer flags every other site as a pad
    state = {"n": 0}

    def pad_scorer(chip):
        state["n"] += 1
        return 0.9 if state["n"] % 2 == 0 else 0.0

    s = build_review_package(cands, tmp_path, chip_fn, pad_scorer=pad_scorer)
    assert s["industrial_vetoed"] == 3
    assert s["for_review"] == 3
    assert len(s["sheets"]) == 1
    assert (tmp_path / "review_list.json").exists()


def test_build_package_handles_missing_chips(tmp_path):
    cands = [_c(32.0, -103.0), _c(33.0, -104.0)]
    s = build_review_package(cands, tmp_path, lambda la, lo: None)
    assert s["for_review"] == 0
    assert s["sheets"] == []


def test_build_package_splits_sheets(tmp_path):
    cands = [_c(32.0 + i * 0.01, -103.0) for i in range(25)]
    s = build_review_package(cands, tmp_path, lambda la, lo: np.zeros((40, 40)),
                             per_sheet=10)
    assert len(s["sheets"]) == 3   # 10 + 10 + 5
