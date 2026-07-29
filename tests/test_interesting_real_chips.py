"""REAL-imagery tests for interesting_intel scorers.

Fixtures in tests/fixtures/interesting_chips/ are actual NAIP / Sentinel-2
chips of the validation controls plus 8 background chips from each control's
own surrounding ring, saved by `python -m interesting_intel.validate`
(2026-07-27). No network at test time.

These exist because synthetic-only validation has repeatedly burned this
project (an agriculture detector passed 6 synthetic tests then scored barren
Mojave as cultivated). Assertion margins are set ~30-50% below the measured
values so they catch real regressions without being flaky to minor numeric
drift.

Measured reference values (2026-07-27):
  bingham   novelty 3.95  geometry 0.197 (bg max 0.16)  radiality 0.33
  racetrack novelty 4.81  geometry 0.190 (bg max 0.13)   [4 km view]
  nazca     novelty 4.02  faint_line 0.54 (geo rank 2/25 in full harness)
  richat    novelty 3.37  radiality 0.36
  kansas    novelty 0.91 (bg median 3.42 -> dull in its own terrain)
"""
from pathlib import Path

import numpy as np
import pytest

from deformation_intel.context import agriculture_score
from interesting_intel import features as F

FIXTURES = Path(__file__).parent / "fixtures" / "interesting_chips"

NAMES = ["bingham_canyon_mine", "racetrack_playa", "minuteman_silo_field",
         "nazca_lines", "richat_structure", "kansas_cropland"]


def _chip(name):
    z = np.load(FIXTURES / f"{name}.npz")
    return z["gray"].astype("float32")


def _bg_vecs(name):
    b = np.load(FIXTURES / f"{name}_background.npz")
    return [F.texture_vector(x.astype("float32")) for x in b["grays"]]


@pytest.fixture(scope="module")
def feats():
    return {n: F.chip_features(_chip(n)) for n in NAMES}


def test_fixtures_exist_and_pass_plausibility():
    for n in NAMES:
        assert (FIXTURES / f"{n}.npz").exists(), f"missing fixture {n}"
        assert F.chip_ok(_chip(n)), f"{n} failed chip_ok"


def test_bingham_mine_is_novel_vs_own_background(feats):
    nov = F.novelty_score(feats["bingham_canyon_mine"]["vector"],
                          _bg_vecs("bingham_canyon_mine"))
    assert nov > 2.5                       # measured 3.95


def test_racetrack_playa_is_novel_at_4km_view(feats):
    nov = F.novelty_score(feats["racetrack_playa"]["vector"],
                          _bg_vecs("racetrack_playa"))
    assert nov > 2.5                       # measured 4.81


def test_nazca_faint_lines_beat_background(feats):
    """The geoglyph lines are invisible to canny (straightedge 0.0) but the
    radon faint-line scorer must rank the chip above its whole ring."""
    fl = feats["nazca_lines"]["faint_line"]
    assert feats["nazca_lines"]["straightedge"] == 0.0
    assert fl > 0.3                        # measured 0.54
    b = np.load(FIXTURES / "nazca_lines_background.npz")
    bg_fl = [F.faint_line_score(x.astype("float32")) for x in b["grays"]]
    assert fl > max(bg_fl)


def test_richat_radiality_beats_background(feats):
    rad = feats["richat_structure"]["radiality"]
    assert rad > 0.25                      # measured 0.36
    b = np.load(FIXTURES / "richat_structure_background.npz")
    bg_rad = [F.radiality_score(x.astype("float32")) for x in b["grays"]]
    assert rad > max(bg_rad)


def test_bingham_pit_reads_as_radial(feats):
    assert feats["bingham_canyon_mine"]["radiality"] > 0.2   # measured 0.33


def test_kansas_cropland_is_dull_in_its_own_terrain(feats):
    """The negative control: cropland must NOT be novel among cropland.
    A ranker that scores everything as novel is ranking noise."""
    nov = F.novelty_score(feats["kansas_cropland"]["vector"],
                          _bg_vecs("kansas_cropland"))
    assert nov < 2.0                       # measured 0.91


def test_kansas_ring_is_recognised_as_cultivated():
    """The agriculture confound must fire on real cultivated ring chips
    (measured 0.67-0.93 on three of the first four) — this is what demotes
    cropland geometry in the funnel."""
    b = np.load(FIXTURES / "kansas_cropland_background.npz")
    scores = [agriculture_score(x.astype("float32")) for x in b["grays"]]
    assert sum(1 for s in scores if s >= 0.4) >= 3


def test_desert_backgrounds_not_cultivated():
    """Mojave/Death-Valley ring chips must NOT read as agriculture (the
    real-data failure mode context.py documents)."""
    b = np.load(FIXTURES / "racetrack_playa_background.npz")
    scores = [agriculture_score(x.astype("float32")) for x in b["grays"]]
    assert sum(1 for s in scores if s >= 0.4) <= 1


def test_geometry_of_controls_beats_their_backgrounds(feats):
    """Bingham and racetrack measured rank 1 of 25 on geometry in their own
    rings; assert they beat the 8 saved background chips outright."""
    for name in ("bingham_canyon_mine", "racetrack_playa"):
        b = np.load(FIXTURES / f"{name}_background.npz")
        bg_geo = [F.geometry_score(x.astype("float32")) for x in b["grays"]]
        assert feats[name]["geometry"] > max(bg_geo), name
