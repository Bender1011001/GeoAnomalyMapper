"""Tests for era-bridge-preserving epoch subsampling (pure function)."""

from deformation_intel.opera import subsample_epochs_preserving_bridges


def _pairs_three_eras():
    """3 eras x 6 epochs; era k's reference date equals an epoch of era k-1."""
    pairs = []
    # era 1: ref 20200101, secondaries Jan..Jun
    for m in range(1, 7):
        pairs.append(("20200101", f"2020{m:02d}15"))
    # era 2: ref = 20200615 (a secondary of era 1 -> the bridge)
    for m in range(7, 13):
        pairs.append(("20200615", f"2020{m:02d}15"))
    # era 3: ref = 20201215
    for m in range(1, 7):
        pairs.append(("20201215", f"2021{m:02d}15"))
    return pairs


def test_bridges_always_kept():
    pairs = _pairs_three_eras()
    keep = subsample_epochs_preserving_bridges(pairs, max_epochs=8)
    kept_secs = {pairs[i][1] for i in keep}
    # era-2 ref (20200615) and era-3 ref (20201215) are bridge secondaries
    assert "20200615" in kept_secs
    assert "20201215" in kept_secs
    # last epoch always kept
    assert (len(pairs) - 1) in keep


def test_budget_roughly_respected_and_sorted():
    pairs = _pairs_three_eras()
    keep = subsample_epochs_preserving_bridges(pairs, max_epochs=8)
    assert keep == sorted(set(keep))
    # bridges are mandatory, so allow small overshoot but not the full set
    assert len(keep) <= 8 + 3
    assert len(keep) < len(pairs)


def test_no_subsample_when_budget_exceeds_count():
    pairs = _pairs_three_eras()
    keep = subsample_epochs_preserving_bridges(pairs, max_epochs=100)
    assert keep == list(range(len(pairs)))
