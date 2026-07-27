"""Tests for the open-ended VLM review harness (no network/model needed)."""
from deformation_intel.vlm_review import (
    build_wide_batches,
    estimate_cost,
    parse_wide_response,
    run_review,
    select_for_focus,
)


def test_wide_batches_one_per_sheet():
    b = build_wide_batches(["a.png", "b.png"])
    assert len(b) == 2 and all(x["kind"] == "wide" for x in b)


def test_parse_wide_response_extracts_records():
    txt = """Here is what I see:
7 | dark circular depression ~200m | collapsed sinkhole or stock pond | 2
12 | bright rectangular clearing | oil well pad | 0
19. | radial line pattern | unclear, possibly old tracks | 3
garbage line without pipes
"""
    recs = parse_wide_response(txt)
    idx = {r["index"]: r for r in recs}
    assert set(idx) == {7, 12, 19}
    assert idx[19]["interest"] == 3
    assert "well pad" in idx[12]["mundane"]


def test_parse_handles_empty_and_junk():
    assert parse_wide_response("") == []
    assert parse_wide_response("nothing notable") == []


def test_select_for_focus_filters_and_sorts():
    recs = [{"index": 1, "interest": 1}, {"index": 2, "interest": 3},
            {"index": 3, "interest": 2}]
    sel = select_for_focus(recs, min_interest=2)
    assert [r["index"] for r in sel] == [2, 3]


def test_two_pass_is_much_cheaper():
    c = estimate_cost(n_sheets=3, n_focus=20)
    assert c["two_pass_usd"] < c["naive_per_chip_usd"]
    assert c["saving_factor"] > 5


def test_run_review_with_injected_model(tmp_path):
    def fake_model(payload):
        return "3 | odd ring feature | plough circle | 2\n4 | nothing | field | 0"

    s = run_review(["s1.png", "s2.png"], fake_model, tmp_path)
    assert s["sheets"] == 2
    assert s["notable"] == 4        # 2 records per sheet
    assert s["for_focus"] == 2      # only interest>=2
    assert (tmp_path / "vlm_focus_queue.json").exists()


def test_run_review_survives_model_errors(tmp_path):
    def broken(payload):
        raise RuntimeError("api down")

    s = run_review(["s1.png"], broken, tmp_path)
    assert s["notable"] == 0
