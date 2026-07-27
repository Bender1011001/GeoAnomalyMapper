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


def test_model_catalog_has_open_and_premium():
    from deformation_intel.vlm_review import MODELS, DEFAULT_MODEL
    assert "open-best" in MODELS and "premium" in MODELS
    # default must be the open-weight flagship, not a paid model
    assert DEFAULT_MODEL == MODELS["open-best"][0]
    assert "qwen" in DEFAULT_MODEL


def test_openrouter_caller_requires_key(monkeypatch):
    import pytest
    from deformation_intel.vlm_review import openrouter_caller
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    call = openrouter_caller()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        call({"image_path": "x.png", "prompt": "p"})


def test_openrouter_caller_builds_request(monkeypatch, tmp_path):
    import json as _json
    from deformation_intel import vlm_review as vr

    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps(
                {"choices": [{"message": {"content": "1 | x | y | 2"}}]}
            ).encode()

    def fake_urlopen(req, timeout=0):
        captured["body"] = _json.loads(req.data)
        captured["auth"] = req.headers.get("Authorization")
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    call = vr.openrouter_caller(model="test/model", api_key="KEY123",
                                detail_note="NOTE")
    out = call({"image_path": str(png), "prompt": "PROMPT"})
    assert out == "1 | x | y | 2"
    assert captured["auth"] == "Bearer KEY123"
    assert captured["body"]["model"] == "test/model"
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["text"].startswith("NOTE")
    assert "PROMPT" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
