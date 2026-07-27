"""Open-ended vision-model review of candidate chips — "what's odd here?"

Everything else in this project asks a NARROW question ("is this a subsidence
void?") and therefore throws away anything that doesn't match. That is a bad
trade when the imagery is free, primary-source, and un-redacted (NAIP 1 m USDA
aerial and declassified CORONA film, neither of which passes through the
blurring/redaction layer commercial map products apply).

A vision model reviewing every chip with a DELIBERATELY OPEN prompt costs
roughly $0.001-0.005 per image — a few dollars for the entire backlog — and can
surface things no detector was written to find. This module builds the batches
and the prompt; the caller supplies the model.

Design notes:
- Two passes. Cheap WIDE pass over contact sheets (100 sites per image) to spot
  which cells deserve attention, then a FOCUSED pass on full-resolution chips
  for the flagged ones. ~100x fewer model calls than per-chip review.
- The prompt asks for anything anomalous, explicitly including categories the
  deformation pipeline is blind to, and REQUIRES a mundane explanation first so
  the model doesn't manufacture mysteries.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# The wide (contact-sheet) prompt: triage which cells are worth a closer look.
WIDE_PROMPT = """This is a grid of satellite/aerial image thumbnails. Each cell
is a separate location; the red crosshair marks the point of interest. The
labels give index, latitude/longitude, and a ground-motion rate.

For EACH cell that contains anything visually notable, report:
  index | one-line description | mundane_explanation | interest (0-3)

Be curious and open-ended. Notable includes, but is NOT limited to:
  - geometric ground patterns: circles, rings, straight lines, grids, spirals
  - large excavations, pits, quarries, spoil heaps, tailings
  - unusual industrial or military-looking installations, antenna arrays,
    bunkers, revetments, airstrips (especially isolated or unmarked ones)
  - craters, collapse features, sinkholes, fissures
  - unexplained colour/tone anomalies, scars, burn marks, dead vegetation
  - abandoned or ruined structures, earthworks, ancient-looking features
  - anything that simply looks WRONG or out of place for the surroundings

Rules:
  - ALWAYS give the most likely mundane explanation first. Most things are
    farms, mines, solar farms, well pads, roads, natural landforms.
  - interest 0 = ordinary, 1 = mildly odd, 2 = genuinely unusual,
    3 = I cannot explain this.
  - Only list cells scoring >= 1. If nothing qualifies, say "nothing notable".
  - Do not speculate about aliens/secret bases; describe what is VISIBLE."""

# The focused prompt: full-resolution single chip.
FOCUS_PROMPT = """A single satellite/aerial image; the crosshair marks the point
of interest. Describe precisely what is visible there and immediately around it.

Answer in this order:
1. What do you actually SEE? (shapes, textures, edges, tone, scale cues)
2. Most likely mundane explanation, and what evidence supports it.
3. What, if anything, does NOT fit that explanation?
4. interest 0-3 (0 ordinary, 3 genuinely unexplained).
5. What additional data would resolve it? (older imagery, different season,
   higher resolution, elevation)

Be concrete and sceptical. Say "I don't know" rather than inventing a story."""


def encode_png(path) -> str:
    """base64-encode an image for a vision API payload."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_wide_batches(sheet_paths: Sequence, prompt: str = WIDE_PROMPT
                       ) -> List[dict]:
    """One request payload per contact sheet (100 sites each)."""
    return [{"image_path": str(p), "prompt": prompt, "kind": "wide"}
            for p in sheet_paths]


def build_focus_batches(chip_paths: Sequence, prompt: str = FOCUS_PROMPT
                        ) -> List[dict]:
    """One request payload per full-resolution chip."""
    return [{"image_path": str(p), "prompt": prompt, "kind": "focus"}
            for p in chip_paths]


def estimate_cost(n_sheets: int, n_focus: int,
                  usd_per_image: float = 0.003) -> dict:
    """Rough cost of a two-pass review. The point of the wide pass is that
    reviewing 100 sites costs ONE image instead of one hundred."""
    naive = (n_sheets * 100 + n_focus) * usd_per_image
    twopass = (n_sheets + n_focus) * usd_per_image
    return {"naive_per_chip_usd": round(naive, 2),
            "two_pass_usd": round(twopass, 2),
            "saving_factor": round(naive / twopass, 1) if twopass else 0.0}


def parse_wide_response(text: str) -> List[dict]:
    """Parse 'index | description | mundane | interest' lines into records.

    Tolerant of chatter around the table; ignores anything unparseable.
    """
    out: List[dict] = []
    for line in (text or "").splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        idx_tok = parts[0].split(".")[0].strip().lstrip("#")
        try:
            idx = int(idx_tok)
        except ValueError:
            continue
        try:
            interest = int(str(parts[-1]).strip()[0])
        except (ValueError, IndexError):
            continue
        out.append({"index": idx, "description": parts[1],
                    "mundane": parts[2], "interest": interest})
    return out


def select_for_focus(records: Sequence[dict], min_interest: int = 2
                     ) -> List[dict]:
    """Which wide-pass hits deserve a full-resolution second look."""
    return sorted([r for r in records if r.get("interest", 0) >= min_interest],
                  key=lambda r: -r.get("interest", 0))


def run_review(sheet_paths: Sequence, call_model: Callable[[dict], str],
               out_dir, *, min_interest: int = 2) -> dict:
    """Run the wide pass over contact sheets with an injected model callable.

    call_model(payload) -> response text. Injected so this module needs no
    particular vendor SDK and stays unit-testable offline.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_records: List[dict] = []
    for payload in build_wide_batches(sheet_paths):
        try:
            text = call_model(payload)
        except Exception:
            continue
        recs = parse_wide_response(text)
        for r in recs:
            r["sheet"] = payload["image_path"]
        all_records.extend(recs)
    focus = select_for_focus(all_records, min_interest)
    (out / "vlm_wide_findings.json").write_text(json.dumps(all_records, indent=1))
    (out / "vlm_focus_queue.json").write_text(json.dumps(focus, indent=1))
    return {"sheets": len(sheet_paths), "notable": len(all_records),
            "for_focus": len(focus)}
