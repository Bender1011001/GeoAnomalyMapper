"""Turn a screened candidate list into something a human can review in minutes.

The funnel (measured rates, 2026-07-27):
    raw anomalies            ~51,000   35 work-days at 20 s each   -- hopeless
    localized only           ~21,000   14 work-days                -- hopeless
    frozen screening rule       ~270    1.5 h                      -- feasible
    + imagery auto-veto         ~150    <1 h                       -- easy
    + CONTACT SHEETS           2 pages  ~5 min                     -- trivial

The screening rule already does the heavy lifting; the remaining problem is
that roughly half of what survives is industrial (a Delaware-Basin survivor sat
directly on a well pad and was only caught by eye). So: auto-score each
survivor's imagery, veto the obvious industry, then lay the rest out as a
labelled grid so a person scans hundreds of sites per minute instead of one.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def rank_for_review(candidates: Sequence[dict]) -> List[dict]:
    """Order survivors by how much they demand a human look.

    Priority = signal strength x how UNEXPLAINED it is. A big accelerating bowl
    with no industrial/agricultural signature outranks a bigger one sitting on a
    well pad, because the second is already explained.
    """
    def score(c: dict) -> float:
        vel = abs(c.get("peak_velocity_cm_yr") or 0.0)
        acc = abs(c.get("accel_cm_yr2") or 0.0)
        cum = abs(c.get("cumulative_cm") or 0.0)
        strength = acc + 0.3 * vel + 0.02 * cum
        ctx = c.get("context") or {}
        scr = c.get("screen") or {}
        pad = ctx.get("industrial_pad") or 0.0
        ag = ctx.get("naip_agriculture") or 0.0
        explained = max(float(pad), float(ag))
        cluster = scr.get("cluster_size") or 1
        # unexplained and isolated -> top of the list
        return strength * (1.0 - 0.8 * explained) / (1.0 + 0.1 * (cluster - 1))

    return sorted(candidates, key=score, reverse=True)


def contact_sheet(chips: Sequence[np.ndarray], labels: Sequence[str],
                  out_path, cols: int = 10, thumb: int = 180,
                  title: str = "") -> Path:
    """Write a labelled grid of thumbnails — the unit of fast human review.

    chips: 2-D grayscale arrays (any size, rescaled). labels: short captions.
    A 10x10 sheet holds 100 sites; a reviewer can triage a sheet in ~2 minutes,
    versus ~20 s per site opened individually (a 6x speedup, and far less
    context-switching).
    """
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    n = len(chips)
    if n == 0:
        raise ValueError("no chips to lay out")
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cap = 18
    W, H = cols * thumb, rows * (thumb + cap) + (30 if title else 0)
    sheet = Image.new("RGB", (W, H), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    y0 = 0
    if title:
        draw.text((6, 8), title, fill=(255, 255, 255))
        y0 = 30
    for i, (chip, lab) in enumerate(zip(chips, labels)):
        r, c = divmod(i, cols)
        a = np.asarray(chip, dtype="float32")
        if a.size and np.isfinite(a).any():
            lo, hi = np.nanpercentile(a, [2, 98])
            a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
            a = np.nan_to_num(a, nan=0.5)
        else:
            a = np.zeros((8, 8), "float32")
        im = Image.fromarray((a * 255).astype("uint8")).convert("RGB")
        im = im.resize((thumb, thumb), Image.NEAREST)
        # centre crosshair marks the anomaly position
        d = ImageDraw.Draw(im)
        m = thumb // 2
        d.line([(m - 14, m), (m - 5, m)], fill=(255, 60, 60), width=2)
        d.line([(m + 5, m), (m + 14, m)], fill=(255, 60, 60), width=2)
        d.line([(m, m - 14), (m, m - 5)], fill=(255, 60, 60), width=2)
        d.line([(m, m + 5), (m, m + 14)], fill=(255, 60, 60), width=2)
        px, py = c * thumb, y0 + r * (thumb + cap)
        sheet.paste(im, (px, py))
        draw.text((px + 3, py + thumb + 3), lab[:34], fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def build_review_package(candidates: Sequence[dict], out_dir,
                         chip_fn: Callable[[float, float], Optional[np.ndarray]],
                         *, pad_scorer: Optional[Callable] = None,
                         max_sites: int = 300, per_sheet: int = 100) -> dict:
    """Fetch a chip per candidate, auto-veto industrial, and emit contact sheets.

    chip_fn(lat, lon) -> 2-D array or None (injected so this stays testable and
    the module has no network dependency of its own).
    Returns a summary dict; writes sheets + review_list.json into out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranked = rank_for_review(candidates)[:max_sites]

    kept: List[dict] = []
    chips: List[np.ndarray] = []
    vetoed = 0
    for c in ranked:
        chip = None
        try:
            chip = chip_fn(c["lat"], c["lon"])
        except Exception:
            chip = None
        if chip is None:
            continue
        if pad_scorer is not None:
            try:
                p = float(pad_scorer(chip))
            except Exception:
                p = 0.0
            c = {**c, "pad_score": p}
            if p >= 0.35:          # context.PAD_THRESHOLD
                vetoed += 1
                continue
        kept.append(c)
        chips.append(chip)

    sheets = []
    for i in range(0, len(kept), per_sheet):
        part = kept[i:i + per_sheet]
        labs = [f"{k+i+1}. {c['lat']:.4f},{c['lon']:.4f} "
                f"{c.get('peak_velocity_cm_yr','?')}cm/yr"
                for k, c in enumerate(part)]
        p = contact_sheet(chips[i:i + per_sheet], labs,
                          out / f"sheet_{i//per_sheet + 1:02d}.png",
                          title=f"review sheet {i//per_sheet + 1} "
                                f"(sites {i+1}-{i+len(part)} of {len(kept)})")
        sheets.append(str(p))
    (out / "review_list.json").write_text(json.dumps(kept, indent=1))
    summary = {"input": len(candidates), "ranked": len(ranked),
               "industrial_vetoed": vetoed, "for_review": len(kept),
               "sheets": sheets}
    (out / "review_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
