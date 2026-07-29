"""Positive/negative-control validation harness.

    python -m interesting_intel.validate

For each known-interesting site: fetch its chip plus a ring of background
chips from the SAME area/source/scale, then ask "does the site outrank its own
background on novelty or geometry?" — exactly the ranking the funnel uses. If
famous sites don't surface against their own surroundings, the system does not
work and nothing downstream matters.

The negative control (central-Kansas cropland) must NOT outrank its
background: a ranker that scores everything as novel is ranking noise.

Side effect: saves the control chips as uint8 .npz fixtures under
tests/fixtures/interesting_chips/ so the CV scorers keep at least one REAL-imagery
test each (synthetic-only validation has burned this project before).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from interesting_intel import features as F
from interesting_intel.pipeline import ChipCache, fetch_many

# name, lat, lon, half_m, px, source, kind
# Scale is chosen per control: the Richat Structure is ~40 km across and
# invisible in a 1 km chip; Nazca geoglyph lines need the finest S2 view.
CONTROLS = [
    dict(name="bingham_canyon_mine", lat=40.523, lon=-112.151, half_m=2000,
         px=500, source="naip", kind="positive"),
    # 4 km view: at 1 km the playa is just one smooth surface among the
    # valley floors of its own background ring (measured FAIL 2026-07-27);
    # the playa-oval-vs-mountains contrast only exists at this scale.
    dict(name="racetrack_playa", lat=36.681, lon=-117.563, half_m=2000,
         px=500, source="naip", kind="positive"),
    # Minuteman Launch Facility G-05 (OSM military=bunker/missile_silo).
    # The spec's approximate ~47.0,-101.5 was plain farmland (measured FAIL);
    # the real field is NW of Minot. Tight 500 m view centres the pad.
    dict(name="minuteman_silo_field", lat=48.05131, lon=-101.85485,
         half_m=250, px=500, source="naip", kind="positive"),
    dict(name="nazca_lines", lat=-14.739, lon=-75.130, half_m=2000,
         px=400, source="s2", kind="positive"),
    dict(name="richat_structure", lat=21.124, lon=-11.401, half_m=25000,
         px=500, source="s2", kind="positive"),
    dict(name="kansas_cropland", lat=38.6, lon=-98.5, half_m=500,
         px=500, source="naip", kind="negative"),
]

N_BACKGROUND = 24
RING_KM = (8.0, 20.0)


def ring_points(lat: float, lon: float, n: int, r_km=RING_KM, seed: int = 11
                ) -> List[dict]:
    """n random points in an annulus around (lat, lon) — same landscape,
    away from the feature itself (for the 25 km Richat chip the ring scales
    with the chip so backgrounds don't overlap the structure)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = rng.uniform(*r_km) * 1000.0
        b = rng.uniform(0, 2 * math.pi)
        dlat = r * math.cos(b) / 111_320.0
        dlon = r * math.sin(b) / (111_320.0 * max(math.cos(math.radians(lat)),
                                                  1e-6))
        out.append({"lat": lat + dlat, "lon": lon + dlon})
    return out


def evaluate_control(ctl: dict, cache: ChipCache, fixtures: Optional[Path],
                     log=print) -> Optional[dict]:
    half_m, px = ctl["half_m"], ctl["px"]
    ring_km = (max(RING_KM[0], 3.2 * half_m / 1000.0),
               max(RING_KM[1], 6.0 * half_m / 1000.0))
    chip = cache.get(ctl["lat"], ctl["lon"], half_m, px, ctl["source"])
    if chip is None or not F.chip_ok(chip["gray"]):
        log(f"  {ctl['name']}: NO IMAGERY — cannot evaluate")
        return None
    bg_pts = ring_points(ctl["lat"], ctl["lon"], N_BACKGROUND, ring_km)
    try:
        chips_map = fetch_many(cache, bg_pts, half_m, px, ctl["source"],
                               workers=8, log=lambda m: None)
    except RuntimeError as exc:
        log(f"  {ctl['name']}: background fetch failed ({exc})")
        return None
    bg = [c for c in chips_map.values() if F.chip_ok(c["gray"])]
    if len(bg) < 10:
        log(f"  {ctl['name']}: only {len(bg)} background chips — skipping")
        return None
    f_c = F.chip_features(chip["gray"])
    f_bg = [F.chip_features(b["gray"]) for b in bg]
    vec_c, geo_c = f_c["vector"], f_c["geometry"]
    vecs = [f["vector"] for f in f_bg]
    geo_bg = [f["geometry"] for f in f_bg]
    nov_c = F.novelty_score(vec_c, vecs)
    nov_bg = [F.novelty_score(vecs[i], vecs[:i] + vecs[i + 1:])
              for i in range(len(vecs))]
    n = len(bg) + 1
    rank_nov = 1 + sum(1 for v in nov_bg if v > nov_c)
    rank_geo = 1 + sum(1 for v in geo_bg if v > geo_c)
    best = min(rank_nov, rank_geo)
    surfaced = best <= max(2, round(0.1 * n))     # top decile of its own area
    res = {**{k: ctl[k] for k in ("name", "lat", "lon", "half_m", "px",
                                  "source", "kind")},
           "n_ranked": n, "novelty": round(nov_c, 2),
           "novelty_bg_median": round(float(np.median(nov_bg)), 2),
           "rank_novelty": rank_nov, "geometry": round(geo_c, 3),
           "geometry_bg_median": round(float(np.median(geo_bg)), 3),
           "rank_geometry": rank_geo, "surfaced_top_decile": bool(surfaced)}
    ok = surfaced if ctl["kind"] == "positive" else not surfaced
    res["pass"] = bool(ok)
    log(f"  {ctl['name']:24s} nov {nov_c:5.2f} (bg med "
        f"{res['novelty_bg_median']:5.2f}, rank {rank_nov}/{n})  geo "
        f"{geo_c:.3f} (rank {rank_geo}/{n})  "
        f"{'PASS' if ok else 'FAIL'} [{ctl['kind']}]")
    if fixtures is not None:
        fixtures.mkdir(parents=True, exist_ok=True)
        g8 = (F.normalize01(chip["gray"]) * 255).astype("uint8")
        np.savez_compressed(fixtures / f"{ctl['name']}.npz", gray=g8,
                            res_m=chip["res_m"], source=chip["source"],
                            half_m=half_m)
        bg8 = np.stack([(F.normalize01(b["gray"]) * 255).astype("uint8")
                        for b in bg[:8]])
        np.savez_compressed(fixtures / f"{ctl['name']}_background.npz",
                            grays=bg8, half_m=half_m)
    return res


def run_controls(out_dir="results/interesting/controls",
                 fixtures_dir="tests/fixtures/interesting_chips",
                 controls=CONTROLS, log=print) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = ChipCache(out / "chip_cache")
    results = []
    for ctl in controls:
        log(f"control: {ctl['name']}")
        r = evaluate_control(ctl, cache,
                             Path(fixtures_dir) if fixtures_dir else None, log)
        if r is not None:
            results.append(r)
    summary = {
        "results": results,
        "positives_passed": sum(1 for r in results
                                if r["kind"] == "positive" and r["pass"]),
        "positives_total": sum(1 for r in results if r["kind"] == "positive"),
        "negatives_passed": sum(1 for r in results
                                if r["kind"] == "negative" and r["pass"]),
        "negatives_total": sum(1 for r in results if r["kind"] == "negative"),
        "fetches": cache.fetches, "cache_hits": cache.hits,
        "sec": round(time.time() - t0, 1),
    }
    (out / "controls.json").write_text(json.dumps(summary, indent=1))
    log(f"controls: {summary['positives_passed']}/{summary['positives_total']}"
        f" positives surfaced, {summary['negatives_passed']}/"
        f"{summary['negatives_total']} negatives stayed dull "
        f"({summary['sec']}s, {cache.fetches} fetches)")
    return summary


if __name__ == "__main__":
    run_controls(log=lambda m: print(m, flush=True))
    sys.exit(0)
