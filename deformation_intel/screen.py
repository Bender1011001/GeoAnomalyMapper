"""Candidate screening for region sweeps — the frozen post-detection rule.

Turns raw localized detections into a reviewed shortlist by applying, in one
place, the confound lessons this project paid for:
  - agriculture / mountain (imagery + slope)   -> context.is_cultivated_confound
  - geothermal / gas subsidence                -> proximity veto (Carson Sink)
  - regional process fragmented into "bowls"   -> cluster veto (>=N within R km)
  - detection-floor context                    -> size/rate stated, not implied

This is the exact rule pre-registered for the arid-basin sweep
(docs/RESEARCH_TRACKS.md 2026-07-25). Promoting it from scratchpad to a tested
module makes the sweep's verdict reproducible (CRITIQUE 3.1). Pure functions;
network confound lookups are injected so the core is unit-testable offline.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def _km(lat1, lon1, lat2, lon2):
    # flat-earth approximation, fine for intra-region clustering (<100 km)
    return 111.0 * float(np.hypot(lat1 - lat2,
                                  (lon1 - lon2) * np.cos(np.radians(lat1))))


def cluster_sizes(candidates: Sequence[dict], radius_km: float = 15.0) -> List[int]:
    """For each candidate, how many candidates (incl. itself) lie within
    radius_km. A large count = a regional process fragmented into pseudo-bowls
    (the Carson Sink lesson), not N independent voids. Pure."""
    pts = [(c["lat"], c["lon"]) for c in candidates]
    out = []
    for i, (la, lo) in enumerate(pts):
        n = sum(1 for (la2, lo2) in pts if _km(la, lo, la2, lo2) <= radius_km)
        out.append(n)
    return out


def min_plant_distance_km(lat: float, lon: float,
                          plants: Sequence[tuple]) -> float:
    """Distance to the nearest (lat, lon) power plant; inf if none supplied."""
    if not plants:
        return float("inf")
    return min(_km(lat, lon, pla, plo) for pla, plo in plants)


def screen_candidates(
    candidates: List[dict],
    *,
    cultivated_fn: Optional[Callable[[dict], bool]] = None,
    plants: Optional[Sequence[tuple]] = None,
    area_max_km2: float = 0.10,
    void_likelihood_min: float = 0.9,
    cluster_radius_km: float = 15.0,
    cluster_max: int = 5,
    plant_veto_km: float = 12.0,
) -> Dict[str, List[dict]]:
    """Apply the frozen candidate rule. Returns
    {'survivors': [...], 'rejected': [...]} where each candidate gets a
    'screen' dict recording every test outcome and, if rejected, 'reject_reason'.

    A candidate SURVIVES iff ALL hold:
      is_localized & accelerating & rate_reliable, area < area_max,
      void_likelihood >= min, NOT cultivated, NOT within plant_veto_km of a
      plant, NOT in a cluster of > cluster_max within cluster_radius.
    cultivated_fn(candidate)->bool lets the caller inject the imagery+slope
    check (context.is_cultivated_confound); default treats unknown as False.
    """
    sizes = cluster_sizes(candidates, cluster_radius_km)
    survivors, rejected = [], []
    for c, csize in zip(candidates, sizes):
        cls = (c.get("classification") or "")
        accel = abs(c.get("accel_cm_yr2") or 0.0)
        pdist = min_plant_distance_km(c["lat"], c["lon"], plants or [])
        cult = bool(cultivated_fn(c)) if cultivated_fn else False
        s = {
            "cluster_size": csize,
            "nearest_plant_km": round(pdist, 1) if np.isfinite(pdist) else None,
            "cultivated": cult,
        }
        c = {**c, "screen": s}
        reason = None
        if not c.get("is_localized"):
            reason = "not_localized"
        elif "regional" in cls:
            reason = "regional"
        elif not c.get("rate_reliable", True):
            reason = "rate_unreliable"
        elif accel < 1e-6 and "accel" not in cls:
            reason = "not_accelerating"
        elif (c.get("area_km2") or 9) >= area_max_km2:
            reason = "too_large"
        elif (c.get("void_likelihood") or 0) < void_likelihood_min:
            reason = "low_void_likelihood"
        elif cult:
            reason = "cultivated"
        elif pdist <= plant_veto_km:
            reason = "near_power_plant"
        elif csize > cluster_max:
            reason = "in_cluster"
        if reason:
            c["reject_reason"] = reason
            rejected.append(c)
        else:
            survivors.append(c)

    def _score(c):
        vl = c.get("void_likelihood") or 0.5
        return vl * (abs(c.get("accel_cm_yr2") or 0)
                     + 0.3 * abs(c.get("peak_velocity_cm_yr") or 0))

    survivors.sort(key=_score, reverse=True)
    return {"survivors": survivors, "rejected": rejected}


def fetch_power_plants(bbox, retries: int = 3) -> List[tuple]:
    """Overpass query for power=plant (lat, lon) within bbox=(lon0,lat0,lon1,
    lat1). Network; kept separate so screen_candidates stays pure/offline."""
    import json
    import time
    import urllib.parse
    import urllib.request
    lon0, lat0, lon1, lat1 = bbox
    q = (f"[out:json][timeout:60];("
         f"node({lat0},{lon0},{lat1},{lon1})[power=plant];"
         f"way({lat0},{lon0},{lat1},{lon1})[power=plant];);out center;")
    url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(q)
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geo/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=90).read())
            out = []
            for el in d.get("elements", []):
                if "lat" in el:
                    out.append((el["lat"], el["lon"]))
                elif "center" in el:
                    out.append((el["center"]["lat"], el["center"]["lon"]))
            return out
        except Exception:
            time.sleep(8 * (a + 1))
    return []


def screen_sweep_output(out_dir, *, fetch_plants: bool = True) -> dict:
    """Apply the frozen rule to a completed run_region_sweep output directory.

    Reads candidates_ranked.json, injects the cultivated check from each
    candidate's stored context (naip_agriculture + slope_deg), optionally
    fetches power plants over the candidate bbox, screens, and writes
    survivors.json + rejected.json + screen_summary.json. Returns the summary.
    """
    import json
    from pathlib import Path
    from deformation_intel.context import is_cultivated_confound

    out = Path(out_dir)
    cands = json.loads((out / "candidates_ranked.json").read_text())
    if not cands:
        (out / "survivors.json").write_text("[]")
        summ = {"input": 0, "survivors": 0, "rejected": 0}
        (out / "screen_summary.json").write_text(json.dumps(summ, indent=1))
        return summ

    def cult(c):
        ctx = c.get("context") or {}
        return is_cultivated_confound(ctx.get("naip_agriculture", float("nan")),
                                      ctx.get("slope_deg", float("nan")))

    plants = []
    if fetch_plants:
        las = [c["lat"] for c in cands]
        los = [c["lon"] for c in cands]
        pad = 0.2
        plants = fetch_power_plants((min(los) - pad, min(las) - pad,
                                     max(los) + pad, max(las) + pad))

    res = screen_candidates(cands, cultivated_fn=cult, plants=plants)
    (out / "survivors.json").write_text(json.dumps(res["survivors"], indent=1))
    (out / "rejected.json").write_text(json.dumps(res["rejected"], indent=1))
    from collections import Counter
    reasons = Counter(c.get("reject_reason") for c in res["rejected"])
    summ = {"input": len(cands), "survivors": len(res["survivors"]),
            "rejected": len(res["rejected"]), "n_plants": len(plants),
            "reject_reasons": dict(reasons)}
    (out / "screen_summary.json").write_text(json.dumps(summ, indent=1))
    return summ
