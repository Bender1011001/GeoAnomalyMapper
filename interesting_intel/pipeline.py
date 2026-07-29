"""The funnel — cheap to expensive, a human only at the end.

    Stage 1  priors.generate_candidates      (regional rasters, no chips)
    Stage 2  chip fetch + CV features        (the real bottleneck: parallel,
                                              disk-cached, never fetch twice)
    Stage 3  cheap VLM on 25-chip contact sheets
    Stage 4  strong VLM on full-res chips
    Stage 5  ranked report + final contact sheet for the human

Operating rules baked in from this project's failure history:
- plausibility gates BEFORE ranking (chip_ok, mosaic-normalised change);
- rank adjustments, not hard vetoes (only visibly-cultivated-on-flat-ground
  and visibly-on-a-pad demote hard, and even they only multiply rank down);
- every stage asserts on ARTIFACTS PRODUCED, not on progress counters;
- resumable: each stage writes a JSON artifact and is skipped when present.

Cost model (measured 2026-07-27): imagery fetch dominates wall-clock; VLM cost
is cents. See run_report.json in any output directory for actuals.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from deformation_intel.context import (
    PAD_THRESHOLD,
    agriculture_score,
    industrial_pad_score,
    is_cultivated_confound,
    osm_infrastructure_sampler,
)
from deformation_intel.review import contact_sheet
from deformation_intel.vlm_review import (
    DEFAULT_MODEL,
    FOCUS_PROMPT,
    MODELS,
    WIDE_PROMPT,
    encode_png,
    parse_wide_response,
)
from interesting_intel import features as F
from interesting_intel import priors as P

logger = logging.getLogger(__name__)

CONUS = (-125.5, 24.0, -66.5, 49.5)          # NAIP exists only here


# ------------------------------------------------------------------ fetching

def _read_bands(url: str, grid, width: int, height: int, bands=(1, 2, 3),
                signed: bool = False, retries: int = 3) -> np.ndarray:
    """Multi-band chip read: windowed DECIMATED read in the source CRS.

    Returns (len(bands), height, width) float32 with nan nodata.

    Deliberately not reproject()-based: warping reads the source window at
    FULL resolution (measured 185 s for one 1 km NAIP chip), while a windowed
    read with an out_shape smaller than the window lets GDAL pick a COG
    overview level (measured seconds). The chip is axis-aligned in the
    source's UTM rather than in lon/lat — a <2 degree skew that is irrelevant
    for ranking/looking but would matter for measurement, so nothing
    downstream may measure geometry off these chips.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as win_from_bounds

    from archaeo_intel.data_access import mpc_sign, vsicurl

    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            path = "/vsicurl/" + mpc_sign(url) if signed else vsicurl(url)
            if not signed:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            with rasterio.open(path) as ds:
                wb = transform_bounds("EPSG:4326", ds.crs, *grid,
                                      densify_pts=21)
                win = win_from_bounds(*wb, ds.transform)
                data = ds.read(list(bands), window=win,
                               out_shape=(len(bands), height, width),
                               boundless=True, masked=True,
                               resampling=Resampling.bilinear,
                               fill_value=ds.nodata)
            out = data.astype("float32").filled(np.nan)
            if ds.nodata is not None:
                out[out == float(ds.nodata)] = np.nan
            return out
        except Exception as exc:
            last = exc
            # a 403 mid-run usually means an expired SAS token that the
            # per-container cache still believes in — drop it so the retry
            # re-signs instead of replaying the dead token
            try:
                from archaeo_intel import data_access
                data_access._mpc_tokens.clear()
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _nanmean_bands(acc: np.ndarray) -> np.ndarray:
    """Band-mean to grayscale without the all-nan-pixel RuntimeWarning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(acc, axis=0).astype("float32")


def chip_grid(lat: float, lon: float, half_m: float):
    dlat = half_m / P.M_PER_DEG_LAT
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def fetch_chip_naip(lat: float, lon: float, half_m: float, px: int
                    ) -> Optional[dict]:
    """NAIP RGB chip via Planetary Computer (signed). ~1 m, CONUS only."""
    from archaeo_intel.data_access import stac_search

    MPC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    grid = chip_grid(lat, lon, half_m)
    feats = stac_search("naip", list(grid),
                        datetime="2018-01-01T00:00:00Z/2026-12-31T23:59:59Z",
                        limit=6, endpoint=MPC)
    acc = np.full((3, px, px), np.nan, "float32")
    for f in feats:
        try:
            g = _read_bands(f["assets"]["image"]["href"], grid, px, px,
                            bands=(1, 2, 3), signed=True)
        except Exception:
            continue
        acc = np.where(np.isfinite(acc), acc, g)
        if np.isfinite(acc[0]).mean() > 0.95:
            break
    if np.isfinite(acc[0]).mean() < 0.5:
        return None
    rgb = np.clip(np.nan_to_num(acc, nan=0.0), 0, 255).astype("uint8")
    return {"gray": _nanmean_bands(acc), "rgb": np.moveaxis(rgb, 0, -1),
            "source": "naip", "res_m": 2 * half_m / px}


def fetch_chip_s2(lat: float, lon: float, half_m: float, px: int
                  ) -> Optional[dict]:
    """Sentinel-2 true-colour (TCI) chip via Earth Search. 10 m, global."""
    from archaeo_intel.data_access import stac_search

    grid = chip_grid(lat, lon, half_m)
    feats = stac_search("sentinel-2-l2a", list(grid),
                        datetime="2023-01-01T00:00:00Z/2026-12-31T23:59:59Z",
                        query={"eo:cloud_cover": {"lt": 15}}, limit=12,
                        sortby=[{"field": "properties.eo:cloud_cover",
                                 "direction": "asc"}])
    for f in feats:
        href = f.get("assets", {}).get("visual", {}).get("href")
        if not href:
            continue
        try:
            g = _read_bands(href, grid, px, px, bands=(1, 2, 3))
        except Exception:
            continue
        if np.isfinite(g[0]).mean() < 0.5:
            continue
        rgb = np.clip(np.nan_to_num(g, nan=0.0), 0, 255).astype("uint8")
        return {"gray": _nanmean_bands(g), "rgb": np.moveaxis(rgb, 0, -1),
                "source": "s2", "res_m": 2 * half_m / px}
    return None


def pick_source(lat: float, lon: float, source: str = "auto") -> str:
    if source != "auto":
        return source
    w, s, e, n = CONUS
    return "naip" if (w <= lon <= e and s <= lat <= n) else "s2"


class ChipCache:
    """Disk cache keyed by rounded lat/lon + source + geometry.

    The imagery fetch is the funnel's bottleneck (~2-4 s per chip serial);
    the cache guarantees we never pay for the same chip twice, including
    across resumed / re-parameterised runs. Failures are cached too (as .fail
    markers) so a resumed run doesn't re-hammer dead locations.
    """

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.fetches = 0
        self.failures = 0

    def _key(self, lat, lon, half_m, px, source) -> str:
        return f"{source}_{lat:+.5f}_{lon:+.5f}_{int(half_m)}_{int(px)}"

    def get(self, lat, lon, half_m: float, px: int, source: str = "auto",
            retry_failed: bool = False) -> Optional[dict]:
        src = pick_source(lat, lon, source)
        key = self._key(lat, lon, half_m, px, src)
        npz = self.dir / (key + ".npz")
        fail = self.dir / (key + ".fail")
        if npz.exists():
            try:
                with np.load(npz) as z:
                    self.hits += 1
                    return {"gray": z["gray"], "rgb": z["rgb"], "source": src,
                            "res_m": 2 * half_m / px}
            except Exception:
                npz.unlink(missing_ok=True)
        if fail.exists() and not retry_failed:
            return None
        fetch = fetch_chip_naip if src == "naip" else fetch_chip_s2
        try:
            chip = fetch(lat, lon, half_m, px)
        except Exception:
            chip = None
        self.fetches += 1
        if chip is None:
            self.failures += 1
            fail.touch()
            return None
        fail.unlink(missing_ok=True)
        np.savez_compressed(npz, gray=chip["gray"], rgb=chip["rgb"])
        return chip


def fetch_many(cache: ChipCache, points: Sequence[dict], half_m: float,
               px: int, source: str = "auto", workers: int = 12,
               log=logger.info) -> Dict[int, dict]:
    """Parallel chip fetch for a list of {lat, lon} dicts -> {index: chip}.

    Aborts loudly if the first 20 attempts ALL fail — an auth/endpoint problem
    looks exactly like this, and the alternative is a counter cheerfully
    reaching 100% while producing nothing.
    """
    out: Dict[int, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cache.get, p["lat"], p["lon"], half_m, px, source): i
                for i, p in enumerate(points)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                chip = fut.result()
            except Exception:
                chip = None
            if chip is not None:
                out[i] = chip
            done += 1
            if done == 20 and not out:
                for f in futs:
                    f.cancel()
                raise RuntimeError("first 20 chip fetches all failed — "
                                   "aborting instead of sweeping nothing")
            if done % 50 == 0:
                log(f"  chips {done}/{len(points)} ({len(out)} ok, "
                    f"{cache.hits} cached)")
    return out


# ----------------------------------------------------------------- VLM layer

def vlm_call(model: str, image_path, prompt: str, *, detail_note: str = "",
             api_key: Optional[str] = None, max_tokens: int = 1500,
             timeout: int = 240) -> dict:
    """One OpenRouter vision call. Returns {text, prompt_tokens,
    completion_tokens} so the pipeline can report ACTUAL cost, not estimates.

    Per the VRSBench/GeoGround lesson, callers must put ground width, north-up
    and sensor/resolution into detail_note — general VLMs are weak on overhead
    scale and orientation without it.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    full = f"{detail_note}\n\n{prompt}" if detail_note else prompt
    body = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": full},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encode_png(image_path)}},
        ]}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "X-Title": "GeoAnomalyMapper"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    usage = data.get("usage", {})
    return {"text": data["choices"][0]["message"]["content"],
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0))}


def model_id(name_or_id: str) -> str:
    """Accept either a MODELS alias ('open-best') or a raw OpenRouter id."""
    if name_or_id in MODELS:
        return MODELS[name_or_id][0]
    return name_or_id or DEFAULT_MODEL


def usd_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    for mid, p_in, p_out in MODELS.values():
        if mid == model:
            return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6
    return (prompt_tokens * 0.5 + completion_tokens * 2.0) / 1e6   # unknown


# Appended to FOCUS_PROMPT: the stock prompt's interest scale measures
# UNEXPLAINABILITY, and a skeptical model correctly rates almost everything 0
# (measured: all 21 Racetrack-region focus reviews scored 0, including a
# crater-like depression and a lone dark outcrop on an empty playa). The
# product metric is "would a human say huh and look" — a fully explained but
# spectacular feature still belongs at the top of the queue.
FOCUS_ADDENDUM = """
6. worth_a_glance 0-3: independent of explainability — would a curious person
   find this visually striking or memorable? A fully explained but spectacular
   feature (a huge open-pit mine, a bizarre natural formation, a lone object
   on an empty plain) still scores 2-3. 0 = nobody would look twice."""


def parse_worth_glance(text: str) -> int:
    """Pull the 0-3 worth_a_glance rating out of a focus response."""
    import re

    t = re.sub(r"\(?\s*0\s*[-–]\s*3\s*\)?", " ", (text or "").lower())
    m = list(re.finditer(r"worth[_ ]?a[_ ]?glance\D{0,20}?([0-3])", t))
    return int(m[-1].group(1)) if m else 0


def parse_focus_interest(text: str) -> int:
    """Pull the 0-3 interest rating out of a FOCUS_PROMPT response.

    Scale mentions like "(0-3)" are stripped first so they can't be mistaken
    for the rating itself.
    """
    import re

    t = re.sub(r"\(?\s*0\s*[-–]\s*3\s*\)?", " ", (text or "").lower())
    m = list(re.finditer(r"interest\D{0,20}?([0-3])", t))
    if m:
        return int(m[-1].group(1))
    m2 = list(re.finditer(r"^\s*4\.\D{0,30}?([0-3])", t, re.MULTILINE))
    return int(m2[-1].group(1)) if m2 else 0


# ------------------------------------------------------------------- ranking

def pct_rank(values: Sequence[float]) -> np.ndarray:
    """Percentile rank in [0,1]; nan values get 0.5 (uninformative, not
    disqualifying — a missing signal must never act as a veto)."""
    v = np.asarray(values, dtype="float64")
    out = np.full(v.shape, 0.5)
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        from scipy.stats import rankdata
        out[ok] = (rankdata(v[ok]) - 1) / (ok.sum() - 1)
    return out


def rank_candidates(records: List[dict]) -> List[dict]:
    """Combine novelty / geometry / prior strength into interest, demote
    (never delete) definitive confounds, and sort descending.

    Weights: novelty leads (the product IS out-of-placeness), geometry second,
    Stage-1 prior strength third. Confound multipliers are reserved for the
    two visually-definitive explanations this project has validated detectors
    for: cultivated-flat-land and industrial pads.
    """
    if not records:
        return []
    nov = pct_rank([r.get("novelty", float("nan")) for r in records])
    geo = pct_rank([r.get("geometry", float("nan")) for r in records])
    pri = pct_rank([r.get("score1", 0.0) for r in records])
    for i, r in enumerate(records):
        base = 0.45 * nov[i] + 0.35 * geo[i] + 0.20 * pri[i]
        mult = 1.0
        why = []
        ag = r.get("agriculture", float("nan"))
        slope = r.get("slope_deg", float("nan"))
        if is_cultivated_confound(ag, slope):
            mult *= 0.3
            why.append("cultivated")
        if (r.get("pad") or 0.0) >= PAD_THRESHOLD:
            mult *= 0.4
            why.append("industrial_pad")
        osm = r.get("osm_infra", float("nan"))
        if np.isfinite(osm):
            mult *= 1.1 if osm == 0 else (0.9 if osm >= 10 else 1.0)
        r["interest"] = round(float(base * mult), 4)
        r["demoted_for"] = why
    return sorted(records, key=lambda r: -r["interest"])


def slope_at(dem: np.ndarray, bbox, lat: float, lon: float, px_m: float,
             win: int = 5) -> float:
    """Mean slope (deg) in a small window of the Stage-1 DEM mosaic."""
    if dem is None:
        return float("nan")
    h, w = dem.shape
    lon_min, lat_min, lon_max, lat_max = bbox
    c = int((lon - lon_min) / (lon_max - lon_min) * w)
    r = int((lat_max - lat) / (lat_max - lat_min) * h)
    r0, r1 = max(r - win, 0), min(r + win + 1, h)
    c0, c1 = max(c - win, 0), min(c + win + 1, w)
    patch = dem[r0:r1, c0:c1]
    if patch.size < 9 or not np.isfinite(patch).any():
        return float("nan")
    p = np.nan_to_num(patch, nan=float(np.nanmedian(patch)))
    gy, gx = np.gradient(p, px_m)
    return float(np.degrees(np.arctan(np.hypot(gx, gy))).mean())


# -------------------------------------------------------------- the funnel

def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def save_png(rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def run_funnel(bbox, out_dir, *, source: str = "auto", half_m: float = 500.0,
               px: int = 500, workers: int = 12, max_stage2: int = 1200,
               n_background: int = 60, max_sheets_sites: int = 500,
               per_sheet: int = 25, vlm_model: str = "open-best",
               focus_model: Optional[str] = None, min_interest: int = 2,
               max_focus: int = 120, use_vlm: bool = True,
               use_osm: bool = False, max_osm: int = 120,
               use_change: bool = True, use_grid: bool = False,
               filter_sheets: bool = True,
               seeds: Sequence[dict] = (), force: bool = False,
               rng_seed: int = 7, log=None) -> dict:
    """Run the full funnel over bbox and write everything under out_dir.

    Resumable: every stage writes an artifact JSON and is skipped when it
    already exists (unless force=True). The chip cache lives in
    out_dir/../chip_cache by default so neighbouring runs share it.
    """
    t_all = time.time()
    out = Path(out_dir)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    log = log or (lambda m: logger.info(m))
    report: dict = _load(art / "run_report.json") or {
        "bbox": list(bbox), "stages": {}, "vlm": {"calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}}
    cache = ChipCache(Path(out_dir).parent / "chip_cache")

    # ---------------- stage 1: priors
    t = time.time()
    s1_path = art / "stage1_candidates.json"
    dem_path = art / "dem.npz"
    cands = None if force else _load(s1_path)
    dem = None
    if dem_path.exists():
        with np.load(dem_path) as z:
            dem = z["dem"]
    if cands is None:
        dem = P.fetch_dem_mosaic(bbox, log=log) if dem is None else dem
        if dem is not None:
            np.savez_compressed(dem_path, dem=dem)
        cands = P.generate_candidates(bbox, dem=dem, use_change=use_change,
                                      use_grid=use_grid, log=log)
        for s in seeds:
            cands.append({"lat": s["lat"], "lon": s["lon"],
                          "priors": {"seed": 5.0}, "score1": 5.0,
                          "seed_name": s.get("name", "seed")})
        _save(s1_path, cands)
        report["stages"]["stage1"] = {"sec": round(time.time() - t, 1),
                                      "candidates": len(cands)}
        _save(art / "run_report.json", report)   # survive a mid-run death
    log(f"stage1: {len(cands)} candidates")

    # ---------------- stage 2: chips + CV features
    t = time.time()
    s2_path = art / "stage2_ranked.json"
    ranked = None if force else _load(s2_path)
    if ranked is None:
        chosen = sorted(cands, key=lambda c: -c["score1"])[:max_stage2]
        chips = fetch_many(cache, chosen, half_m, px, source, workers, log)
        # background sample: random locations in the same region — novelty is
        # DEFINED as distance from these
        rng = np.random.default_rng(rng_seed)
        lon_min, lat_min, lon_max, lat_max = bbox
        bg_pts = [{"lat": float(rng.uniform(lat_min, lat_max)),
                   "lon": float(rng.uniform(lon_min, lon_max))}
                  for _ in range(n_background)]
        bg_chips = fetch_many(cache, bg_pts, half_m, px, source, workers, log)
        bg_vecs = [F.texture_vector(c["gray"]) for c in bg_chips.values()
                   if F.chip_ok(c["gray"])]
        if len(bg_vecs) < 8:
            log(f"WARNING stage2: only {len(bg_vecs)} background chips — "
                "novelty will be nan and ranking falls back to geometry")
        px_m_dem = 30.0

        def score_one(item):
            i, c = item
            chip = chips.get(i)
            if chip is None or not F.chip_ok(chip["gray"]):
                return None
            g = chip["gray"]
            feats = F.chip_features(g)
            rec = dict(c)
            rec["source"] = chip["source"]
            rec["res_m"] = round(chip["res_m"], 2)
            rec["geometry"] = round(feats["geometry"], 4)
            rec["novelty"] = round(F.novelty_score(feats["vector"], bg_vecs), 3)
            rec["straightedge"] = round(feats["straightedge"], 3)
            rec["faint_line"] = round(feats["faint_line"], 3)
            rec["radiality"] = round(feats["radiality"], 3)
            rec["agriculture"] = round(agriculture_score(g), 3)
            rec["pad"] = round(industrial_pad_score(g), 3)
            rec["slope_deg"] = round(slope_at(dem, bbox, c["lat"], c["lon"],
                                              px_m_dem), 2)
            rec["center_contrast"] = round(F.center_contrast(g), 2)
            return rec

        with ThreadPoolExecutor(max_workers=max(workers // 2, 2)) as ex:
            records = [r for r in ex.map(score_one, enumerate(chosen))
                       if r is not None]
        if not records:
            raise RuntimeError("stage2 produced zero scored records — check "
                               "imagery access before believing any output")
        if use_osm:
            sampler = osm_infrastructure_sampler()
            pre = rank_candidates(records)
            for r in pre[:max_osm]:
                r["osm_infra"] = sampler(r["lat"], r["lon"])
        ranked = rank_candidates(records)
        for n, r in enumerate(ranked):
            r["rank"] = n + 1
        _save(s2_path, ranked)
        report["stages"]["stage2"] = {
            "sec": round(time.time() - t, 1), "scored": len(ranked),
            "fetches": cache.fetches, "cache_hits": cache.hits,
            "fetch_failures": cache.failures}
        _save(art / "run_report.json", report)
    log(f"stage2: {len(ranked)} scored & ranked")

    # ---------------- stage 3: contact sheets + wide VLM
    t = time.time()
    s3_path = art / "stage3_wide.json"
    sheets_dir = out / "sheets"
    wide = None if force else _load(s3_path)
    top = ranked[:max_sheets_sites]
    if wide is None:
        sheet_paths: List[str] = []
        for si in range(0, len(top), per_sheet):
            part = top[si:si + per_sheet]
            chips, labels = [], []
            for r in part:
                chip = cache.get(r["lat"], r["lon"], half_m, px, source)
                if chip is None:
                    chip = {"gray": np.zeros((8, 8), "float32")}
                chips.append(chip.get("rgb", chip["gray"]))
                labels.append(f"{r['rank']}. {r['lat']:.4f},{r['lon']:.4f}")
            p = contact_sheet(chips, labels,
                              sheets_dir / f"sheet_{si//per_sheet + 1:03d}.png",
                              cols=5, thumb=256,
                              title=f"interesting_intel sites "
                                    f"{si+1}-{si+len(part)}")
            sheet_paths.append(str(p))
        if not sheet_paths or not all(Path(s).exists() for s in sheet_paths):
            raise RuntimeError("stage3 wrote no contact sheets")
        wide = {"sheets": sheet_paths, "records": []}
        if use_vlm:
            wm = model_id(vlm_model)
            note = (f"Context: each thumbnail cell covers {2*half_m:.0f} m x "
                    f"{2*half_m:.0f} m of ground, north is up, source is "
                    f"aerial/satellite imagery at ~{2*half_m/px:.1f} m per "
                    f"pixel, true colour. Cell labels give a rank index and "
                    f"latitude,longitude.")
            for sp in sheet_paths:
                try:
                    resp = vlm_call(wm, sp, WIDE_PROMPT, detail_note=note)
                except Exception as exc:
                    log(f"stage3: VLM call failed on {sp}: {exc}")
                    continue
                report["vlm"]["calls"] += 1
                report["vlm"]["prompt_tokens"] += resp["prompt_tokens"]
                report["vlm"]["completion_tokens"] += resp["completion_tokens"]
                report["vlm"]["usd"] += usd_cost(wm, resp["prompt_tokens"],
                                                 resp["completion_tokens"])
                for rec in parse_wide_response(resp["text"]):
                    rec["sheet"] = sp
                    wide["records"].append(rec)
        _save(s3_path, wide)
        report["stages"]["stage3"] = {"sec": round(time.time() - t, 1),
                                      "sheets": len(wide["sheets"]),
                                      "notable": len(wide["records"])}
        _save(art / "run_report.json", report)
    log(f"stage3: {len(wide['sheets'])} sheets, "
        f"{len(wide['records'])} notable cells")

    # ---------------- stage 4: focus VLM on full-res chips
    t = time.time()
    s4_path = art / "stage4_focus.json"
    focus = None if force else _load(s4_path)
    if focus is None:
        by_rank = {r["rank"]: r for r in ranked}
        picks = sorted((w for w in wide["records"]
                        if w.get("interest", 0) >= min_interest
                        and w.get("index") in by_rank),
                       key=lambda w: -w.get("interest", 0))[:max_focus]
        focus = []
        fm = model_id(focus_model or "open-best")
        big_half, big_px = half_m * 1.5, 768
        note = (f"Context: the image covers {2*big_half:.0f} m x "
                f"{2*big_half:.0f} m of ground, north is up, "
                f"aerial/satellite imagery at ~{2*big_half/big_px:.1f} m "
                f"per pixel, true colour.")
        for wrec in picks:
            r = by_rank[wrec["index"]]
            chip = cache.get(r["lat"], r["lon"], big_half, big_px, source)
            if chip is None:
                continue
            png = save_png(chip["rgb"],
                           out / "chips_full" / f"rank_{r['rank']:04d}.png")
            entry = {"rank": r["rank"], "lat": r["lat"], "lon": r["lon"],
                     "wide_interest": wrec.get("interest", 0),
                     "wide_description": wrec.get("description", ""),
                     "png": str(png)}
            if use_vlm:
                try:
                    resp = vlm_call(fm, png, FOCUS_PROMPT + FOCUS_ADDENDUM,
                                    detail_note=note)
                    report["vlm"]["calls"] += 1
                    report["vlm"]["prompt_tokens"] += resp["prompt_tokens"]
                    report["vlm"]["completion_tokens"] += resp["completion_tokens"]
                    report["vlm"]["usd"] += usd_cost(fm, resp["prompt_tokens"],
                                                     resp["completion_tokens"])
                    entry["focus_interest"] = parse_focus_interest(resp["text"])
                    entry["worth_glance"] = parse_worth_glance(resp["text"])
                    entry["focus_text"] = resp["text"]
                except Exception as exc:
                    log(f"stage4: VLM call failed rank {r['rank']}: {exc}")
            focus.append(entry)
        _save(s4_path, focus)
        report["stages"]["stage4"] = {"sec": round(time.time() - t, 1),
                                      "focused": len(focus)}
        _save(art / "run_report.json", report)
    log(f"stage4: {len(focus)} full-res reviews")

    # ---------------- stage 5: final queue for the human
    t = time.time()
    finals = sorted(focus, key=lambda e: (-(e.get("worth_glance") or 0),
                                          -(e.get("focus_interest") or 0),
                                          -(e.get("wide_interest") or 0),
                                          e["rank"]))
    queue = finals[:100]
    if not queue:      # no VLM (or nothing flagged): fall back to CV ranking
        queue = [{"rank": r["rank"], "lat": r["lat"], "lon": r["lon"]}
                 for r in ranked[:100]]
    by_rank = {r["rank"]: r for r in ranked}
    if queue:
        chips, labels = [], []
        for e in queue:
            chip = cache.get(e["lat"], e["lon"], half_m, px, source)
            chips.append(chip.get("rgb", chip["gray"]) if chip
                         else np.zeros((8, 8), "float32"))
            labels.append(f"{e['rank']}. w{e.get('worth_glance', '?')}"
                          f"i{e.get('focus_interest', '?')} "
                          f"{e['lat']:.4f},{e['lon']:.4f}")
        contact_sheet(chips, labels, out / "final_queue.png", cols=10,
                      thumb=200, title="FINAL QUEUE — worth a glance?")
    # multi-filter dossier per queue site (NAIP false-IR, S2 true/SWIR,
    # iron-oxide + clay ratios, hillshade) — skipped when already on disk,
    # so a resumed run only fills gaps
    fpaths: Dict[int, str] = {}
    if filter_sheets and queue:
        from interesting_intel.filters import filter_sheet

        def one_sheet(e):
            p = out / "filters" / f"rank_{e['rank']:04d}_filters.png"
            if p.exists():
                return e["rank"], str(p)
            try:
                got = filter_sheet(e["lat"], e["lon"], p,
                                   title=f"rank {e['rank']}  "
                                         f"{e['lat']:.4f}, {e['lon']:.4f}")
                return e["rank"], (str(got) if got else None)
            except Exception as exc:
                log(f"stage5: filter sheet failed rank {e['rank']}: {exc}")
                return e["rank"], None

        with ThreadPoolExecutor(max_workers=max(workers // 3, 2)) as ex:
            for rank, p in ex.map(one_sheet, queue):
                if p:
                    fpaths[rank] = p
        log(f"stage5: {len(fpaths)}/{len(queue)} filter sheets")

    lines = ["# interesting_intel run report", "",
             f"bbox: {list(bbox)}", "",
             f"funnel: {len(cands)} candidates -> {len(ranked)} scored -> "
             f"{len(top)} sheeted -> {len(focus)} focused -> "
             f"{len(queue)} in final queue", ""]
    for e in queue:
        r = by_rank.get(e["rank"], {})
        lines += [f"## rank {e['rank']}  ({e['lat']:.4f}, {e['lon']:.4f})  "
                  f"worth_a_glance {e.get('worth_glance', '?')}/3  "
                  f"unexplained {e.get('focus_interest', '?')}/3",
                  f"- priors: {r.get('priors', {})}  novelty "
                  f"{r.get('novelty')}  geometry {r.get('geometry')}"
                  f"  demoted_for: {r.get('demoted_for', [])}",
                  f"- wide pass: {e.get('wide_description', '')}",
                  f"- chip: {e.get('png', '')}",
                  f"- filters: {fpaths.get(e['rank'], 'n/a')}", ""]
        if e.get("focus_text"):
            lines += ["```", e["focus_text"].strip(), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    report["stages"]["stage5"] = {"sec": round(time.time() - t, 1),
                                  "queue": len(queue),
                                  "filter_sheets": len(fpaths)}
    report["total_sec"] = round(time.time() - t_all, 1)
    report["vlm"]["usd"] = round(report["vlm"]["usd"], 4)
    _save(art / "run_report.json", report)
    log(f"done in {report['total_sec']}s, VLM ${report['vlm']['usd']}, "
        f"queue={len(queue)} -> {out / 'report.md'}")
    return report
