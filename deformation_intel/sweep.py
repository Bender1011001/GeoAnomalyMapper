"""Region-scale deformation sweep driver — push-button, resumable, granule-major.

Turns a bounding box into ranked subsidence candidates using the fast
granule-major reader (opera.build_frame_cubes), the validated detector
(detect.detect_anomalies), and the confound screen (context.is_cultivated_
confound). Everything streams to a per-tile cache + per-tile result JSON, so a
re-launch resumes where it stopped.

Pure helpers (tile_grid, rank_candidates) are unit-tested; the network path is
exercised by the end-to-end benchmarks in scratchpad.

CLI:
    python -m deformation_intel.sweep --bbox -103.6 31.6 -102.6 32.4 \
        --out data/research/permian_sweep --tile-km 24 --workers 8

IMPORTANT (Windows / macOS spawn): run_region_sweep spawns a process pool, so a
script that calls it MUST guard its entry point:

    if __name__ == "__main__":
        run_region_sweep(...)

Without the guard, each pool worker re-imports and re-runs the caller's
top-level code (a mini fork-bomb). The `-m deformation_intel.sweep` CLI is
already guarded; only ad-hoc caller scripts need this.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]   # (lon_min, lat_min, lon_max, lat_max)


def tile_grid(bbox: BBox, tile_km: float = 24.0) -> Dict[str, Tuple[float, float]]:
    """Regular grid of AOI tile centres covering `bbox`.

    Longitude spacing widens with latitude (cos correction) so tiles stay
    ~tile_km on a side. Keys are stable, coordinate-derived strings so a resumed
    run reproduces the same tiling. Pure — no network.
    """
    lon0, lat0, lon1, lat1 = bbox
    if lon1 <= lon0 or lat1 <= lat0:
        raise ValueError("bbox must be (lon_min, lat_min, lon_max, lat_max)")
    dlat = tile_km / 111.0
    tiles: Dict[str, Tuple[float, float]] = {}
    lat = lat0 + dlat / 2.0
    while lat < lat1:
        dlon = tile_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
        lon = lon0 + dlon / 2.0
        while lon < lon1:
            key = f"t_{lat:.4f}_{lon:.4f}"
            tiles[key] = (round(lat, 4), round(lon, 4))
            lon += dlon
        lat += dlat
    return tiles


def _neighborhood(tiles: Dict[str, Tuple[float, float]], seed_key: str,
                  span_deg: float = 3.0) -> Dict[str, Tuple[float, float]]:
    """Tiles within span_deg of the seed — one call ~= one frame's worth, so the
    frame the centroid selects actually covers most of what we pass."""
    slat, slon = tiles[seed_key]
    return {k: (la, lo) for k, (la, lo) in tiles.items()
            if abs(la - slat) <= span_deg and abs(lo - slon) <= span_deg}


def rank_candidates(candidates: List[dict]) -> List[dict]:
    """Sort localized candidates by a void-priority score (pure, testable).

    score = void_likelihood * (|accel| + 0.3*|peak_velocity|). Regional and
    non-localized anomalies rank below any localized one.
    """
    def score(c: dict) -> float:
        loc = 1.0 if c.get("is_localized") else 0.0
        vl = c.get("void_likelihood") or c.get("confidence") or 0.5
        acc = abs(c.get("accel_cm_yr2") or 0.0)
        vel = abs(c.get("peak_velocity_cm_yr") or 0.0)
        return loc * 100 + vl * (acc + 0.3 * vel)

    return sorted(candidates, key=score, reverse=True)


def _anomaly_to_dict(a, tile_key: str) -> dict:
    d = {k: getattr(a, k) for k in (
        "lat", "lon", "kind", "classification", "confidence",
        "peak_velocity_cm_yr", "mean_velocity_cm_yr", "accel_cm_yr2",
        "area_km2", "n_pixels", "sigma", "cumulative_cm", "is_localized",
        "source_depth_m", "source_volume_rate_m3_yr", "void_likelihood")
        if hasattr(a, k)}
    d["tile"] = tile_key
    ctx = getattr(a, "context", None)
    if ctx:
        d["context"] = ctx
    return d


def run_region_sweep(
    bbox: BBox,
    out_dir: str | Path,
    *,
    tile_km: float = 24.0,
    half_width_km: float = 11.0,
    workers: int = 8,
    max_epochs: Optional[int] = None,
    coherence_threshold: float = 0.6,
    min_epochs: int = 8,
    with_context: bool = True,
    resume: bool = True,
) -> dict:
    """Run the full region sweep. Returns a summary dict; writes per-tile JSON
    results and an aggregate ranked candidate list under out_dir.

    Resumable: tiles whose result JSON already exists are skipped. Frame-by-
    frame — each iteration builds one frame's tiles granule-major, detects,
    screens confounds, writes results, then moves to the next unbuilt tile.
    """
    from deformation_intel.detect import detect_anomalies
    from deformation_intel.opera import build_frame_cubes

    out = Path(out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    cache_root = out / "cache"

    samplers = None
    if with_context:
        try:
            from archaeo_intel.data_access import read_grid, stac_search
            from deformation_intel.context import make_default_samplers
            samplers = make_default_samplers(read_grid_fn=read_grid,
                                             stac_search_fn=stac_search)
        except Exception as exc:
            logger.warning("context samplers unavailable (%s); running bare",
                           type(exc).__name__)

    all_tiles = tile_grid(bbox, tile_km)
    logger.info("region %s -> %d tiles", bbox, len(all_tiles))

    remaining = dict(all_tiles)
    if resume:
        for k in list(remaining):
            if (out / "tiles" / f"{k}.json").exists():
                remaining.pop(k)
        logger.info("resume: %d/%d tiles already done",
                    len(all_tiles) - len(remaining), len(all_tiles))

    n_built = 0
    n_cand = 0
    guard = 0
    while remaining and guard < len(all_tiles) + 5:
        guard += 1
        seed = next(iter(remaining))
        batch = _neighborhood(remaining, seed)
        try:
            cubes, unassigned = build_frame_cubes(
                batch, half_width_km=half_width_km,
                coherence_threshold=coherence_threshold, max_epochs=max_epochs,
                workers=workers, min_epochs=min_epochs, cache_root=cache_root,
                progress=True)
        except Exception as exc:
            logger.warning("frame build failed at seed %s: %s -> parking batch",
                           seed, type(exc).__name__)
            # park the seed so we don't loop forever on a bad frame
            (out / "tiles" / f"{seed}.json").write_text(
                json.dumps({"tile": seed, "error": type(exc).__name__,
                            "anoms": []}))
            remaining.pop(seed, None)
            continue

        built_keys = set(batch) - set(unassigned)
        for k in built_keys:
            cube = cubes.get(k)
            anoms: List[dict] = []
            if cube is not None:
                try:
                    found = detect_anomalies(
                        cube, context_samplers=samplers,
                        pixel_size_m=half_width_km * 2000.0 / cube["cube"].shape[1])
                    anoms = [_anomaly_to_dict(a, k) for a in found]
                except Exception as exc:
                    logger.warning("detect failed %s: %s", k, type(exc).__name__)
            (out / "tiles" / f"{k}.json").write_text(
                json.dumps({"tile": k, "lat": all_tiles[k][0],
                            "lon": all_tiles[k][1], "anoms": anoms}))
            n_built += 1
            n_cand += len(anoms)
            remaining.pop(k, None)

        if not built_keys:
            # centroid frame covered none of the batch — park the seed
            (out / "tiles" / f"{seed}.json").write_text(
                json.dumps({"tile": seed, "note": "no frame coverage",
                            "anoms": []}))
            remaining.pop(seed, None)

    # aggregate + rank
    agg: List[dict] = []
    for f in (out / "tiles").glob("*.json"):
        try:
            agg.extend(json.loads(f.read_text()).get("anoms", []))
        except Exception:
            pass
    localized = [c for c in agg if c.get("is_localized")]
    ranked = rank_candidates(localized)
    (out / "candidates_ranked.json").write_text(json.dumps(ranked, indent=1))
    summary = {"bbox": bbox, "tiles_total": len(all_tiles),
               "tiles_built": n_built, "anomalies": n_cand,
               "localized": len(localized)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("SWEEP DONE: %s", summary)
    return summary


def _main(argv=None):
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Region-scale OPERA deformation sweep")
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--out", required=True)
    p.add_argument("--tile-km", type=float, default=24.0)
    p.add_argument("--half-width-km", type=float, default=11.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-context", action="store_true")
    a = p.parse_args(argv)
    run_region_sweep(tuple(a.bbox), a.out, tile_km=a.tile_km,
                     half_width_km=a.half_width_km, workers=a.workers,
                     max_epochs=a.max_epochs, with_context=not a.no_context)


if __name__ == "__main__":
    _main()
