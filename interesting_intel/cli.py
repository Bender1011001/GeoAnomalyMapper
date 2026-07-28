"""CLI:  python -m interesting_intel --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --out DIR

Runs the full funnel (priors -> CV -> VLM -> report). Common variants:

    # no VLM spend, CV ranking only
    python -m interesting_intel --bbox -117.75 36.45 -117.35 36.85 \
        --out results/interesting/racetrack --no-vlm

    # seed positive controls into the candidate pool
    python -m interesting_intel --bbox ... --out ... \
        --seed 36.681 -117.563 racetrack_playa

    # positive/negative-control validation harness (no region sweep)
    python -m interesting_intel.validate
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="interesting_intel",
        description="Rank free satellite imagery by 'worth a human glance'.")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--source", default="auto", choices=["auto", "naip", "s2"])
    ap.add_argument("--half-m", type=float, default=500.0,
                    help="chip half-width in metres (default 500)")
    ap.add_argument("--px", type=int, default=500,
                    help="chip size in pixels (default 500)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-stage2", type=int, default=1200,
                    help="candidates that get imagery + CV scores")
    ap.add_argument("--max-sheets-sites", type=int, default=500,
                    help="candidates that reach VLM contact sheets")
    ap.add_argument("--max-focus", type=int, default=120,
                    help="chips that reach the full-res VLM pass")
    ap.add_argument("--vlm-model", default="open-best",
                    help="MODELS alias or OpenRouter id for the wide pass")
    ap.add_argument("--focus-model", default=None,
                    help="model for the full-res pass (default: wide model)")
    ap.add_argument("--min-interest", type=int, default=2,
                    help="wide-pass interest needed for a full-res look")
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip both VLM passes (CV ranking only)")
    ap.add_argument("--osm", action="store_true",
                    help="query Overpass for mapped infrastructure (slow, "
                         "polite-rate; adjusts rank only)")
    ap.add_argument("--no-change", action="store_true",
                    help="skip the S2 change-over-time prior")
    ap.add_argument("--no-filter-sheets", action="store_true",
                    help="skip the per-site multi-filter dossier panels")
    ap.add_argument("--grid", action="store_true",
                    help="add unbiased grid candidates (coverage floor)")
    ap.add_argument("--grid-m", type=float, default=500.0)
    ap.add_argument("--seed", nargs=3, action="append", default=[],
                    metavar=("LAT", "LON", "NAME"),
                    help="inject a known site into the candidate pool "
                         "(repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="recompute stages even when artifacts exist")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    from interesting_intel.pipeline import run_funnel

    seeds = [{"lat": float(a), "lon": float(b), "name": c}
             for a, b, c in args.seed]
    report = run_funnel(
        tuple(args.bbox), args.out, source=args.source, half_m=args.half_m,
        px=args.px, workers=args.workers, max_stage2=args.max_stage2,
        max_sheets_sites=args.max_sheets_sites, max_focus=args.max_focus,
        vlm_model=args.vlm_model, focus_model=args.focus_model,
        min_interest=args.min_interest, use_vlm=not args.no_vlm,
        use_osm=args.osm, use_change=not args.no_change, use_grid=args.grid,
        filter_sheets=not args.no_filter_sheets,
        seeds=seeds, force=args.force, log=lambda m: print(m, flush=True))
    print(f"VLM cost ${report['vlm']['usd']}, "
          f"total {report['total_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
