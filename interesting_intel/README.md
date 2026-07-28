# interesting_intel — a general "worth a glance" ranker

Looks at large volumes of free satellite/aerial imagery and surfaces anything
a curious human would want to see — archaeology, weird geometry, industrial
oddities, craters, abandoned installations, unexplained scars. The output is
not a label. It is a QUEUE: success means a human opens the top 100 and
repeatedly says "huh, what is that?".

## The funnel

```
candidate locations (regional rasters, no chips)          priors.py
  |  DEM relief/depression outliers + Sentinel-2 spectral outliers
  |  + change-over-years (+ optional unbiased grid, + injected seeds)
  v
imagery chips + CV features (parallel, disk-cached)       features.py
  |  novelty-vs-own-region, geometry, confound scores; rank, never veto
  v
cheap VLM over 25-chip contact sheets                     pipeline.py
  v
strong VLM over full-res colour chips
  v
human reviews final_queue.png + report.md
```

## Run it

```bash
python -m interesting_intel --bbox -117.75 36.45 -117.35 36.85 \
    --out results/interesting/racetrack \
    --seed 36.681 -117.563 racetrack_playa
```

Useful flags: `--no-vlm` (CV ranking only, zero spend), `--source s2|naip`,
`--osm` (Overpass mapped-ness rank adjustment), `--force` (recompute), and
`--seed LAT LON NAME` to inject positive controls. Every stage writes an
artifact under `<out>/artifacts/` and is skipped on re-run; chips cache in
`<out>/../chip_cache` and are never fetched twice. `run_report.json` records
wall-time per stage and ACTUAL VLM token spend in USD.

## Validation

```bash
python -m interesting_intel.validate
```

ranks each positive control (Bingham Canyon Mine, Racetrack Playa, a
Minuteman silo field, the Nazca lines, the Richat Structure) against ~24
background chips from its own surroundings, and requires the central-Kansas
negative control to stay dull. Results land in
`results/interesting/controls/controls.json`; the control chips are saved to
`tests/fixtures/interesting_chips/` as the real-imagery test fixtures for
`tests/test_interesting_real_chips.py` — synthetic-only validation is not
accepted in this repo.

Measured results and the honest precision number live in
`docs/INTERESTING_FINDINGS.md`.

## Rules this module lives by (project failure history)

- plausibility gates BEFORE ranking; rank adjustments, not hard vetoes
- novelty is RELATIVE TO SURROUNDINGS, never absolute
- assert on artifacts produced, not progress counters
- every scorer gets at least one real-imagery test
- no discovery claims; everything is a lead until verified
- do not publish precise coordinates of archaeological sites in conflict
  zones (round to ~0.1 degree publicly)
