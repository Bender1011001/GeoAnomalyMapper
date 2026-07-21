---
project: GeoAnomalyMapper-1
status: working
updated: 2026-07-21
---

# GeoAnomalyMapper — Deformation Intelligence + Archaeology Surface Proxies

## Current Mission
- Detect, classify, and FORECAST active subsurface processes (growing
  sinkholes, salt dissolution, mine settling, induced subsidence) from free
  satellite InSAR, with every capability claim backed by a ground-truth
  experiment (`deformation_intel/`).
- Second validated line: archaeological surface-proxy screening from free
  data (`archaeo_intel/` + the pre-registered research program in
  docs/RESEARCH_TRACKS.md, governed by docs/DISCOVERY_SOP.md).
- All outputs are candidates requiring ground follow-up, never confirmed
  voids/sites. Candidate coordinates in conflict zones stay LOCAL (redacted
  in anything public).
- Do not revive removed/unverified claims: SAR Doppler vibrometry FAILED
  blind validation and was deleted; legacy gravity/mineral performance
  numbers are UNVERIFIED until re-derived through the blind harness.

## Resume (what is actually open, 2026-07-21)
- **Bare-desert OPERA sweep running: 39/65 tiles done.** Drivers
  desert_sweep_v2.py / v2b.py (live processes; scripts preserved in
  data/research/scripts/). Per-tile JSON + caches in
  data/research/desert_sweep/ — a killed tile resumes from cache. When all
  65 finish: triage with desert_triage.py; US public-land coordinates are
  publishable after verification.
- **Queued research** (see the status summary atop docs/RESEARCH_TRACKS.md):
  Hunt 9 moisture-lag; closure-phase archaeology stretch test + early-warning
  fold-in; anisotropy into the combined-ML ensemble; Cap-1 hyper-arid
  re-point or seasonal-baseline rule; CORONA stereo DEMs (bench).
- **Next deformation-product build:** groundwater/seasonality rejection layer
  (aquifer well-cones vs karst) + ascending/descending decomposition.
- National scan (12 AOIs): DONE — docs/NATIONAL_SCAN_REPORT.md (2026-07-11).

## Working System
```
OPERA DISP-S1 (9.5-yr time series)  ┐
                                    ├─> deformation_intel.detect
HyP3 short pairs (fast movers)      ┘      cluster -> classify -> Mogi source
                                            -> forecast -> void_likelihood

DEM + S2 + thermal + CORONA ──> archaeo_intel: prominence ranking -> VLM
                                triage (prompt v3) -> human review, redacted
```
- `deformation_intel/` — validated deformation engine (see its README).
- `archaeo_intel/` — archaeology surface-proxy package (see its README);
  corona.py = CORONA atlas access + human-GCP georeferencing (operational).
- `tools/insar_prototype/` — HyP3 channel scripts (Wink-validated).
- `blind_validation.py` + `geoanomaly.py` — blind validation harness + CLI.
- `slc_data_fetcher.py` — Sentinel-1/ASF search, auth, download utilities.
- `data/research/scripts/` — 149 preserved experiment scripts referenced by
  the RESEARCH_TRACKS ledger (gitignored; vet coordinates before publishing).

## Validation Ledger (what is proven)
- Wink TX: active subsidence recovered by BOTH channels (HyP3 bowl at
  0.69 km; OPERA accelerating cluster + automatic 2023 regime change).
- Tampa Sinkhole Alley: 42 localized accelerating Mogi-consistent candidates.
- Central Valley: aquifer province correctly classified regional (201/264).
- Archaeology channels, measured ceilings (Menze-Ur 14,324-site truth):
  prominence 0.547 / BSI 0.553 / texture 0.595 / thermal 0.622 /
  combined-ML 0.616 / **anisotropy 0.639 + 0.608 replicated (best single;
  ranking-feature only)**. TDA-H1 = validated ring feature, not autonomous
  at 30 m DEM. Closure phase = validated disturbance detector with ~6-week
  early-warning lead. CORONA GCP georef operational (~150-200 m local).
- Failed & removed: vibrometry. Retired: ICA (controls mis-sited twice).
  Killed: Hunt 11 B-perp altimeter (floor ~3 m; target regime unreachable).
  Unverified: mineral prospectivity numbers.
- Records: docs/experiment_records/, docs/RESEARCH_TRACKS.md (full ledger).

## Tech Stack
Python; numpy/scipy; rasterio/pyproj; xarray+h5netcdf; earthaccess, asf_search,
hyp3_sdk; matplotlib. No torch in the current system. Wolfram Mathematica
available via wolframscript (used in georef attempt #4).

## Trap Diary
- OPERA granules: 0-360 lon in some products; reference-era resets MUST be
  stitched (timeseries.stitch_reference_eras) or velocities are garbage.
- ASF hosting is split: some frames cloud-lazy-readable, others datapool-only
  (401s everything except asf_search's ASFSession download) — opera.py's
  3-tier fallback exists for this reason; don't simplify it away.
- 20x4-look HyP3 products alias away small fast bowls; use 10x2 + short pairs.
- Report velocity as the robust mean rate, never the quadratic endpoint.
- Windows: run deformation tests in a separate pytest process from torch
  suites (native DLL load-order crash).
- A detector's validated ENVELOPE must match the deployment AOI: the
  bare-desert coherence rule on vegetated steppe produced 21 pseudo-events
  from one rain cell + seasonal dry-down (Cap-1 sweep #1).
- Scene-wide events survive quiet-zone mean normalization via their spatial
  gradient; apply the >30%-same-onset-date scene-event veto.
- Verify a positive control's signal exists in the RAW data before
  registering its coordinates (two ICA tests voided by mis-sited controls).
- DISP-S1 retains only coherent pixels: a "farmland" control actually
  measures farm buildings. Flat nulls need geometry-selected road pixels.
- Session scratchpads are EPHEMERAL: any script referenced by a permanent
  ledger must be copied to data/research/scripts/ in-session, same day.
- Long remote-I/O pipelines hang silently past internal watchdogs; wrap each
  work unit in subprocess.run(timeout=...) — the OS-level kill is the only
  unconditional one (desert sweep v2 pattern).
- MPC STAC has real outages (504s): retest a known-good query before blaming
  your code; use a recovery watcher to auto-relaunch.
- Prompt changes to VLM triage need a labeled anchor set + repeats (v2->v3
  went 10/20 -> 19/20; single-call checks can't see systematic errors).

## Build / Verify
- `python geoanomaly.py health --json --skip-gpu`
- `pytest tests/test_deformation_*.py -q` (deformation suite)
- `pytest tests/test_archaeo_intel.py -q` (archaeology analytics)
- `pytest tests/ -q --ignore=tests/test_deformation_timeseries.py
  --ignore=tests/test_deformation_sources.py --ignore=tests/test_deformation_detect.py`
  (validation-harness suite)
