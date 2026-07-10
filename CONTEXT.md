---
project: GeoAnomalyMapper-1
status: working
updated: 2026-07-09
---

# GeoAnomalyMapper — Ground-Deformation Intelligence

## Current Mission
- Detect, classify, and FORECAST active subsurface processes (growing
  sinkholes, salt dissolution, mine settling, induced subsidence) from free
  satellite InSAR, with every capability claim backed by a ground-truth
  experiment.
- All outputs are candidates requiring ground follow-up, never confirmed voids.
- Do not revive removed/unverified claims: the SAR Doppler vibrometry pipeline
  FAILED blind validation (Carlsbad vs barren control) and was deleted; the
  legacy gravity/mineral performance numbers are UNVERIFIED and must not be
  used until re-derived through the blind-validation harness.

## Resume
- Pick up at: national scan queue (12 high-value AOIs) — consolidated
  fixed-detector pass when all cubes are cached under data/national_scan/.
- Next planned build: groundwater/seasonality rejection layer (separates
  aquifer well-cones from karst voids) + ascending/descending decomposition.

## Working System
```
OPERA DISP-S1 (9.5-yr time series)  ┐
                                    ├─> deformation_intel.detect
HyP3 short pairs (fast movers)      ┘      cluster -> classify -> Mogi source
                                            -> forecast -> void_likelihood
```
- `deformation_intel/` — the validated engine (see its README).
- `tools/insar_prototype/` — HyP3 channel scripts (Wink-validated).
- `blind_validation.py` + `geoanomaly.py` — methodology-agnostic blind
  validation harness and CLI (real execution now requires an explicit
  pipeline_executor; the legacy executor was removed).
- `slc_data_fetcher.py` — Sentinel-1/ASF search, auth, download utilities.

## Validation Ledger (what is proven)
- Wink TX: documented active subsidence recovered by BOTH channels (HyP3 bowl
  at 0.69 km; OPERA accelerating cluster + automatic 2023 regime change).
- Tampa Sinkhole Alley: 42 localized accelerating Mogi-consistent candidates.
- Central Valley: aquifer province correctly classified regional (201/264).
- Failed & removed: vibrometry (site-independent artifacts). Unverified:
  mineral prospectivity numbers. Records in docs/experiment_records/.

## Tech Stack
Python; numpy/scipy; rasterio/pyproj; xarray+h5netcdf; earthaccess, asf_search,
hyp3_sdk; matplotlib. No torch in the current system.

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

## Build / Verify
- `python geoanomaly.py health --json --skip-gpu`
- `pytest tests/test_deformation_*.py -q` (28 tests)
- `pytest tests/ -q --ignore=tests/test_deformation_timeseries.py
  --ignore=tests/test_deformation_sources.py --ignore=tests/test_deformation_detect.py`
  (validation-harness suite)
