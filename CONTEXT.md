---
project: SAR-project
status: wip
updated: 2026-03-14
---

# GeoAnomalyMapper

## Resume
- **Pick up at**: [Review and update]
- **Last session**: [Auto-migrated to CONTEXT v2]
- **Blocked on**: Nothing

## Status
- **Working**: Core geophysical data fusion pipeline, InSAR pre-processing, anomaly detection (`detect_voids.py`), and visualization toolings (PNG, KMZ).
- **Broken**: None explicitly known, but requires external data sourcing which limits full end-to-end automated testing without mocks.

## Tech Stack
- Python 3.9+ 
- GDAL/Rasterio (assumed based on GeoTIFF processing)
- Leaflet (for the interactive `docs/` visualization)
- `pmtiles` (for map tiling)

## Key Files
- `multi_resolution_fusion.py` — Core script that resamples and combines layers using uncertainty-aware weighting.
- `detect_voids.py` — Calculates probability scores for potential subsurface voids based on the fused layers.
- `process_data.py` & `process_insar_data.py` — Data normalization layers placing everything onto a common grid.
- `validate_against_known_features.py` — Quality check script to sample fused rasters against known ground truths.

## Architecture Quirks
- The `data/` directory is strictly EXCLUDED from version control (`.gitignore`). It relies on a local file structure (`data/raw/...` and `data/processed/...`).
- Scripts are designed as self-contained CLI tools that do NOT download data. Getting raw data (EGM2008, EMAG2, Copernicus DEM) is a manual prerequisite.

## Trap Diary
- **Missing Data Folders**: Because `data/` is out of version control, running scripts on a fresh clone without first populating `data/raw/` and creating the `data/processed/` hierarchy will cause pathing errors.

## Anti-Patterns (DO NOT)
- Do not add raw or processed GeoTIFF / geospatial data files to the git repository. They bloat the tree.
- Do not build automated downloaders into the core scripts; the project philosophy specifically demands data gathering remain a manual, external step.

## Build / Verify
Execute CLI tools with `--help` to ensure environment is intact (e.g., `python process_data.py --help`).
