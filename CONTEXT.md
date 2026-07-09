---
project: GeoAnomalyMapper-1
status: working
updated: 2026-05-09
---

# GeoAnomalyMapper - Underground Object/Void Search

## Current Mission
- Current goal: find underground objects or voids, with deeper credible bodies ranked higher than shallow bodies.
- This is not currently a mineral-prospectivity project. Do not revive old gravity/mineral performance claims unless a reproducible validation artifact is added.
- Treat all outputs as model-derived candidates requiring independent ground truth. The code can rank candidates; it does not prove structures exist.

## Resume
- Pick up at: run focused tests after the depth-prioritized scoring and AlphaEarth utility changes.
- Task in progress: integrate useful surface-context data without pretending it is direct subsurface evidence.
- Blocked on: no real 64-band AlphaEarth GeoTIFF exports are present yet, so embedding processing is implemented and unit-tested but not run on field data.

## Working Pipeline
```
SLC Fetch -> Doppler Vibrometry -> PINN Inversion -> 3D Visualization/Extraction
                                      |
                                      v
                              wave_speed_volume.npy
                              void_probability_volume.npy
                              detected_anomalies.csv
```

Blind validation now layers on top of the existing candidate CSV output:

```
public validation manifest -> blind_validation.py run -> frozen_candidates.csv + run_manifest.json
withheld labels -----------> blind_validation.py score -> score_report.json
score/artifacts -----------> blind_validation.py package-report -> redacted report package
```

## Tech Stack
- Python 3.10+, PyTorch/CUDA, NumPy, SciPy, rasterio, scikit-image, Matplotlib
- Data: Umbra X-band SLC when available, Sentinel-1 SLC via ASF/Earthdata as fallback
- Optional surface context: Google Satellite Embedding / AlphaEarth annual 64-band GeoTIFF exports from Earth Engine

## Key Files
- `slc_data_fetcher.py` - Umbra and Sentinel-1 SLC acquisition/extraction.
- `sar_vibrometry.py` - Doppler sub-aperture vibrometry and deterministic synthetic fixtures.
- `pinn_vibro_inversion.py` - PINN 3D wave-speed / void-probability inversion.
- `visualize_3d_subsurface.py` - connected body extraction, depth-prioritized ranking, reports, CSVs, and visualizations.
- `run_biondi_exploration.py` - end-to-end phase runner and target definitions.
- `blind_validation.py` - validation-first blind known-void runner/scorer harness. The runner reads only public manifests and freezes candidate CSVs; the scorer reads withheld labels after outputs are frozen.
- `geoanomaly.py` - stable top-level validation-first CLI wrapper; delegates `geoanomaly.py validation ...` to `blind_validation.py`.
- `satellite_embeddings.py` - real AlphaEarth/Satellite Embedding GeoTIFF inspection, annual dot-product change scoring, and Earth Engine export-script generation.
- `validation_examples/` - safe fixture manifests/templates only; no private real ground-truth labels.
- `docs/VALIDATION_FIRST_WORKFLOW.md` - exact validation-first commands and claims boundary.
- `tests/` - focused unit tests for validation-critical behavior.

## Current Ranking Logic
- `visualize_3d_subsurface.extract_anomaly_bodies()` now ranks connected void/object candidates by `deep_target_score`.
- `deep_target_score` keeps evidence in the loop and boosts deeper bodies:
  - `void_evidence_score`: combines mean void probability, wave-speed drop, edge sharpness, and artificiality/geometric cues.
  - `depth_priority_score`: uses centroid depth and bottom depth fraction inside the inversion volume.
  - `deep_target_rank`: sorted output rank, where rank 1 is the best deep target candidate.
- Centroids use voxel-center coordinates, not top/left voxel edges.
- Morphological cleanup is disabled by default (`morphology_iterations: 0`) because erosion can erase narrow tunnels, shafts, or small deep bodies before connected-component labeling.
- `run_biondi_exploration.py` stores the top five ranked deep targets in phase JSON and summarizes the best target in phase reports.
- `run_biondi_exploration.py --resolution deep` is the profile aligned with "deeper the better": up to 5000 m max depth, 2000 m domain width, 160 z slices, 0.25 Hz excitation, and dynamic deep report slices.

## AlphaEarth / Google Satellite Embedding Notes
- Official Earth Engine collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- Each annual image has 64 bands (`A00` through `A63`) at 10 m resolution. All bands must be used together as one embedding vector.
- Unit-length embeddings can be compared across years with a dot product. `satellite_embeddings.py compare` writes:
  - similarity GeoTIFF: 64D dot-product similarity
  - change GeoTIFF: `1 - clipped_similarity`
  - JSON summary: valid pixels, score distribution, and threshold counts
- `run_biondi_exploration.py --use-embeddings` can pass explicit anomaly rasters or directory fallback rasters to PINN training as a weak surface prior when `--surface-prior-weight` is positive.
- Important boundary: AlphaEarth embeddings summarize annual surface conditions. They are useful for surface disturbance, access routes, subsidence context, false-positive review, and target triage. They do not replace SAR SLC phase history and are not direct depth evidence.

## Architecture Quirks
- Void probability uses sigmoid mapping, not linear: `1/(1+exp(-(speed-thresh)/temp))`.
- Temperature is `background_wave_speed * 0.05` for sharp detection.
- Current void threshold is 0.35 to catch weaker Sentinel-1-derived anomalies.
- Sentinel-1 C-band and TOPSAR geometry are limited for deep/small-feature interpretation; Umbra Spotlight or other higher-quality SLC data is preferred when available.
- Deeper targets need wider/deeper inversion domains and lower excitation frequencies, but increasing depth also increases non-uniqueness and hallucination risk.

## Last Several Changes
- Added `satellite_embeddings.py` for real 64-band AlphaEarth GeoTIFF export scripting, inspection, and year-to-year change rasters.
- Added `tests/test_satellite_embeddings.py` to verify vector math, real GeoTIFF writes, summary generation, and Earth Engine script output.
- Updated `visualize_3d_subsurface.py` to rank connected bodies by depth-aware target score instead of mean void probability alone.
- Hardened extraction so thin deep features survive by default and single-voxel/narrow crops do not crash edge-sharpness scoring.
- Updated `run_biondi_exploration.py` so consolidated phase reports include the top deep target candidate.
- Added a real `deep` resolution profile and dynamic depth-slice selection so deep runs are not silently capped to shallow report views.
- Updated `README.md` to state the current mission as underground object/void search and document AlphaEarth as surface context.
- Earlier cleanup aligned README claims with the actual SAR/PINN codebase and removed unsupported continental mineral validation language.
- Added the blind known-void validation foundation: public no-label manifests, withheld scorer labels, dry-run/real-run separation, deterministic score reports, safe fixtures, validation report packaging, stable top-level validation CLI delegation, and unit tests.

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| Goal drift back to mineral prospectivity | README had old gravity/mineral language | Keep project framed as underground object/void search unless evidence changes |
| Treating AlphaEarth as a detector | Embeddings are surface-condition features, not depth measurements | Use as context/change screening only |
| Thin tunnels disappear before scoring | Morphological erosion removes 1-voxel-wide connected bodies | Keep morphology disabled by default; rely on `min_anomaly_voxels` for filtering |
| NaN loss during training | AMP mixed precision in physics loss | Disable AMP for physics branch |
| CUDA OOM on H100 | Batch size too large | Reduce batch size / use gradient accumulation |
| Void probability always near 0 | Linear mapping with partial volume effect | Use sigmoid mapping with tight temperature |
| Khafre and Khufu Phase 1 overlap | Same SAR scene and broad domain | Use tight per-target domains when separating nearby targets |
| Label leakage into validation runs | Known void geometry can bias target setup or scoring thresholds | Keep `blind_validation.py run` limited to public manifests; use withheld labels only with `blind_validation.py score` after candidate CSVs are frozen |
| Secrets in validation artifacts | Logs or notes can accidentally contain tokens | Do not package `.env`; `package-report` redacts secret-like keys and bearer/token strings but inputs should avoid secrets |

## Anti-Patterns
- Do not write simulated, mock, pseudo, placeholder, TODO, or pass-through code.
- Do not claim a detected void/object is real without independent validation.
- Do not use AlphaEarth embeddings as if they provide subsurface depth.
- Do not sort candidate reports by shallow/high-probability blobs alone; preserve deep target ranking.
- Do not set `void_threshold` above 0.4 for Sentinel-1-heavy runs without reviewing the volume distribution.
- Do not run large deep domains without recording the exact grid, depth, domain width, frequency, and data source.
- Do not use `quick`, `standard`, or `high` profiles when the actual objective is maximum depth; use `--resolution deep` and expect a much heavier run.
- Do not put known void geometry, expected void depths, or withheld-label paths in public validation manifests.
- Do not enable synthetic fallback for blind real-world known-void validation; use it only for explicit controls.

## Useful Commands
```bash
python -m unittest discover -s tests
python blind_validation.py validate-public --manifest validation_examples/public_manifest_fixture.json
python blind_validation.py run --manifest validation_examples/public_manifest_fixture.json --output-dir data/blind_validation/fixture_run
python blind_validation.py score --run-manifest data/blind_validation/fixture_run/run_manifest.json --labels validation_examples/withheld_labels_fixture.json --output data/blind_validation/fixture_score.json
python blind_validation.py package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --output-dir data/blind_validation/fixture_report_package
python geoanomaly.py commands
python run_biondi_exploration.py --phase 3 --resolution deep
python visualize_3d_subsurface.py --volume data/inversion_3d/outputs/wave_speed_volume.npy --void-prob data/inversion_3d/outputs/void_probability_volume.npy
python satellite_embeddings.py export-ee-script --lat 38.3512 --lon -121.986 --buffer-deg 0.02 --years 2023 2024 --out-js data/alphaearth_exports/vacaville_export.js
python satellite_embeddings.py compare --before data/alphaearth_exports/site_2023.tif --after data/alphaearth_exports/site_2024.tif --change-out data/alphaearth_exports/site_change.tif --summary-out data/alphaearth_exports/site_change_summary.json
```
