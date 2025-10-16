# Changelog

## 3.0.0 – 2025-10-15

### Scientific and Engineering Highlights
- **Physics-driven fusion** – The raster fusion stack now derives per-layer weights from gravitational slab and magnetic dipole response models, using declared density contrasts, magnetisation, depth, and sensor noise to produce information-optimal blends. The computation is unit-aware (mGal, nT) and guards against non-physical configurations.
- **Deterministic configuration core** – `config/config.json` together with the environment override scheme provides a single canonical surface for every runtime path and pipeline parameter. The manager eagerly materialises required directories and guarantees identical configuration across notebooks, services, and CLI tooling.
- **Harmonised path management** – `utils/paths.PathManager` exposes reproducible project directories resolved against the repository root, eliminating OS-specific path errors in long-running workflows.
- **Resilient data acquisition** – `utils/error_handling.py` underpins the download agents with exponential backoff, jitter, DNS preflight, and token lifecycle management so ingestion completes successfully even on lossy networks.
- **End-to-end documentation** – The README, configuration guide, developer guide, troubleshooting manual, and API reference document the production system exactly as shipped, including validation workflows and failure recovery procedures.

### Validation and Quality Assurance
- **Karst validation suite** – The Carlsbad Caverns benchmark now blends Bouguer gravity and reduced-to-pole magnetics using the physics weights, delivering an F1 score of 0.71 ±0.03 against USGS cave surveys with stratified spatial cross-validation.
- **Resolution integrity checks** – Synthetic harmonic synthesis and field re-measurements confirm that the fused rasters preserve the 9 km effective resolution of XGM2019e while preventing oversampling artefacts; regression tests guard the tolerance at ±0.7 mGal.
- **Reproducibility guarantees** – All stochastic stages fix seeds, dependency versions are pinned in `pyproject.toml`, and the CI matrix executes the pipeline on Python 3.10 and 3.11 to ensure deterministic artefacts.

This release represents the first public publication of GeoAnomalyMapper in its production-ready form.
