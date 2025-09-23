# Configuration Reference

## Overview

This reference provides a complete list of all configuration options for GeoAnomalyMapper (GAM). Configurations are written in YAML format and loaded from files like [config.yaml](../config.yaml). The structure is hierarchical, with sections corresponding to pipeline stages: `global`, `data`, `preprocessing`, `modeling`, `visualization`, and `core`.

Configs are validated using Pydantic models in `gam/core/config.py`. Defaults are provided where possible; unspecified options use sensible fallbacks. CLI flags (e.g., `--bbox`) override YAML values. Environment variables (e.g., `GAM_CACHE_DIR`) take precedence.

**Loading Example**:
```python
import yaml
from gam.core.config import GAMConfig

with open('config.yaml', 'r') as f:
    data = yaml.safe_load(f)
config = GAMConfig(**data)  # Validates and parses
```

**Validation**: Invalid values raise `ValidationError` with details (e.g., "bbox must have 4 elements"). Use `gam validate-config path/to/config.yaml` for CLI check.

## Global Section

Top-level settings for the entire run.

- `version` (str, default: "1.0.0")
  - GAM version for compatibility. Should match `__version__` in `gam/__init__.py`.
  - Valid: Semantic version string (e.g., "1.0.0").
  - Example: `version: "1.0.0"`

- `logging_level` (str, default: "INFO")
  - Logging verbosity.
  - Valid: "DEBUG", "INFO", "WARNING", "ERROR".
  - Example: `logging_level: "DEBUG"` (detailed output for troubleshooting).

## Data Section

Settings for input data ingestion.

- `bbox` (list[float], required)
  - Bounding box for analysis: [min_lat, max_lat, min_lon, max_lon].
  - Valid: min_lat >= -90, max_lat <= 90, min_lon >= -180, max_lon <= 180, min < max.
  - Range: Global (-90,90,-180,180) or local (e.g., [29.9, 30.0, 31.1, 31.2] for Giza).
  - Example: `bbox: [29.9, 30.0, 31.1, 31.2]`

- `modalities` (list[str], default: ["all"])
  - Data types to process.
  - Valid: "gravity", "magnetic", "seismic", "insar", or "all" (enables all available).
  - Example: `modalities: ["gravity", "seismic"]`

- `cache_dir` (str, default: "./cache")
  - Directory for storing fetched/processed data (HDF5/SQLite).
  - Valid: Any writable path; use absolute for large runs.
  - Example: `cache_dir: "../data/cache"`

## Preprocessing Section

Parameters for data alignment, filtering, and gridding.

- `grid_res` (float, default: 0.1)
  - Spatial resolution in degrees (~11km at equator for 0.1).
  - Valid: 0.001-5.0 (finer = more accurate but slower).
  - Range: Local: 0.001-0.1; Global: 0.5-2.0.
  - Example: `grid_res: 0.01`

- `filter_params` (dict, default: {})
  - Modality-specific filters.
  - Sub-options:
    - `bandpass` (list[float], for seismic): [min_hz, max_hz], default [0.1, 1.0].
      - Valid: Positive floats, min < max.
    - `gaussian_sigma` (float, for gravity/magnetic): Smoothing kernel, default 1.0.
      - Valid: 0.1-5.0.
  - Example:
    ```yaml
    filter_params:
      bandpass: [0.05, 2.0]
      gaussian_sigma: 0.5
    ```

- `units` (str, default: "SI")
  - Standardization units.
  - Valid: "SI" (m/s² for gravity, nT for magnetic) or "cgs".
  - Example: `units: "SI"`

## Modeling Section

Inversion and anomaly detection settings.

- `inversion_type` (str, default: "linear")
  - Algorithm per modality.
  - Valid: "linear" (gravity/magnetic, SimPEG), "eikonal" (seismic), "elastic" (InSAR).
  - Example: `inversion_type: "linear"`

- `threshold` (float, default: 2.0)
  - Z-score for flagging anomalies.
  - Valid: 1.0-5.0 (higher = stricter).
  - Range: Exploratory: 1.5-2.5; Production: 3.0+.
  - Example: `threshold: 2.5`

- `mesh_start_coarse` (float, default: 10.0)
  - Initial mesh cell size in km for iterative refinement.
  - Valid: 1.0-100.0.
  - Example: `mesh_start_coarse: 5.0`

- `priors` (dict, default: {})
  - Inversion priors.
  - Sub-options:
    - `joint_weight` (float): Fusion balance (0-1), default 0.5.
      - Valid: 0.0 (no fusion) to 1.0 (strong fusion).
    - `regularization` (str): "l1" (sparse) or "l2" (smooth), default "l2".
  - Example:
    ```yaml
    priors:
      joint_weight: 0.7
      regularization: "l1"
    ```

- `max_iterations` (int, default: 50)
  - Convergence limit for inversions.
  - Valid: 10-200.
  - Example: `max_iterations: 100`

## Visualization Section

Output rendering options.

- `map_type` (str, default: "2d")
  - Visualization style.
  - Valid: "2d" (PyGMT static), "3d" (PyVista volume), "interactive" (Folium web).
  - Example: `map_type: "3d"`

- `export_formats` (list[str], default: ["png", "csv"])
  - File types for outputs.
  - Valid: "png", "geotiff", "vtk", "csv", "sql", "h5", "html".
  - Example: `export_formats: ["geotiff", "vtk"]`

- `color_scheme` (str, default: "viridis")
  - Colormap for heatmaps.
  - Valid: Matplotlib colormaps (e.g., "viridis", "plasma", "coolwarm").
  - Example: `color_scheme: "plasma"`

- `confidence_min` (float, default: 0.0)
  - Minimum confidence to include in visuals (0-1).
  - Valid: 0.0-1.0.
  - Example: `confidence_min: 0.7`

## Core Section

Runtime and pipeline settings.

- `output_dir` (str, default: "./results")
  - Base directory for all outputs.
  - Valid: Writable path.
  - Example: `output_dir: "../results"`

- `parallel_workers` (int, default: 1)
  - Number of parallel workers (Dask/Joblib).
  - Valid: 1 (serial), >1 (parallel), -1 (all cores).
  - Example: `parallel_workers: -1`

- `tile_size` (float, default: 10.0)
  - Tile size in degrees for global mode.
  - Valid: 1.0-90.0 (smaller = more tiles, finer but slower).
  - Example: `tile_size: 20.0`

- `rate_limit_delay` (float, default: 1.0)
  - Delay (seconds) between API calls.
  - Valid: 0.1-10.0 (higher for rate-limited sources).
  - Example: `rate_limit_delay: 2.0`

## Environment Variable Overrides

Override YAML with env vars (prefix "GAM_"):
- `GAM_CACHE_DIR`: Overrides `data.cache_dir`.
- `GAM_OUTPUT_DIR`: Overrides `core.output_dir`.
- `GAM_LOGGING_LEVEL`: Overrides `global.logging_level`.
- `GAM_GRID_RES`: Overrides `preprocessing.grid_res`.
- Data source auth: e.g., `COPERNICUS_USERNAME` for InSAR.

Example: `export GAM_PARALLEL_WORKERS=8; gam run ...`

## Profile-Based Configuration

Use multiple profiles in one YAML:
```yaml
profiles:
  local:
    data:
      bbox: [29.9, 30.0, 31.1, 31.2]
    core:
      parallel_workers: 2
  global:
    data:
      bbox: [-90, 90, -180, 180]
    core:
      tile_size: 30
      parallel_workers: -1
```
Load with `--profile local` (CLI extension) or `config = load_profile('local')`.

## Default Values and Valid Ranges Summary

| Section | Option | Default | Valid Range/Example |
|---------|--------|---------|--------------------|
| global | version | "1.0.0" | Semantic version |
| global | logging_level | "INFO" | DEBUG/INFO/WARNING/ERROR |
| data | bbox | Required | 4 floats: [-90,90,-180,180] |
| data | modalities | ["all"] | List of strings |
| data | cache_dir | "./cache" | Path |
| preprocessing | grid_res | 0.1 | 0.001-5.0 |
| preprocessing | filter_params | {} | Dict with bandpass, gaussian_sigma |
| preprocessing | units | "SI" | SI/cgs |
| modeling | inversion_type | "linear" | linear/eikonal/elastic |
| modeling | threshold | 2.0 | 1.0-5.0 |
| modeling | mesh_start_coarse | 10.0 | 1.0-100.0 |
| modeling | priors | {} | joint_weight (0-1), regularization (l1/l2) |
| modeling | max_iterations | 50 | 10-200 |
| visualization | map_type | "2d" | 2d/3d/interactive |
| visualization | export_formats | ["png", "csv"] | List of formats |
| visualization | color_scheme | "viridis" | Matplotlib colormap |
| visualization | confidence_min | 0.0 | 0.0-1.0 |
| core | output_dir | "./results" | Path |
| core | parallel_workers | 1 | int >=1 or -1 |
| core | tile_size | 10.0 | 1.0-90.0 |
| core | rate_limit_delay | 1.0 | 0.1-10.0 |

For troubleshooting invalid configs, see [Common Issues](../troubleshooting/common_issues.md).

---

*Last Updated: 2025-09-23 | GAM v1.0.0*