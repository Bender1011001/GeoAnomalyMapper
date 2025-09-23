# User Guide

## Overview

This comprehensive manual covers advanced usage of GeoAnomalyMapper (GAM) for experienced users. It details the command-line interface (CLI), configuration options, supported data sources and formats, output handling, visualization choices, and best practices. For beginners, start with the [Quickstart Guide](quickstart.md).

GAM's design emphasizes flexibility: use CLI for quick runs, Python API for customization, and YAML configs for reproducibility. All features support local-to-global scales via parallel processing.

## CLI Reference

GAM's CLI is built with Click and accessible via the `gam` command after installation. Run `gam --help` for overview. The main command is `gam run`, with subcommands for cache management and utilities.

### Main Command: `gam run`

Orchestrates the full pipeline. Basic syntax:
```bash
gam run [OPTIONS] [ARGS]
```

**Required Arguments**:
- None (uses config.yaml if present).

**Options**:
- `--bbox TEXT`: Bounding box as "min_lat max_lat min_lon max_lon" (e.g., "29.9 30.0 31.1 31.2"). Overrides config['data']['bbox'].
- `--modalities TEXT`: Comma-separated list (e.g., "gravity,seismic"). Options: gravity, magnetic, seismic, insar, all. Defaults to config.
- `--config PATH`: Path to YAML config file (e.g., "config_giza.yaml"). Loads and overrides defaults.
- `--output PATH`: Output directory (e.g., "results/region"). Defaults to config['core']['output_dir'].
- `--global / --no-global`: Enable global tiling mode. Uses config['core']['tile_size'] for worldwide processing.
- `--parallel-workers INTEGER`: Number of parallel workers (Dask/Joblib). -1 for all cores; defaults to config['core']['parallel_workers'].
- `--grid-res FLOAT`: Preprocessing grid resolution in degrees. Overrides config['preprocessing']['grid_res'].
- `--threshold FLOAT`: Anomaly detection z-score threshold. Overrides config['modeling']['threshold'].
- `--map-type TEXT`: Visualization type: "2d", "3d", "interactive". Defaults to config['visualization']['map_type'].
- `--export-formats TEXT`: Comma-separated (e.g., "geotiff,vtk,sql"). Defaults to config.
- `--verbose / -v`: Set logging to DEBUG (more output).
- `--quiet / -q`: Set logging to WARNING (less output).
- `--help`: Show help.

**Examples**:
- Regional analysis: `gam run --bbox 40.7 40.8 -74.0 -73.9 --modalities all --output nyc_results`
- Global run: `gam run --global --modalities seismic --parallel-workers 8`
- Config-driven: `gam run --config advanced.yaml --map-type interactive`

**Subcommands**:
- `gam cache`: Manage data cache.
  - `gam cache clear --modality gravity`: Delete cached gravity data.
  - `gam cache list`: Show cached items (size, timestamp).
  - `gam cache size`: Report total cache usage.
- `gam version`: Print GAM version and dependencies.
- `gam validate-config PATH`: Check YAML config for errors (uses Pydantic schema).

**Error Handling**: CLI raises custom exceptions (e.g., DataFetchError) with helpful messages. Use `--verbose` for traces.

## Configuration File Documentation

GAM uses YAML files for reproducible runs. Default: [config.yaml](../config.yaml). Structure is modular, mirroring the pipeline.

### Loading and Validation
Load in Python: 
```python
import yaml
from gam.core.config import validate_config  # Validates with Pydantic

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = validate_config(config)  # Raises ValidationError if invalid
```

CLI loads automatically if `--config` provided. Environment vars override (e.g., `GAM_CACHE_DIR` for cache_dir).

### Full Schema

#### global
- `version` (str): GAM version string (e.g., "1.0.0"). For compatibility checks.
- `logging_level` (str): "DEBUG", "INFO", "WARNING", "ERROR". Defaults to "INFO".

#### data
- `bbox` (list[float]): [min_lat, max_lat, min_lon, max_lon]. Must be valid WGS84 coords (-90 to 90 lat, -180 to 180 lon).
- `modalities` (list[str]): ["gravity", "magnetic", "seismic", "insar", "all"]. "all" enables all available.
- `cache_dir` (str): Path for HDF5/SQLite caches (e.g., "data/cache"). Defaults to "./cache".

#### preprocessing
- `grid_res` (float): Grid resolution in degrees (0.001-1.0). Finer = more compute.
- `filter_params` (dict):
  - `bandpass` (list[float]): [min_hz, max_hz] for seismic filtering.
  - `gaussian_sigma` (float): Smoothing kernel size for gravity/magnetic.
- `units` (str): "SI" (default) or "cgs". Standardizes inputs.

#### modeling
- `inversion_type` (str): "linear" (gravity/magnetic), "eikonal" (seismic), "elastic" (insar).
- `threshold` (float): Z-score for anomaly flagging (1.0-5.0). Higher = fewer false positives.
- `mesh_start_coarse` (float): Initial mesh size in km (1-100).
- `priors` (dict):
  - `joint_weight` (float): 0-1; balances modalities in fusion.
  - `regularization` (str): "l1" (sparse), "l2" (smooth).
- `max_iterations` (int): Inversion convergence limit (10-100).

#### visualization
- `map_type` (str): "2d" (PyGMT static), "3d" (PyVista volume), "interactive" (Folium web).
- `export_formats` (list[str]): ["png", "geotiff", "vtk", "csv", "sql", "h5"].
- `color_scheme` (str): Matplotlib colormap (e.g., "viridis", "plasma").
- `confidence_min` (float): Filter outputs by confidence (0-1).

#### core
- `output_dir` (str): Base path for results (e.g., "data/output").
- `parallel_workers` (int): -1 (all cores), 1 (serial), or specific number.
- `tile_size` (int/float): Degrees per tile for global mode (5-30).
- `rate_limit_delay` (float): Seconds between API calls (0.5-5.0) to respect limits.

**Profiles**: Use multiple YAMLs (e.g., local.yaml, global.yaml) and switch via `--config`. For overrides, env vars like `GAM_GRID_RES=0.05`.

**Validation**: Configs are validated against schemas in `gam/core/config.py`. Common errors: Invalid bbox (e.g., min > max), unknown modality.

## Data Sources and Formats

GAM ingests from public APIs. Configured in [data_sources.yaml](../data_sources.yaml). No proprietary data required.

### Supported Sources
- **Gravity**: USGS Magnetic and Gravity Data (GeoJSON). Base URL: `https://mrdata.usgs.gov/services/gravity?bbox={bbox}&format=geojson`. Format: Points with anomaly values (mGal).
- **Magnetic**: USGS (similar to gravity). Anomalies in nT.
- **Seismic**: IRIS FDSN Web Services via ObsPy. Queries stations/channels by bbox/time. Format: MiniSEED streams (velocity/displacement).
- **InSAR**: Copernicus Sentinel-1 via SentinelAPI. SLC/GRD products. Requires free account for downloads. Format: SAR interferograms (phase/displacement).

**Adding Custom Sources**: Edit data_sources.yaml or implement plugins (see Developer Guide). All sources support bbox filtering; time ranges via config.

### Input Data Formats
- Raw: API responses (JSON for gravity/magnetic, MiniSEED for seismic, ZIP for InSAR).
- Internal: `RawData` dataclass (metadata dict + values as np.ndarray/xarray.Dataset/ObsPy Stream).
- Processed: xarray.Dataset (lat/lon/depth coords, 'data' and 'uncertainty' vars).

**Supported Projections**: WGS84 (EPSG:4326) default; reprojects via PyProj if needed.

**Data Quality**: GAM handles missing data with interpolation; logs gaps. For custom datasets, use `gam.ingestion.load_local(path)` API.

## Output Formats and Visualization Options

### Output Structure
Results in `output_dir/`:
- `anomalies.csv`: Detected anomalies (see Quickstart for schema).
- `{modality}_model.h5`: Per-modality inversion results (HDF5).
- `fused_model.h5`: Joint fusion (xarray.Dataset).
- Visuals: `anomaly_map.{ext}` (PNG for static, HTML for interactive, GeoTIFF/VTK for 3D).
- `report.pdf/html`: Summary with stats, maps, interpretations (via ReportLab or Jinja).
- Logs: `gam_run_{timestamp}.log`.

### Visualization Options
- **2D Maps**: PyGMT/Matplotlib heatmaps. Options: Confidence overlay, contour lines.
- **3D Volumes**: PyVista isosurfaces of anomaly density. Export VTK for external viewers.
- **Interactive**: Folium maps with popups (lat/lon/confidence). Embed in Jupyter.
- **Custom**: API allows `generate_visualization(anomalies, type='custom', kwargs={'cmap': 'coolwarm'})`.

**Export Formats**:
- **CSV/SQL**: Tabular for GIS/DB import.
- **GeoTIFF**: Raster maps with CRS.
- **VTK/HDF5**: 3D models for Paraview/VisIt.
- **JSON**: Metadata and summaries.

View in QGIS (for GeoTIFF) or online viewers (Folium HTML).

## Best Practices and Tips

### Performance Optimization
- **Small Regions First**: Test with 0.1° bbox before global.
- **Parallelism**: Set `parallel_workers: -1` for multi-core; monitor RAM (use Dask dashboard: `dask dashboard`).
- **Caching**: Reuse cache for repeated runs; clear periodically (`gam cache clear`).
- **Global Runs**: Use tile_size=10-20; process in batches if memory-limited.
- **Profiling**: Add `--verbose` and check logs for bottlenecks (e.g., ingestion slowest for InSAR).

### Data Handling
- **Modality Selection**: Start with gravity (fastest); add seismic/InSAR for depth accuracy.
- **Bbox Tips**: Ensure overlap for fusion; use tools like GeoPandas to define from shapefiles.
- **API Limits**: Respect rate_limit_delay (1-2s); for high-volume, use proxies or batch queries.
- **Offline Mode**: Pre-download data via `fetch_data(bbox, save=True)`; load with `load_cached()`.

### Analysis Tips
- **Threshold Tuning**: 2.0 for exploratory, 3.0+ for high-confidence. Visualize distributions first.
- **Validation**: Compare with known sites (e.g., Giza voids); use uncertainty fields.
- **Workflows**: Chain runs (e.g., gravity scout → full multi-modal). Export to ML pipelines (anomalies.csv as input).
- **Reproducibility**: Pin versions in requirements.txt; use Docker for environments (see Deployment Guide).
- **Common Pitfalls**: Invalid bbox (lat >90), missing extras (install [geophysics] for modeling), low RAM (reduce grid_res).

### Security and Ethics
- Public data only; respect API terms (no scraping).
- For sensitive sites (archaeology), anonymize outputs.
- Cite sources in reports (e.g., "Gravity data from USGS").

For code-level customization, see [API Reference](../developer/api_reference.md). Report issues at GitHub.

---

*Last Updated: 2025-09-23 | GAM v1.0.0*