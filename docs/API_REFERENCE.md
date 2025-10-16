# API Reference and Technical Documentation

**Comprehensive Reference for GeoAnomalyMapper Components**

This document provides technical details on the production GeoAnomalyMapper APIs. It covers the physics-weighted fusion stack, InSAR processing pipeline, resilience framework, configuration/path systems, and integration points. For user guides, see [README.md](README.md) and specialized docs.

All APIs are in Python 3.10+ and integrate with the unified config system. Import from `GeoAnomalyMapper` or `utils`.

## 1. Configuration and Path Resolution System

### ConfigManager (utils/config.py)
Central loader for `config/config.json` with optional environment overrides.

**Class: ConfigManager**
- **__init__(self, config_path=None, env_prefix='GAM')**: Loads JSON defaults and
  applies overrides from environment variables that start with the prefix (for
  example, `GAM__FUSION__DYNAMIC_WEIGHTING=true`).
- **get(self, key, default=None)**: Retrieves a value using dotted notation.
- **get_path(self, key, default=None)**: Returns a `pathlib.Path`, resolving
  relative entries against the repository root.
- **set(self, key, value)**: Updates the in-memory configuration for the current
  process.
- **items(self)** / **as_dict(self)**: Helpers for diagnostics and debugging.

**Technical Details**:
- Loads `.env` automatically when `python-dotenv` is installed.
- Environment overrides use double underscores to traverse nested keys
  (`GAM__PATHS__RAW_DATA=/custom/raw`).
- Directories declared under `project.*` and `paths.*` are created during
  initialisation to simplify local setup.

**Example**:
```python
from utils.config import ConfigManager
config = ConfigManager()
data_root = config.get_path('project.data_root')  # Path('./data')
if config.get('data_sources.insar.enabled'):
    # Proceed
    pass
```

### PathManager (utils/paths.py)
Convenience wrapper that exposes commonly used directories as properties.

**Class: PathManager**
- **__init__(self)**: Reads resolved directories from `ConfigManager` and caches
  them.
- **data_dir / output_dir / processed_dir / cache_dir**: Properties returning the
  corresponding directories.
- **get_path(self, key, default=None)**: Access additional entries by key.
- **join(self, base_key, *subpaths)**: Append relative components to a known base
  directory.
- **resolve(self, relative_path)**: Convert a repository-relative string into an
  absolute path.

**Technical Details**:
- Relative paths are resolved against the repository root.
- Directories are created when the manager initialises so downstream code can
  assume they exist.

**Integration**: Import `from utils.paths import paths` to reuse the singleton
instance across modules.

### Environment overrides

All configuration values can be overridden using environment variables with the
`GAM__` prefix.  Each double underscore denotes a traversal into the nested JSON
structure:

```bash
export GAM__PROJECT__DATA_ROOT=/mnt/geo/data
export GAM__FUSION__DYNAMIC_WEIGHTING=true
```

Values are parsed as JSON when possible (`true`, `false`, numbers, quoted
strings).  Use `.env` to persist local overrides without committing them.

## 2. Physics-Weighted Fusion

### `physics_weighting` (gam/fusion/weight_calculator.py)
Derives per-layer weights using analytical geophysical response models.

**Function: `physics_weighting(layer_configs)`**
- **Inputs**: Mapping of layer name → configuration dictionary. Each dictionary must declare:
  - `model`: One of `gravity_slab`, `magnetic_dipole`, or `topography_gradient`.
  - `resolution`: Nominal ground sampling distance in metres.
  - Model parameters (for example, `density_contrast_kg_m3`, `target_thickness_m`, `magnetization_a_m`, `anomaly_volume_m3`).
  - `noise_floor`: Instrument noise floor in native units (mGal, nT, metres).
- **Output**: `WeightResult` with weights that sum to 1.

**Technical Details**:
- **Gravity slab**: Uses the infinite Bouguer slab equation \( \Delta g = 2\pi G \Delta\rho t \) with exponential decay by depth and converts to mGal.
- **Magnetic dipole**: Approximates the vertical component of a buried dipole \( B_z = \frac{\mu_0}{4\pi} \frac{2M\cos I}{r^3} \) and converts to nanoTesla.
- **Topography gradient**: Relates characteristic relief and slope to gradient magnitude per metre.
- Information score per layer is `(response/noise_floor)^2 / resolution`, ensuring high-signal, low-noise, fine-resolution rasters dominate the fusion.
- Validation guards prevent zero or negative information content and provide descriptive errors when required parameters are missing.

**Example**:
```python
from gam.fusion.weight_calculator import physics_weighting

weights = physics_weighting(
    {
        "gravity": {
            "model": "gravity_slab",
            "resolution": 1000,
            "density_contrast_kg_m3": 420,
            "target_thickness_m": 180,
            "target_depth_m": 600,
            "noise_floor": 0.08,
        },
        "magnetics": {
            "model": "magnetic_dipole",
            "resolution": 1000,
            "magnetization_a_m": 9.5,
            "anomaly_volume_m3": 2.2e5,
            "target_depth_m": 600,
            "inclination_deg": 63,
            "noise_floor": 1.2,
        },
    }
)
print(weights.weights)
```

> Example output: `{'gravity': 0.609, 'magnetics': 0.391}` for the Carlsbad configuration.

### Fusion driver (`gam/fusion/multi_resolution_fusion.py`)
Executes the weighted fusion and writes Cloud Optimised GeoTIFF outputs.

- **`fuse_layers(layers, output_path)`**: Loads single-band rasters, applies `physics_weighting`, harmonises profiles, performs a weighted average ignoring nodata, and writes a COG to `output_path`.
- **`run(config_path, output_dir)`**: Iterates over products declared in `config/fusion.yaml`, calling `fuse_layers` for each.
- **`main(argv=None)`**: CLI entry point supporting `run --config <yaml> --output <dir>`.

**Technical Details**:
- Raster IO relies on `rasterio`; nodata propagation preserves blanks where no contributing pixels exist.
- Weights are logged with three decimal precision to aid QA/QC.
- The driver enforces single-band rasters to avoid silent misuse of multi-band products.
- Outputs use IEEE float32 with `NaN` nodata, compatible with downstream ML and GIS tooling.

## 3. GraphTemplateProcessor for InSAR (utils/snap_templates.py)

### Class: GraphTemplateProcessor
Generates and executes dynamic SNAP graphs.

- **__init__(self, snap_path=None, config=None)**: Sets GPT path (auto-detects).
- **parse_safe_metadata(self, safe_dir)**: Extracts {'orbit', 'baseline', 'polarization', ...}.
- **generate_template(self, metadata, graph_type='interferogram')**: Builds XML string.
  - Types: 'interferogram', 'timeseries', 'velocity'.
  - Adaptive: Baseline <150m → tighter filtering.
- **execute_graph(self, xml, input_dir, output_dir)**: Runs `gpt graph.xml -Pinput=...`.
  - Returns: Output paths list.
  - Handles: Retries via RobustDownloader.

**Technical Details**:
- XML Templating: Jinja2 for dynamic params (e.g., {{baseline}} in unwrapping).
- Optimization: Goldstein filter (alpha=0.5); SNAPHU for unwrapping.
- Batch: Supports stack processing.
- Outputs: GeoTIFFs (phase, coherence, velocity).

**Example**:
```python
from utils.snap_templates import GraphTemplateProcessor
gtp = GraphTemplateProcessor()
meta = gtp.parse_safe_metadata('S1A.SAFE')
xml = gtp.generate_template(meta)
results = gtp.execute_graph(xml, 'data/raw/insar', 'data/processed/insar')
```

**Configuration**:
- `"snap.template_params.filter_alpha"`: 0.1-1.0.
- `"snap.unwrap_method"`: 'snaphu-mcf' or 'mcf'.

## 4. RobustDownloader and Error Handling Framework (utils/error_handling.py)

### Class: RobustDownloader
Resilient file/API downloader.

- **__init__(self, max_retries=5, base_delay=1, config=None)**: Sets policy.
- **download_with_retry(self, url, path, auth_service=None, checksum=None)**: Downloads with resume/validation.
  - Auth: Integrates TokenManager.
  - Returns: bool success.
- **stream_download(self, url, callback=None)**: Chunked for large files (tqdm progress).

**Technical Details**:
- Session: requests.Session with adapters (pool=10).
- Retries: @retry_with_backoff for transients.
- Integrity: Size/checksum; removes failures.
- Throttling: Configurable bytes/sec.

### CircuitBreaker
- **__init__(threshold=5, timeout=60)**.
- **call(func, *args, **kwargs)**: Context manager; skips if open.

### TokenManager
- **get_token(self)**: Refreshes if expired.
- Services: Copernicus, Earthdata (URLs from config).

**Example**:
```python
from utils.error_handling import RobustDownloader, TokenManager
tm = TokenManager('copernicus')
downloader = RobustDownloader(auth_service='copernicus')
downloader.download_with_retry('https://example.com/data.zip', 'data.zip', tm.get_token())
```

**Error Hierarchy**:
- Base: GeoAnomalyError.
- RetryableError → ConnectionError, Timeout, RateLimitError.
- PermanentError → ValueError, FileNotFoundError.

## Integration and Best Practices

- **Config First**: Always load ConfigManager early.
- **Path Safety**: Use PathManager for all I/O.
- **Robust Calls**: Wrap external (requests, subprocess) with retry/circuit.
- **Validation**: Call validate() post-load.
- **Logging**: Use `logging.getLogger(__name__)`; config sets level.
- **Testing**: Mock utils in tests (e.g., patch PathManager).

For full source: See utils/ and scripts. Extend via inheritance.

### CLI Utilities

#### gam_data_agent.py
Unified data acquisition agent for geophysical datasets.

**CLI Interface**:
```bash
python gam_data_agent.py status [--report]
python gam_data_agent.py download [free|all] [--dataset <key>] [--bbox "lon1,lat1,lon2,lat2"] [--dry-run]
```

- **status**: Shows download status from `data/data_status.json`.
- **download**: Fetches datasets (e.g., EMAG2, EGM2008, Sentinel-1). Supports bbox for regional data.
- Datasets: emag2_magnetic, egm2008_gravity, xgm2019e_gravity, srtm_dem, insar_sentinel1.
- Integrates RobustDownloader for resilience; paths resolve via `ConfigManager`.

**Example**:
```bash
python gam_data_agent.py download free --bbox "-105,32,-104,33"
```

Tracks progress in `data/data_status.json`; idempotent runs.

#### setup_environment.py
Environment diagnostics and optional setup utility.

**CLI Interface**:
```bash
python setup_environment.py check [--deep] [--yes] [--network] [--json <file>]
python setup_environment.py setup [--yes]
python setup_environment.py requirements
```

- **check**: Runs diagnostics (system, packages, paths, config mode, Stage 1-5 components). `--deep` imports modules (may create dirs); `--network` DNS preflight.
- **setup**: Creates data/ structure and optional .env copy using the canonical configuration.
- **requirements**: Shows dependency summaries.

**Example**:
```bash
python setup_environment.py check --deep --yes --json diag.json
```

Non-invasive by default; explicit confirmation for side effects.

## Integration and Best Practices

- **Config First**: Always load ConfigManager early.
- **Path Safety**: Use PathManager for all I/O.
- **Robust Calls**: Wrap external (requests, subprocess) with retry/circuit.
- **Validation**: Call validate() post-load.
- **Logging**: Use `logging.getLogger(__name__)`; config sets level.
- **Testing**: Mock utils in tests (e.g., patch PathManager).
- **Configuration**: Import `ConfigManager` and `PathManager` directly; no compatibility layers exist.

For full source: See utils/ and scripts. Extend via inheritance.

*Updated: October 2025*