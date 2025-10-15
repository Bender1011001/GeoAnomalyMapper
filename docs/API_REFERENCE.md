# API Reference and Technical Documentation

**Comprehensive Reference for GeoAnomalyMapper Components**

This document provides technical details on the core APIs, classes, and functions introduced in v2.0 following the scientific code review. It covers the new dynamic fusion capabilities (WeightCalculator), InSAR processing (GraphTemplateProcessor), robustness framework (RobustDownloader), configuration/path systems, and integration points. For user guides, see [README.md](README.md) and specialized docs.

All APIs are in Python 3.9+ and integrate with the unified config system. Import from `GeoAnomalyMapper` or `utils`.

## 1. Configuration and Path Resolution System

### ConfigManager (utils/config.py)
Central loader for `config.json` + `.env` with validation.

**Class: ConfigManager**
- **__init__(self, config_file='config/config.json', env_file='.env')**: Loads and parses.
- **get(self, key, default=None)**: Retrieves value (e.g., `config.get('fusion.dynamic_weighting')`).
- **get_path(self, key)**: Resolves path with substitution (e.g., `${data_root}/raw`).
- **set(self, key, value)**: Runtime override.
- **validate(self)**: Schema check; raises ValueError on invalid.

**Technical Details**:
- Uses `json` + `python-dotenv` for loading.
- Variable substitution: `${VAR}` from env/config.
- Schema: Pydantic-like validation for types (int/float/bool/path).
- Cross-Platform: Integrates with PathManager for OS-aware paths.

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
Dynamic path resolution using pathlib.

**Class: PathManager**
- **__init__(self, config=None)**: Initializes from ConfigManager.
- **get(self, key)**: Returns Path (e.g., `pm.get('paths.raw_data')`).
- **resolve(self, key)**: Expands and normalizes (handles ~, env vars).
- **ensure_dir(self, key)**: Creates if missing.
- **validate(self)**: Checks existence/writability.

**Technical Details**:
- Substitutes `${key}` from config/env.
- OS-Aware: Uses `pathlib.Path` for / vs \.
- Caching: Memoizes resolved paths.
- Errors: Raises PermanentError on invalid (e.g., non-writable).

**Integration**: All scripts use PathManager for data I/O.

## 2. Dynamic Fusion Capabilities

### WeightCalculator (multi_resolution_fusion.py)
Computes adaptive weights for data fusion.

**Class: WeightCalculator**
- **__init__(self, config=None)**: Loads params (uncertainty, confidence).
- **compute_weights(self, layers, metadata)**: Returns dict of weights per layer.
  - Params: layers (list of rasters), metadata (dict: resolution, uncertainty).
  - Returns: {'insar': 0.85, 'gravity': 0.15, ...}.
- **spectral_weights(self, layers, cutoff=10)**: Band-specific (FFT-based).
- **update_confidence(self, validation_results)**: Adjusts c_i from known features.

**Technical Details**:
- Formula: \( w_i = \frac{1}{\sigma_i^2 + \epsilon} \times c_i \).
- σ_i: From metadata + local std dev.
- c_i: From validation (default 0.5; updated via ROC).
- Spectral: FFT decomposition; low-freq favors gravity.
- Efficiency: Vectorized with NumPy; per-pixel optional (`fast_mode=True`).

**Example**:
```python
from multi_resolution_fusion import WeightCalculator
wc = WeightCalculator()
layers = {'insar': insar_raster, 'gravity': gravity_raster}
meta = {'insar': {'resolution': 10, 'uncertainty': 0.05}, ...}
weights = wc.compute_weights(layers, meta)
fused = np.average([insar, gravity], weights=[weights['insar'], weights['gravity']])
```

**Configuration**:
- `"fusion.base_uncertainty"`: ε (0.01-0.1).
- `"fusion.confidence_threshold"`: Min c_i.

### Fusion Pipeline (multi_resolution_fusion.py)
High-level API for weighted fusion.

**Function: process_multi_resolution(bbox, output, config=None, dynamic=True)**
- Inputs: bbox (tuple lon/lat), output (str path).
- Processes: Downloads (if needed), weights, fuses, saves TIFF.
- Returns: Path to fused raster.

**Technical Details**:
- Resampling: GDAL warp with cubic/average based on direction.
- Uncertainty Propagation: Outputs sigma map.
- Validation Hook: Calls WeightCalculator.update_confidence post-fusion.

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

*Updated: October 2025 - v2.0 (Technical APIs)*