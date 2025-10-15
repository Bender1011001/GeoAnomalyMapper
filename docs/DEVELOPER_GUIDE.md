# Developer Guide for New Utilities

**Extending GeoAnomalyMapper's Modular Utilities**

The scientific code review emphasized modularity to replace monolithic scripts. The new `utils/` directory provides reusable components: `paths.py` for cross-platform resolution, `error_handling.py` for robustness, and `snap_templates.py` for InSAR processing. This guide covers implementation, extension, and best practices for developers contributing to or customizing the project.

## Introduction

Utilities are designed for:
- **Reusability**: Import into scripts or other projects.
- **Testability**: Unit tests in `tests/utils/`.
- **Configuration Integration**: Use `utils/config.py` for params.
- **Documentation**: Inline docstrings + this guide.

**Import Pattern**:
```python
from utils import paths, error_handling, snap_templates
from utils.config import ConfigManager
```

**Setup for Development**:
```bash
pip install -e ".[dev]"  # Includes testing deps
python setup_environment.py validate
pytest tests/utils/  # Run utility tests
```

### v2 Integration Overview
Stages 1-5 introduced modular config/paths, data agent, fusion enhancements, validation, and environment setup. This guide covers extending these for v2.

- **Staged Approach**: v2 uses shims for backward compatibility. Enable via `GAM_USE_V2_CONFIG=true`. Future stages build on this (e.g., Stage 6: docs convergence).
- **Key Changes**: Structured config replaces hardcoded values; shims allow gradual migration.

## 1. paths.py - Cross-Platform Path Resolution

### Purpose
Replaces hardcoded paths with dynamic, OS-aware resolution using pathlib. Handles substitution from config.json (e.g., `${data_root}`).

### Key Classes/Functions
- **PathManager**: Central resolver.
  ```python
  from utils.paths import PathManager

  pm = PathManager()  # Loads from config
  raw_dir = pm.get('paths.raw_data')  # Path('./data/raw')
  insar_dir = pm.resolve('insar_dir')  # Ensures exists
  ```

- **validate_paths()**: Checks writability, existence.
  ```python
  from utils.paths import validate_paths
  if not validate_paths():
      raise ValueError("Path issues detected")
  ```

### Extension
- Add new keys to config.json (e.g., `"custom_dir": "${data_root}/custom"`).
- Override: `pm.set_base('/alternative/root')`.
- Custom Resolver: Subclass PathManager for project-specific logic.

**Best Practices**:
- Always use `pm.get()` over string literals.
- Handle non-existent: `pm.ensure_dir('output_dir')`.
- Testing: Mock config in unit tests.

**Example in Script**:
```python
# In data_agent.py
pm = PathManager()
raw_path = pm.get('paths.raw_data')
if not raw_path.exists():
    raw_path.mkdir(parents=True)
```

### Shim System for Compatibility
`utils/config_shim.py` and `utils/paths_shim.py` provide v1/v2 bridges:
- **When v2 disabled** (`GAM_USE_V2_CONFIG=false`): Returns hardcoded defaults (e.g., `get_data_dir() -> Path('data')`).
- **When enabled**: Delegates to ConfigManager/PathManager.
- **Gradual Adoption**: Update imports to shims; enable flag to activate v2 without code changes.

**Example**:
```python
from utils import config_shim, paths_shim

data_dir = config_shim.get_data_dir()  # v1: 'data'; v2: from config.json
output_path = paths_shim.get_output_dir() / 'report.txt'
```

This maintains v1 workflows while allowing opt-in to v2 features like validation and substitution.

## 2. error_handling.py - Robustness Framework

### Purpose
Provides retry, circuit breaker, and error categorization for reliable operations (downloads, API calls).

### Key Classes/Functions
- **RobustDownloader**: Core downloader with resilience.
  ```python
  from utils.error_handling import RobustDownloader

  downloader = RobustDownloader(max_retries=5, base_delay=1)
  try:
      success = downloader.download_with_retry(url, output_path, auth='copernicus')
  except PermanentError as e:
      logger.error(f"Failed permanently: {e}")
  ```

- **@retry_with_backoff**: Decorator for functions.
  ```python
  from utils.error_handling import retry_with_backoff

  @retry_with_backoff(max_retries=3)
  def fetch_metadata(url):
      return requests.get(url).json()
  ```

- **CircuitBreaker**: State management.
  ```python
  from utils.error_handling import CircuitBreaker

  cb = CircuitBreaker(threshold=5, timeout=60)
  if cb.is_open():
      return None  # Skip
  with cb:
      result = risky_operation()
  ```

- **TokenManager**: Auth handling.
  ```python
  from utils.error_handling import TokenManager

  tm = TokenManager(service='copernicus')
  token = tm.get_token()  # Refreshes if needed
  headers = {'Authorization': f'Bearer {token}'}
  ```

- **ensure_dns(hosts)**: Pre-flight check.
  ```python
  ensure_dns(['urs.earthdata.nasa.gov'])
  ```

### Extension
- Custom Exceptions: Inherit from RetryableError/PermanentError.
- New Services: Add to `DEFAULT_SERVICES` dict (auth URLs, hosts).
- Retry Policies: Override `get_delay()` for custom backoff.
- Integration: Wrap external calls (e.g., gdal) with decorator.

**Best Practices**:
- Categorize errors explicitly (raise AuthError for 401).
- Log context: Include service/URL in exceptions.
- Graceful Degradation: Skip non-critical (e.g., optional InSAR).
- Testing: Use `pytest-mock` to simulate failures; assert retries.
- **v2 Integration**: Use with config for retry params (e.g., `max_retries = config.get('robustness.max_retries')`).

**Example Extension** (Custom Downloader):
```python
class CustomDownloader(RobustDownloader):
    def __init__(self):
        super().__init__(max_retries=10)  # Override

    def download_gdal(self, source, dest):
        @retry_with_backoff()
        def gdal_call():
            subprocess.run(['gdalwarp', source, dest])
        gdal_call()
```

## 3. snap_templates.py - Dynamic InSAR Processing

### Purpose
Generates adaptive SNAP Graph XML from Sentinel-1 metadata, replacing static templates for varying acquisitions.

### Key Classes/Functions
- **GraphTemplateProcessor**: Builds and executes graphs.
  ```python
  from utils.snap_templates import GraphTemplateProcessor

  gtp = GraphTemplateProcessor(snap_path='/path/to/gpt')
  metadata = parse_safe_metadata('path/to/.SAFE')  # Orbit, baseline, etc.
  graph_xml = gtp.generate_template(metadata, target='interferogram')
  result = gtp.execute_graph(graph_xml, input_dir='data/raw/insar', output_dir='data/processed/insar')
  ```

- **parse_safe_metadata(path)**: Extracts params from .SAFE.
  ```python
  meta = parse_safe_metadata('S1A_IW_SLC__20230101.SAFE')
  # Returns: {'orbit': 'ascending', 'baseline': 120, 'polarization': 'VV'}
  ```

- **default_params()**: Configurable defaults (filter alpha, unwrap method).

### Extension
- Add Processors: Subclass for custom nodes (e.g., atmospheric correction).
- Metadata Parsers: Extend for other formats (e.g., ALOS).
- Params: Override via config (`"snap.template_params"`).
- Batch: `gtp.process_batch(input_dirs)` for stacks.

**Best Practices**:
- Validate Metadata: Check baseline <150m.
- Error Handling: Wrap with RobustDownloader for GPT calls.
- Outputs: Standardize to GeoTIFF with CRS.
- Testing: Mock XML generation; use sample .SAFE.
- **v2 Integration**: Load params from config (e.g., `filter_alpha = config.get('snap.template_params.filter_alpha')`).

**Example in Pipeline**:
```python
# In data_agent.py post-download
if config.get('insar.auto_process'):
    gtp = GraphTemplateProcessor()
    for safe_dir in insar_dirs:
        meta = parse_safe_metadata(safe_dir)
        if meta['baseline'] < 150:
            xml = gtp.generate_template(meta)
            gtp.execute_graph(xml, safe_dir, processed_dir)
```

## Extending the Utilities System

### Adding New Utilities
1. Create `utils/new_utility.py` with docstrings.
2. Add to `__init__.py`: `from . import new_utility`.
3. Config Integration: Use ConfigManager for params.
4. Tests: `tests/test_new_utility.py` with pytest.
5. Docs: Update this guide + inline examples.

### Integration Patterns
- **In Scripts**: Import and use (e.g., data_agent.py uses all three).
- **External Projects**: `pip install -e .` then import.
- **Hooks**: Config `"extensions": ["custom_module"]` for plugins.
- **v2 Guidance**: Use shims for config/paths in new code; enable flags for features like dynamic weighting.

### Testing and CI
- Run: `pytest tests/utils/ -v`.
- Coverage: `pytest --cov=utils`.
- Linting: `black utils/`, `flake8 utils/`.

## Best Practices

- **Modularity**: Keep utils stateless; pass config.
- **Error Propagation**: Raise custom exceptions; don't swallow.
- **Performance**: Cache resolutions (e.g., paths); async for I/O.
- **Documentation**: NumPy-style docstrings; examples in code.
- **Versioning**: Semantic changes → bump utils version in pyproject.toml.
- **Security**: No secrets in utils; defer to .env.
- **Staged Integration**: Follow Stages 1-5 pattern: config first, then paths, robustness, etc. Document shims for compatibility.

For API details: [API_REFERENCE.md](API_REFERENCE.md).

Contribute via PRs; see CONTRIBUTING.md.

*Updated: October 2025 - v2.0 (Stage 1-5 Integration)*