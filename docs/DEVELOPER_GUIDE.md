# Developer Guide for GeoAnomalyMapper Utilities

GeoAnomalyMapper's utility layer provides hardened building blocks that the
scientific pipeline depends on: deterministic configuration management, robust
path resolution, resilient data acquisition, and adaptive InSAR templating. This
guide documents the design goals of each component, recommended extension
patterns, and code snippets that illustrate best practices.

## Development environment

```bash
pip install -e .[all]
python -m utils.config       # Inspect the loaded configuration tree
pytest                       # Execute the full test suite if present
```

All utilities assume Python 3.10+ and rely on the central configuration file
(`config/config.json`).

## Configuration integration

`utils.config.ConfigManager` is a singleton that loads the JSON configuration,
applies environment overrides, and materialises directories on demand. Use it to
fetch scalar settings or resolved paths:

```python
from utils.config import ConfigManager

config = ConfigManager()
cache_path = config.get_path("project.cache_dir")
robust_retries = config.get("robustness.max_retries", 5)
```

`ConfigManager` is thread-safe for read operations and caches lookups to avoid
filesystem churn. Call `ConfigManager().reload()` to re-read configuration after
modifying `config.json` at runtime (for example, in integration tests).

## Path management (`utils/paths.py`)

`PathManager` exposes common directories as `pathlib.Path` objects resolved
relative to the repository root. It keeps path handling consistent across
platforms and honours overrides from `config.json` or environment variables.

```python
from utils.paths import paths

raw_tiles = paths.raw_data
reports_dir = paths.join("output_dir", "reports")
```

* Add new canonical locations by extending the dictionary in `PathManager.__init__`.
* Use `paths.resolve("relative/path")` when you need an absolute path from the
  repository root.
* When writing tests, temporarily modify configuration keys with
  `ConfigManager().set("paths.raw_data", tmp_path)` to point utilities at a
  sandboxed directory.

## Resilience framework (`utils/error_handling.py`)

The resilience toolkit underpins every network-bound component. It provides
structured error classes, retry semantics with exponential backoff and jitter,
circuit breakers, DNS verification, and credential management.

### Key abstractions

- **`retry_with_backoff` decorator** – Wraps any function and retries predictable
  transient failures. Defaults to five attempts with exponential backoff.
- **`RobustDownloader`** – High-level helper that combines retries, checksum
  validation, and resume support for large payloads.
- **`CircuitBreaker`** – Guards critical sections by halting repeated failures
  and reopening after a cooldown.
- **`TokenManager`** – Centralises OAuth-style token caching and refresh.

### Usage pattern

```python
from utils.error_handling import RobustDownloader, PermanentError

agent = RobustDownloader(max_retries=6, base_delay=1.5)
try:
    agent.download_with_retry(url, output_path, auth="copernicus")
except PermanentError as exc:
    logger.error("Acquisition aborted: %s", exc)
```

Tune behaviour through `config.json` under the `robustness` section (retry
counts, thresholds, cache locations). Log records emitted by the framework are
structured and include dataset identifiers, attempt counts, and latency metrics,
which simplifies monitoring.

## SNAP template generation (`utils/snap_templates.py`)

`GraphTemplateProcessor` builds Sentinel-1 processing graphs dynamically. It
inspects SAFE metadata to configure orbital parameters, baselines, polarisation,
and processing chains, then executes SNAP GPT with the resulting XML.

```python
from utils.snap_templates import GraphTemplateProcessor

processor = GraphTemplateProcessor()
metadata = processor.parse_safe_metadata("S1A_IW_SLC__20230101T123456.SAFE")
graph_xml = processor.generate_template(metadata, target="interferogram")
processor.execute_graph(
    graph_xml,
    input_dir="data/raw/sentinel1",
    output_dir="data/processed/insar",
)
```

Best practices:

- Validate baselines and Doppler centroids before launching GPT to avoid wasted
  processing time.
- Persist generated XML alongside outputs for auditability; the processor
  returns the graph path for convenience.
- Customise default parameters through the `snap.template_params` block in
  `config.json` (e.g., filtering kernels, unwrapping strategies).

## Extending the utilities

1. **Follow the configuration contract** – Any new helper should expose
   tunables via `config.json` rather than module-level constants.
2. **Document invariants** – Provide docstrings that explain assumptions about
   CRS, resolution, or I/O semantics.
3. **Write deterministic code paths** – Utility functions should avoid global
   state beyond the shared configuration singletons so they remain testable.
4. **Surface metrics** – When adding long-running operations, emit structured
   logs (`INFO` for progress, `WARNING` for retries, `ERROR` for unrecoverable
   failures) to ensure observability.
5. **Keep imports lightweight** – Utilities intentionally avoid heavy
   dependencies so they can be reused by notebooks, CLI tools, or API services.

Adhering to these guidelines keeps GeoAnomalyMapper's infrastructure predictable
and production-ready while allowing advanced users to tailor behaviour to their
projects.
