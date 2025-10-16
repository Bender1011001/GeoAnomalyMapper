# Configuration Guide

GeoAnomalyMapper centralises all runtime configuration in `config/config.json`.
The file defines the project directories used by each processing stage, points
to the YAML recipe files that describe tiling, fusion, training, and serving, and
exposes tunable runtime settings such as logging levels and service endpoints.
Environment variables can override any key without editing the JSON on disk,
ensuring reproducible defaults with machine-specific flexibility.

## Getting started

1. Clone the repository. A fully populated `config/config.json` is tracked so
   the pipeline runs immediately.
2. Copy `config/config.json.example` if you prefer to maintain a private,
   writable variant:
   ```bash
   cp config/config.json.example config/config.json
   ```
3. (Optional) Duplicate `.env.example` into `.env` to store credentials or local
   overrides that should not be committed.
4. Install project dependencies and execute the workflow normally.

Updating `config.json` lets you point the pipeline to alternative data roots or
adjust behaviour without editing source code.

## File structure

The top-level keys mirror the codebase. The default file looks like this:

```json
{
  "project": {
    "name": "GeoAnomalyMapper",
    "version": "3.0.0",
    "data_root": "data",
    "outputs_dir": "data/outputs",
    "processed_dir": "data/processed",
    "cache_dir": "data/cache"
  },
  "paths": {
    "raw_data": "data/raw",
    "interim": "data/interim",
    "features": "data/features",
    "models": "data/models",
    "logs": "data/outputs/logs"
  },
  "pipelines": {
    "tiling_config": "config/tiling_zones.yaml",
    "data_sources_config": "config/data_sources.yaml",
    "fusion_config": "config/fusion.yaml",
    "training_config": "config/training.yaml",
    "serving_config": "config/serving.yaml"
  },
  "fusion": {
    "dynamic_weighting": true,
    "temperature": 1.0,
    "target_resolution_m": 250,
    "uncertainty_threshold": 0.25
  },
  "serving": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false
  },
  "logging": {
    "level": "INFO",
    "structured": true
  }
}
```

* `project.*` – canonical directories created on start-up.
* `paths.*` – additional folders used by preprocessors and exporters.
* `pipelines.*` – pointers to the YAML recipes for tiling, fusion, training, and
  serving. YAML remains ideal for these structured tables; the JSON file merely
  records their locations.
* `fusion`, `serving`, `logging` – runtime options consumed directly by code.

`ConfigManager` automatically creates any directories referenced by `project.*`
and `paths.*`, so fresh environments do not require manual scaffolding.

## Environment overrides

Any key can be overridden using the `GAM__` prefix with double-underscore
separators. For example:

```bash
# Use an alternate data directory without editing config.json
echo "GAM__PROJECT__DATA_ROOT=/mnt/geo/data" >> .env

# Enable dynamic fusion weighting on the fly
export GAM__FUSION__DYNAMIC_WEIGHTING=true
```

Values are parsed as JSON when possible, so `true`, `false`, `42`, or `"string"`
are handled automatically. If parsing fails, the literal string is used.

## Access from code

Use `ConfigManager` and `PathManager` for consistent access. Both classes live in
`utils` and are singletons.

```python
from utils.config import ConfigManager
from utils.paths import paths

config = ConfigManager()
data_root = config.get_path("project.data_root")
fusion_enabled = config.get("fusion.dynamic_weighting", False)

raw_dir = paths.raw_data
outputs = paths.join("output_dir", "reports")
```

`ConfigManager.get("section.key")` returns raw values. Use `get_path` for
filesystem-aware values – it resolves relative paths against the repository root
and expands `~` automatically. `PathManager` exposes the most common paths as
properties (`data_dir`, `output_dir`, `processed_dir`, etc.).

## Validation and troubleshooting

* Confirm the loader sees your changes:
  ```bash
  python -c "from utils.config import ConfigManager; print(ConfigManager().get('project.data_root'))"
  ```
* Run `python -m utils.config` to print a short summary of the loaded keys.
* If a directory is missing, ensure your path points to a directory rather than
  a file. The manager creates directories referenced under `project.*` and
  `paths.*` when it initialises.
* Remove stale overrides by deleting the relevant entries from `.env` or by
  unsetting the environment variables.
