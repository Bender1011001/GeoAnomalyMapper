# Configuration Guide

GeoAnomalyMapper now ships with a single, version-controlled configuration
surface located at `config/config.json`.  The file captures the directories used
by the processing pipeline, points to the YAML recipe files that describe each
stage, and exposes tunable runtime settings (logging, serving, fusion knobs).
Environment variables can override any key without editing the JSON on disk,
allowing reproducible defaults with per-machine customisation.

## Getting started

1. Copy the tracked defaults if you need a writable copy:
   ```bash
   cp config/config.json.example config/config.json
   ```
2. (Optional) Duplicate `.env.example` into `.env` to store credentials or
   overrides you do not want to commit.
3. Install project dependencies and run the pipeline as normal.

The repository includes `config/config.json` so fresh clones work immediately.
Updating `config.json` lets you point the pipeline to alternative data roots or
change fusion behaviour without touching source code.

## File structure

The top-level keys are intentionally small and map directly onto the code base.
The default file looks like this:

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
    "dynamic_weighting": false,
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
* `pipelines.*` – pointers to existing YAML recipes (tiling, fusion, training,
  serving).  These files remain YAML because they describe large structured
  tables, but their locations are centralised here.
* `fusion`, `serving`, `logging` – runtime options consumed directly by code.

The `ConfigManager` automatically creates any directories referenced by
`project.*` and `paths.*` to simplify local setup.

## Environment overrides

Any key can be overridden using the `GAM__` prefix with double-underscore
separators.  For example:

```bash
# Use an alternate data directory without editing config.json
echo "GAM__PROJECT__DATA_ROOT=/mnt/geo/data" >> .env

# Enable dynamic fusion weighting on the fly
export GAM__FUSION__DYNAMIC_WEIGHTING=true
```

Values are parsed as JSON when possible, so `true`, `false`, `42`, or `"string"`
are handled automatically.  If parsing fails, the literal string is used.

## Access from code

Use `ConfigManager` and `PathManager` for consistent access.  Both classes live
in `utils` and are singletons.

```python
from utils.config import ConfigManager
from utils.paths import paths

config = ConfigManager()
data_root = config.get_path("project.data_root")
fusion_enabled = config.get("fusion.dynamic_weighting", False)

raw_dir = paths.raw_data
outputs = paths.join("output_dir", "reports")
```

`ConfigManager.get("section.key")` returns raw values.  Use `get_path` for
filesystem-aware values – it resolves relative paths against the repository root
and expands `~` automatically.  `PathManager` exposes the most common paths as
properties (`data_dir`, `output_dir`, `processed_dir`, etc.).

## Validation and troubleshooting

* Confirm the loader sees your changes: `python -c "from utils.config import ConfigManager; print(ConfigManager().get('project.data_root'))"`.
* Run `python -m utils.config` to print a short summary of the loaded keys.
* If a directory is missing, ensure your path points to a directory rather than
  a file.  The manager creates directories referenced under `project.*` and
  `paths.*` when it initialises.
* Remove stale overrides by deleting the relevant entries from `.env` or by
  unsetting the environment variables.

## Configuration discipline

Configuration is treated as production code.  Changes to `config.json` are
validated in CI, configuration diffs are reviewed alongside code, and the
registry of environment overrides lives in `docs/API_REFERENCE.md`.  This
ensures every execution environment – from interactive notebooks to scheduled
pipelines – observes the identical, version-controlled configuration tree.
