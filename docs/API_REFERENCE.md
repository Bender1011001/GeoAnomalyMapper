# API Reference and Technical Documentation

This reference documents the core Python APIs and CLI entry points that power
GeoAnomalyMapper. All modules target Python 3.10+ and expect configuration to be
provided by `config/config.json`.

## Configuration and path management

### `utils.config.ConfigManager`

Singleton loader for the JSON configuration. Responsibilities:

- Load `config/config.json` (or a supplied path) and apply overrides defined via
  environment variables prefixed with `GAM__`.
- Provide `get(key, default=None)` and `get_path(key, default=None)` helpers for
  dotted lookups and filesystem-aware paths.
- Materialise directories referenced under `project.*` and `paths.*` when the
  manager initialises, ensuring downstream code can rely on them.
- Expose `reload()`, `items()`, and `as_dict()` for diagnostics and testing.

```python
from utils.config import ConfigManager

config = ConfigManager()
model_dir = config.get_path("paths.models")
use_dynamic_weights = config.get("fusion.dynamic_weighting", False)
```

### `utils.paths.PathManager`

Thin wrapper around `ConfigManager` that exposes common directories as
properties (`data_dir`, `output_dir`, `processed_dir`, etc.) and provides
helpers such as `join(base_key, *subpaths)` and `resolve(relative_path)`.

```python
from utils.paths import paths

raw_tiles = paths.raw_data
reports_dir = paths.join("output_dir", "reports")
```

## Fusion pipeline (`gam.fusion.multi_resolution_fusion`)

The fusion module combines multiple rasters into a harmonised product using
resolution-aware weights.

### `resolution_weighting(resolutions: Dict[str, float], temperature: float = 1.0)`

Located in `gam.fusion.weight_calculator`. Accepts a mapping from layer name to
nominal ground sampling distance (metres) and returns normalised weights that
favour higher resolution layers. The optional temperature parameter controls the
sharpness of the weighting distribution.

```python
from gam.fusion.weight_calculator import resolution_weighting

weights = resolution_weighting({"insar": 10.0, "gravity": 250.0}).weights
# {'insar': 0.96, 'gravity': 0.04}
```

### `fuse_layers(layers: Dict[str, Dict[str, Any]], output_path: Path, *, dynamic: bool = True, temperature: float = 1.0)`

Reads each raster, applies either dynamic or uniform weights, and writes a Cloud
Optimised GeoTIFF via `gam.io.cogs.write_cog`. The `dynamic` flag switches
between resolution-aware weights and equal weights; the `temperature` argument
controls how sharply the distribution favours fine-resolution layers. Each
layer definition must include `path` and `resolution` keys. Dataset handles are
closed properly to avoid descriptor leaks.

```python
from pathlib import Path
from gam.fusion.multi_resolution_fusion import fuse_layers

layers = {
    "insar": {"path": Path("data/features/insar.tif"), "resolution": 10.0},
    "gravity": {"path": Path("data/features/gravity.tif"), "resolution": 250.0},
}
fused_path = fuse_layers(layers, Path("data/products/fused.tif"), temperature=0.8)
```

### Command-line interface

Use the module as a CLI to process all products defined in `config/fusion.yaml`:

```bash
python -m gam.fusion.multi_resolution_fusion run \
  --config config/fusion.yaml \
  --output data/products/fusion \
  --temperature 0.9
```

Specify `--static` to disable dynamic weighting for sensitivity checks. Each
product entry in the YAML file should define a `name` and a `layers` list with
`path` and `resolution` fields.

## InSAR graph generation (`utils.snap_templates`)

`GraphTemplateProcessor` generates SNAP GPT graphs based on Sentinel-1 SAFE
metadata and executes them using `gpt`.

### `GraphTemplateProcessor(template_path: str, config: Optional[Dict] = None)`

- Loads the XML template and prepares a substitution map.
- Optional `config` overrides defaults such as subswath, polarisation, and burst
  indices.

### `extract_sentinel1_params(safe_path: str) -> Dict`

- Parses `manifest.safe`, measurement TIFFs, and annotation XML to determine
  acquisition mode, available subswaths, polarisations, burst indices, and
  approximate AOI.
- Returns a dictionary containing defaults suitable for interferogram
  generation.

### `validate_parameters(params: Dict, master_safe: str, slave_safe: str) -> bool`

- Ensures the master/slave pair share mode, subswath, and polarisation.
- Computes burst overlap and adjusts `first_burst`/`last_burst` accordingly.
- Checks bounding box overlap and raises descriptive errors on mismatches.

### `generate_graph(template_params: Dict, output_path: str) -> str`

- Substitutes template variables (e.g., `${SUBSWATH}`) and writes a ready-to-run
  graph XML file.

### `process_interferogram(master_safe: str, slave_safe: str, output_dir: str, manual_params: Optional[Dict] = None) -> Dict`

- High-level helper that extracts parameters, validates the pair, generates the
  graph, and executes SNAP GPT. Returns a dictionary containing the graph path,
  output file, and parameters used.

```python
from utils.snap_templates import GraphTemplateProcessor

processor = GraphTemplateProcessor("config/templates/interferogram.xml")
result = processor.process_interferogram(
    master_safe="data/raw/sentinel1/master.SAFE",
    slave_safe="data/raw/sentinel1/slave.SAFE",
    output_dir="data/processed/insar",
)
print(result["output_file"])
```

## Resilience utilities (`utils.error_handling`)

The resilience toolkit offers reusable primitives for reliable data acquisition.

### Key components

- `retry_with_backoff` – Decorator implementing exponential backoff with jitter.
- `RobustDownloader` – Wraps HTTP downloads with retries, checksum validation,
  and resume support.
- `CircuitBreaker` – Protects external services from repeated failures.
- `TokenManager` – Manages OAuth-style tokens for authenticated services.
- Exception hierarchy (`GeoAnomalyError`, `RetryableError`, `PermanentError`,
  etc.) – Allows fine-grained error handling.

```python
from utils.error_handling import RobustDownloader, PermanentError

agent = RobustDownloader(max_retries=6, base_delay=1.5)
try:
    agent.download_with_retry("https://example.com/data.zip", "data/raw/data.zip")
except PermanentError as exc:
    logger.error("Acquisition aborted: %s", exc)
```

## Command-line entry points

### STAC indexing

```bash
python -m gam.agents.stac_index init --out data/stac
```

Initialises a local STAC catalogue for tracking dataset provenance.

### Data synchronisation

```bash
python -m gam.agents.gam_data_agent sync --config config/data_sources.yaml
```

Downloads and updates datasets defined in the configuration, recording progress
in `data/stac/status.json`.

### Raster harmonisation

```bash
python -m gam.io.reprojection run --tiling config/tiling_zones.yaml
```

Reprojects and tiles source rasters into the unified grid.

### Feature generation

```bash
python -m gam.features.rolling_features run \
  --tiling config/tiling_zones.yaml \
  --schema data/feature_schema.json
```

Computes contextual metrics and emits Cloud Optimised GeoTIFF feature layers.

### Model training and inference

```bash
python -m gam.models.train run \
  --dataset data/labels/training_points.csv \
  --schema data/feature_schema.json \
  --output artifacts

python -m gam.models.infer_tiles run \
  --features data/features \
  --model artifacts/selected_model.pkl \
  --schema data/feature_schema.json \
  --output data/products
```

The training command logs metrics to MLflow (configured in `config.json`) and
emits calibrated models. Inference consumes the persisted schema to guarantee
feature alignment.

### Post-processing and vectorisation

```bash
python -m gam.models.postprocess run \
  --probabilities data/products \
  --output data/products/vectors \
  --threshold from:mlflow
```

Transforms probability rasters into vector features using watershed segmentation
and topology validation.

### API serving

```bash
uvicorn gam.api.main:app --host 0.0.0.0 --port 8080
```

Hosts the FastAPI service providing `/predict/points` and `/predict/bbox`
endpoints that operate directly against the STAC catalogue and trained models.
