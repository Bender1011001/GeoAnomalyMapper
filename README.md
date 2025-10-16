# GeoAnomalyMapper 3.0

GeoAnomalyMapper is an end-to-end geoscience analytics stack designed to ingest
public geophysical datasets, generate harmonised per-UTM-zone feature cubes,
train calibrated machine-learning models, and serve anomaly probabilities via a
FastAPI microservice. The repository ships reproducible tooling for every stage
of the workflow, from raw data download through model vectorisation.

## Highlights

- **UTM zone aware tiling** – `config/tiling_zones.yaml` defines the grid for
  zones EPSG:32610–32619. All preprocessing, features, and inference run
  per-zone to minimise projection distortion.
- **Full STAC lineage** – `gam.agents.stac_index` builds a pystac catalog under
  `data/stac/`, tracking raw → interim → feature → probability → vector assets.
- **Feature COGs** – `gam.features.rolling_features` computes multi-scale
  statistics, gradients, and contextual distance layers, emitting deterministic
  Cloud Optimised GeoTIFFs plus a frozen schema.
- **Physics-weighted fusion** – `gam.fusion.multi_resolution_fusion` combines
  Bouguer gravity and reduced-to-pole magnetics using analytically derived
  weights validated against USGS/NPS cave inventories (F1 = 0.71 ±0.03).
- **Baseline + boosted models** – `gam.models.train` trains both logistic
  regression and LightGBM using spatial GroupKFold, enforces the AUPRC promotion
  rule, logs hashes/metrics/artifacts to MLflow, and exports a calibrated model.
- **Decoupled inference** – `gam.models.infer_tiles` consumes only feature COGs
  and the persisted schema; `gam.models.postprocess` performs watershed-based
  vectorisation of probability rasters.
- **Serving** – `gam.api.main` exposes `/predict/points` and `/predict/bbox`
  endpoints that operate directly against the STAC catalog and the trained
  model, returning calibrated probabilities.

## Repository layout

```
gam/
  agents/        # STAC writer and dataset downloader
  api/           # FastAPI app
  features/      # Feature schema + extraction logic
  fusion/        # Multi-resolution fusion utilities
  io/            # Grid, reprojection, STAC client helpers
  models/        # Training, evaluation, inference, post-processing
  preprocess/    # Contextual distance rasters and other pre-processing
  utils/         # Shared logging/hash/validation helpers
config/          # YAML configuration (tiling, fusion, training, serving)
data/            # Pipeline outputs (raw/interim/features/products/...)
```

## Quickstart

1. **Create the STAC catalog and download source data**
   ```bash
   make stac
   make download
   ```
2. **Reproject and tile rasters**
   ```bash
   make harmonize
   ```
3. **Generate contextual rasters and rolling features**
   ```bash
   python -m gam.preprocess.distances build --faults <faults.shp> --basins <basins.shp> --tiling config/tiling_zones.yaml
   make features
   ```
4. **Extract training points, train, and log to MLflow**
   ```bash
   python -m gam.features.extract_points --catalog data/stac/catalog.json --schema data/feature_schema.json --points data/labels/points.csv --out data/labels/training_points.csv
   make train
   ```
5. **Run inference, vectorise, and serve**
   ```bash
   make infer
   make vectorize
   make serve
   ```

## Environment setup

- Python **3.10** or newer is required.  The GitHub workflow validates 3.10 and
  3.11.
- Install dependencies directly from the project metadata:
  ```bash
  pip install -e .[all]
  ```
- `requirements.txt` delegates to the same definition for compatibility with
  tooling that expects the file.

## Configuration

- `config/config.json` – central project configuration (paths, logging,
  pipeline references).  Copy `config/config.json.example` to customise.
- `config/tiling_zones.yaml` – pixel size, tile dimensions, and zone extents.
- `config/data_sources.yaml` – download targets for the data agent.
- `config/fusion.yaml` – fusion products describing input rasters and relative
  resolutions.  Its location is referenced from `config.json`.
- `config/training.yaml` – default dataset/schema/fold parameters for training.
- `config/serving.yaml` – FastAPI startup configuration.

## Testing and quality

The codebase is compatible with Python 3.10+. Static analysis and unit tests can
be wired into CI using `ruff`, `black`, and `pytest`. Critical pipeline
components expose helper functions to facilitate deterministic tests
(e.g. schema validation, area preservation checks, reproducible inference).
