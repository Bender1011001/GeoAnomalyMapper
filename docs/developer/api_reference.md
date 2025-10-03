# API Reference

This reference documents GeoAnomalyMapper (GAM)'s public API. It's organized by module, with key classes, functions, data structures, and usage examples. For full docstrings, build with Sphinx (see [Sphinx Setup](../conf.py)). All APIs use type hints (Python 3.10+) and follow PEP 484.

GAM's API is modular: import from submodules (e.g., `from gam.core.pipeline import run_pipeline`). Core interfaces are abstract (ABC) for extensibility.

## Core Module (`gam.core`)

The core orchestrates the pipeline, handles config, parallelism, and utilities.

### Key Functions

- `run_pipeline(bbox: Tuple[float, float, float, float], modalities: List[str], config: Dict[str, Any] = None, output_dir: str = "results") -> Dict[str, Any]`
  - Runs the full GAM pipeline: ingestion → preprocessing → modeling → detection → visualization.
  - **Parameters**:
    - `bbox`: (min_lat, max_lat, min_lon, max_lon).
    - `modalities`: List of "gravity", "magnetic", "seismic", "insar", or "all".
    - `config`: Optional dict or path to YAML; defaults load from 'config.yaml'.
    - `output_dir`: Where to save results.
  - **Returns**: Dict with 'anomalies' (pd.DataFrame), 'models' (dict of xarray.Dataset), 'visualizations' (dict of file paths).
  - **Raises**: GAMError subclasses (e.g., DataFetchError).
  - **Example**:
    ```python
    from gam.core.pipeline import run_pipeline
    results = run_pipeline((29.9, 30.0, 31.1, 31.2), ["gravity"], output_dir="giza_results")
    print(results['anomalies'].head())
    ```

- `load_config(path: str = "config.yaml") -> Dict[str, Any]`
  - Loads and validates YAML config using Pydantic.
  - **Returns**: Validated dict.
  - **Example**:
    ```python
    config = load_config("custom_config.yaml")
    ```

- `validate_config(config: Dict[str, Any]) -> Dict[str, Any]`
  - Validates against schema (bbox, modalities, etc.).
  - **Raises**: ValidationError.

### Key Classes

- `PipelineOrchestrator`
  - Manages workflow steps. Instantiate with config, call `run()`.
  - **Methods**:
    - `ingest()`: Calls ingestion.manager.fetch_data.
    - `preprocess()`: Calls preprocessing.manager.process.
    - `model()`: Calls modeling.manager.invert_and_fuse.
    - `visualize(anomalies: pd.DataFrame)`: Calls visualization.manager.generate.
  - **Example**:
    ```python
    from gam.core.pipeline import PipelineOrchestrator
    orch = PipelineOrchestrator(config)
    anomalies = orch.model(orch.preprocess(orch.ingest()))
    ```

- `GAMPipeline`
  - High-level orchestrator for the full GAM workflow: ingestion → preprocessing → modeling → visualization.
  - **Parameters**:
    - `config_path`: Path to YAML config (str, optional).
    - `config`: GAMConfig object (optional; overrides path).
    - `use_dask`: Enable parallel processing (bool, default: True).
    - `n_workers`: Dask workers (int, default: 4).
    - `cache_dir`: Cache path (Path, optional).
    - Managers: Optional injections for ingestion, preprocessing, etc.
  - **Methods**:
    - `run_analysis(bbox: Tuple[float,...], modalities: List[str], output_dir: str, ...)`: Executes pipeline, returns PipelineResults.
    - `close()`: Shuts down Dask client.
  - **Returns**: PipelineResults dataclass with raw_data, processed_data, inversion_results, anomalies, visualizations.
  - **Example**:
    ```python
    from gam.core.pipeline import GAMPipeline
    pipeline = GAMPipeline(config_path='config.yaml', use_dask=True)
    results = pipeline.run_analysis(bbox=(29.9, 30.0, 31.1, 31.2), modalities=['gravity'])
    pipeline.close()
    ```
  - **Notes**: Supports global tiling; validates bbox; logs progress.

- `ParallelExecutor` (uses Dask/Joblib)
  - Handles tiling and distribution.
  - **Methods**:
    - `submit_tasks(tasks: List[Callable]) -> List[futures]`.
    - `compute(futures: List) -> List[results]`.

### Data Structures
- `PipelineResults` (dataclass): anomalies (pd.DataFrame), models (Dict[str, xarray.Dataset]), metadata (Dict).

## Ingestion Module (`gam.ingestion`)

Handles data fetching and caching from public sources.

### Key Classes

- `DataSource` (ABC)
  - Abstract base for fetchers.
  - **Methods**:
    - `fetch(bbox: Tuple[float, ...], params: Dict[str, Any]) -> RawData`: Abstract; implement in subclasses.
  - **Subclasses**:
    - `GravitySource`: Fetches from USGS API.
    - `SeismicSource`: Uses ObsPy FDSN client.
    - `InSARSource`: Uses SentinelAPI.

- `CacheManager` (ABC)
  - **Methods**:
    - `store(key: str, data: Any, compress: bool = True) -> None`.
    - `retrieve(key: str) -> Optional[Any]`.
    - `delete(key: str) -> None`.
  - **Subclasses**:
    - `HDF5Cache`: For large arrays.
    - `SQLCache`: For metadata/anomalies (SQLAlchemy).

### Key Functions

- `fetch_data(bbox: Tuple, modalities: List[str], sources: Dict = None) -> Dict[str, RawData]`
  - Fetches and caches data.
  - **Returns**: Dict[modality, RawData].
  - **Example**:
    ```python
    from gam.ingestion.manager import fetch_data
    raw = fetch_data((0, 10, 0, 10), ["gravity"])
    ```

- `RawData` (dataclass): metadata (Dict), values (Union[np.ndarray, xarray.Dataset, obspy.Stream]).

## Preprocessing Module (`gam.preprocessing`)

Aligns, filters, and grids raw data.

### Key Classes

- `Preprocessor` (ABC)
  - **Methods**:
    - `process(raw_data: RawData, grid_res: float, config: Dict) -> xarray.Dataset`: Abstract.
  - **Subclasses**:
    - `GravityPreprocessor`: Unit conversion, Gaussian filter.
    - `SeismicPreprocessor`: Bandpass, decimation (ObsPy).
    - `InSARPreprocessor`: Phase unwrapping (MintPy integration).

- `SeismicPreprocessor`
  - Preprocessor for seismic waveforms: bandpass filtering and STA/LTA picking for travel times.
  - **Parameters**: Inherited from Preprocessor.
  - **Methods**:
    - `process(data: RawData, **kwargs) -> ProcessedGrid`: Filters (lowcut/highcut Hz), picks onsets (sta_len/lta_len s, threshold), grids times (s).
  - **Returns**: ProcessedGrid with 'data' as travel times (s), metadata: modality='seismic', feature='travel_time'.
  - **Example**:
    ```python
    from gam.preprocessing.processors import SeismicPreprocessor
    proc = SeismicPreprocessor()
    grid = proc.process(raw_stream, lowcut=0.1, highcut=1.0, threshold=3.5)
    ```
  - **Notes**: Uses ObsPy Stream; outputs for tomography input. Reference: Trnkoczy (2010).

### Key Functions

- `preprocess_data(raw_data: Dict[str, RawData], grid_res: float, filter_params: Dict) -> Dict[str, xarray.Dataset]`
  - Applies modality-specific processing.
  - **Returns**: Processed grids.
  - **Example**:
    ```python
    from gam.preprocessing.manager import preprocess_data
    processed = preprocess_data(raw, 0.1, {"gaussian_sigma": 1.0})
    ```

- `ProcessedGrid` (type alias): xarray.Dataset with coords ['lat', 'lon', 'depth'], vars ['data', 'uncertainty'].

## Modeling Module (`gam.modeling`)

Performs inversion, fusion, and anomaly detection.

### Key Classes

- `Inverter` (ABC)
  - **Methods**:
    - `invert(processed: xarray.Dataset, mesh: Any, priors: Dict) -> Dict[str, Any]`: Returns {'model', 'uncertainty', 'converged'}.
    - `fuse(models: List[Dict], joint_weight: float) -> xarray.Dataset`: Bayesian joint inversion.
  - **Subclasses**:
    - `GravityInverter` (SimPEG).
    - `SeismicInverter` (eikonal tomography).
    - `FusionInverter`: Joint model.
    - `InSARInverter` (elastic models).

- `GravityInverter`
  - SimPEG-based gravity inversion for 3D density contrasts with terrain correction.
  - **Parameters**:
    - `base_density`: Reference density (kg/m³, default: 2000).
    - `g`: Gravitational constant (default: 6.67e-11).
  - **Methods**:
    - `invert(data: ProcessedGrid, **kwargs) -> InversionResults`: TreeMesh setup, prism terrain correction, L1/L2 reg, returns density model (kg/m³).
    - `fuse(models: List[InversionResults], **kwargs) -> np.ndarray`: Weighted average fusion.
  - **Returns**: InversionResults with model (3D xarray), uncertainty, metadata (converged, iterations, params).
  - **Example**:
    ```python
    from gam.modeling.gravity import GravityInverter
    inverter = GravityInverter(base_density=2200)
    results = inverter.invert(processed_grid, regularization={'alpha_s': 1e-4})
    ```
  - **Notes**: Adaptive meshing; units mGal input, kg/m³ output. Refs: SimPEG, Oldenburg (1996).

- `InSARInverter`
  - Elastic half-space inversion for deformation sources using Mogi/Okada models.
  - **Parameters**:
    - `poisson`: Poisson ratio (default: 0.25).
    - `modulus`: Young's modulus (Pa, default: 80e9).
  - **Methods**:
    - `invert(data: ProcessedGrid, **kwargs) -> InversionResults`: Atm correction, grid search + LSQ, Mogi/Okada forward, returns volume change (m³).
    - `fuse(models: List[InversionResults], **kwargs) -> np.ndarray`: Weighted sum.
  - **Returns**: InversionResults with 3D model (volume density), metadata (source_type, volume_change, pressure).
  - **Example**:
    ```python
    from gam.modeling.insar import InSARInverter
    inverter = InSARInverter(poisson=0.25)
    results = inverter.invert(los_grid, source_type='okada', inc=0.3)
    ```
  - **Notes**: LOS in m; PyCoulomb for Okada. Refs: Okada (1985), Chen & Zebker (2001).

- `AnomalyDetector`
  - **Methods**:
    - `detect(fused_model: xarray.Dataset, threshold: float) -> pd.DataFrame`: Z-score based.
  - **Example**:
    ```python
    from gam.modeling.anomaly_detection import AnomalyDetector
    detector = AnomalyDetector(threshold=2.0)
    anomalies = detector.detect(fused_model)
    ```

### Key Functions

- `invert_and_fuse(processed: Dict[str, xarray.Dataset], config: Dict) -> xarray.Dataset`
  - Full modeling pipeline.
  - **Returns**: Fused model.

- `AnomalyOutput` (pd.DataFrame schema): columns ['lat', 'lon', 'depth', 'confidence', 'anomaly_type', 'score'].

## Visualization Module (`gam.visualization`)

Generates maps and exports.

### Key Classes

- `Visualizer` (ABC)
  - **Methods**:
    - `render(anomalies: pd.DataFrame, output_type: str, config: Dict) -> Union[plt.Figure, folium.Map, str]`: Abstract.
  - **Subclasses**:
    - `Map2DVisualizer` (PyGMT/Matplotlib).
    - `Volume3DVisualizer` (PyVista).
    - `InteractiveVisualizer` (Folium).

### Key Functions

- `generate_visualization(anomalies: pd.DataFrame, type: str = "2d", export_formats: List[str] = None, config: Dict = None) -> Dict[str, str]`
  - Generates and saves visuals.
  - **Returns**: Dict[format, file_path].
  - **Example**:
    ```python
    from gam.visualization.manager import generate_visualization
    paths = generate_visualization(anomalies, "interactive", ["html", "png"])
    ```

## Utilities and Exceptions

- **Exceptions** (`gam.core.exceptions`): GAMError (base), DataFetchError, PreprocessingError, InversionConvergenceError, CacheError.
- **Utils** (`gam.core.utils`): bbox_to_string, hash_bbox, validate_modality.

## Usage Patterns

### End-to-End Script
```python
import yaml
from gam.core.pipeline import run_pipeline
from gam.visualization.manager import generate_visualization

config = yaml.safe_load(open('config.yaml'))
results = run_pipeline(**config['data'], config=config)
viz_paths = generate_visualization(results['anomalies'], config=config['visualization'])
```

### Plugin Extension
Subclasses ABCs and register via entry_points in setup.py.

### Error Handling
```python
from gam.core.exceptions import DataFetchError
try:
    raw = fetch_data(bbox, modalities)
except DataFetchError as e:
    print(f"Fetch failed: {e.source} - {e.message}")
```

For internal details, see source code. This API is stable for v1.0+; check changelog for changes.

## Visualization API: Tiles and Scenes

### Overview
The API serves Cesium 3D Tiles under `/tiles` and provides scene artifact endpoints used by the 3D Globe page. These endpoints support visualization of analysis results in CesiumJS-based globes. Relevant implementation files:

- [`main.py`](GeoAnomalyMapper/gam/api/main.py)
- [`artifacts.py`](GeoAnomalyMapper/gam/core/artifacts.py)
- [`globe_viewer.py`](GeoAnomalyMapper/gam/visualization/globe_viewer.py)

### Tiles Endpoints
The `/tiles` path is mounted as static files from `data/outputs/tilesets`, serving Cesium 3D Tilesets. Individual tileset metadata is available via dedicated endpoints.

- **Base mount**: `GET /tiles/<path:path>`  
  Serves static files (e.g., `.b3dm`, `.json`) from the tileset directory. Uses FastAPI's `StaticFiles` with automatic caching headers (e.g., `ETag`, `Last-Modified`).

- **Tileset metadata**: `GET /tilesets/{name}/tileset.json`  
  Returns the `tileset.json` content as JSON for the named tileset.  
  - **Parameters**: `name` (path param, str) – Tileset directory name.  
  - **Response**: JSON object (Cesium tileset spec).  
  - **Status**: 200 OK, 404 if tileset not found, 500 if invalid JSON.

- **Tileset listing**: `GET /tiles/tilesets.json`  
  Lists available tilesets.  
  - **Response**: `{"tilesets": [{"name": str, "url": "/tiles/{name}/tileset.json"}]}` (sorted by name).  
  - **Status**: 200 OK (empty list if no tilesets).

**Example URLs** (API server at `localhost:8000`):  
- Tileset JSON: `http://localhost:8000/tiles/myset/tileset.json`

**cURL example**:  
```
curl -I http://localhost:8000/tiles/myset/tileset.json
```

### Scene Artifact Endpoints
Scene configurations (JSON exported from the GlobeViewer) are stored per analysis and retrievable via the API. Saving is handled client-side (e.g., in the dashboard); no upload endpoint is exposed.

- **Read scene**: `GET /analysis/{analysis_id}/scene.json`  
  Returns the scene configuration JSON for the analysis.  
  - **Parameters**: `analysis_id` (path param, str) – Analysis identifier.  
  - **Response**: Raw JSON object (GlobeViewer scene config: layers, camera). Loaded from `data/outputs/state/{analysis_id}/scene.json`.  
  - **Status**: 200 OK, 404 if scene not found, 500 if invalid JSON.

**Example URL** (API server at `localhost:8000`):  
- Scene JSON: `http://localhost:8000/analysis/myanalysis/scene.json`

**cURL example**:  
```
curl -s http://localhost:8000/analysis/<analysis_id>/scene.json | jq .
```

### Response Schemas
Responses use raw JSON (no Pydantic models exposed).  

- **Tileset listing** (`/tiles/tilesets.json`):  
  ```json
  {
    "tilesets": [
      {
        "name": "myset",
        "url": "/tiles/myset/tileset.json"
      }
    ]
  }
  ```

- **Scene JSON** (`/analysis/{id}/scene.json`):  
  Arbitrary JSON from GlobeViewer export (e.g., `{"layers": [...], "camera": {...}}`). See [`globe_viewer.py`](GeoAnomalyMapper/gam/visualization/globe_viewer.py) for structure.

### Cross-Origin and Reverse Proxy Notes
In Docker deployment, NGINX reverse proxy routes requests as follows:  
- `/` → Dashboard (Streamlit on port 8501)  
- `/analysis/` → API (FastAPI on port 8000)  
- `/tiles/` → API (FastAPI on port 8000)  

See [`nginx.conf`](GeoAnomalyMapper/deployment/docker/nginx.conf) and [`docker-compose.yml`](GeoAnomalyMapper/deployment/docker/docker-compose.yml) for configuration. The proxy listens on port 8080.  

**Deployed example URLs** (proxy at `localhost:8080`):  
- Tileset: `http://localhost/tiles/myset/tileset.json`  
- Scene: `http://localhost/analysis/<analysis_id>/scene.json`

### Authentication/Config Notes
No authentication is required for these endpoints by default (open access).  

The 3D Globe visualization requires a Cesium Ion access token configured in the dashboard environment (unrelated to API calls; used client-side for terrain/imagery). See:  
- [`installation.md`](GeoAnomalyMapper/docs/user/installation.md)  
- [`quickstart.md`](GeoAnomalyMapper/docs/user/quickstart.md)  
- [`globe_viewer.md`](GeoAnomalyMapper/docs/user/globe_viewer.md)

### Error Cases and Troubleshooting Pointers
- **404 on `/tiles/...`**: Verify the tileset directory exists under `data/outputs/tilesets` and contains valid files. Check mount in [`main.py`](GeoAnomalyMapper/gam/api/main.py).  
- **404 on `/analysis/{id}/scene.json`**: Confirm analysis artifacts are generated and the scene.json file exists at `data/outputs/state/{id}/scene.json`. See [`artifacts.py`](GeoAnomalyMapper/gam/core/artifacts.py) for path logic.  
- **Reverse proxy path mismatch**: Ensure NGINX routes are correctly configured for `/tiles` and `/analysis`. Inspect logs and verify upstreams in [`nginx.conf`](GeoAnomalyMapper/deployment/docker/nginx.conf).  

For more issues, see [`common_issues.md`](GeoAnomalyMapper/docs/troubleshooting/common_issues.md).

## Tiles Utilities: ECEF Reprojection and py3dtiles Wrapper

These utilities in `gam.visualization.tiles_builder` support 3D tiles generation for Cesium visualization.

### reproject_wgs84_to_ecef(lon: float | np.ndarray, lat: float | np.ndarray, height: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

- **Purpose**: Reprojects coordinates from EPSG:4979 (WGS84 3D: longitude, latitude, ellipsoidal height in meters) to EPSG:4978 (ECEF: Earth-centered, Earth-fixed Cartesian coordinates in meters).
- **Semantics**: Explicit longitude-first ordering (lon, lat, height); uses `pyproj.Transformer(always_xy=True)` for vectorization support. Inputs can be scalars or array-like (broadcast to common shape).
- **Dependencies**: Requires `pyproj` at runtime (included in core requirements).

**Minimal Example**:
```python
from GeoAnomalyMapper.gam.visualization.tiles_builder import reproject_wgs84_to_ecef
x, y, z = reproject_wgs84_to_ecef(0.0, 0.0, 0.0)  # x ≈ 6378137 m, y ≈ 0, z ≈ 0
import numpy as np
x2, y2, z2 = reproject_wgs84_to_ecef([0, 10], [0, 0], [0, 0])
assert x2.shape == y2.shape == z2.shape == (2,)
```

**Implementation**: [`reproject_wgs84_to_ecef()`](GeoAnomalyMapper/gam/visualization/tiles_builder.py:160)

### run_py3dtiles_convert(input_path: str | Path, out_dir: str | Path, threads: int = 1, timeout: int = 300, allow_overwrite: bool = False, extra_args: List[str] = None) -> Dict[str, Any]

- **Purpose**: Safe subprocess wrapper for the `py3dtiles convert` CLI, with guardrails: checks tool availability (via PATH), validates input/output paths, enforces timeout, whitelists flags (only `--overwrite` allowed beyond basics), and structures results.
- **Parameters**:
  - `input_path`: Path to input file (e.g., `points.xyz` with space-delimited X Y Z).
  - `out_dir`: Output directory for tileset (created if needed).
  - `threads`: Number of threads for conversion (passed as `--threads`).
  - `timeout`: Seconds before subprocess timeout.
  - `allow_overwrite`: If True, allows `--overwrite` flag.
  - `extra_args`: Limited list; only `--overwrite` is whitelisted (others raise ValueError).
- **Behavior**: Composes command (e.g., `py3dtiles convert input_path --out out_dir --threads N`), runs via `subprocess.run`, captures stdout/stderr, raises on failure.
- **Returns**: Dict with keys: `command` (list[str]), `returncode` (int), `stdout_tail` (str, last 500 chars), `out_dir` (str).
- **Exceptions**:
  - `FileNotFoundError`: If `py3dtiles` not in PATH or input_path missing.
  - `TimeoutError`: If subprocess exceeds timeout.
  - `RuntimeError`: If returncode != 0, with stdout/stderr details.

**Minimal Example**:
```python
from GeoAnomalyMapper.gam.visualization.tiles_builder import run_py3dtiles_convert
result = run_py3dtiles_convert("points.xyz", "tiles_out", threads=4, timeout=120, allow_overwrite=True)
assert result["returncode"] == 0
```

**Implementation**: [`run_py3dtiles_convert()`](GeoAnomalyMapper/gam/visualization/tiles_builder.py:241)

### Pitfalls/Caveats

- **Lon/Lat Order**: `reproject_wgs84_to_ecef` requires (lon, lat, height) due to `always_xy=True`; lat/lon swap will produce incorrect ECEF.
- **Height Semantics**: Input height is ellipsoidal (WGS84); orthometric/geoid heights differ by ~30-100m—use consistent datum.
- **py3dtiles Wrapper Restrictions**: Only safe flags allowed (e.g., no `--attributes` or custom); extend via direct CLI if needed, but wrapper prevents injection.
- **Runtime Dependencies**: `pyproj` for reprojection (ImportError if missing); `py3dtiles` CLI for conversion (FileNotFoundError if not installed/accessible).

---

*Generated: 2025-09-23 | For auto-generation, use Sphinx apidoc.