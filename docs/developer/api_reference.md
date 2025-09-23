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

---

*Generated: 2025-09-23 | For auto-generation, use Sphinx apidoc.*