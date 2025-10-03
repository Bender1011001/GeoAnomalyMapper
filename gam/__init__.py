"""
GeoAnomalyMapper (GAM): Open-source tool for geophysical data fusion and subsurface anomaly detection.

GAM integrates gravity, magnetic, seismic, and InSAR datasets through a modular pipeline:
1. Ingestion: Fetch and cache raw data from public sources (USGS, IRIS, ESA).
2. Preprocessing: Clean, filter, grid, and standardize to common coordinates.
3. Modeling: Perform geophysical inversions and multi-modal fusion.
4. Visualization: Generate maps, 3D renders, and export anomalies.

Unified API
-----------
High-level: Use GAMPipeline for end-to-end workflows.
>>> from gam import GAMPipeline
>>> pipeline = GAMPipeline(config_path='config.yaml')
>>> results = pipeline.run_analysis(
...     bbox=(29.0, 31.0, 30.0, 32.0),
...     modalities=['gravity', 'magnetic'],
...     output_dir='./results',
...     use_cache=True
... )
>>> print(f"Detected {len(results['anomalies'])} anomalies")

Low-level: Access individual managers for custom workflows.
>>> from gam import IngestionManager, PreprocessingManager
>>> ingestion = IngestionManager()
>>> raw_data = ingestion.fetch_multiple(['gravity'], bbox=(29.0, 31.0, 30.0, 32.0))
>>> preprocessing = PreprocessingManager()
>>> processed = preprocessing.process(raw_data)

Key Data Structures
-------------------
- RawData: From ingestion (dataclass with metadata and np.ndarray/xarray.Dataset values).
- ProcessedGrid: From preprocessing (xarray.Dataset with lat/lon/depth dimensions).
- InversionResults: From modeling (dict with model, uncertainty).
- AnomalyOutput: Final anomalies (pd.DataFrame with lat, lon, depth, confidence, anomaly_type).

Configuration
-------------
- config.yaml: Pipeline settings (modalities, resolutions, fusion params).
- data_sources.yaml: API endpoints and credentials (use env vars for keys).
- Logging: Configured at package level; override in config.yaml.

Installation & Usage
--------------------
pip install gam
gam run --bbox 29 31 30 32 --modalities gravity magnetic --output results/

For global processing: gam global --region global --tiles 10 --workers 4

Dependencies: NumPy, SciPy, xarray, Dask, PyGMT, PyVista, SimPEG, obspy, h5py, click.

See docs/architecture.md for detailed system design and module interfaces.
"""

import logging
import sys

# Package metadata
__version__ = "0.1.0"
__author__ = "GeoAnomalyMapper Team"
__license__ = "MIT"
__description__ = "Geophysical data fusion for subsurface anomaly detection."

# Package-level logging configuration
# Set root logger for 'gam' to INFO with console handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('gam')
log.info(f"GAM {__version__} initialized")

# Lazy export strategy to keep import-time light (no heavy geospatial deps required for CLI --help)
from typing import TYPE_CHECKING
import importlib

__all__ = [
    # High-level
    'GAMPipeline',
    # Data structures
    'RawData', 'ProcessedGrid', 'InversionResults', 'AnomalyOutput',
    # Managers
    'IngestionManager', 'PreprocessingManager', 'ModelingManager', 'VisualizationManager',
    # Core
    'GAMConfig', 'GAMError', 'PipelineError', 'validate_bbox'
]

# Map attribute names to (module_path, symbol_name) for lazy resolution
_lazy_exports = {
    # High-level
    'GAMPipeline': ('gam.core.pipeline', 'GAMPipeline'),

    # Ingestion
    'IngestionManager': ('gam.ingestion.manager', 'IngestionManager'),
    'RawData': ('gam.ingestion.data_structures', 'RawData'),
    'HDF5CacheManager': ('gam.ingestion.cache_manager', 'HDF5CacheManager'),
    'GravityFetcher': ('gam.ingestion.fetchers', 'USGSGravityFetcher'),
    'SeismicFetcher': ('gam.ingestion.fetchers', 'SeismicFetcher'),
    'MagneticFetcher': ('gam.ingestion.fetchers', 'USGSMagneticFetcher'),
    'InSARFetcher': ('gam.ingestion.fetchers', 'ESAInSARFetcher'),
    'DataSource': ('gam.ingestion.base', 'DataSource'),

    # Preprocessing
    'PreprocessingManager': ('gam.preprocessing.manager', 'PreprocessingManager'),
    'ProcessedGrid': ('gam.preprocessing.data_structures', 'ProcessedGrid'),
    'Preprocessor': ('gam.preprocessing.base', 'Preprocessor'),
    'GravityPreprocessor': ('gam.preprocessing.processors', 'GravityPreprocessor'),
    'MagneticPreprocessor': ('gam.preprocessing.processors', 'MagneticPreprocessor'),
    'SeismicPreprocessor': ('gam.preprocessing.processors', 'SeismicPreprocessor'),
    'InSARPreprocessor': ('gam.preprocessing.processors', 'InSARPreprocessor'),
    'DaskPreprocessor': ('gam.preprocessing.parallel', 'DaskPreprocessor'),

    # Modeling (optional; may raise ImportError when accessed if extras missing)
    'ModelingManager': ('gam.modeling.manager', 'ModelingManager'),
    'InversionResults': ('gam.modeling.data_structures', 'InversionResults'),
    'AnomalyOutput': ('gam.modeling.data_structures', 'AnomalyOutput'),
    'Inverter': ('gam.modeling.base', 'Inverter'),
    'GravityInverter': ('gam.modeling', 'GravityInverter'),
    'MagneticInverter': ('gam.modeling', 'MagneticInverter'),
    'SeismicInverter': ('gam.modeling', 'SeismicInverter'),
    'InSARInverter': ('gam.modeling', 'InSARInverter'),
    'JointInverter': ('gam.modeling', 'JointInverter'),
    'AnomalyDetector': ('gam.modeling.anomaly_detection', 'AnomalyDetector'),
    'MeshGenerator': ('gam.modeling.mesh', 'MeshGenerator'),

    # Visualization (optional)
    'VisualizationManager': ('gam.visualization.manager', 'VisualizationManager'),

    # Core utilities
    'GAMConfig': ('gam.core.config', 'GAMConfig'),
    'GAMError': ('gam.core.exceptions', 'GAMError'),
    'PipelineError': ('gam.core.exceptions', 'PipelineError'),
    'validate_bbox': ('gam.core.utils', 'validate_bbox'),
}

if TYPE_CHECKING:
    # Optionally expose types to static analyzers without importing at runtime
    pass


def __getattr__(name: str):
    """Lazily import attributes on first access to avoid heavy deps at import-time."""
    target = _lazy_exports.get(name)
    if target is None:
        raise AttributeError(f"module 'gam' has no attribute {name!r}")
    module_path, symbol = target
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(
            f"Failed to import '{name}' from '{module_path}'. "
            f"Install required optional dependencies to use this feature."
        ) from e
    try:
        return getattr(module, symbol)
    except AttributeError as e:
        raise AttributeError(
            f"Attribute '{symbol}' not found in module '{module_path}'."
        ) from e


def __dir__():
    return sorted(list(globals().keys()) + __all__)
