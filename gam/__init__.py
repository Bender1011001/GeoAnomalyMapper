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

# High-level API: GAMPipeline from core
from .core.pipeline import GAMPipeline

# Low-level APIs: Managers from each module
from .ingestion import (
    IngestionManager, RawData, HDF5CacheManager,
    GravityFetcher, SeismicFetcher, MagneticFetcher, InSARFetcher,
    DataSource
)
from .preprocessing import (
    PreprocessingManager, ProcessedGrid, Preprocessor,
    GravityPreprocessor, MagneticPreprocessor, SeismicPreprocessor, InSARPreprocessor,
    DaskPreprocessor
)
from .modeling import (
    ModelingManager, InversionResults, AnomalyOutput, Inverter,
    GravityInverter, MagneticInverter, SeismicInverter, InSARInverter,
    JointInverter, AnomalyDetector, MeshGenerator
)
try:
    from .visualization import VisualizationManager
except Exception as e:
    if "GMT" in str(e):
        VisualizationManager = None
        log.warning(f"VisualizationManager not available due to missing GMT/PyGMT: {str(e)}; visualization features disabled.")
    else:
        raise

# Core utilities (config, exceptions, etc.)
from .core.config import GAMConfig
from .core.exceptions import GAMError, PipelineError
from .core.utils import validate_bbox

# CLI entry (if using Click/Argparse in core.cli)
from .core.cli import cli  # For direct invocation if needed

__all__ = [
    # High-level
    'GAMPipeline',
    # Data structures
    'RawData', 'ProcessedGrid', 'InversionResults', 'AnomalyOutput',
    # Managers
    'IngestionManager', 'PreprocessingManager', 'ModelingManager', 'VisualizationManager',
    # Core
    'GAMConfig', 'GAMError', 'PipelineError', 'validate_bbox', 'cli'
]