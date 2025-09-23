"""Preprocessing Module.

Cleans, aligns, filters, and grids heterogeneous geophysical data to a common resolution.
Includes resampling to WGS84 grids, noise reduction (e.g., Gaussian filters), unit conversions,
and parallel processing with Dask arrays. Supports ObsPy for seismic, MintPy for InSAR interferograms.

Main classes:
- PreprocessingManager: Orchestrates processing pipelines
- ProcessedGrid: Standardized output data structure (xarray.Dataset)
- Processors: GravityPreprocessor, MagneticPreprocessor, SeismicPreprocessor, InSARPreprocessor
- DaskPreprocessor: Parallel wrapper for large datasets

Example usage:
>>> from gam.preprocessing import PreprocessingManager
>>> manager = PreprocessingManager()
>>> processed = manager.preprocess_data(raw_data, modality='gravity')
"""

import logging
from .base import Preprocessor
from .data_structures import ProcessedGrid
from .manager import PreprocessingManager
from .parallel import DaskPreprocessor
from .processors import (
    GravityPreprocessor,
    MagneticPreprocessor,
    SeismicPreprocessor,
    InSARPreprocessor
)

# Set up module logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

__all__ = [
    'Preprocessor',
    'ProcessedGrid',
    'PreprocessingManager',
    'DaskPreprocessor',
    'GravityPreprocessor',
    'MagneticPreprocessor',
    'SeismicPreprocessor',
    'InSARPreprocessor'
]