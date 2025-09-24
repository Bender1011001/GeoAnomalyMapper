r"""
Data Ingestion Module for GeoAnomalyMapper (GAM).

This module handles the automated fetching, caching, and standardization of
geophysical datasets from public sources including USGS (gravity, magnetic),
IRIS (seismic), and ESA (InSAR/Sentinel-1). It provides a plugin-based architecture
for extensibility, robust error handling with retries, rate limiting, and
HDF5-based caching for performance and resumability.

Key Components
--------------
- IngestionManager: Central coordinator for fetching modalities with caching and parallel support.
- RawData: Dataclass for standardized raw data representation with validation.
- Fetchers: Modality-specific classes (GravityFetcher, SeismicFetcher, etc.) implementing DataSource ABC.
- HDF5CacheManager: Persistent storage for RawData with versioning and expiration.
- Exceptions: Custom errors (DataFetchError, CacheError, APITimeoutError) and decorators (@retry_fetch, @rate_limit).

Dependencies
------------
- requests: For HTTP API calls (USGS, ESA).
- obspy: For seismic data handling (IRIS FDSN).
- h5py: For HDF5 caching.
- tenacity: For retry logic with exponential backoff.
- PyYAML: For configuration loading from data_sources.yaml.
- numpy: For array-based data (gravity/magnetic values).
- concurrent.futures: For parallel fetching.

Configuration
-------------
- data_sources.yaml: API endpoints and auth (e.g., ESA_API_KEY via env).
- Environment variables: For sensitive keys (e.g., export ESA_API_KEY='your_key').

Usage
-----
>>> from gam.ingestion import IngestionManager
>>> manager = IngestionManager()
>>> gravity_data = manager.fetch_modality('gravity', bbox=(29.0, 31.0, 30.0, 32.0))
>>> print(f"Fetched {len(gravity_data.values)} gravity measurements")
>>> results = manager.fetch_multiple(['gravity', 'magnetic'], (29.0, 31.0, 30.0, 32.0))

Notes
-----
Follows architecture in docs/architecture.md. All operations are thread-safe.
For production, configure logging centrally in config.yaml (level: INFO).
Extensibility: Subclass DataSource and register via entry_points.
"""

from .manager import IngestionManager
from .data_structures import RawData
from .cache_manager import HDF5CacheManager
from .fetchers import (
    GravityFetcher, SeismicFetcher, MagneticFetcher, InSARFetcher
)
from .base import DataSource
from .exceptions import (
    DataFetchError, CacheError, APITimeoutError
)

from .fetchers import retry_fetch

__all__ = [
    'IngestionManager', 'RawData', 'HDF5CacheManager',
    'GravityFetcher', 'SeismicFetcher', 'MagneticFetcher', 'InSARFetcher',
    'DataSource', 'DataFetchError', 'CacheError', 'APITimeoutError'
]

# Logging configuration for the module
import logging
log = logging.getLogger('gam.ingestion')
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)