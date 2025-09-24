
"""Main ingestion manager coordinating fetchers and caching for GAM."""

import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cache_manager import HDF5CacheManager
from .fetchers import GravityFetcher, SeismicFetcher, MagneticFetcher, InSARFetcher
from .data_structures import RawData
from .exceptions import DataFetchError
from .base import DataSource


logger = logging.getLogger(__name__)


class IngestionManager:
    r"""
    Coordinates data fetchers and caching for multiple geophysical modalities.

    This class provides a unified interface for fetching data from various sources,
    integrating caching to avoid redundant API calls, and supporting parallel
    fetching for efficiency. It also loads and validates configuration from
    data_sources.yaml, supporting environment variable overrides for API keys.

    Parameters
    ----------
    cache_dir : str, default 'data/cache'
        Directory for HDF5 cache files.
    fetchers : Dict[str, DataSource], optional
        Custom fetchers by modality. Defaults to built-in.

    Attributes
    ----------
    cache : HDF5CacheManager
        Cache instance.
    fetchers : Dict[str, DataSource]
        Available fetchers by modality name.
    config : Dict[str, Any]
        Loaded configuration from data_sources.yaml with env overrides.

    Methods
    -------
    fetch_modality(modality, bbox, **kwargs)
        Fetch single modality data with caching.
    fetch_multiple(modalities, bbox, max_workers, **kwargs)
        Fetch multiple modalities in parallel.
    clear_cache(expire_days)
        Clear cache via manager.
    _load_config()
        Load and validate configuration (internal).

    Notes
    -----
    Modalities: 'gravity', 'seismic', 'magnetic', 'insar'.
    Caching uses key from metadata hash. Parallel uses ThreadPoolExecutor (max_workers=4).
    Errors in parallel fetches are logged and result set to None.
    Configuration: data_sources.yaml with sections per modality (e.g., url, api_key).
    Validation: Required keys per modality; api_key from env if specified.

    Examples
    --------
    >>> manager = IngestionManager()
    >>> gravity_data = manager.fetch_modality('gravity', (29.0, 31.0, 30.0, 32.0))
    >>> print(f"Fetched {len(gravity_data.values)} gravity measurements")
    >>> results = manager.fetch_multiple(['gravity', 'magnetic'], (29.0, 31.0, 30.0, 32.0))
    """

    def __init__(self, cache_dir: str = 'data/cache', fetchers: Optional[Dict[str, DataSource]] = None):
        self.cache = HDF5CacheManager(cache_dir)
        self.fetchers = fetchers or {
            'gravity': GravityFetcher(),
            'seismic': SeismicFetcher(),
            'magnetic': MagneticFetcher(),
            'insar': InSARFetcher()
        }
        self.config = self._load_config()
        logger.info("IngestionManager initialized with cache and fetchers")

    def _load_config(self) -> Dict[str, Any]:
        r"""
        Load and validate configuration from data_sources.yaml.

        Supports environment variables for API keys (e.g., ESA_API_KEY).
        Validates required keys per modality.

        Returns
        -------
        Dict[str, Any]
            Loaded config.

        Raises
        ------
        ValueError
            If config file missing or invalid (e.g., missing required keys).
        FileNotFoundError
            If data_sources.yaml not found.

        Notes
        -----
        Expected structure:
        gravity:
          url: https://mrdata.usgs.gov/services/gravity
        insar:
          api_key: ESA_API_KEY  # Will load from env
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed; config loading disabled. Install with: pip install PyYAML")
            return {}
        config_path = 'data_sources.yaml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found. Create it with API endpoints and keys.")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        # Override with env vars
        for modality, mod_config in config.items():
            if isinstance(mod_config, dict) and 'api_key' in mod_config:
                env_key = mod_config['api_key'].upper()
                mod_config['api_key'] = os.getenv(env_key, mod_config.get('default_key', ''))
                if not mod_config['api_key']:
                    logger.warning(f"API key for {modality} ({env_key}) not set in environment")
            # Basic validation
            required = ['url'] if modality != 'insar' else ['api_key']
            missing = [k for k in required if k not in mod_config]
            if missing:
                raise ValueError(f"Missing required config for {modality}: {missing}")
        logger.info("Configuration loaded and validated")
        return config

    def fetch_modality(self, modality: str, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Fetch data for a single modality with caching.

        Checks cache first; fetches and caches if miss.

        Parameters
        ----------
        modality : str
            Modality name ('gravity', 'seismic', etc.).
        bbox : Tuple[float, float, float, float]
            Bounding box (min_lat, max_lat, min_lon, max_lon).
        **kwargs : dict
            Parameters passed to fetcher (e.g., starttime for seismic).

        Returns
        -------
        RawData
            Fetched or cached data.

        Raises
        ------
        ValueError
            If unknown modality.
        DataFetchError
            From fetcher (wrapped).

        Notes
        -----
        Cache key based on metadata hash. Uses config if fetcher supports.

        Examples
        --------
        >>> data = manager.fetch_modality('gravity', (29.0, 31.0, 30.0, 32.0))
        """
        if modality not in self.fetchers:
            raise ValueError(f"Unknown modality: {modality}. Available: {list(self.fetchers.keys())}")
        fetcher = self.fetchers[modality]
        metadata = {
            'source': modality,
            'timestamp': datetime.now(),
            'bbox': bbox,
            'parameters': kwargs
        }
        temp_data = RawData(metadata, None)  # For key generation
        key = self.cache._generate_key(temp_data)
        if self.cache.exists(key):
            logger.info(f"Cache hit for {modality}: {key}")
            return self.cache.load_data(key)
        logger.info(f"Cache miss for {modality}; fetching...")
        try:
            data = fetcher.fetch_data(bbox, **kwargs)
            # Ensure metadata source matches
            data.metadata['source'] = modality
            self.cache.save_data(data)
            return data
        except DataFetchError as e:
            logger.error(f"Fetch failed for {modality}: {e}")
            raise

    def fetch_multiple(self, modalities: List[str], bbox: Tuple[float, float, float, float], max_workers: int = 4, **kwargs) -> Dict[str, RawData]:
        r"""
        Fetch multiple modalities in parallel with caching.

        Uses ThreadPoolExecutor for concurrent fetches. Continues on individual failures.

        Parameters
        ----------
        modalities : List[str]
            List of modalities to fetch.
        bbox : Tuple[float, float, float, float]
            Shared bounding box.
        max_workers : int, default 4
            Max concurrent threads.
        **kwargs : dict
            Passed to all fetchers.

        Returns
        -------
        Dict[str, RawData]
            Results by modality; None on failure.

        Notes
        -----
        Each fetch uses individual caching. Thread-safe via cache lock.

        Examples
        --------
        >>> results = manager.fetch_multiple(['gravity', 'magnetic'], (29.0, 31.0, 30.0, 32.0))
        """
        results = {mod: None for mod in modalities}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mod = {
                executor.submit(self.fetch_modality, mod, bbox, **kwargs): mod
                for mod in modalities
            }
            for future in as_completed(future_to_mod):
                mod = future_to_mod[future]
                try:
                    results[mod] = future.result()
                except Exception as e:
                    logger.error(f"Parallel fetch failed for {mod}: {e}")
                    results[mod] = None
        logger.info(f"Parallel fetch completed for {len(modalities)} modalities")
        return results

    def clear_cache(self, expire_days: Optional[int] = None) -> int:
        r"""
        Clear cache entries.

        Parameters
        ----------
        expire_days : int, optional
            Delete entries older than this; None clears all.

        Returns
        -------
        int
            Number deleted.

        Examples
        --------
        >>> deleted = manager.clear_cache(7)
        """
        return self.cache.clear_cache(expire_days)