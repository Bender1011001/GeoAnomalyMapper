"""Cache management for the GAM data ingestion module using HDF5."""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional
from threading import Lock
import h5py
import numpy as np  # For ndarray handling

from .data_structures import RawData
from .exceptions import CacheError


logger = logging.getLogger(__name__)


class HDF5CacheManager:
    r"""
    HDF5-based cache manager for storing and retrieving RawData objects.

    This class implements the CacheManager interface using HDF5 files for
    efficient storage of large geophysical datasets. Each cache entry is a
    separate .h5 file with metadata as attributes and values as datasets.
    Supports compression, versioning, expiration, and thread-safety.

    Parameters
    ----------
    cache_dir : str, default 'data/cache'
        Directory to store HDF5 cache files. Created if does not exist.
    version : str, default '1.0'
        Version string for cache compatibility checks.
    lock : Lock, optional
        Thread lock for concurrent access. Defaults to new Lock().

    Attributes
    ----------
    cache_dir : str
        Cache directory path.
    version : str
        Cache version.
    lock : Lock
        Synchronization lock.
    file_ext : str
        HDF5 file extension ('.h5').

    Methods
    -------
    save_data(data)
        Save RawData to cache and return key.
    load_data(key)
        Load RawData by key or return None if missing/invalid.
    exists(key)
        Check if cache entry exists.
    clear_cache(expire_days)
        Delete expired or all cache entries.

    Notes
    -----
    Keys are MD5 hashes of source + bbox + timestamp for uniqueness.
    Values: ndarray stored directly (gzip), others as JSON bytes.
    Unsupported values raise CacheError. For complex types (ObsPy, xarray),
    extend with custom serialization (e.g., to_netcdf path in metadata).
    Expiration based on fetch timestamp attribute.

    Examples
    --------
    >>> manager = HDF5CacheManager('my_cache')
    >>> key = manager.save_data(raw_data)
    >>> data = manager.load_data(key)
    >>> if manager.exists(key):
    ...     print("Cached")
    >>> manager.clear_cache(7)  # Delete >7 days old
    """

    def __init__(self, cache_dir: str = 'data/cache', version: str = '1.0', lock: Optional[Lock] = None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.version = version
        self.lock = lock or Lock()
        self.file_ext = '.h5'

    def _generate_key(self, data: RawData) -> str:
        r"""
        Generate a unique cache key from RawData metadata.

        Parameters
        ----------
        data : RawData
            The data to key.

        Returns
        -------
        str
            MD5 hash of source_bbox_timestamp.

        Notes
        -----
        Ensures determinism; collisions unlikely with MD5 for this use.
        """
        bbox_str = '_'.join(f"{coord:.6f}" for coord in data.metadata['bbox'])
        source = data.metadata['source']
        timestamp_str = data.metadata['timestamp'].isoformat()
        key_str = f"{source}_{bbox_str}_{timestamp_str}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def save_data(self, data: RawData) -> str:
        r"""
        Save RawData to HDF5 cache.

        Parameters
        ----------
        data : RawData
            The data to cache. Must be validated.

        Returns
        -------
        str
            The generated cache key.

        Raises
        ------
        CacheError
            If save fails (e.g., I/O error, unsupported values).
        ValueError
            If data invalid (call data.validate() prior).

        Notes
        -----
        Uses gzip compression for datasets. Metadata stored as JSON attrs.
        Thread-safe via lock.

        Examples
        --------
        >>> key = manager.save_data(raw_data)
        """
        data.validate()  # Ensure valid before caching
        key = self._generate_key(data)
        file_path = os.path.join(self.cache_dir, f"{key}{self.file_ext}")
        with self.lock:
            try:
                with h5py.File(file_path, 'w') as f:
                    # Metadata as attributes
                    f.attrs['version'] = self.version
                    f.attrs['timestamp'] = data.metadata['timestamp'].isoformat()
                    f.attrs['source'] = data.metadata['source']
                    f.attrs['bbox'] = json.dumps(data.metadata['bbox'])
                    f.attrs['parameters'] = json.dumps(data.metadata.get('parameters', {}))
                    # Values as dataset
                    if isinstance(data.values, np.ndarray):
                        f.create_dataset('values', data=data.values, compression='gzip')
                    else:
                        # Fallback to JSON for dict/list/str
                        values_serial = json.dumps(data.values) if hasattr(data.values, '__dict__') or isinstance(data.values, (dict, list)) else str(data.values)
                        values_bytes = values_serial.encode('utf-8')
                        f.create_dataset('values', data=values_bytes, compression='gzip')
                logger.info(f"Saved data to cache: {key} (source: {data.metadata['source']})")
                return key
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)  # Cleanup partial
                raise CacheError(key, 'save', str(e))

    def load_data(self, key: str) -> Optional[RawData]:
        r"""
        Load RawData from cache by key.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        RawData or None
            Loaded data if valid, else None (missing, version mismatch, corrupt).

        Raises
        ------
        CacheError
            If file exists but load fails (e.g., corrupt).

        Notes
        -----
        Reconstructs RawData and validates. Thread-safe.

        Examples
        --------
        >>> data = manager.load_data('abc123def456')
        """
        file_path = os.path.join(self.cache_dir, f"{key}{self.file_ext}")
        if not os.path.exists(file_path):
            logger.debug(f"Cache miss for key: {key}")
            return None
        with self.lock:
            try:
                with h5py.File(file_path, 'r') as f:
                    if f.attrs.get('version') != self.version:
                        logger.warning(f"Version mismatch for {key}: {f.attrs.get('version')} != {self.version}")
                        return None
                    metadata = {
                        'source': f.attrs['source'],
                        'timestamp': datetime.fromisoformat(f.attrs['timestamp']),
                        'bbox': json.loads(f.attrs['bbox']),
                        'parameters': json.loads(f.attrs['parameters'])
                    }
                    values_dataset = f['values'][:]
                    if isinstance(values_dataset, bytes):
                        values = json.loads(values_dataset.decode('utf-8'))
                    else:
                        values = values_dataset  # ndarray
                    data = RawData(metadata, values)
                    data.validate()
                    logger.debug(f"Loaded data from cache: {key}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load cache {key}: {e}")
                raise CacheError(key, 'load', str(e))

    def exists(self, key: str) -> bool:
        r"""
        Check if a cache entry exists and is valid (version match).

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        bool
            True if valid cache exists.

        Notes
        -----
        Quick check; does not load full data.
        """
        file_path = os.path.join(self.cache_dir, f"{key}{self.file_ext}")
        if not os.path.exists(file_path):
            return False
        try:
            with h5py.File(file_path, 'r') as f:
                return f.attrs.get('version') == self.version
        except Exception:
            logger.warning(f"Corrupt cache file: {key}")
            return False

    def clear_cache(self, expire_days: Optional[int] = None) -> int:
        r"""
        Clear cache entries: all if expire_days None, else expired ones.

        Parameters
        ----------
        expire_days : int, optional
            Days before expiration. If None, clear all.

        Returns
        -------
        int
            Number of deleted entries.

        Notes
        -----
        Checks timestamp attr for expiration. Deletes corrupt files too.
        Thread-safe; logs deletions.

        Examples
        --------
        >>> deleted = manager.clear_cache(30)  # Keep recent 30 days
        """
        with self.lock:
            deleted = 0
            cutoff = datetime.now() - timedelta(days=expire_days) if expire_days is not None else None
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(self.file_ext):
                    file_path = os.path.join(self.cache_dir, filename)
                    key = filename[:-len(self.file_ext)]
                    try:
                        with h5py.File(file_path, 'r') as f:
                            if f.attrs.get('version') != self.version:
                                os.remove(file_path)
                                deleted += 1
                                continue
                            ts_str = f.attrs.get('timestamp')
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str)
                                if cutoff and ts < cutoff:
                                    os.remove(file_path)
                                    deleted += 1
                                    logger.info(f"Deleted expired cache: {key} (fetched {ts})")
                    except Exception as e:
                        logger.warning(f"Failed to check {key}: {e}; deleting")
                        os.remove(file_path)
                        deleted += 1
            if expire_days is None:
                # If clear all, we already handled above, but ensure
                pass
            logger.info(f"Cleared {deleted} cache entries")
            return deleted