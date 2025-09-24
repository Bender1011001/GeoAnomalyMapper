"""
Pytest configuration and fixtures for GeoAnomalyMapper (GAM) integration tests.

This file provides shared fixtures for synthetic data, mocking external dependencies,
temporary directories, configurations, and caching/database setup. Ensures tests are
isolated, deterministic, and fast by avoiding real API calls and using in-memory/tmp storage.

Usage in tests:
- Use synthetic_raw_data to get mock RawData for modalities.
- monkeypatch via mock_external_apis for API simulation.
- test_config for pipeline params.
- tmp_output_dir for verifying outputs.
- cache_manager for testing caching flows.

Markers:
- @pytest.mark.integration: For end-to-end tests.
- @pytest.mark.skipif(not has_obspy, reason="Requires obspy") for seismic-specific.

Dependencies: pytest, numpy, xarray, pandas, h5py, sqlite3, unittest.mock.
Optional: obspy (for seismic mocks), requests (for mocking).
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from unittest.mock import MagicMock, patch

# Set numpy seed for deterministic synthetic data
np.random.seed(42)

# Optional dependency checks
has_obspy = False
try:
    import obspy
    has_obspy = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_bbox():
    """Standard test bounding box (small region for quick processing)."""
    return (29.0, 31.0, 30.0, 32.0)  # lon_min, lon_max, lat_min, lat_max

@pytest.fixture(scope="function")
def synthetic_raw_data(request):
    from gam.ingestion.data_structures import RawData
    """
    Parametrized fixture for synthetic RawData across modalities.
    
    Yields RawData dataclass with:
    - values: np.ndarray or xarray.Dataset (2D grid for simplicity)
    - metadata: Dict with bbox, units, source, timestamp
    
    Modalities: 'gravity' (mGal), 'magnetic' (nT), 'seismic' (velocity km/s), 'insar' (displacement mm).
    """
    modality = request.param if hasattr(request, 'param') else 'gravity'
    lat = np.linspace(30.0, 32.0, 10)
    lon = np.linspace(29.0, 31.0, 10)
    coords = {'lat': lat, 'lon': lon}
    
    if modality == 'gravity':
        # Synthetic gravity anomalies around 9.8 mGal with Gaussian noise
        values = 9.8 + np.random.normal(0, 0.1, (10, 10))
        units = 'mGal'
        source = 'USGS'
    elif modality == 'magnetic':
        # Synthetic magnetic field around 50 nT
        values = 50 + np.random.normal(0, 5, (10, 10))
        units = 'nT'
        source = 'USGS'
    elif modality == 'seismic':
        if not has_obspy:
            pytest.skip("Seismic tests require obspy")
        # Synthetic P-wave velocity
        values = 6.0 + np.random.normal(0, 0.2, (10, 10))  # km/s
        units = 'km/s'
        source = 'IRIS'
    elif modality == 'insar':
        # Synthetic displacement
        values = np.random.normal(0, 1, (10, 10))  # mm
        units = 'mm'
        source = 'ESA'
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    # Create xarray Dataset for grid data
    da = xr.DataArray(values, coords=coords, dims=['lat', 'lon'])
    dataset = da.to_dataset(name='data')
    
    metadata = {
        'bbox': (29.0, 31.0, 30.0, 32.0),
        'units': units,
        'source': source,
        'timestamp': pd.Timestamp.now().isoformat(),
        'resolution': 0.2  # degrees
    }
    
    return RawData(values=dataset, metadata=metadata)

@pytest.fixture(scope="function")
def synthetic_raw_data_multi():
    from gam.ingestion.data_structures import RawData
    """Multi-modality synthetic data for fusion tests."""
    modalities = ['gravity', 'magnetic']
    data = {}
    for mod in modalities:
        request_param = type('Obj', (), {'param': mod})()
        data[mod] = synthetic_raw_data(request_param)
    return data

@pytest.fixture(scope="function")
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs (maps, exports)."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    yield out_dir
    # Cleanup not needed as tmp_path handles it

@pytest.fixture(scope="session")
def test_config():
    """Standard test configuration as dict (mimics loaded YAML)."""
    return {
        'pipeline': {
            'modalities': ['gravity'],
            'resolution': 0.1,
            'use_cache': False,
            'parallel_workers': 1  # Sequential for tests
        },
        'ingestion': {
            'cache_dir': str(Path('tests/cache')),  # Will use tmp in tests
            'retry_attempts': 1
        },
        'preprocessing': {
            'filter_window': 3,
            'grid_method': 'bilinear'
        },
        'modeling': {
            'inversion_type': 'simple',  # Mock/simple for tests
            'fusion_method': 'bayesian'
        },
        'visualization': {
            'output_formats': ['png', 'csv'],
            'map_projection': 'WGS84'
        },
        'bbox': (29.0, 31.0, 30.0, 32.0)
    }

@pytest.fixture(scope="session")
def minimal_config():
    """Minimal config for basic/single-modality tests."""
    base = test_config()
    base['pipeline']['modalities'] = ['gravity']
    base['modeling']['inversion_type'] = 'minimal'
    return base

@pytest.fixture(scope="session")
def performance_config():
    """Config optimized for performance benchmarks (larger data, parallel)."""
    base = test_config()
    base['pipeline']['parallel_workers'] = 2
    base['bbox'] = (20.0, 40.0, 20.0, 40.0)  # Larger region
    return base

@pytest.fixture(scope="function")
def mock_external_apis(monkeypatch):
    from gam.ingestion.exceptions import DataFetchError
    """Mock external API calls to return synthetic data, avoiding network."""
    def mock_get(url, *args, **kwargs):
        if 'usgs' in url.lower():
            return MagicMock(status_code=200, json=lambda: {
                'gravity': np.random.normal(9.8, 0.1, 100).tolist(),
                'locations': [{'lat': 30.5, 'lon': 30.0}] * 100
            })
        elif 'iris' in url.lower():
            if not has_obspy:
                raise ImportError("obspy required for seismic mock")
            return MagicMock(status_code=200, content=b'synthetic seismic data')
        elif 'esa' in url.lower():
            return MagicMock(status_code=200, json=lambda: {
                'displacement': np.random.normal(0, 1, 100).tolist(),
                'coordinates': [{'lat': 31.0, 'lon': 30.0}] * 100
            })
        raise DataFetchError(f"Mock failed for {url}")
    
    monkeypatch.setattr('requests.get', mock_get)
    
    # Mock obspy for seismic
    if has_obspy:
        def mock_fdsn_client(*args, **kwargs):
            client = MagicMock()
            client.get_waveforms = MagicMock(return_value=obspy.read())  # Empty trace
            return client
        monkeypatch.setattr('obspy.clients.fdsn.Client', mock_fdsn_client)
    
    yield

@pytest.fixture(scope="function")
def tmp_cache_dir(tmp_path):
    """Temporary cache directory for HDF5 caching tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    yield cache_dir

@pytest.fixture(scope="function")
def mock_cache_manager(tmp_cache_dir, monkeypatch):
    from gam.ingestion.cache_manager import HDF5CacheManager
    from gam.ingestion.data_structures import RawData
    import xarray as xr
    """Mock HDF5CacheManager to use tmp dir and return synthetic cached data."""
    def create_mock_manager():
        manager = HDF5CacheManager(cache_dir=str(tmp_cache_dir))
        # Patch save/load to use synthetic data
        def mock_save(key, data):
            with h5py.File(tmp_cache_dir / f"{key}.h5", 'w') as f:
                if isinstance(data.values, xr.Dataset):
                    data.values.to_netcdf(f.create_dataset('data', data=data.values.to_netcdf().encode()))
                else:
                    f.create_dataset('data', data=data.values)
            logger.info(f"Mock saved {key} to cache")
        
        def mock_load(key):
            cache_file = tmp_cache_dir / f"{key}.h5"
            if cache_file.exists():
                with h5py.File(cache_file, 'r') as f:
                    # Simplified load; in real, reconstruct RawData
                    values = np.array(f['data'][:]) if 'data' in f else np.random.normal(0, 1, (10, 10))
                    metadata = {'cached': True, 'key': key}
                    return RawData(values=xr.Dataset({'data': (['lat', 'lon'], values)}), metadata=metadata)
            raise FileNotFoundError(f"Cache miss for {key}")
        
        manager.save = mock_save
        manager.load = mock_load
        return manager
    
    mock_mgr = create_mock_manager()
    monkeypatch.setattr('gam.ingestion.cache_manager.HDF5CacheManager', lambda *args, **kwargs: mock_mgr)
    yield mock_mgr

@pytest.fixture(scope="function")
def sqlite_db(tmp_path):
    """In-memory or tmp SQLite DB for testing exports/caching persistence."""
    db_path = tmp_path / "test_gam.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY,
            lat REAL, lon REAL, depth REAL,
            confidence REAL, anomaly_type TEXT
        )
    """)
    conn.commit()
    yield conn
    conn.close()
    db_path.unlink(missing_ok=True)

@pytest.fixture(autouse=True)
def setup_logging():
    """Auto-configure logging for tests (suppress INFO if needed)."""
    logging.getLogger('gam').setLevel(logging.WARNING)  # Reduce noise
    yield
    logging.getLogger('gam').setLevel(logging.INFO)

@pytest.mark.tryfirst
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")

def pytest_runtest_setup(item):
    """Skip tests requiring optional deps."""
    if "seismic" in item.name and not has_obspy:
        pytest.skip("Requires obspy for seismic tests")