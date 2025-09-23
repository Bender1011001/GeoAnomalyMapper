"""Unit tests for the GAM data ingestion module."""

import os
import json
import pytest
import responses
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

import h5py
from obspy import Stream, Trace

from gam.ingestion import (
    RawData, HDF5CacheManager, GravityFetcher, MagneticFetcher,
    SeismicFetcher, InSARFetcher, IngestionManager, DataFetchError,
    retry_fetch, rate_limit
)


@pytest.fixture
def sample_metadata():
    """Fixture for sample metadata."""
    return {
        'source': 'Test Source',
        'timestamp': datetime.now(),
        'bbox': (29.0, 31.0, 30.0, 32.0),
        'parameters': {'test': 'param'}
    }


@pytest.fixture
def sample_raw_data(sample_metadata):
    """Fixture for sample RawData."""
    data = RawData(sample_metadata, np.array([1.0, 2.0, 3.0]))
    data.validate()
    return data


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture for temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


def test_raw_data_init_and_validate(sample_metadata):
    """Test RawData initialization and validation."""
    values = np.array([1.0, 2.0])
    data = RawData(sample_metadata, values)
    assert data.metadata == sample_metadata
    assert np.array_equal(data.values, values)
    data.validate()  # Should succeed

    # Invalid bbox
    invalid_meta = sample_metadata.copy()
    invalid_meta['bbox'] = (31.0, 29.0, 32.0, 30.0)  # min > max
    with pytest.raises(ValueError, match="Invalid bbox"):
        invalid_data = RawData(invalid_meta, values)
        invalid_data.validate()

    # Invalid timestamp
    invalid_meta = sample_metadata.copy()
    invalid_meta['timestamp'] = "invalid"
    with pytest.raises(ValueError, match="timestamp must be a datetime"):
        invalid_data = RawData(invalid_meta, values)
        invalid_data.validate()

    # None values
    with pytest.raises(ValueError, match="values cannot be None"):
        RawData(sample_metadata, None)


def test_raw_data_serialization(sample_raw_data):
    """Test RawData to/from dict roundtrip."""
    data_dict = sample_raw_data.to_dict()
    assert 'metadata' in data_dict
    assert 'values' in data_dict
    assert isinstance(data_dict['metadata']['timestamp'], str)  # ISO

    reconstructed = RawData.from_dict(data_dict)
    reconstructed.validate()
    assert reconstructed.metadata['source'] == sample_raw_data.metadata['source']
    assert np.array_equal(reconstructed.values, sample_raw_data.values)


def test_hdf5_cache_manager(temp_cache_dir):
    """Test HDF5CacheManager operations."""
    manager = HDF5CacheManager(cache_dir=temp_cache_dir)
    sample_raw_data = RawData({'source': 'test', 'timestamp': datetime.now(), 'bbox': (0,1,0,1), 'parameters': {}}, np.array([1.0]))

    # Save and load
    key = manager.save_data(sample_raw_data)
    assert len(key) == 32  # MD5 hash
    loaded = manager.load_data(key)
    assert loaded is not None
    assert loaded.metadata['source'] == 'test'
    assert np.array_equal(loaded.values, np.array([1.0]))

    # Exists
    assert manager.exists(key)
    assert not manager.exists('nonexistent')

    # Clear cache
    deleted = manager.clear_cache()
    assert deleted == 1  # One file deleted
    assert not manager.exists(key)

    # Version mismatch
    with h5py.File(os.path.join(temp_cache_dir, f"{key}.h5"), 'w') as f:
        f.attrs['version'] = '0.0'  # Mismatch
    assert not manager.exists(key)  # Should return False on version check


def test_gravity_fetcher():
    """Test GravityFetcher with mocked API."""
    bbox = (29.0, 31.0, 30.0, 32.0)
    mock_response = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"gravity": 123.45}},
            {"properties": {"gravity": 678.90}}
        ]
    }
    with responses.RequestsMock() as rsps:
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/gravity', json=mock_response, status=200)
        fetcher = GravityFetcher()
        data = fetcher.fetch_data(bbox)
        data.validate()
        assert data.metadata['source'] == 'USGS Gravity'
        assert len(data.values) == 2
        assert np.isclose(data.values, np.array([123.45, 678.90])).all()

    # Empty response
    with responses.RequestsMock() as rsps:
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/gravity', json={"features": []}, status=200)
        with pytest.raises(DataFetchError, match="No data found"):
            fetcher.fetch_data(bbox)


def test_magnetic_fetcher():
    """Test MagneticFetcher with mocked API."""
    bbox = (29.0, 31.0, 30.0, 32.0)
    mock_response = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"magnetic": 456.78}},
            {"properties": {"magnetic": 901.23}}
        ]
    }
    with responses.RequestsMock() as rsps:
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/magnetic', json=mock_response, status=200)
        fetcher = MagneticFetcher()
        data = fetcher.fetch_data(bbox)
        data.validate()
        assert data.metadata['source'] == 'USGS Magnetic'
        assert len(data.values) == 2
        assert np.isclose(data.values, np.array([456.78, 901.23])).all()


@patch('obspy.clients.fdsn.Client.get_waveforms')
def test_seismic_fetcher(mock_get_waveforms, bbox=(29.0, 31.0, 30.0, 32.0)):
    """Test SeismicFetcher with mocked ObsPy client."""
    mock_stream = Stream(traces=[Trace(data=np.random.rand(100))])
    mock_get_waveforms.return_value = mock_stream

    fetcher = SeismicFetcher()
    kwargs = {'starttime': '2023-01-01', 'endtime': '2023-01-02'}
    data = fetcher.fetch_data(bbox, **kwargs)
    data.validate()
    assert data.metadata['source'] == 'IRIS Seismic'
    assert isinstance(data.values, Stream)
    assert len(data.values) == 1
    mock_get_waveforms.assert_called_once()


def test_insar_fetcher():
    """Test InSARFetcher raises NotImplementedError."""
    fetcher = InSARFetcher()
    bbox = (29.0, 31.0, 30.0, 32.0)
    with pytest.raises(NotImplementedError, match="placeholder"):
        fetcher.fetch_data(bbox)


def test_ingestion_manager(temp_cache_dir):
    """Test IngestionManager fetch_modality with caching."""
    manager = IngestionManager(cache_dir=temp_cache_dir)
    bbox = (29.0, 31.0, 30.0, 32.0)

    # Cache miss
    mock_response = {"features": [{"properties": {"gravity": 123.45}}]}
    with responses.RequestsMock() as rsps:
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/gravity', json=mock_response, status=200)
        data1 = manager.fetch_modality('gravity', bbox)
        assert data1 is not None
        assert len(data1.values) == 1

    # Cache hit (same key)
    data2 = manager.fetch_modality('gravity', bbox)
    assert data2 is not None
    assert id(data1) != id(data2)  # New object, but same content

    # Unknown modality
    with pytest.raises(ValueError, match="Unknown modality"):
        manager.fetch_modality('unknown', bbox)


def test_fetch_multiple_success_and_failure():
    """Test fetch_multiple with mixed results."""
    manager = IngestionManager()
    bbox = (29.0, 31.0, 30.0, 32.0)

    # Mock success for gravity, failure for magnetic
    with responses.RequestsMock() as rsps:
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/gravity', json={"features": [{"properties": {"gravity": 123}}]}, status=200)
        rsps.add(rsps.GET, 'https://mrdata.usgs.gov/services/magnetic', status=500)  # Failure

        results = manager.fetch_multiple(['gravity', 'magnetic'], bbox)
        assert results['gravity'] is not None
        assert results['magnetic'] is None  # Failure set to None


def test_retry_fetch():
    """Test retry_fetch decorator attempts 3 times."""
    calls = []

    @retry_fetch()
    def flaky_func():
        calls.append(1)
        raise DataFetchError("test", "flaky")

    with pytest.raises(DataFetchError):
        flaky_func()
    assert len(calls) == 3  # Retried 3 times


def test_rate_limit():
    """Test rate_limit decorator sleeps on excess calls."""
    import time

    calls = []
    sleep_times = []

    @rate_limit(1)  # 1 per minute, but test with window=0.1 for quick
    def limited_func():
        calls.append(1)
        return "ok"

    # First call
    result1 = limited_func()
    assert result1 == "ok"
    assert len(calls) == 1

    # Second call immediately - should sleep
    with patch('time.sleep') as mock_sleep:
        result2 = limited_func()
        mock_sleep.assert_called_once_with(0.1)  # Approx window
    assert result2 == "ok"
    assert len(calls) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])