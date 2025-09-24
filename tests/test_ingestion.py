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
from shapely.geometry import box
import numpy as np
from datetime import datetime, timedelta

from gam.ingestion import (
    RawData, HDF5CacheManager, GravityFetcher, MagneticFetcher,
    SeismicFetcher, InSARFetcher, IngestionManager, DataFetchError,
    APITimeoutError, retry_fetch, rate_limit
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

@pytest.fixture
def insar_fetcher():
    """Fixture for InSARFetcher instance."""
    return InSARFetcher()

@pytest.fixture
def mock_products():
    """Fixture for mock Sentinel-1 products."""
    product_id = 'S1A_IW_SLC__1SDV_20240115T054321_20240115T054348_052123_063456_1234'
    return {
        product_id: {
            'title': product_id,
            'platformname': 'Sentinel-1',
            'producttype': 'SLC',
            'orbitdirection': 'ASCENDING',
            'beginposition': datetime(2024, 1, 15, 5, 43, 21),
            'endposition': datetime(2024, 1, 15, 5, 43, 48),
            'footprint': box(30.0, 29.0, 32.0, 31.0).wkt  # Full bbox match
        }
    }

@pytest.fixture
def mock_raster_src():
    """Fixture for mock rasterio dataset."""
    mock_src = MagicMock()
    mock_src.width = 400
    mock_src.height = 300
    mock_src.window_transform.return_value = (30.0, 0.01, 0, 29.0, 0, -0.01)  # Affine approx
    return mock_src

@pytest.fixture
def sample_bbox():
    """Sample bbox for tests."""
    return (29.0, 31.0, 30.0, 32.0)


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

class TestInSARFetcher:
    """Test suite for InSARFetcher functionality."""

    def test_insar_fetcher_initialization(self, insar_fetcher):
        """Test InSAR fetcher initialization and config"""
        assert insar_fetcher.api_url == "https://apihub.copernicus.eu/apihub/"
        assert isinstance(insar_fetcher.cache, HDF5CacheManager)
        assert insar_fetcher.source == "ESA Sentinel-1"
        assert insar_fetcher.timeout == 300.0
        assert np.isclose(insar_fetcher.wavelength, 0.05546576)

    @pytest.mark.parametrize("product_type", ["SLC", "GRD"])
    @patch('gam.ingestion.fetchers.HDF5CacheManager.load_data')
    @patch('gam.ingestion.fetchers.hashlib.md5')
    @patch('gam.ingestion.fetchers.shutil.rmtree')
    @patch('gam.ingestion.fetchers.tempfile.mkdtemp')
    @patch('gam.ingestion.fetchers.os.walk')
    @patch('gam.ingestion.fetchers.os.path.exists')
    @patch('gam.ingestion.fetchers.zipfile.ZipFile')
    @patch('gam.ingestion.fetchers.rasterio.open')
    @patch('gam.ingestion.fetchers.SentinelAPI')
    def test_insar_fetch_data_success(self, mock_sentinel, mock_rasterio, mock_zip, mock_exists, mock_walk, mock_mkdtemp, mock_rmtree, mock_md5, mock_load_cache, insar_fetcher, mock_products, mock_raster_src, sample_bbox, monkeypatch, product_type, tmp_path):
        """Test successful InSAR data fetching with SLC products"""
        # Setup mocks
        mock_load_cache.return_value = None  # Cache miss
        mock_md5.return_value.hexdigest.return_value = 'fixed_key'
        mock_mkdtemp.side_effect = ['/tmp/download', '/tmp/extract']
        mock_exists.return_value = True
        mock_zip.return_value.extractall = MagicMock()
        mock_walk.return_value = [('measurement', [], ['iw1.vv.tiff'])]
        mock_tiff_path = '/tmp/extract/measurement/iw1.vv.tiff'
        mock_rasterio.return_value.__enter__.return_value = mock_raster_src
        height, width = 15, 20  # Subsampled
        if product_type == 'SLC':
            mock_data = np.exp(1j * np.linspace(0, 2*np.pi, height*width)).reshape(height, width).astype(np.complex64)
            expected_disp = np.angle(mock_data) * insar_fetcher.wavelength / (4 * np.pi)
        else:
            mock_data = np.random.randint(0, 256, (height, width)).astype(np.float32)
            max_amp = np.max(mock_data)
            expected_disp = (mock_data / max_amp * 1000).flatten()
        mock_raster_src.read.return_value = mock_data

        def mock_xy(transform, rows, cols):
            # Generate points, some in bbox for mask
            total = len(rows)
            lons = np.linspace(29.5, 32.5, total)  # Some outside
            lats = np.linspace(28.5, 31.5, total)
            points = list(zip(lons, lats))
            return points  # (lon, lat) tuples
        mock_raster_src.transform.xy = MagicMock(side_effect=mock_xy)

        mock_api = mock_sentinel.return_value
        mock_api.query.return_value = mock_products
        product_id = list(mock_products.keys())[0]
        mock_product = mock_products[product_id].copy()
        mock_product['producttype'] = product_type
        mock_api.download.return_value = {product_id: {'path': f'/tmp/download/{product_id}.zip'}}

        # Env for creds
        monkeypatch.setenv('ESA_USERNAME', 'test_user')
        monkeypatch.setenv('ESA_PASSWORD', 'test_pass')

        # Call
        kwargs = {'product_type': product_type, 'start_date': '2024-01-01', 'end_date': '2024-01-15'}
        with patch('gam.ingestion.fetchers.HDF5CacheManager.save_data') as mock_save:
            data = insar_fetcher.fetch_data(sample_bbox, **kwargs)

        # Assertions
        data.validate()
        assert data.metadata['source'] == 'ESA Sentinel-1'
        assert len(data.values) > 0
        np.testing.assert_allclose(data.values, expected_disp.flatten(), atol=1e-6)
        assert 'lat' in data.metadata
        assert 'lon' in data.metadata
        assert len(data.metadata['lat']) == len(data.values)
        # Verify filtering: not all points, some masked out
        assert len(data.values) < height * width  # Some outside bbox
        assert data.metadata['parameters']['product_type'] == product_type
        assert data.metadata['units'] == ('m' if product_type == 'SLC' else 'mm (amplitude proxy)')
        mock_api.query.assert_called_once_with(
            box(30.0, 29.0, 32.0, 31.0).wkt,
            date=('2024-01-01', '2024-01-15'),
            platformname='Sentinel-1',
            producttype=product_type,
            polarisationmode='VV'
        )
        mock_api.download.assert_called_once()
        mock_rasterio.assert_called_once_with(mock_tiff_path)
        mock_raster_src.read.assert_called_once()
        mock_save.assert_called_once()

    def test_insar_fetch_data_grd_products(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test InSAR fetching with GRD products"""
        # Reuse success test logic, but isolate for GRD
        # (Parametrized above covers it; this can be additional if needed)
        pass  # Covered in parametrized test

    def test_insar_authentication(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test ESA credentials handling"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        # Should succeed (as in success test)
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            mock_api.return_value.query.return_value = {'id': {}}
            insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
        mock_api.assert_called_once_with('user', 'pass', insar_fetcher.api_url)

        # Kwargs override
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            insar_fetcher.fetch_data(sample_bbox, username='kw_user', password='kw_pass', start_date='2024-01-01')
        mock_api.assert_called_once_with('kw_user', 'kw_pass', insar_fetcher.api_url)

    def test_insar_bbox_filtering(self, insar_fetcher, sample_bbox):
        """Test geographic filtering of products"""
        # Mock query with product footprint intersecting bbox
        intersecting_products = {'id1': {'footprint': box(30.5, 29.5, 31.5, 30.5).wkt}}  # Partial overlap
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            mock_api.return_value.query.return_value = intersecting_products
            with patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None):
                data = insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01', product_type='SLC')
        # Assert selected product (max beginposition, but since one, ok)
        assert len(data.values) > 0  # Filtered coords in bbox

        # No intersection (mock empty after filter, but impl selects before; test via mask none)
        # For full no products, see below

    def test_insar_date_filtering(self, insar_fetcher, sample_bbox):
        """Test temporal filtering with custom date ranges"""
        custom_start = '2023-12-01'
        custom_end = '2023-12-31'
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            mock_api.return_value.query.return_value = mock_products  # Fixture
            with patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None):
                insar_fetcher.fetch_data(sample_bbox, start_date=custom_start, end_date=custom_end, product_type='SLC')
        mock_api.return_value.query.assert_called_with(
            ANY,  # footprint
            date=(custom_start, custom_end),
            platformname='Sentinel-1',
            producttype='SLC',
            polarisationmode='VV'
        )

    def test_insar_no_products_found(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test handling when no products match criteria"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            mock_api.return_value.query.return_value = {}
            with pytest.raises(DataFetchError, match="No SLC products found"):
                insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01', product_type='SLC')

    def test_insar_authentication_failure(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test authentication error handling"""
        monkeypatch.delenv('ESA_USERNAME', raising=False)
        monkeypatch.delenv('ESA_PASSWORD', raising=False)
        with pytest.raises(ValueError, match="ESA credentials required"):
            insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')

        # Mock API init fail
        monkeypatch.setenv('ESA_USERNAME', 'invalid')
        monkeypatch.setenv('ESA_PASSWORD', 'invalid')
        with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api, pytest.raises(DataFetchError):
            mock_api.side_effect = Exception("Auth failed")
            insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')

    @patch('gam.ingestion.fetchers.SentinelAPI')
    def test_insar_download_failure(self, mock_sentinel, insar_fetcher, sample_bbox, monkeypatch):
        """Test download error handling and retries"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        mock_sentinel.return_value.query.return_value = mock_products
        mock_sentinel.return_value.download.side_effect = [Timeout(), {}]  # Retry once, then success (but test fail)
        # For pure fail: side_effect = Timeout()
        with pytest.raises(APITimeoutError):
            mock_sentinel.return_value.download.side_effect = Timeout()
            insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
        # Assert retry (but decorator handles; mock calls ==3 for max_attempts=3)

    @patch('gam.ingestion.fetchers.SentinelAPI')
    @patch('gam.ingestion.fetchers.os.path.exists')
    def test_insar_processing_failure(self, mock_exists, mock_sentinel, insar_fetcher, sample_bbox, monkeypatch):
        """Test SAFE file processing error handling"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        mock_sentinel.return_value.query.return_value = mock_products
        mock_sentinel.return_value.download.return_value = {'id': {'path': '/tmp/zip.zip'}}
        mock_exists.return_value = True  # Zip exists
        mock_exists.side_effect = [True, False]  # Zip yes, TIFF no
        with pytest.raises(DataFetchError, match="VV TIFF not found"):
            insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')

        # Rasterio fail
        with patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio:
            mock_rasterio.side_effect = Exception("Invalid TIFF")
            mock_exists.return_value = True  # TIFF exists
            with pytest.raises(DataFetchError, match="Processing failed"):
                insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')

        # No data in bbox (mask all False)
        with patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio, patch('rasterio.transform.xy') as mock_xy:
            mock_src = MagicMock()
            mock_src.read.return_value = np.zeros((1,1), np.complex64)
            mock_rasterio.return_value.__enter__.return_value = mock_src
            mock_xy.return_value = [(33.0, 28.0)]  # Outside bbox
            mock_sentinel.return_value.download.return_value = {'id': {'path': '/tmp/zip.zip'}}
            mock_exists.return_value = True
            with patch('gam.ingestion.fetchers.os.walk') as mock_walk:
                mock_walk.return_value = [('measurement', [], ['vv.tiff'])]
                with pytest.raises(DataFetchError, match="No data points within bbox"):
                    insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')

    @patch('gam.ingestion.fetchers.hashlib.md5')
    def test_insar_cache_integration(self, mock_md5, insar_fetcher, sample_bbox, monkeypatch, sample_raw_data):
        """Test HDF5 cache save/load functionality"""
        mock_md5.return_value.hexdigest.return_value = 'cache_key'
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')

        # Cache hit
        with patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=sample_raw_data):
            with patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:  # Not called
                data = insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
            mock_api.assert_not_called()
            assert data == sample_raw_data

        # Cache miss + save
        with patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None), \
             patch('gam.ingestion.fetchers.HDF5CacheManager.save_data') as mock_save, \
             patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
            mock_api.return_value.query.return_value = mock_products
            mock_api.return_value.download.return_value = {'id': {'path': '/tmp/zip'}}
            # Minimal mocks for flow
            with patch('gam.ingestion.fetchers.os.path.exists', return_value=True), \
                 patch('gam.ingestion.fetchers.zipfile.ZipFile'), \
                 patch('gam.ingestion.fetchers.os.walk'), \
                 patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio:
                mock_src = MagicMock()
                mock_src.read.return_value = np.array([[0.001]])
                mock_rasterio.return_value.__enter__.return_value = mock_src
                with patch('rasterio.transform.xy', return_value=[(31.0, 30.5)]):
                    data = insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
            mock_save.assert_called_once_with(data)

    def test_insar_data_format_validation(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test RawData output format and validation"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        with patch('gam.ingestion.fetchers.SentinelAPI'), \
             patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None), \
             patch('gam.ingestion.fetchers.os.path.exists') as mock_exists, \
             patch('gam.ingestion.fetchers.zipfile.ZipFile') as mock_zip, \
             patch('gam.ingestion.fetchers.os.walk') as mock_walk, \
             patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio:
                mock_exists.return_value = True
                mock_zip.return_value.extractall = MagicMock()
                mock_walk.return_value = [('measurement', [], ['vv.tiff'])]
                # Setup to produce valid data
                mock_src = MagicMock()
                mock_src.read.return_value = np.array([[0.001, 0.002]])
                mock_rasterio.return_value.__enter__.return_value = mock_src
                with patch('rasterio.transform.xy', return_value=[(30.5, 29.5), (31.0, 30.0)]):
                    data = insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
                    data.validate()  # Should pass
                    assert isinstance(data.values, np.ndarray)
                    assert all(key in data.metadata for key in ['source', 'timestamp', 'bbox', 'parameters', 'lat', 'lon', 'units', 'note'])
                    assert isinstance(data.metadata['lat'], list)
                    assert len(data.metadata['lat']) == len(data.values)

    def test_insar_coordinate_generation(self, insar_fetcher, sample_bbox, monkeypatch):
        """Test lat/lon coordinate array generation"""
        monkeypatch.setenv('ESA_USERNAME', 'user')
        monkeypatch.setenv('ESA_PASSWORD', 'pass')
        with patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio, \
             patch('gam.ingestion.fetchers.SentinelAPI'), \
             patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None):
            mock_src = MagicMock()
            mock_src.read.return_value = np.ones((2,2), np.complex64)
            mock_rasterio.return_value.__enter__.return_value = mock_src
            def mock_xy(transform, rows, cols):
                return [(30.1, 29.1), (30.2, 29.2), (31.1, 30.1), (31.2, 30.2)]
            with patch('rasterio.transform.xy', side_effect=mock_xy):
                # Other mocks minimal
                with patch('gam.ingestion.fetchers.os.walk') as mock_walk, \
                     patch('gam.ingestion.fetchers.os.path.exists', return_value=True) as mock_exists, \
                     patch('gam.ingestion.fetchers.zipfile.ZipFile') as mock_zip, \
                     patch('gam.ingestion.fetchers.SentinelAPI') as mock_api:
                    mock_walk.return_value = [('measurement', [], ['vv.tiff'])]
                    mock_zip.return_value.extractall = MagicMock()
                    mock_api.return_value.query.return_value = mock_products
                    mock_api.return_value.download.return_value = {'id': {'path': '/tmp/zip.zip'}}
                    data = insar_fetcher.fetch_data(sample_bbox, start_date='2024-01-01')
            lats, lons = data.metadata['lat'], data.metadata['lon']
            assert all(29.0 <= lat <= 31.0 for lat in lats)
            assert all(30.0 <= lon <= 32.0 for lon in lons)
            assert len(lats) == len(data.values) == 4  # All in bbox

    def test_insar_displacement_calculation(self, insar_fetcher, mock_products):
        """Test displacement value calculation from phase/amplitude"""
        # SLC phase
        os.environ['ESA_USERNAME'] = 'user'
        os.environ['ESA_PASSWORD'] = 'pass'
        try:
            with patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio:
                mock_src = MagicMock()
                phase_data = np.array([[np.pi/2, np.pi], [3*np.pi/2, 0]], dtype=np.complex64)
                mock_src.read.return_value = phase_data
                mock_rasterio.return_value.__enter__.return_value = mock_src
                with \
                     patch('gam.ingestion.fetchers.SentinelAPI') as mock_api, \
                     patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None), \
                     patch('gam.ingestion.fetchers.os.path.exists', return_value=True), \
                     patch('gam.ingestion.fetchers.os.walk', return_value=[('measurement', [], ['vv.tiff'])]), \
                     patch('gam.ingestion.fetchers.zipfile.ZipFile') as mock_zip, \
                     patch('gam.ingestion.fetchers.tempfile.mkdtemp', return_value='/tmp'), \
                     patch('gam.ingestion.fetchers.shutil.rmtree'), \
                     patch('rasterio.transform.xy', return_value=[(30.5, 29.5)] * 4):
                    mock_api.return_value.query.return_value = mock_products
                    mock_api.return_value.download.return_value = {'id': {'path': '/tmp.zip'}}
                    mock_zip.return_value.extractall = MagicMock()
                    data = insar_fetcher.fetch_data((29, 31, 30, 32), product_type='SLC', start_date='2024-01-01')
                expected_phase = np.angle(phase_data).flatten()
                expected_disp = expected_phase * insar_fetcher.wavelength / (4 * np.pi)
                np.testing.assert_allclose(data.values, expected_disp, atol=1e-6)
        finally:
            if 'ESA_USERNAME' in os.environ:
                del os.environ['ESA_USERNAME']
            if 'ESA_PASSWORD' in os.environ:
                del os.environ['ESA_PASSWORD']

        # GRD amp
        os.environ['ESA_USERNAME'] = 'user'
        os.environ['ESA_PASSWORD'] = 'pass'
        try:
            with patch('gam.ingestion.fetchers.rasterio.open') as mock_rasterio:
                mock_src = MagicMock()
                amp_data = np.array([[100, 200], [150, 255]], dtype=np.float32)
                mock_src.read.return_value = amp_data
                mock_rasterio.return_value.__enter__.return_value = mock_src
                with \
                     patch('gam.ingestion.fetchers.SentinelAPI') as mock_api, \
                     patch('gam.ingestion.fetchers.HDF5CacheManager.load_data', return_value=None), \
                     patch('gam.ingestion.fetchers.os.path.exists', return_value=True), \
                     patch('gam.ingestion.fetchers.os.walk', return_value=[('measurement', [], ['vv.tiff'])]), \
                     patch('gam.ingestion.fetchers.zipfile.ZipFile') as mock_zip, \
                     patch('gam.ingestion.fetchers.tempfile.mkdtemp', return_value='/tmp'), \
                     patch('gam.ingestion.fetchers.shutil.rmtree'), \
                     patch('rasterio.transform.xy', return_value=[(30.5, 29.5)] * 4):
                    mock_api.return_value.query.return_value = mock_products
                    mock_api.return_value.download.return_value = {'id': {'path': '/tmp.zip'}}
                    mock_zip.return_value.extractall = MagicMock()
                    data = insar_fetcher.fetch_data((29, 31, 30, 32), product_type='GRD', start_date='2024-01-01')
                max_amp = 255
                expected_amp = (amp_data / max_amp * 1000).flatten()
                np.testing.assert_allclose(data.values, expected_amp, atol=1e-6)
        finally:
            if 'ESA_USERNAME' in os.environ:
                del os.environ['ESA_USERNAME']
            if 'ESA_PASSWORD' in os.environ:
                del os.environ['ESA_PASSWORD']

    def test_insar_invalid_bbox(self, insar_fetcher):
        """Test invalid bbox raises ValueError"""
        invalid_bbox = (31.0, 29.0, 32.0, 30.0)  # min > max
        with pytest.raises(ValueError, match="Invalid bbox"):
            insar_fetcher.fetch_data(invalid_bbox, start_date='2024-01-01')

    # Additional edge cases covered in failures above


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