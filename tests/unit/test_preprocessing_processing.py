"""Unit tests for core data transformation components in preprocessing module."""

import pytest
import numpy as np
import numpy.testing as npt
import xarray as xr
from obspy import Stream, Trace
from unittest.mock import patch

from gam.ingestion.data_structures import RawData
from gam.preprocessing.data_structures import ProcessedGrid
from gam.preprocessing.filters import NoiseFilter, BandpassFilter, OutlierFilter, SpatialFilter
from gam.preprocessing.gridding import RegularGridder, AdaptiveGridder, CoordinateAligner
from gam.preprocessing.processors import GravityPreprocessor, MagneticPreprocessor, SeismicPreprocessor, InSARPreprocessor


class TestNoiseFilter:
    """Test NoiseFilter class."""

    def test_apply_to_ndarray(self):
        """Test Gaussian filter on numpy array."""
        # Create synthetic 2D grid with known pattern
        x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        data = np.exp(-(x**2 + y**2))  # Gaussian peak

        filter_obj = NoiseFilter(sigma=1.0)
        result = filter_obj.apply(data)

        # Result should be smoothed version
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        # Original peak should be reduced
        assert result.max() < data.max()

    def test_apply_to_xarray(self):
        """Test Gaussian filter on xarray Dataset."""
        lat = np.linspace(0, 1, 5)
        lon = np.linspace(0, 1, 5)
        data = xr.DataArray(
            np.random.rand(5, 5),
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon}
        )
        ds = xr.Dataset({'data': data})

        filter_obj = NoiseFilter(sigma=0.5)
        result = filter_obj.apply(ds)

        assert isinstance(result, ProcessedGrid)
        assert 'data' in result.ds
        assert result.ds['data'].shape == (5, 5)

    def test_apply_to_rawdata(self):
        """Test Gaussian filter on RawData."""
        data = np.random.rand(10, 10)
        metadata = {'units': 'm/s²', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, data)

        filter_obj = NoiseFilter(sigma=2.0, mode='constant')
        result = filter_obj.apply(raw)

        assert isinstance(result, RawData)
        assert result.values.shape == data.shape

    def test_with_nan_values(self):
        """Test filter handles NaN values."""
        data = np.random.rand(10, 10)
        data[5, 5] = np.nan

        filter_obj = NoiseFilter(sigma=1.0)
        result = filter_obj.apply(data)

        # NaN should be preserved or handled
        assert result.shape == data.shape

    def test_different_sigma_values(self):
        """Test different sigma values affect smoothing."""
        data = np.zeros((10, 10))
        data[5, 5] = 1.0  # Impulse

        filter_small = NoiseFilter(sigma=0.5)
        filter_large = NoiseFilter(sigma=2.0)

        result_small = filter_small.apply(data)
        result_large = filter_large.apply(data)

        # Larger sigma should spread more
        assert np.sum(result_large > 0.1) > np.sum(result_small > 0.1)


class TestBandpassFilter:
    """Test BandpassFilter class."""

    def test_apply_to_stream(self):
        """Test bandpass filter on ObsPy Stream."""
        # Create synthetic seismic trace
        npts = 1000
        dt = 0.01
        t = np.arange(npts) * dt
        # Mix of low and high frequency
        data = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)

        trace = Trace(data=data, header={'delta': dt, 'npts': npts})
        stream = Stream([trace])

        metadata = {'units': 'm/s', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, stream)

        filter_obj = BandpassFilter(lowcut=0.05, highcut=2.0)
        result = filter_obj.apply(raw)

        assert isinstance(result, RawData)
        assert isinstance(result.values, Stream)
        # High frequency component should be attenuated
        original_power = np.sum(data**2)
        filtered_power = np.sum(result.values[0].data**2)
        assert filtered_power < original_power

    def test_invalid_input(self):
        """Test error on non-Stream input."""
        data = np.random.rand(100)
        metadata = {'units': 'm/s', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, data)

        filter_obj = BandpassFilter()
        with pytest.raises(Exception):  # PreprocessingError
            filter_obj.apply(raw)


class TestOutlierFilter:
    """Test OutlierFilter class."""

    def test_zscore_method(self):
        """Test z-score outlier detection."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        # Add outliers
        data[10] = 10.0
        data[20] = -10.0

        filter_obj = OutlierFilter(method='zscore', threshold=3.0)
        result = filter_obj.apply(data)

        # Outliers should be replaced with NaN
        assert np.isnan(result[10])
        assert np.isnan(result[20])
        # Most values should remain
        assert np.sum(~np.isnan(result)) > 90

    def test_iqr_method(self):
        """Test IQR outlier detection."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier

        filter_obj = OutlierFilter(method='iqr', threshold=1.5)
        result = filter_obj.apply(data)

        assert np.isnan(result[-1])  # Last element (100) should be NaN

    def test_with_xarray(self):
        """Test on xarray Dataset."""
        lat = np.linspace(0, 1, 5)
        lon = np.linspace(0, 1, 5)
        data = xr.DataArray(
            np.random.rand(5, 5),
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon}
        )
        data.values[2, 2] = 100.0  # Outlier
        ds = xr.Dataset({'data': data})

        filter_obj = OutlierFilter(method='zscore', threshold=2.0)
        result = filter_obj.apply(ds)

        assert isinstance(result, ProcessedGrid)
        assert np.isnan(result.ds['data'].values[2, 2])


class TestSpatialFilter:
    """Test SpatialFilter class."""

    def test_median_filter(self):
        """Test median spatial filter."""
        # Create grid with salt-and-pepper noise
        data = np.ones((10, 10))
        data[5, 5] = 0.0  # Pepper
        data[3, 3] = 2.0  # Salt

        filter_obj = SpatialFilter(size=3)
        result = filter_obj.apply(data)

        assert result.shape == data.shape
        # Noise should be reduced
        assert abs(result[5, 5] - 1.0) < 0.5
        assert abs(result[3, 3] - 1.0) < 0.5

    def test_different_sizes(self):
        """Test different kernel sizes."""
        data = np.random.rand(20, 20)

        filter_small = SpatialFilter(size=3)
        filter_large = SpatialFilter(size=7)

        result_small = filter_small.apply(data)
        result_large = filter_large.apply(data)

        # Larger kernel should smooth more
        assert np.var(result_large) < np.var(result_small)


class TestRegularGridder:
    """Test RegularGridder class."""

    def test_uniform_scattered_data(self):
        """Test gridding uniform scattered points."""
        # Create regular scattered points
        lats = np.linspace(0, 1, 20)
        lons = np.linspace(0, 1, 20)
        values = np.sin(2 * np.pi * lats) * np.cos(2 * np.pi * lons) + np.random.normal(0, 0.01, len(lats))

        points = np.column_stack((lats, lons, values))
        metadata = {'bbox': (0, 1, 0, 1), 'units': 'm'}

        gridder = RegularGridder(resolution=0.1, method='linear')
        result = gridder.apply(points)

        assert isinstance(result, ProcessedGrid)
        assert 'data' in result.ds
        # Should create 11x11 grid (0 to 1 with 0.1 spacing)
        assert result.ds['data'].shape == (11, 11)
        assert len(result.ds.coords['lat']) == 11
        assert len(result.ds.coords['lon']) == 11

    def test_non_uniform_scattered_data(self):
        """Test gridding non-uniform scattered points."""
        np.random.seed(42)
        n_points = 50
        lats = np.random.uniform(0, 1, n_points)
        lons = np.random.uniform(0, 1, n_points)
        values = lats + lons  # Linear function

        points = np.column_stack((lats, lons, values))
        metadata = {'bbox': (0, 1, 0, 1), 'units': 'm'}

        gridder = RegularGridder(resolution=0.2, method='cubic')
        result = gridder.apply(points)

        assert result.ds['data'].shape == (6, 6)  # 0,0.2,0.4,0.6,0.8,1.0

    def test_different_methods(self):
        """Test different interpolation methods."""
        lats = np.array([0.1, 0.5, 0.9])
        lons = np.array([0.1, 0.5, 0.9])
        values = np.array([1.0, 2.0, 3.0])

        points = np.column_stack((lats, lons, values))

        for method in ['linear', 'cubic', 'nearest']:
            gridder = RegularGridder(resolution=0.5, method=method)
            result = gridder.apply(points)
            assert result.ds['data'].shape == (3, 3)


class TestAdaptiveGridder:
    """Test AdaptiveGridder class."""

    def test_adaptive_gridding(self):
        """Test adaptive resolution based on density."""
        # Create clustered points
        np.random.seed(42)
        # Dense cluster
        lats1 = np.random.normal(0.5, 0.05, 30)
        lons1 = np.random.normal(0.5, 0.05, 30)
        # Sparse points
        lats2 = np.random.uniform(0, 1, 10)
        lons2 = np.random.uniform(0, 1, 10)

        lats = np.concatenate([lats1, lats2])
        lons = np.concatenate([lons1, lons2])
        values = np.ones_like(lats)

        points = np.column_stack((lats, lons, values))

        gridder = AdaptiveGridder(base_resolution=0.1, density_factor=0.5)
        result = gridder.apply(points)

        assert isinstance(result, ProcessedGrid)
        # Should adjust resolution based on density
        assert 'adaptive_adjustment' in result.metadata


class TestCoordinateAligner:
    """Test CoordinateAligner class."""

    def test_same_crs_no_change(self):
        """Test no transformation when CRS matches."""
        lats = np.array([0.1, 0.5])
        lons = np.array([0.1, 0.5])
        values = np.array([1.0, 2.0])

        points = np.column_stack((lats, lons, values))
        metadata = {'coordinate_system': 'EPSG:4326', 'bbox': (0, 1, 0, 1)}

        aligner = CoordinateAligner(target_crs='EPSG:4326')
        result = aligner.apply(points)

        # Should return similar data
        npt.assert_allclose(result[:, :2], points[:, :2], rtol=1e-10)

    @patch('gam.preprocessing.gridding.Transformer')
    def test_crs_transformation(self, mock_transformer):
        """Test coordinate transformation."""
        # Mock transformer
        mock_transformer.from_crs.return_value.transform.return_value = ([0.2, 0.6], [0.2, 0.6])

        lats = np.array([0.1, 0.5])
        lons = np.array([0.1, 0.5])
        values = np.array([1.0, 2.0])

        points = np.column_stack((lats, lons, values))

        aligner = CoordinateAligner(target_crs='EPSG:3857', source_crs='EPSG:4326')
        result = aligner.apply(points)

        # Check transformation was called
        mock_transformer.from_crs.assert_called_once()


class TestGravityPreprocessor:
    """Test GravityPreprocessor class."""

    def test_process_basic(self):
        """Test basic gravity processing."""
        # Create synthetic gravity data
        n_points = 100
        lats = np.random.uniform(0, 1, n_points)
        lons = np.random.uniform(0, 1, n_points)
        # Simple gravity anomaly pattern
        values = 10 + 5 * np.sin(2 * np.pi * lats) + np.random.normal(0, 0.1, n_points)

        metadata = {
            'units': 'mGal',
            'bbox': (0, 1, 0, 1),
            'elevation': np.random.uniform(0, 100, n_points)
        }
        raw = RawData(metadata, values)

        processor = GravityPreprocessor()
        result = processor.process(raw, grid_resolution=0.2)

        assert isinstance(result, ProcessedGrid)
        assert result.units == 'm/s²'
        assert 'modality' in result.metadata
        assert result.metadata['modality'] == 'gravity'

    def test_without_elevation(self):
        """Test processing without elevation data."""
        values = np.random.rand(50)
        metadata = {'units': 'mGal', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, values)

        processor = GravityPreprocessor()
        result = processor.process(raw, apply_filters=False)

        assert isinstance(result, ProcessedGrid)


class TestMagneticPreprocessor:
    """Test MagneticPreprocessor class."""

    def test_process_basic(self):
        """Test basic magnetic processing."""
        n_points = 50
        values = np.random.normal(50000, 100, n_points)  # nT values
        igrf = np.random.normal(49500, 50, n_points)  # Reference field

        metadata = {
            'units': 'nT',
            'bbox': (0, 1, 0, 1),
            'inclination': 60.0,
            'declination': 0.0
        }
        raw = RawData(metadata, values)

        processor = MagneticPreprocessor()
        result = processor.process(raw, igrf_model=igrf, grid_resolution=0.2)

        assert isinstance(result, ProcessedGrid)
        assert result.units == 'nT'
        assert result.metadata['modality'] == 'magnetic'

    def test_without_igrf(self):
        """Test processing without IGRF model."""
        values = np.random.rand(30)
        metadata = {'units': 'nT', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, values)

        processor = MagneticPreprocessor()
        result = processor.process(raw, apply_rtp=False)

        assert isinstance(result, ProcessedGrid)


class TestSeismicPreprocessor:
    """Test SeismicPreprocessor class."""

    def test_process_basic(self):
        """Test basic seismic processing."""
        # Create synthetic seismic traces
        traces = []
        for i in range(5):
            npts = 1000
            dt = 0.01
            t = np.arange(npts) * dt
            # P-wave like signal
            data = np.sin(2 * np.pi * 10 * t) * np.exp(-t / 0.5)
            trace = Trace(
                data=data,
                header={
                    'delta': dt,
                    'npts': npts,
                    'coordinates': {'latitude': 0.1 * i, 'longitude': 0.1 * i}
                }
            )
            traces.append(trace)

        stream = Stream(traces)
        metadata = {'units': 'm/s', 'bbox': (0, 0.5, 0, 0.5)}
        raw = RawData(metadata, stream)

        processor = SeismicPreprocessor()
        result = processor.process(raw, grid_resolution=0.2)

        assert isinstance(result, ProcessedGrid)
        assert result.units == 's'
        assert result.metadata['modality'] == 'seismic'
        assert result.metadata['feature'] == 'travel_time'


class TestInSARPreprocessor:
    """Test InSARPreprocessor class."""

    def test_process_basic(self):
        """Test basic InSAR processing."""
        # Create synthetic phase data
        lat = np.linspace(0, 1, 10)
        lon = np.linspace(0, 1, 10)
        phase_data = np.random.uniform(-np.pi, np.pi, (10, 10))  # Wrapped phase

        phase_da = xr.DataArray(
            phase_data,
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon}
        )
        ds = xr.Dataset({'phase': phase_da})

        metadata = {'units': 'radians', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, ds)

        processor = InSARPreprocessor()

        # Mock SNAPHU since it's external
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            # Create mock unwrapped file
            with patch('numpy.fromfile') as mock_fromfile:
                mock_fromfile.return_value = phase_data  # Mock unwrapped as same for simplicity

                result = processor.process(raw, grid_resolution=0.2, apply_atm_correction=False)

                assert isinstance(result, ProcessedGrid)
                assert result.units == 'm'
                assert result.metadata['modality'] == 'insar'

    def test_without_snaphu(self):
        """Test fallback when SNAPHU fails."""
        lat = np.linspace(0, 1, 5)
        lon = np.linspace(0, 1, 5)
        phase_data = np.random.rand(5, 5)

        phase_da = xr.DataArray(
            phase_data,
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon}
        )
        ds = xr.Dataset({'phase': phase_da})

        metadata = {'units': 'radians', 'bbox': (0, 1, 0, 1)}
        raw = RawData(metadata, ds)

        processor = InSARPreprocessor()

        # Mock SNAPHU failure
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("SNAPHU not found")

            result = processor.process(raw, apply_atm_correction=False)

            # Should still complete with wrapped phase
            assert isinstance(result, ProcessedGrid)