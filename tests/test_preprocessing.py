"""Unit tests for the GAM preprocessing module."""

import pytest
import numpy as np
import xarray as xr
from obspy import Stream, Trace
from obspy.core.utls import Stats
import yaml

from gam.ingestion.data_structures import RawData
from gam.preprocessing import (
    PreprocessingManager, ProcessedGrid, NoiseFilter, BandpassFilter,
    OutlierFilter, SpatialFilter, RegularGridder, AdaptiveGridder,
    CoordinateAligner, UnitConverter, GravityPreprocessor, MagneticPreprocessor,
    SeismicPreprocessor, InSARPreprocessor, DaskPreprocessor
)
from gam.core.exceptions import PreprocessingError


@pytest.fixture
def synthetic_raw_data():
    """Synthetic RawData for testing."""
    metadata = {
        'source': 'test',
        'timestamp': np.datetime64('2023-01-01'),
        'bbox': (30.0, 31.0, 0.0, 1.0),
        'units': 'mGal',
        'parameters': {}
    }
    values = np.random.randn(100)
    return RawData(metadata, values)


@pytest.fixture
def synthetic_xr_dataset():
    """Synthetic xarray.Dataset."""
    lats = np.linspace(30, 31, 10)
    lons = np.linspace(0, 1, 10)
    data = np.random.randn(10, 10)
    ds = xr.Dataset(
        {'data': (['lat', 'lon'], data)},
        coords={'lat': lats, 'lon': lons}
    )
    ds.attrs['units'] = 'mGal'
    ds.attrs['coordinate_system'] = 'EPSG:4326'
    return ds


@pytest.fixture
def synthetic_seismic_stream():
    """Synthetic ObsPy Stream."""
    tr = Trace(data=np.random.randn(1000), header=Stats({'delta': 0.01}))
    tr.stats.coordinates = Stats({'latitude': 30.5, 'longitude': 0.5})
    return Stream([tr])


def test_processed_grid_creation_and_validation(synthetic_xr_dataset):
    """Test ProcessedGrid creation and validation."""
    grid = ProcessedGrid(synthetic_xr_dataset)
    grid.validate()
    assert grid.ds.dims == {'lat': 10, 'lon': 10}
    assert 'data' in grid.ds
    assert grid.units == 'mGal'


def test_processed_grid_transform_crs(synthetic_xr_dataset):
    """Test coordinate transformation."""
    grid = ProcessedGrid(synthetic_xr_dataset)
    new_grid = grid.transform_crs('EPSG:3857')
    new_grid.validate()
    assert new_grid.coordinate_system == 'EPSG:3857'
    # Check coords changed (approx)
    assert not np.allclose(grid.ds.coords['lat'], new_grid.ds.coords['lat'])


def test_processed_grid_unit_conversion(synthetic_xr_dataset):
    """Test unit conversion."""
    grid = ProcessedGrid(synthetic_xr_dataset)
    new_grid = grid.convert_units('m/s²', 1e-5)  # mGal to m/s²
    assert new_grid.units == 'm/s²'
    np.testing.assert_allclose(new_grid.ds['data'], grid.ds['data'] * 1e-5, rtol=1e-5)


def test_noise_filter(synthetic_raw_data):
    """Test Gaussian noise filter."""
    filter_ = NoiseFilter(sigma=1.0)
    filtered = filter_.apply(synthetic_raw_data)
    assert len(filtered.values) == len(synthetic_raw_data.values)
    # Check variance reduced
    orig_var = np.var(synthetic_raw_data.values)
    filt_var = np.var(filtered.values)
    assert filt_var < orig_var


def test_outlier_filter(synthetic_raw_data):
    """Test outlier detection."""
    # Add outliers
    values = synthetic_raw_data.values.copy()
    values[0] = 10 * np.std(values)
    outlier_data = RawData(synthetic_raw_data.metadata, values)
    filter_ = OutlierFilter(threshold=3.0)
    filtered = filter_.apply(outlier_data)
    # Check outlier replaced with NaN or mean
    assert np.isnan(filtered.values[0]) or np.isclose(filtered.values[0], np.mean(values[1:]))


def test_spatial_filter(synthetic_xr_dataset):
    """Test median spatial filter."""
    filter_ = SpatialFilter(size=3)
    filtered = filter_.apply(synthetic_xr_dataset)
    assert filtered.ds['data'].shape == synthetic_xr_dataset['data'].shape


def test_bandpass_filter(synthetic_seismic_stream):
    """Test bandpass filter for seismic."""
    filter_ = BandpassFilter(lowcut=0.1, highcut=1.0)
    raw_data = RawData({'source': 'seismic', 'bbox': (0,0,0,0), 'timestamp': np.datetime64('now')}, synthetic_seismic_stream)
    filtered = filter_.apply(raw_data)
    assert isinstance(filtered.values, Stream)
    assert len(filtered.values) == len(synthetic_seismic_stream)


def test_regular_gridder(synthetic_raw_data):
    """Test regular gridding."""
    # Scattered points
    points = np.random.rand(50, 2) * np.array([[1,1]]) + np.array([[30, 0]])
    values = np.random.rand(50)
    scattered_data = RawData(synthetic_raw_data.metadata, np.column_stack((points, values)))
    gridder = RegularGridder(resolution=0.1)
    grid = gridder.apply(scattered_data)
    grid.validate()
    assert grid.ds['data'].shape == (11, 11)  # Approx for bbox


def test_adaptive_gridder(synthetic_raw_data):
    """Test adaptive gridding."""
    gridder = AdaptiveGridder(base_resolution=0.1)
    grid = gridder.apply(synthetic_raw_data)
    assert isinstance(grid, ProcessedGrid)
    # Check metadata has adjustment
    assert 'adaptive_adjustment' in grid.ds.attrs


def test_coordinate_aligner(synthetic_xr_dataset):
    """Test coordinate alignment."""
    aligner = CoordinateAligner(target_crs='EPSG:4326')
    aligned = aligner.apply(synthetic_xr_dataset)
    assert aligned.coordinate_system == 'EPSG:4326'


def test_unit_converter():
    """Test unit conversions."""
    converter = UnitConverter()
    assert converter.validate_unit('gravity', 'mGal')
    factor = converter.get_factor('gravity', 'mGal', 'm/s²')
    assert np.isclose(factor, 1e-5)
    converted = converter.convert(100.0, 'mGal', 'm/s²', 'gravity')
    assert np.isclose(converted, 0.001)


def test_gravity_preprocessor(synthetic_raw_data):
    """Test gravity processor."""
    proc = GravityPreprocessor()
    # Mock elevation
    kwargs = {'elevation': np.zeros_like(synthetic_raw_data.values)}
    grid = proc.process(synthetic_raw_data, **kwargs)
    assert isinstance(grid, ProcessedGrid)
    assert grid.units == 'm/s²'


def test_magnetic_preprocessor(synthetic_raw_data):
    """Test magnetic processor."""
    proc = MagneticPreprocessor()
    kwargs = {'igrf_model': np.zeros_like(synthetic_raw_data.values)}
    grid = proc.process(synthetic_raw_data, **kwargs)
    assert grid.units == 'nT'


def test_seismic_preprocessor(synthetic_seismic_stream):
    """Test seismic processor."""
    metadata = {'source': 'seismic', 'bbox': (30,31,0,1), 'timestamp': np.datetime64('now')}
    raw_data = RawData(metadata, synthetic_seismic_stream)
    proc = SeismicPreprocessor()
    grid = proc.process(raw_data)
    assert grid.units == 'm/s'


def test_insar_preprocessor():
    """Test InSAR processor."""
    # Synthetic phase dataset
    ds = xr.Dataset({'phase': (['lat', 'lon'], np.random.randn(10,10))}, coords={'lat': np.linspace(30,31,10), 'lon': np.linspace(0,1,10)})
    metadata = {'source': 'insar', 'bbox': (30,31,0,1), 'timestamp': np.datetime64('now')}
    raw_data = RawData(metadata, ds)
    proc = InSARPreprocessor()
    grid = proc.process(raw_data)
    assert grid.units == 'm'


def test_dask_preprocessor(synthetic_raw_data):
    """Test Dask parallel wrapper."""
    proc = GravityPreprocessor()
    dask_proc = DaskPreprocessor(proc, n_workers=1)  # Small for test
    grid = dask_proc.process(synthetic_raw_data)
    assert isinstance(grid, ProcessedGrid)
    # Check metadata has parallel info
    assert 'parallel_processing' in grid.ds.attrs


def test_preprocessing_manager_full(synthetic_raw_data):
    """Test manager full pipeline."""
    manager = PreprocessingManager()
    grid = manager.preprocess_data(synthetic_raw_data, modality='gravity')
    assert isinstance(grid, ProcessedGrid)


def test_preprocessing_manager_pipeline(synthetic_raw_data):
    """Test manager custom pipeline."""
    manager = PreprocessingManager()
    grid = manager.preprocess_data(synthetic_raw_data, modality='gravity', pipeline=['filter', 'grid'])
    assert isinstance(grid, ProcessedGrid)


def test_preprocessing_manager_config(tmp_path):
    """Test config loading."""
    config_path = tmp_path / 'test_config.yaml'
    config_data = {'preprocessing': {'grid_resolution': 0.05, 'filters': ['noise']}}
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    manager = PreprocessingManager(str(config_path))
    assert manager.config['preprocessing']['grid_resolution'] == 0.05


def test_preprocessing_errors():
    """Test error handling."""
    manager = PreprocessingManager()
    with pytest.raises(PreprocessingError):
        manager.preprocess_data(RawData({'source': 'unknown'}, np.array([])), modality='unknown')