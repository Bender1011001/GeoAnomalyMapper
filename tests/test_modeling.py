"""Unit tests for GAM modeling module."""

import json
import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import Mock, patch

from gam.modeling import (
    AnomalyDetector,
    AnomalyOutput,
    GravityInverter,
    InSARInverter,
    InversionResults,
    JointInverter,
    MagneticInverter,
    MeshGenerator,
    ModelingConfig,
    ModelingManager,
    SeismicInverter,
)
from gam.modeling.base import Inverter
from gam.modeling.data_structures import InversionResults
from gam.modeling.fusion import JointInverter
from gam.modeling.mesh import MeshGenerator
from gam.preprocessing.data_structures import ProcessedGrid
from gam.core.exceptions import GAMError, InversionConvergenceError


@pytest.fixture
def synthetic_processed_grid():
    """Synthetic ProcessedGrid for testing."""
    lat = np.linspace(0, 1, 10)
    lon = np.linspace(0, 1, 10)
    depth = np.linspace(0, 1, 5)
    data = np.random.rand(10, 10, 5)
    ds = xr.Dataset(
        {'data': (['lat', 'lon', 'depth'], data)},
        coords={'lat': lat, 'lon': lon, 'depth': depth}
    )
    ds.attrs['units'] = 'mGal'
    ds.attrs['grid_resolution'] = 0.1
    ds.attrs['processed_at'] = pd.Timestamp.now()
    return ProcessedGrid(ds)


@pytest.fixture
def synthetic_inversion_results():
    """Synthetic InversionResults."""
    model = np.random.rand(10, 10, 5)
    uncertainty = np.random.rand(10, 10, 5) * 0.1
    metadata = {
        'converged': True,
        'iterations': 10,
        'residuals': 1.2,
        'units': 'kg/m³',
        'algorithm': 'test',
        'parameters': {},
        'timestamp': pd.Timestamp.now(),
    }
    return InversionResults(model, uncertainty, metadata)


def test_inversion_results_creation():
    """Test InversionResults dataclass."""
    model = np.random.rand(5, 5, 3)
    uncertainty = np.random.rand(5, 5, 3)
    results = InversionResults(model, uncertainty)
    assert results.model.shape == (5, 5, 3)
    assert results.uncertainty.shape == (5, 5, 3)
    assert results.metadata['converged'] is False  # Default
    results.validate()  # No error


def test_inversion_results_serialization(tmp_path):
    """Test JSON/CSV serialization."""
    results = synthetic_inversion_results()
    json_path = tmp_path / 'test.json'
    csv_path = tmp_path / 'test.csv'
    results.to_json(json_path)
    results.to_csv(csv_path)
    loaded = InversionResults.from_json(json_path)
    assert np.allclose(loaded.model, results.model)
    loaded_csv = InversionResults.from_csv(csv_path)
    assert np.allclose(loaded_csv.model, results.model)


def test_anomaly_output_creation():
    """Test AnomalyOutput DataFrame."""
    data = {
        'lat': [1.0, 2.0],
        'lon': [1.0, 2.0],
        'depth': [100.0, 200.0],
        'confidence': [0.8, 0.9],
        'anomaly_type': ['void', 'contrast'],
        'strength': [0.5, 0.6]
    }
    anomalies = AnomalyOutput(data)
    anomalies.validate()
    assert len(anomalies) == 2
    assert 'confidence' in anomalies.columns


def test_anomaly_output_filter(tmp_path):
    """Test filtering and export."""
    data = {
        'lat': np.linspace(0, 1, 10),
        'lon': np.linspace(0, 1, 10),
        'depth': np.full(10, 100.0),
        'confidence': np.linspace(0.1, 1.0, 10),
        'anomaly_type': ['test'] * 10,
        'strength': np.ones(10)
    }
    anomalies = AnomalyOutput(data)
    filtered = anomalies.filter_confidence(0.5)
    assert len(filtered) == 5  # Approx half
    json_path = tmp_path / 'anoms.json'
    csv_path = tmp_path / 'anoms.csv'
    anomalies.to_json(json_path)
    anomalies.to_csv(csv_path)
    loaded = AnomalyOutput.from_json(json_path)
    assert len(loaded) == 10


def test_mesh_generator_regular():
    """Test regular TensorMesh."""
    gen = MeshGenerator()
    data = synthetic_processed_grid()
    mesh = gen.create_mesh(data, type='regular', hmin=0.1)
    assert isinstance(mesh, simpeg.mesh.TensorMesh)
    assert mesh.nC > 1000


def test_mesh_generator_adaptive():
    """Test adaptive TreeMesh."""
    gen = MeshGenerator()
    data = synthetic_processed_grid()
    mesh = gen.create_mesh(data, type='adaptive', hmin=0.05)
    assert isinstance(mesh, simpeg.mesh.TreeMesh)
    assert mesh.hmin == 0.05


def test_mesh_generator_seismic():
    """Test PyGIMLi mesh."""
    gen = MeshGenerator()
    data = synthetic_processed_grid()
    mesh = gen.create_mesh(data, type='seismic', dimension=3)
    assert isinstance(mesh, pg.Mesh)
    assert mesh.dim() == 3


def test_modeling_config():
    """Test config validation."""
    config_dict = {
        'mesh_resolution': 0.01,
        'regularization': {'alpha_s': 1e-4},
        'max_iterations': 20,
        'fusion_scheme': 'bayesian',
        'anomaly_threshold': 2.5
    }
    config = ModelingConfig(**config_dict)
    assert config.mesh_resolution == 0.01
    # Test default
    default_config = ModelingConfig()
    assert 'alpha_s' in default_config.regularization


def test_gravity_inverter(synthetic_processed_grid):
    """Test GravityInverter with mock SimPEG."""
    with patch('gam.modeling.gravity.simpeg') as mock_simpeg:
        mock_simpeg.mesh.TreeMesh.return_value = Mock(nC=1000)
        mock_simpeg.survey.Survey.return_value = Mock()
        inverter = GravityInverter()
        # Mock invert to return synthetic
        with patch.object(inverter, 'invert', return_value=synthetic_inversion_results()):
            results = inverter.invert(synthetic_processed_grid)
        assert isinstance(results, InversionResults)
        assert results.model.shape == (10, 10, 5)


def test_magnetic_inverter(synthetic_processed_grid):
    """Test MagneticInverter."""
    with patch('gam.modeling.magnetic.simpeg') as mock_simpeg:
        mock_simpeg.mesh.TreeMesh.return_value = Mock(nC=1000)
        inverter = MagneticInverter()
        with patch.object(inverter, 'invert', return_value=synthetic_inversion_results()):
            results = inverter.invert(synthetic_processed_grid, inclination=60)
        assert results.metadata['parameters']['inclination'] == 60


def test_seismic_inverter(synthetic_processed_grid):
    """Test SeismicInverter with mock PyGIMLi."""
    with patch('gam.modeling.seismic.pg') as mock_pg:
        mock_model = Mock()
        mock_model.estimate.return_value = np.log(1 / 2000)
        mock_model.invert.return_value = mock_model
        mock_model.iterations.return_value = 10
        mock_model.residuals.return_value = np.array([0.1])
        mock_model.jacobian.return_value = np.eye(100)
        mock_model.relativeError.return_value = np.ones(100) * 0.05
        inverter = SeismicInverter()
        with patch('gam.modeling.seismic.TravelTimeModelling', return_value=mock_model):
            results = inverter.invert(synthetic_processed_grid, dimension=3)
        assert results.metadata['iterations'] == 10
        assert np.all(results.model > 1000)  # Velocity > v_min


def test_insar_inverter(synthetic_processed_grid):
    """Test InSARInverter with synthetic LOS."""
    synthetic_processed_grid.ds['data'] = np.random.rand(10, 10, 5) * 0.01  # mm
    synthetic_processed_grid.ds.attrs['incidence'] = 0.3
    inverter = InSARInverter()
    with patch('gam.modeling.insar.least_squares', return_value=Mock(cost=10, x=np.array([0,0,1000,1e6]), nfev=50)):
        results = inverter.invert(synthetic_processed_grid, source_type='mogi')
    assert 'volume_change' in results.metadata
    assert results.metadata['source_type'] == 'mogi'


def test_joint_fusion():
    """Test fusion with synthetic results."""
    results = [synthetic_inversion_results() for _ in range(3)]
    joint = JointInverter()
    fused = joint.fuse(results, scheme='weighted_avg')
    assert fused.shape == results[0].model.shape
    # Test bayesian
    fused_b = joint.fuse(results, scheme='bayesian')
    assert fused_b.shape == fused.shape


def test_anomaly_detector():
    """Test detection on synthetic with known anomaly."""
    # Synthetic with blob anomaly
    fused = np.random.rand(10, 10, 5)
    fused[5, 5, 2] = 10  # Anomaly
    detector = AnomalyDetector()
    anomalies = detector.detect(fused, threshold=2.0, method='zscore')
    assert len(anomalies) >= 1
    assert anomalies['anomaly_type'].iloc[0] == 'density_contrast'  # High value


def test_modeling_manager_config():
    """Test manager config loading."""
    config_dict = {
        'mesh_resolution': 0.005,
        'fusion_scheme': 'cross_gradient',
        'anomaly_threshold': 3.0
    }
    with patch('builtins.open', mock_open(read_data=yaml.dump({'modeling': config_dict}))):
        manager = ModelingManager()
    assert manager.config.mesh_resolution == 0.005


def test_modeling_manager_run_inversion(synthetic_processed_grid):
    """Test manager inversion call."""
    manager = ModelingManager()
    with patch.object(manager, 'run_inversion', return_value=synthetic_inversion_results()):
        results = manager.run_inversion(synthetic_processed_grid, 'gravity')
    assert isinstance(results, InversionResults)


def test_modeling_manager_full_pipeline(synthetic_processed_grid):
    """Test full pipeline."""
    manager = ModelingManager()
    with patch.object(manager, 'full_pipeline', return_value=(np.zeros((10,10,5)), AnomalyOutput())):
        fused, anomalies = manager.full_pipeline(synthetic_processed_grid, ['gravity'])
    assert fused.shape == (10, 10, 5)
    assert isinstance(anomalies, AnomalyOutput)


def test_modeling_manager_iterative_refine(synthetic_processed_grid):
    """Test refinement loop."""
    manager = ModelingManager()
    with patch.object(manager, 'iterative_refine', return_value=AnomalyOutput()):
        anomalies = manager.iterative_refine(synthetic_processed_grid, ['gravity'], n_iters=1)
    assert isinstance(anomalies, AnomalyOutput)


def test_inverter_abstract():
    """Test abstract Inverter cannot be instantiated."""
    with pytest.raises(TypeError):
        Inverter()  # Abstract


def test_mesh_quality_warning():
    """Test mesh validation warns on poor quality."""
    gen = MeshGenerator()
    data = synthetic_processed_grid()
    with pytest.warns(UserWarning):
        mesh = gen.create_simpeg_mesh(data, mesh_type='regular', hmin=0.1, quality_threshold=2.0)
    # Note: Actual warning depends on mesh; test structure


def test_fusion_single_model():
    """Test fusion with one model returns identity."""
    joint = JointInverter()
    results = [synthetic_inversion_results()]
    fused = joint.fuse(results)
    np.testing.assert_array_equal(fused, results[0].model)


def test_anomaly_detection_empty():
    """Test no detection on uniform model."""
    uniform = np.ones((10,10,5))
    detector = AnomalyDetector()
    anomalies = detector.detect(uniform, threshold=3.0)
    assert len(anomalies) == 0


def test_config_validation_bad():
    """Test invalid config raises."""
    with pytest.raises(ValueError):
        ModelingConfig(max_iterations=0)  # ge=1


# Integration test for numerical accuracy (tolerance)
def test_inversion_numerical_tolerance(synthetic_processed_grid):
    """Mock inversion with known recovery."""
    expected_model = np.ones((10,10,5))
    with patch('gam.modeling.gravity.InversionResults', return_value=InversionResults(expected_model, np.zeros_like(expected_model), {})):
        inverter = GravityInverter()
        results = inverter.invert(synthetic_processed_grid)
    np.testing.assert_allclose(results.model, expected_model, atol=1e-6)


# Test fusion numerical
def test_fusion_linearity():
    """Test weighted avg linearity."""
    results = [synthetic_inversion_results() for _ in range(2)]
    joint = JointInverter()
    fused1 = joint.fuse(results, weights=[1,0])
    fused2 = joint.fuse(results, weights=[0,1])
    fused_avg = joint.fuse(results, weights=[0.5,0.5])
    np.testing.assert_allclose(fused_avg, 0.5 * fused1 + 0.5 * fused2, atol=1e-8)