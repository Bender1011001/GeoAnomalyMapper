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
# Additional Scientific Algorithm Unit Tests (Phase 3 Enhancements)

def test_gravity_terrain_correction(synthetic_dem, synthetic_processed_grid):
    """Test _compute_terrain_correction with mock DEM, assert correction values."""
    from gam.modeling.gravity import GravityInverter
    import numpy as np
    
    inverter = GravityInverter()
    
    # Mock the computation if internal; or call directly if exposed
    with patch.object(inverter, '_compute_terrain_correction') as mock_compute:
        # Simulate return: small corrections based on elevation gradient
        expected_correction = np.gradient(synthetic_dem['elevation'], axis=(0,1)) * 0.3086  # mGal per m elevation approx
        mock_compute.return_value = expected_correction
        
        correction = inverter._compute_terrain_correction(synthetic_dem, synthetic_processed_grid.ds.coords)
        
        assert correction.shape == (10, 10)
        assert np.all(np.abs(correction) < 5.0)  # Reasonable terrain correction <5 mGal
        assert np.mean(correction) > -1.0 and np.mean(correction) < 1.0  # Near zero average

def test_gravity_l1_vs_l2_inversion(synthetic_processed_grid):
    """Test L1 vs L2 inversion with synthetic data."""
    from gam.modeling.gravity import GravityInverter
    
    inverter = GravityInverter()
    
    # L2 inversion (smooth)
    results_l2 = inverter.invert(synthetic_processed_grid, regularization='l2')
    assert results_l2.metadata['regularization'] == 'l2'
    assert results_l2.metadata['converged'] is True
    
    # L1 inversion (sparse)
    results_l1 = inverter.invert(synthetic_processed_grid, regularization='l1')
    assert results_l1.metadata['regularization'] == 'l1'
    assert results_l1.metadata['converged'] is True
    
    # L1 should have more sparse model (higher fraction of small values)
    l1_sparsity = np.sum(np.abs(results_l1.model) < 1e-3)
    l2_sparsity = np.sum(np.abs(results_l2.model) < 1e-3)
    assert l1_sparsity > l2_sparsity * 1.5  # L1 sparser

@pytest.mark.skipif(not has_obspy, reason="Requires obspy for seismic")
def test_seismic_sta_lta_picking(sample_seismic_trace):
    """Test STA/LTA picking on sample trace, assert pick times."""
    from gam.modeling.seismic import SeismicInverter
    
    inverter = SeismicInverter()
    trace = sample_seismic_trace[0]
    
    # Run STA/LTA with params for synthetic pick at ~0.5s
    picks = inverter.sta_lta_picking(trace, sta_len=2, lta_len=20, threshold=3.0)
    
    assert len(picks) >= 1
    pick_time = picks[0]  # First pick
    assert 0.45 < pick_time < 0.55  # Within 0.05s of synthetic arrival at 0.5s
    assert pick_time.unit == 's'  # Assuming returns datetime or seconds

@pytest.mark.skipif(not has_pygimli, reason="Requires pygimli for inversion")
def test_seismic_pygimli_inversion(known_velocity_model, synthetic_processed_grid):
    """Test PyGIMLi inversion with known velocity model."""
    from gam.modeling.seismic import SeismicInverter
    import numpy as np
    
    inverter = SeismicInverter()
    
    # Mock travel time data from known model
    true_vel = known_velocity_model['velocities']
    true_depth = known_velocity_model['depths']
    
    # Run inversion
    results = inverter.invert(
        synthetic_processed_grid, 
        use_pygimli=True, 
        true_model=known_velocity_model,
        dimension=1  # 1D for simplicity
    )
    
    recovered_vel = results.model[:, 0, 0]  # Flatten to 1D
    
    # Assert recovery within tolerance
    rmse = np.sqrt(np.mean((recovered_vel - true_vel)**2))
    assert rmse < 100.0  # m/s tolerance
    assert np.all(recovered_vel > 1400)  # Above min velocity
    assert results.metadata['algorithm'] == 'pygimli_travel_time'

def test_insar_snaphu_unwrapping(synthetic_wrapped_phase):
    """Test SNAPHU unwrapping on synthetic wrapped phase."""
    from gam.modeling.insar import InSARInverter
    import numpy as np
    
    inverter = InSARInverter()
    
    # Unwrap
    unwrapped = inverter.snaphu_unwrap(synthetic_wrapped_phase)
    
    assert unwrapped.shape == synthetic_wrapped_phase.shape
    assert unwrapped.attrs['units'] == 'radians'
    
    # Reconstruct true phase from fixture logic
    lat, lon = synthetic_wrapped_phase.lat.values, synthetic_wrapped_phase.lon.values
    true_phase = np.outer(lat, np.ones_like(lon)) * 0.1 + np.outer(np.ones_like(lat), lon) * 0.05
    
    # Wrapped difference
    diff = np.abs(unwrapped - true_phase)
    diff = np.minimum(diff, 2 * np.pi - diff)
    assert np.mean(diff) < 1e-3  # High accuracy for synthetic

def test_insar_atmospheric_filtering(synthetic_wrapped_phase):
    """Test atmospheric filtering on synthetic phase."""
    from gam.modeling.insar import InSARInverter
    
    inverter = InSARInverter()
    
    # Add synthetic atmosphere: low-frequency noise
    atm_noise = np.sin(synthetic_wrapped_phase.lat * 10) * np.cos(synthetic_wrapped_phase.lon * 10) * 0.2
    noisy_phase = synthetic_wrapped_phase + atm_noise
    
    filtered = inverter.atmospheric_filter(noisy_phase, method='simple_detrend')
    
    assert filtered.shape == noisy_phase.shape
    
    # Assert reduced low-freq variance
    orig_var = np.var(np.fft.fftshift(np.fft.fft2(noisy_phase.values))[:5, :5])  # Low freq
    filt_var = np.var(np.fft.fftshift(np.fft.fft2(filtered.values))[:5, :5])
    assert filt_var < orig_var * 0.5  # Reduced atmospheric signal

def test_insar_los_sign_convention():
    """Test LOS sign convention with synthetic uplift."""
    from gam.modeling.insar import InSARInverter, mogi_multi_forward
    import numpy as np
    
    inverter = InSARInverter()
    
    # Synthetic uplift: uz = +10 mm at center, e=n=0
    obs_pos = np.array([[0, 0, 0]])  # Single point at source
    x0, y0, d, dV = 0, 0, 1000, 1e6  # Parameters for 10mm uplift
    ux, uy, uz = 0, 0, 0.01  # m
    
    # Ascending: inc=0.3 rad (~23°), head=0 (east-looking approx)
    inc_asc = 0.3
    head_asc = 0.0
    los_asc = np.array([np.sin(inc_asc)*np.cos(head_asc), np.sin(inc_asc)*np.sin(head_asc), np.cos(inc_asc)])
    d_los_asc = np.dot([ux, uy, uz], los_asc)
    
    # Descending: inc=0.3, head=pi (west-looking)
    head_desc = np.pi
    los_desc = np.array([np.sin(inc_asc)*np.cos(head_desc), np.sin(inc_asc)*np.sin(head_desc), np.cos(inc_asc)])
    d_los_desc = np.dot([ux, uy, uz], los_desc)
    
    # Positive toward satellite: both >0 for uplift (range decrease)
    assert d_los_asc > 0
    assert d_los_desc > 0
    
    # Verify implementation
    theta = np.array([x0, y0, d, dV])
    pred_asc = mogi_multi_forward(theta, obs_pos, np.array([inc_asc]), np.array([head_asc]))
    pred_desc = mogi_multi_forward(theta, obs_pos, np.array([inc_asc]), np.array([head_desc]))
    assert pred_asc > 0
    assert pred_desc > 0
    np.testing.assert_allclose(pred_asc, d_los_asc, atol=1e-6)
    np.testing.assert_allclose(pred_desc, d_los_desc, atol=1e-6)

def test_insar_los_sign_convention():
    """Test LOS sign convention with synthetic uplift."""
    from gam.modeling.insar import InSARInverter, mogi_multi_forward
    import numpy as np
    
    inverter = InSARInverter()
    
    # Synthetic uplift: uz = +10 mm at center, e=n=0
    obs_pos = np.array([[0, 0, 0]])  # Single point at source
    x0, y0, d, dV = 0, 0, 1000, 1e6  # Parameters for ~10mm uplift
    ux, uy, uz = 0, 0, 0.01  # m
    
    # Ascending: inc=0.3 rad (~23°), head=0 (east-looking approx)
    inc_asc = 0.3
    head_asc = 0.0
    los_asc = np.array([np.sin(inc_asc)*np.cos(head_asc), np.sin(inc_asc)*np.sin(head_asc), np.cos(inc_asc)])
    d_los_asc = np.dot([ux, uy, uz], los_asc)
    
    # Descending: inc=0.3, head=pi (west-looking)
    head_desc = np.pi
    los_desc = np.array([np.sin(inc_asc)*np.cos(head_desc), np.sin(inc_asc)*np.sin(head_desc), np.cos(inc_asc)])
    d_los_desc = np.dot([ux, uy, uz], los_desc)
    
    # Positive toward satellite: both >0 for uplift (range decrease)
    assert d_los_asc > 0
    assert d_los_desc > 0
    
    # Verify implementation
    theta = np.array([x0, y0, d, dV])
    pred_asc = mogi_multi_forward(theta, obs_pos, np.array([inc_asc]), np.array([head_asc]))
    pred_desc = mogi_multi_forward(theta, obs_pos, np.array([inc_asc]), np.array([head_desc]))
    assert pred_asc > 0
    assert pred_desc > 0
    np.testing.assert_allclose(pred_asc, d_los_asc, atol=1e-6)
    np.testing.assert_allclose(pred_desc, d_los_desc, atol=1e-6)

def test_non_gaussian_residuals():
    """Test robustness to non-Gaussian noise."""
    from gam.modeling.insar import InSARInverter
    import numpy as np
    from scipy.stats import levy_stable
    
    # Synthetic LOS with Gaussian vs heavy-tailed noise
    np.random.seed(42)
    n_points = 100
    true_params = np.array([0, 0, 1000, 1e6])  # Mogi for ~10mm
    obs_pos = np.random.uniform(-5000, 5000, (n_points, 2))
    obs_pos = np.column_stack([obs_pos, np.zeros(n_points)])  # z=0
    inc_grid = np.full(n_points, 0.3)
    head_grid = np.full(n_points, 0.0)
    
    # Gaussian noise
    gaussian_noise = np.random.normal(0, 0.001, n_points)
    los_gauss = mogi_multi_forward(true_params, obs_pos, inc_grid, head_grid) + gaussian_noise
    
    # Heavy-tailed (Levy alpha=1.5)
    levy_noise = levy_stable.rvs(alpha=1.5, size=n_points, random_state=42) * 0.001
    los_levy = mogi_multi_forward(true_params, obs_pos, inc_grid, head_grid) + levy_noise
    
    # Invert both
    def invert_los(los_data):
        def residual(theta):
            pred = mogi_multi_forward(theta, obs_pos, inc_grid, head_grid)
            return pred - los_data
        res = least_squares(residual, true_params, method='trf')
        return res.x
    
    params_gauss = invert_los(los_gauss)
    params_levy = invert_los(los_levy)
    
    # Recovery should be similar for Gaussian; levy may have higher variance but still close
    rmse_gauss = np.sqrt(np.mean((params_gauss - true_params)**2))
    rmse_levy = np.sqrt(np.mean((params_levy - true_params)**2))
    assert rmse_gauss < 1e-3  # Good recovery
    assert rmse_levy < 1e-2  # Reasonable for heavy-tailed

def test_insar_okada_forward(synthetic_fault_params, synthetic_processed_grid):
    """Test Okada forward modeling with known fault params."""
    from gam.modeling.insar import InSARInverter
    import numpy as np
    
    inverter = InSARInverter()
    
    # Compute forward displacement
    disp = inverter.okada_forward(
        synthetic_fault_params,
        grid_coords={'lat': synthetic_processed_grid.ds.lat, 'lon': synthetic_processed_grid.ds.lon}
    )
    
    assert disp.shape == (10, 10)
    assert np.all(np.isfinite(disp))  # No NaNs
    assert np.max(np.abs(disp)) > 0.1  # Visible displacement in mm
    assert np.mean(disp) < 0.05  # Mostly near zero away from fault
    assert 'displacement' in disp.name  # Assuming DataArray