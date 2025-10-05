import numpy as np
import xarray as xr
import numpy.testing as npt

from gam.modeling.data_structures import ProcessedGrid, InversionResults
from gam.modeling.anomaly_detection import detect_anomalies
from gam.modeling.fusion import fuse_grids
from gam.modeling.joint import JointInverter
from gam.modeling.mesh import MeshGenerator
from gam.core.geodesy import geodetic_to_projected, ensure_crs


def make_simple_grid(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, crs: str = "EPSG:4326"):
    """
    Helper: create an xarray.DataArray with coords and CRS suitable for ProcessedGrid.
    """
    da = xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    da.attrs["crs"] = crs
    return da


def test_detect_anomalies_percentile_single_peak():
    # Create a 5x5 grid with a single high anomaly at center
    lat = np.linspace(10.0, 14.0, 5)
    lon = np.linspace(-120.0, -116.0, 5)
    base = np.zeros((5, 5), dtype=float)
    base[2, 2] = 10.0  # strong anomaly
    model_da = make_simple_grid(base, lat, lon)
    uncertainty_da = xr.zeros_like(model_da) + 0.1

    inv = InversionResults(model=model_da, uncertainty=uncertainty_da, metadata={"test": True})
    inv.validate()

    # Detect anomalies above 90th percentile
    df = detect_anomalies(inv, p=90)
    # Expect exactly one anomaly at center
    assert len(df) == 1
    detected = df.iloc[0]
    npt.assert_allclose(detected["anomaly_score"], 10.0, atol=1e-8)
    npt.assert_allclose(detected["lat"], lat[2], atol=1e-8)
    npt.assert_allclose(detected["lon"], lon[2], atol=1e-8)


def test_detect_anomalies_no_anomalies_returns_empty():
    # Uniform grid -> no anomalies for high percentile
    lat = np.linspace(0.0, 4.0, 5)
    lon = np.linspace(0.0, 4.0, 5)
    base = np.ones((5, 5)) * 5.0
    model_da = make_simple_grid(base, lat, lon)
    uncertainty_da = xr.zeros_like(model_da) + 0.1

    inv = InversionResults(model=model_da, uncertainty=uncertainty_da, metadata={})
    inv.validate()

    df = detect_anomalies(inv, p=99)
    # No values exceed the 99th percentile in a uniform grid
    assert df.shape[0] == 0
    # DataFrame has correct columns and dtypes
    assert list(df.columns) == ["lat", "lon", "anomaly_score"]


def test_fuse_grids_mean_and_mad():
    # Two simple 2x2 grids with known numbers
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    g1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    g2 = np.array([[1.0, 3.0], [3.0, 5.0]])

    da1 = make_simple_grid(g1, lat, lon)
    da1.attrs["crs"] = "EPSG:4326"
    da2 = make_simple_grid(g2, lat, lon)
    da2.attrs["crs"] = "EPSG:4326"

    pg1 = ProcessedGrid(da1)
    pg2 = ProcessedGrid(da2)
    pg1.validate()
    pg2.validate()

    inv = fuse_grids([pg1, pg2])
    # Expected model: mean across modalities
    expected_model = (g1 + g2) / 2.0
    npt.assert_allclose(inv.model.values, expected_model, atol=1e-12)

    # Compute expected MAD manually:
    stacked = np.stack([g1, g2], axis=0)  # (modality, lat, lon)
    median_vals = np.median(stacked, axis=0)
    deviations = np.abs(stacked - median_vals)
    mad = np.median(deviations, axis=0)
    expected_uncertainty = mad * 1.4826
    npt.assert_allclose(inv.uncertainty.values, expected_uncertainty, atol=1e-12)

    # Metadata checks
    assert inv.metadata.get("fusion_method") == "robust_mean_mad"
    assert inv.metadata.get("n_modalities") == 2


def test_joint_inverter_fuse_behavior():
    # JointInverter.fuse in joint.py has a simple identity for single model and weighted avg fallback
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    grid_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    grid_b = np.array([[2.0, 3.0], [4.0, 5.0]])
    da_a = make_simple_grid(grid_a, lat, lon)
    da_b = make_simple_grid(grid_b, lat, lon)

    inv_a = InversionResults(model=da_a, uncertainty=xr.zeros_like(da_a) + 0.1, metadata={})
    inv_b = InversionResults(model=da_b, uncertainty=xr.zeros_like(da_b) + 0.1, metadata={})

    ji = JointInverter()
    # Single model -> returns the model itself (xarray.DataArray)
    out_single = ji.fuse([inv_a])
    # For single model the same DataArray is returned (or equal values)
    npt.assert_allclose(out_single.values, da_a.values, atol=1e-12)

    # Multiple models with explicit weights -> weighted average
    weights = np.array([0.25, 0.75])
    out_weighted = ji.fuse([inv_a, inv_b], weights=weights)
    expected = weights[0] * da_a.values + weights[1] * da_b.values
    npt.assert_allclose(out_weighted, expected, atol=1e-12)


def test_mesh_generator_transformer_and_project_coords():
    # Verify MeshGenerator projects lat/lon using pyproj transformer consistently with geodetic_to_projected
    lat = np.array([10.0, 11.0])
    lon = np.array([-120.0, -119.0])
    mg = MeshGenerator()
    # Use a common projected CRS for testing
    target_crs = ensure_crs("EPSG:3857")
    # Setup transformer using the public method (private method _setup_transformer is used)
    mg._setup_transformer("EPSG:4326", "EPSG:3857")
    x_mg, y_mg = mg._project_coords(lon, lat)

    x_ref, y_ref = geodetic_to_projected(lon, lat, target_crs)
    # Ensure shapes match and values are close
    assert x_mg.shape == x_ref.shape
    assert y_mg.shape == y_ref.shape
    npt.assert_allclose(x_mg, x_ref, rtol=1e-6, atol=1e-6)
    npt.assert_allclose(y_mg, y_ref, rtol=1e-6, atol=1e-6)