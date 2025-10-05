import json
from datetime import datetime
import os
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import xarray as xr
import pytest

_repo_root = pathlib.Path(__file__).resolve().parents[2]  # GeoAnomalyMapper directory

# Load exceptions module and register under its package name so imports like
# "from gam.core.exceptions import DataValidationError" succeed when other modules import it.
_exceptions_path = _repo_root / "gam" / "core" / "exceptions.py"
_spec = importlib.util.spec_from_file_location("gam.core.exceptions", str(_exceptions_path))
_ex_mod = importlib.util.module_from_spec(_spec)
# register before execution so submodules can import it
sys.modules["gam.core.exceptions"] = _ex_mod
_spec.loader.exec_module(_ex_mod)
DataValidationError = _ex_mod.DataValidationError

# Load modeling data structures and register under the package name to ensure
# dataclasses and relative imports resolve correctly.
_ds_path = _repo_root / "gam" / "modeling" / "data_structures.py"
_spec2 = importlib.util.spec_from_file_location("gam.modeling.data_structures", str(_ds_path))
_ds_mod = importlib.util.module_from_spec(_spec2)
sys.modules["gam.modeling.data_structures"] = _ds_mod
_spec2.loader.exec_module(_ds_mod)
ProcessedGrid = _ds_mod.ProcessedGrid
InversionResults = _ds_mod.InversionResults
AnomalyOutput = _ds_mod.AnomalyOutput

# Load base.py for Inverter ABC and register under its package name.
_base_path = _repo_root / "gam" / "modeling" / "base.py"
_spec3 = importlib.util.spec_from_file_location("gam.modeling.base", str(_base_path))
_base_mod = importlib.util.module_from_spec(_spec3)
sys.modules["gam.modeling.base"] = _base_mod
_spec3.loader.exec_module(_base_mod)
Inverter = _base_mod.Inverter


def make_dataarray(shape=(2, 3), coords=None, crs="EPSG:4326"):
    arr = np.arange(np.prod(shape)).reshape(shape).astype(float)
    if coords is None:
        coords = {"lat": np.linspace(0, 1, shape[0]), "lon": np.linspace(0, 1, shape[1])}
    da = xr.DataArray(arr, coords=coords, dims=["lat", "lon"], attrs={"crs": crs})
    return da


# -------------------------
# Tests for ProcessedGrid
# -------------------------
@pytest.mark.parametrize(
    "data,expected_exception",
    [
        (make_dataarray(), None),  # valid
        (xr.Dataset({"a": make_dataarray()}), DataValidationError),  # wrong type
        (xr.DataArray(np.ones((2, 2)), coords={"lat": [0, 1]}, dims=["lat", "lon"]), DataValidationError),  # missing lon
    ],
)
def test_processed_grid_validate_parametrized(data, expected_exception):
    if expected_exception is None:
        pg = ProcessedGrid(data)
        # should not raise
        pg.validate()
    else:
        with pytest.raises(DataValidationError):
            ProcessedGrid(data).validate()


def test_processed_grid_missing_crs_raises():
    da = make_dataarray()
    da.attrs.pop("crs", None)
    with pytest.raises(DataValidationError):
        ProcessedGrid(da).validate()


# -------------------------
# Tests for InversionResults
# -------------------------
def test_inversionresults_validate_and_serialization_roundtrip(tmp_path):
    model = make_dataarray((2, 2))
    uncertainty = xr.DataArray(np.full((2, 2), 0.1), coords=model.coords, dims=model.dims)
    meta = {"units": "kg/m3", "algorithm": "test"}

    res = InversionResults(model=model, uncertainty=uncertainty, metadata=meta)
    # validate should pass
    res.validate()

    d = res.to_dict()
    assert "model" in d and "uncertainty" in d and "metadata" in d
    # recreate from dict
    res2 = InversionResults.from_dict(d)
    assert isinstance(res2, InversionResults)
    assert res2.model.shape == res.model.shape
    assert res2.metadata == meta

    # repr shows shape
    assert "InversionResults" in repr(res2)


@pytest.mark.parametrize(
    "modify,exc",
    [
        (lambda m, u: (None, u), DataValidationError),  # model None
        (lambda m, u: (m, None), DataValidationError),  # uncertainty None
        (lambda m, u: (m.isel(lat=0), u), DataValidationError),  # shape mismatch
    ],
)
def test_inversionresults_invalid_cases(modify, exc):
    model = make_dataarray((2, 2))
    uncertainty = xr.DataArray(np.full((2, 2), 0.1), coords=model.coords, dims=model.dims)
    m2, u2 = modify(model, uncertainty)
    with pytest.raises(DataValidationError):
        InversionResults(model=m2, uncertainty=u2, metadata={}).validate()


def test_inversionresults_coords_mismatch_raises():
    model = make_dataarray((2, 2))
    # Create uncertainty with different coords
    uncertainty = xr.DataArray(np.full((2, 2), 0.1), coords={"lat": [0, 1], "lon": [10, 20]}, dims=model.dims)
    with pytest.raises(DataValidationError):
        InversionResults(model=model, uncertainty=uncertainty, metadata={}).validate()


# -------------------------
# Tests for AnomalyOutput
# -------------------------
def make_valid_anomaly_df(n=3):
    # Create sensible default anomaly records for testing.
    # Ensure confidence values are strictly within [0, 1] for any n.
    data = {
        "lat": [40.0 + i * 0.01 for i in range(n)],
        "lon": [-100.0 + i * 0.01 for i in range(n)],
        "depth": [500.0 + i * 10 for i in range(n)],
        "confidence": list(np.linspace(0.1, 0.9, n)),
        "anomaly_type": ["void"] * n,
        "strength": [-0.5 + i * 0.1 for i in range(n)],
    }
    return data


def test_anomalyoutput_valid_and_repr_and_timestamp():
    data = make_valid_anomaly_df(2)
    ao = AnomalyOutput(data)
    assert ao.validate() is True
    assert "timestamp" in ao.columns
    s = repr(ao)
    assert "AnomalyOutput" in s


def test_anomalyoutput_missing_required_columns_raises():
    df = pd.DataFrame({"lat": [1.0], "lon": [2.0]})
    with pytest.raises(DataValidationError):
        AnomalyOutput(df)


def test_anomalyoutput_confidence_range_and_depth_and_duplicates():
    data = make_valid_anomaly_df(2)
    # invalid confidence
    data_bad_conf = data.copy()
    data_bad_conf["confidence"] = [1.5, 0.5]
    with pytest.raises(DataValidationError):
        AnomalyOutput(data_bad_conf)

    # negative depth
    data_bad_depth = data.copy()
    data_bad_depth["depth"] = [-1.0, 100.0]
    with pytest.raises(DataValidationError):
        AnomalyOutput(data_bad_depth)

    # duplicates
    data_dup = make_valid_anomaly_df(2)
    # make both rows identical
    for k in data_dup:
        data_dup[k] = [data_dup[k][0], data_dup[k][0]]
    with pytest.raises(DataValidationError):
        AnomalyOutput(data_dup)


def test_filter_confidence_and_invalid_arg():
    data = make_valid_anomaly_df(4)
    ao = AnomalyOutput(data)
    with pytest.raises(ValueError):
        ao.filter_confidence(-0.1)
    filtered = ao.filter_confidence(0.5)
    assert isinstance(filtered, AnomalyOutput)
    # all returned must have confidence >= 0.5
    if len(filtered) > 0:
        assert filtered["confidence"].min() >= 0.5


def test_compute_confidence_zscore_zero_std():
    # strengths all equal => std == 0 -> confidence = 0.5
    data = make_valid_anomaly_df(3)
    data["strength"] = [1.0, 1.0, 1.0]
    ao = AnomalyOutput(data)
    ao.compute_confidence(method="zscore")
    assert np.allclose(ao["confidence"].values, 0.5)


def test_compute_confidence_percentile():
    data = make_valid_anomaly_df(4)
    ao = AnomalyOutput(data)
    ao.compute_confidence(method="percentile")
    # confidence values should be in [0,1]
    assert ao["confidence"].min() >= 0
    assert ao["confidence"].max() <= 1


def test_compute_confidence_unknown_method_raises():
    data = make_valid_anomaly_df(2)
    ao = AnomalyOutput(data)
    with pytest.raises(ValueError):
        ao.compute_confidence(method="unknown_method")


def test_serialization_csv_json_roundtrip(tmp_path):
    data = make_valid_anomaly_df(3)
    # include modality_contributions dict to exercise serialization logic
    data["modality_contributions"] = [{"grav": 0.6}, {"grav": 0.4}, {"grav": 0.5}]
    ao = AnomalyOutput(data)

    csv_path = tmp_path / "anoms.csv"
    ao.to_csv(str(csv_path), index=False)
    ao2 = AnomalyOutput.from_csv(str(csv_path))
    assert isinstance(ao2, AnomalyOutput)
    assert len(ao2) == len(ao)

    json_path = tmp_path / "anoms.json"
    ao.to_json(str(json_path))
    ao3 = AnomalyOutput.from_json(str(json_path))
    assert isinstance(ao3, AnomalyOutput)
    assert len(ao3) == len(ao)


# add_cluster_labels requires sklearn; test guarded by import
def test_add_cluster_labels_if_available():
    data = make_valid_anomaly_df(5)
    ao = AnomalyOutput(data)
    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except Exception:
        pytest.skip("scikit-learn not available; skipping clustering test")
    # should not raise
    ao.add_cluster_labels(n_clusters=2)
    assert "cluster_id" in ao.columns


# -------------------------
# Tests for Inverter ABC
# -------------------------
def test_inverter_is_abstract():
    with pytest.raises(TypeError):
        Inverter()  # cannot instantiate abstract base class


def test_concrete_inverter_can_be_instantiated_and_used():
    # define a minimal concrete inverter
    class ConcreteInverter(Inverter):
        def invert(self, data: ProcessedGrid, **kwargs):
            # return a simple InversionResults using the input grid
            model = data.data.copy()
            uncertainty = xr.zeros_like(model) + 1.0
            return InversionResults(model=model, uncertainty=uncertainty, metadata={"converged": True})

        def fuse(self, models, **kwargs):
            # simple average fuse: models are InversionResults -> average model values
            arrays = [m.model.values for m in models]
            return np.mean(arrays, axis=0)

    # Prepare ProcessedGrid
    da = make_dataarray((2, 2))
    pg = ProcessedGrid(da)
    pg.validate()

    inv = ConcreteInverter()
    res = inv.invert(pg)
    assert isinstance(res, InversionResults)
    # test fuse
    fused = inv.fuse([res, res])
    assert isinstance(fused, np.ndarray)
    assert fused.shape == res.model.shape