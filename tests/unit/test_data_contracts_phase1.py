import pytest
import xarray as xr
from pydantic import ValidationError

from GeoAnomalyMapper.gam.core.data_contracts import RawData, ProcessedGrid, InversionResult, Anomaly


def test_rawdata_valid():
    rd = RawData(
        source="gravity",
        bbox=(-120.0, 30.0, -119.0, 31.0),
        data={"a": 1},
        metadata={"units": "mGal"},
    )
    assert rd.source == "gravity"
    assert rd.bbox == (-120.0, 30.0, -119.0, 31.0)
    assert rd.data == {"a": 1}
    assert rd.metadata == {"units": "mGal"}


def test_rawdata_invalid_bbox_order():
    # min_lon > max_lon should raise a ValidationError
    with pytest.raises(ValidationError):
        RawData(
            source="gravity",
            bbox=(-119.0, 30.0, -120.0, 31.0),  # invalid: min_lon >= max_lon
            data={"a": 1},
            metadata={},
        )


def test_processedgrid_valid():
    ds = xr.Dataset(
        {"v": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [30.0, 31.0], "lon": [-120.0, -119.0]},
    )
    pg = ProcessedGrid(grid=ds)
    assert isinstance(pg.grid, xr.Dataset)
    assert "lat" in pg.grid.coords and "lon" in pg.grid.coords


def test_processedgrid_missing_coords():
    ds = xr.Dataset({"v": (("y", "x"), [[1.0]])}, coords={"y": [30.0], "x": [-120.0]})
    with pytest.raises(ValidationError):
        ProcessedGrid(grid=ds)


def test_inversionresult_dim_match():
    model = xr.Dataset(
        {"m": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [30.0, 31.0], "lon": [-120.0, -119.0]},
    )
    uncertainty = xr.Dataset(
        {"u": (("lat", "lon"), [[0.1, 0.1], [0.1, 0.1]])},
        coords={"lat": [30.0, 31.0], "lon": [-120.0, -119.0]},
    )
    inv = InversionResult(model=model, uncertainty=uncertainty, metadata={})
    assert isinstance(inv.model, xr.Dataset)
    assert isinstance(inv.uncertainty, xr.Dataset)


def test_inversionresult_dim_mismatch():
    model = xr.Dataset(
        {"m": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [30.0, 31.0], "lon": [-120.0, -119.0]},
    )
    # uncertainty has different dims/sizes
    uncertainty = xr.Dataset(
        {"u": (("lat", "lon", "depth"), [[ [0.1], [0.1] ], [ [0.1], [0.1] ] ])},
        coords={"lat": [30.0, 31.0], "lon": [-120.0, -119.0], "depth": [10.0]},
    )
    with pytest.raises(ValidationError):
        InversionResult(model=model, uncertainty=uncertainty, metadata={})


def test_anomaly_valid_range():
    a = Anomaly(
        latitude=30.0,
        longitude=-120.0,
        depth_meters=10.0,
        confidence=0.0,
        anomaly_type="positive",
    )
    assert a.latitude == 30.0
    assert a.longitude == -120.0
    assert a.depth_meters == 10.0
    assert 0.0 <= a.confidence <= 1.0
    assert a.anomaly_type == "positive"


def test_anomaly_invalid_confidence():
    with pytest.raises(ValidationError):
        Anomaly(
            latitude=30.0,
            longitude=-120.0,
            depth_meters=5.0,
            confidence=1.5,  # invalid
            anomaly_type="negative",
        )