"""Tests for the granule-major / process-parallel OPERA read path.

The network parts can't be unit-tested, but the index math and the
many-AOIs-from-one-open contract can — and those are where correctness lives.
"""
import numpy as np
import pytest

from deformation_intel.opera import (
    read_many_aois,
    read_windows_parallel,
    window_indices,
)


def test_window_indices_centre():
    x = np.arange(0, 1000, 10.0)      # 0..990 m
    y = np.arange(1000, 0, -10.0)     # descending, like a north-up raster
    sl = window_indices(x, y, cx=500.0, cy=500.0, half=50.0)
    assert sl is not None
    ysl, xsl = sl
    assert x[xsl].min() >= 450 and x[xsl].max() <= 550
    assert y[ysl].min() >= 450 and y[ysl].max() <= 550


def test_window_indices_off_grid_returns_none():
    x = np.arange(0, 1000, 10.0)
    y = np.arange(1000, 0, -10.0)
    assert window_indices(x, y, cx=99999.0, cy=500.0, half=50.0) is None
    assert window_indices(x, y, cx=500.0, cy=-99999.0, half=50.0) is None


def test_window_indices_width_scales_with_half():
    x = np.arange(0, 1000, 10.0)
    y = np.arange(1000, 0, -10.0)
    _, x_small = window_indices(x, y, 500.0, 500.0, 20.0)
    _, x_big = window_indices(x, y, 500.0, 500.0, 200.0)
    n_small = len(x[x_small])
    n_big = len(x[x_big])
    assert n_big > n_small


class _FakeDS:
    """Minimal stand-in for an opened OPERA granule."""

    def __init__(self, n=100):
        self.closed = False
        self.opens = 0
        import xarray as xr
        xs = np.arange(n) * 30.0
        ys = (np.arange(n) * -30.0)[::-1]
        disp = np.ones((n, n), "float32")
        coh = np.ones((n, n), "float32")
        self._ds = xr.Dataset(
            {"displacement": (("y", "x"), disp),
             "temporal_coherence": (("y", "x"), coh)},
            coords={"x": xs, "y": ys},
        )
        self._ds["spatial_ref"] = 0
        self._ds["spatial_ref"].attrs["crs_wkt"] = "EPSG:32615"

    def __getitem__(self, k):
        return self._ds[k]

    @property
    def variables(self):
        return self._ds.variables

    def close(self):
        self.closed = True


def test_read_many_aois_opens_granule_once(monkeypatch):
    """The whole point of granule-major: N AOIs must cost ONE open."""
    ds = _FakeDS()
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return ds

    # patch the projection so lat/lon map into the fake grid
    import deformation_intel.opera as op

    class _T:
        @staticmethod
        def from_crs(a, b, always_xy=True):
            class _I:
                @staticmethod
                def transform(lon, lat):
                    return (lon, lat)
            return _I()

    monkeypatch.setattr(op, "Transformer", _T, raising=False)
    import pyproj
    monkeypatch.setattr(pyproj, "Transformer", _T)

    aois = {"a": (-600.0, 600.0), "b": (-900.0, 900.0), "c": (-1200.0, 1200.0)}
    out = read_many_aois(factory, aois, half=200.0, coherence_threshold=0.5)
    assert calls["n"] == 1, "granule must be opened exactly once for all AOIs"
    assert set(out) == {"a", "b", "c"}
    assert ds.closed is True


def test_read_windows_parallel_serial_fallback(monkeypatch):
    """workers<=1 must not spawn a pool (safe inside worker processes)."""
    import deformation_intel.opera as op
    seen = []

    def fake_read(task):
        seen.append(task[0])
        return ("disp", "x", "y", "crs")

    monkeypatch.setattr(op, "_pool_read", fake_read)
    out = op.read_windows_parallel(["u1", "u2"], 1.0, 2.0, 100.0, 0.6,
                                   workers=1)
    assert seen == ["u1", "u2"]
    assert len(out) == 2 and out[0][0] == "disp"


def test_build_aoi_cube_accepts_workers_kwarg():
    import inspect
    from deformation_intel.opera import build_aoi_cube
    sig = inspect.signature(build_aoi_cube)
    assert "workers" in sig.parameters
    assert sig.parameters["workers"].default == 1   # opt-in, no behaviour change
