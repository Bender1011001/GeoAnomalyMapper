import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from GeoAnomalyMapper.gam.visualization.tiles_builder import (
    reproject_wgs84_to_ecef,
    run_py3dtiles_convert,
)


def _as_float(x) -> float:
    """Convert scalar or 0-D numpy array to a Python float."""
    arr = np.asarray(x)
    return float(arr.reshape(()))


def test_reproject_wgs84_to_ecef_scalar_accuracy():
    # Skip cleanly if pyproj is not available in the environment
    pytest.importorskip("pyproj")

    x, y, z = reproject_wgs84_to_ecef(0.0, 0.0, 0.0)
    tol = 5e-3  # meters

    xf = _as_float(x)
    yf = _as_float(y)
    zf = _as_float(z)

    assert abs(xf - 6378137.0) <= tol, f"x expected ~6378137.0, got {xf}"
    assert abs(yf - 0.0) <= tol, f"y expected ~0.0, got {yf}"
    assert abs(zf - 0.0) <= tol, f"z expected ~0.0, got {zf}"


def test_reproject_wgs84_to_ecef_vectorization():
    # Also guard this vectorized path on pyproj presence
    pytest.importorskip("pyproj")

    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 0.0])
    h = np.array([0.0, 0.0])

    x, y, z = reproject_wgs84_to_ecef(lon, lat, h)

    assert x.shape == (2,), f"x shape expected (2,), got {x.shape}"
    assert y.shape == (2,), f"y shape expected (2,), got {y.shape}"
    assert z.shape == (2,), f"z shape expected (2,), got {z.shape}"

    assert np.all(np.isfinite(x)), "x contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"
    assert np.all(np.isfinite(z)), "z contains non-finite values"


def test_run_py3dtiles_convert_missing_binary_raises():
    # Ensure that when py3dtiles is not on PATH, we raise early and never invoke subprocess.run
    with patch("GeoAnomalyMapper.gam.visualization.tiles_builder.shutil.which", return_value=None) as mock_which, \
         patch("GeoAnomalyMapper.gam.visualization.tiles_builder.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError) as excinfo:
            run_py3dtiles_convert("dummy_input.xyz", "dummy_out")
        mock_run.assert_not_called()
        msg = str(excinfo.value).lower()
        assert "py3dtiles" in msg and "path" in msg, (
            f"Error message should mention 'py3dtiles' and 'PATH', got: {excinfo.value}"
        )
        mock_which.assert_called_once_with("py3dtiles")


def test_run_py3dtiles_convert_success_command_and_result(tmp_path: Path):
    # Create minimal valid input/output
    input_path = tmp_path / "points.xyz"
    input_path.write_text("")
    out_dir = tmp_path / "tiles_out"

    # Return a CompletedProcess-like object
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with patch("GeoAnomalyMapper.gam.visualization.tiles_builder.shutil.which", return_value="/usr/bin/py3dtiles"), \
         patch("GeoAnomalyMapper.gam.visualization.tiles_builder.subprocess.run", side_effect=fake_run) as mock_run:
        result = run_py3dtiles_convert(
            input_path,
            out_dir,
            threads=4,
            timeout=120,
            allow_overwrite=True,
        )

        # subprocess.run called once with a list command and correct kwargs
        assert mock_run.call_count == 1, "subprocess.run should be called exactly once"
        call_args, call_kwargs = mock_run.call_args

        # Command must be list and match the expected order exactly
        assert isinstance(call_args[0], list), "Command should be a list (shell=False semantics)"
        expected_cmd = [
            "py3dtiles",
            "convert",
            str(input_path),
            "--out",
            str(out_dir),
            "--jobs",
            "4",
            "--overwrite",
        ]
        assert call_args[0] == expected_cmd, f"Command mismatch.\nExpected: {expected_cmd}\nGot: {call_args[0]}"

        # Keyword arguments
        assert call_kwargs.get("cwd") == str(out_dir), f"cwd expected {str(out_dir)}, got {call_kwargs.get('cwd')}"
        assert call_kwargs.get("capture_output") is True, "capture_output should be True"
        assert call_kwargs.get("text") is True, "text should be True"
        assert float(call_kwargs.get("timeout")) == 120.0, f"timeout expected 120.0, got {call_kwargs.get('timeout')}"
        assert call_kwargs.get("check") is False, "check should be False"
        # Explicitly ensure shell was not requested
        assert call_kwargs.get("shell", False) is False, "shell should be False or omitted"

        # Result dictionary shape and values
        assert isinstance(result, dict), "Result should be a dict"
        for k in ("command", "returncode", "stdout_tail", "out_dir"):
            assert k in result, f"Missing key in result: {k}"
        assert result["returncode"] == 0, f"Expected returncode 0, got {result['returncode']}"


def test_run_py3dtiles_convert_timeout_raises(tmp_path: Path):
    # Prepare minimal input/output for validation before subprocess
    input_path = tmp_path / "points.xyz"
    input_path.write_text("")
    out_dir = tmp_path / "tiles_out"

    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["py3dtiles"], timeout=0.1)

    with patch("GeoAnomalyMapper.gam.visualization.tiles_builder.shutil.which", return_value="/usr/bin/py3dtiles"), \
         patch("GeoAnomalyMapper.gam.visualization.tiles_builder.subprocess.run", side_effect=raise_timeout):
        with pytest.raises(TimeoutError) as excinfo:
            run_py3dtiles_convert(input_path, out_dir, timeout=0.1)
        msg = str(excinfo.value).lower()
        assert ("timed out" in msg) or ("0.1" in msg) or ("py3dtiles" in msg), (
            f"TimeoutError message should mention the timeout and/or the command; got: {excinfo.value}"
        )