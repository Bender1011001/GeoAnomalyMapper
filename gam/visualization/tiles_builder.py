# Copyright (c) GeoAnomalyMapper
# SPDX-License-Identifier: MIT
"""
Tiles Builder utilities.

This module provides helpers to:
- Convert a gridded anomaly field to scattered point samples suitable for 3D viewing.
- Optionally reproject lon/lat/height (EPSG:4979) to ECEF XYZ (EPSG:4978).
- Write a simple XYZ file and optionally invoke py3dtiles to build a basic tileset.

Notes
- py3dtiles expects coordinates in a projected or geocentric CRS with meters as units
  (e.g., EPSG:4978 WGS84 geocentric/ECEF). If your points are in lon/lat degrees,
  reproject to ECEF using reproject_llh_to_ecef() before building tiles.
- This module performs no I/O side effects except writing to the provided out_dir when
  build_3dtiles_from_points() is called.

See also
- Tiling conventions (2D WGS84 degree tiles) are defined elsewhere in the project, e.g.,
  GeoAnomalyMapper/gam/core/tiles.py. This module focuses on 3D point visualization artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Sequence, Union, Dict, List

import logging
import numpy as np
import subprocess
import shutil

logger = logging.getLogger(__name__)


def anomalies_to_points(
    anom: np.ndarray,
    bbox: Tuple[float, float, float, float],
    z_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert a 2D anomalies grid into point samples [lon_deg, lat_deg, height_m, value_0_1].

    Parameters
    ----------
    anom : np.ndarray
        2D array (ny, nx) of anomaly intensities. May contain NaNs.
    bbox : tuple(min_lon, min_lat, max_lon, max_lat)
        Geographic bounds in degrees for the grid coverage (WGS84/EPSG:4326).
        min_lon < max_lon and min_lat < max_lat are required.
    z_scale : float, optional
        Height scale factor in meters applied to the normalized anomaly value (default 1000.0).

    Returns
    -------
    np.ndarray
        Array of shape (N, 4) with columns:
        [lon_deg, lat_deg, height_m, value_0_1], where N = ny * nx.

    Notes
    -----
    - Normalization is computed nan-safely using finite values only; non-finite inputs are
      assigned a normalized value of 0.0.
    - The height is computed as normalized * z_scale (meters). If all inputs are non-finite,
      normalized values are 0.0 and height is 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> pts = anomalies_to_points(a, bbox=(-10.0, 50.0, -9.0, 51.0), z_scale=1000.0)
    >>> pts.shape
    (4, 4)
    >>> # Columns are [lon_deg, lat_deg, height_m, value_0_1]
    """
    if not isinstance(anom, np.ndarray) or anom.ndim != 2:
        raise ValueError("anom must be a 2D numpy array")
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        raise ValueError("bbox must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)")

    min_lon, min_lat, max_lon, max_lat = bbox
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")

    ny, nx = anom.shape

    # Build coordinate vectors and grids
    lons = np.linspace(float(min_lon), float(max_lon), num=int(nx), dtype=float)
    lats = np.linspace(float(min_lat), float(max_lat), num=int(ny), dtype=float)
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="xy")

    # Nan-safe normalization to [0, 1]
    finite_mask = np.isfinite(anom)
    norm = np.zeros_like(anom, dtype=float)
    if np.any(finite_mask):
        a_min = float(np.nanmin(anom[finite_mask]))
        a_max = float(np.nanmax(anom[finite_mask]))
        rng = a_max - a_min
        eps = 1e-12
        norm[finite_mask] = (anom[finite_mask] - a_min) / (rng + eps)
    # else: keep zeros if no finite values

    height = norm * float(z_scale)

    points = np.column_stack(
        (lon_grid.ravel().astype(float), lat_grid.ravel().astype(float), height.ravel().astype(float), norm.ravel().astype(float))
    )
    return points


def reproject_llh_to_ecef(points_llh: np.ndarray) -> np.ndarray:
    """
    Reproject lon/lat/height (EPSG:4979) to ECEF XYZ (EPSG:4978).

    Parameters
    ----------
    points_llh : np.ndarray
        Array of shape (N, >=3) with columns [lon_deg, lat_deg, height_m].

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns [X_m, Y_m, Z_m] in ECEF (EPSG:4978).

    Raises
    ------
    ImportError
        If pyproj is not installed. Install pyproj (>=3.6.0) or provide pre-converted ECEF points.

    Examples
    --------
    >>> import numpy as np
    >>> pts_llh = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 1000.0]])
    >>> xyz = reproject_llh_to_ecef(pts_llh)
    >>> xyz.shape
    (2, 3)
    """
    if not isinstance(points_llh, np.ndarray) or points_llh.ndim != 2 or points_llh.shape[1] < 3:
        raise ValueError("points_llh must be a 2D numpy array with at least 3 columns [lon, lat, height]")

    try:
        import pyproj  # type: ignore
    except Exception as e:
        raise ImportError(
            "pyproj is required for reproject_llh_to_ecef but is not available. "
            "Install pyproj (e.g., via requirements or conda env), or supply points already in ECEF (EPSG:4978)."
        ) from e

    src = pyproj.CRS.from_epsg(4979)  # WGS84 (lon, lat, ellipsoidal height)
    dst = pyproj.CRS.from_epsg(4978)  # WGS84 geocentric (ECEF)
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)

    lon = points_llh[:, 0].astype(float)
    lat = points_llh[:, 1].astype(float)
    h = points_llh[:, 2].astype(float)

    X, Y, Z = transformer.transform(lon, lat, h)
    return np.column_stack((np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)))


def reproject_wgs84_to_ecef(lon_deg, lat_deg, height_m=0.0):
    """
    Reproject WGS84 3D geographic coordinates (EPSG:4979) to Earth-Centered, Earth-Fixed
    geocentric Cartesian coordinates (ECEF, EPSG:4978) using pyproj with always_xy=True.

    Ordering and units:
    - Inputs are (lon, lat, height) where longitude and latitude are in degrees, and height
      is ellipsoidal height above the WGS84 ellipsoid in meters (not orthometric height).
    - Output is (x, y, z) in meters in an Earth-centered frame (origin at Earth's center).

    Behavior:
    - Accepts scalars or array-like inputs. Inputs are converted to numpy arrays and
      broadcast to a common shape. If broadcasting fails due to incompatible shapes,
      a ValueError is raised that includes the input shapes.
    - Internally forwards to reproject_llh_to_ecef(), which uses
      pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True) to ensure
      (lon, lat, h) ordering.

    Parameters
    ----------
    lon_deg : float or array-like of float
        Longitude(s) in degrees.
    lat_deg : float or array-like of float
        Latitude(s) in degrees.
    height_m : float or array-like of float, optional
        Ellipsoidal height(s) above WGS84 in meters. Default is 0.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x, y, z) arrays as float64 with the broadcasted shape of the inputs.

    Raises
    ------
    ValueError
        If lon_deg, lat_deg, height_m cannot be broadcast to a common shape.

    Notes
    -----
    - CRS: EPSG:4979 (WGS84 3D lon/lat/ellipsoidal height) -> EPSG:4978 (WGS84 geocentric/ECEF).
    - Numerical note: ECEF uses Earth's center as origin, with axes fixed to Earth.
      WGS84 reference ellipsoid has an equatorial radius of approximately 6378137.0 meters.

    Examples
    --------
    Scalar example:
    >>> x, y, z = reproject_wgs84_to_ecef(0.0, 0.0, 0.0)
    >>> round(x.item(), 3), round(y.item(), 3), round(z.item(), 3)
    (6378137.0, 0.0, 0.0)

    Vectorized example:
    >>> import numpy as np
    >>> x, y, z = reproject_wgs84_to_ecef([0, 10], [0, 0], [0, 0])
    >>> x.shape == y.shape == z.shape == (2,)
    True
    """
    # Convert to arrays and validate broadcastability
    lon_arr = np.asarray(lon_deg, dtype=float)
    lat_arr = np.asarray(lat_deg, dtype=float)
    h_arr = np.asarray(height_m, dtype=float)

    try:
        lon_b, lat_b, h_b = np.broadcast_arrays(lon_arr, lat_arr, h_arr)
    except ValueError as e:
        raise ValueError(
            f"lon_deg, lat_deg, height_m are not broadcastable: "
            f"shapes {lon_arr.shape}, {lat_arr.shape}, {h_arr.shape}"
        ) from e

    # Flatten to (N, 3) for underlying conversion, then reshape back
    llh = np.column_stack((lon_b.ravel(), lat_b.ravel(), h_b.ravel()))
    xyz = reproject_llh_to_ecef(llh)  # (N, 3)

    out_shape = lon_b.shape
    x = np.asarray(xyz[:, 0], dtype=float).reshape(out_shape)
    y = np.asarray(xyz[:, 1], dtype=float).reshape(out_shape)
    z = np.asarray(xyz[:, 2], dtype=float).reshape(out_shape)
    return x, y, z


def run_py3dtiles_convert(
    input_path,
    out_dir,
    threads: Optional[int] = None,
    timeout: float = 600.0,
    allow_overwrite: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> dict:
    """
    Safe wrapper around the py3dtiles "convert" CLI with strict guardrails.

    Purpose
    -------
    Provides a robust, minimal, and explicit interface to invoke:
      py3dtiles convert <input_path> --out <out_dir>
    while enforcing:
    - Tool availability checks
    - Input/output path validation and out_dir creation
    - A very limited safe-flag whitelist (no arbitrary passthrough)
    - Controlled parallelism via threads -> "--jobs"
    - Timeouts, structured result, and clear exceptions

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to a point file (e.g., XYZ/CSV/PLY) or directory accepted by py3dtiles.
        Must exist; otherwise FileNotFoundError is raised.
    out_dir : str or os.PathLike
        Output directory for tiles. Created with mkdir(parents=True, exist_ok=True).
        If creation fails, a RuntimeError is raised.
    threads : Optional[int], default None
        If provided, mapped to the py3dtiles parallelism flag: "--jobs <threads>".
    timeout : float, default 600.0
        Seconds to wait for the subprocess before timing out. On timeout, a TimeoutError is raised.
    allow_overwrite : bool, default False
        If True, appends the safe flag "--overwrite" to the command.
    extra_args : Optional[Sequence[str]], default None
        Additional flags are strictly controlled by a whitelist. Allowed values: ["--overwrite"] only.
        Any value not in the whitelist raises ValueError. This is intentionally restrictive to avoid
        unexpected side effects; use explicit parameters instead of arbitrary pass-through.

    Command Construction
    --------------------
    Base:
        ["py3dtiles", "convert", str(input_path), "--out", str(out_dir)]
    threads:
        If provided -> ["--jobs", str(threads)]
    overwrite:
        If allow_overwrite -> ["--overwrite"]
        If extra_args includes "--overwrite", it will be de-duplicated (added only once).
    extra_args:
        Only allowed values are appended; others raise ValueError.

    Subprocess Semantics
    --------------------
    - Executed with shell=False and cwd set to out_dir for stable relative outputs.
    - capture_output=True, text=True, check=False.
    - On timeout: logs error and raises TimeoutError including the timeout and the command.
    - On nonzero return code: logs stderr at error level and raises RuntimeError including the return code,
      a tail of stderr, and the command string.
    - On success: returns a structured dict with fields:
        {
          "command": cmd (list[str]),
          "returncode": 0,
          "stdout_tail": last 1000 chars of stdout,
          "out_dir": str(out_dir),
        }

    Example
    -------
    >>> result = run_py3dtiles_convert("points.xyz", "tiles_out", threads=4, timeout=120, allow_overwrite=True)
    >>> result["returncode"] == 0
    True
    """
    # Validate CLI availability
    if shutil.which("py3dtiles") is None:
        raise FileNotFoundError("py3dtiles CLI not found on PATH. Install py3dtiles and ensure it is on your PATH.")

    in_path = Path(input_path)
    out_path = Path(out_dir)

    # Validate input existence
    if not in_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_path.resolve()}")

    # Ensure output directory exists or can be created
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Explicitly re-raise as RuntimeError to provide actionable context
        raise RuntimeError(f"Failed to create output directory '{out_path}': {e}") from e

    # Validate and normalize extra_args (strict whitelist)
    allowed: set[str] = {"--overwrite"}
    validated_extra: List[str] = []
    if extra_args is not None:
        for arg in extra_args:
            if arg not in allowed:
                raise ValueError(f"Unsupported extra argument '{arg}'. Allowed: {sorted(allowed)}")
            validated_extra.append(arg)

    # Build command
    cmd: List[str] = ["py3dtiles", "convert", str(in_path), "--out", str(out_path)]
    if threads is not None:
        # Convert to int in case a numeric-like value is provided
        try:
            cmd.extend(["--jobs", str(int(threads))])
        except Exception:
            # Keep minimal assumptions; let py3dtiles handle invalid values if str(int(threads)) fails
            cmd.extend(["--jobs", str(threads)])
    # Add overwrite if allowed and not already present from validated extras
    if allow_overwrite and "--overwrite" not in validated_extra:
        cmd.append("--overwrite")
    # Append validated extras, de-duplicating if needed
    for arg in validated_extra:
        if arg not in cmd:
            cmd.append(arg)

    # Execute subprocess with timeout and without shell
    logger.info("Invoking py3dtiles: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(out_path),
            capture_output=True,
            text=True,
            timeout=float(timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        logger.error("py3dtiles timed out after %.1f seconds: %s", float(timeout), " ".join(cmd))
        raise TimeoutError(f"py3dtiles convert timed out after {timeout:.1f}s: {' '.join(cmd)}") from e
    except Exception as e:
        # Unexpected execution failure (e.g., permissions)
        logger.error("Failed to execute py3dtiles: %s", e)
        raise

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        tail = stderr[-1000:]
        logger.error("py3dtiles failed (code %s). stderr tail:\n%s", proc.returncode, tail)
        if stdout:
            logger.debug("py3dtiles stdout:\n%s", stdout)
        raise RuntimeError(
            f"py3dtiles convert failed with code {proc.returncode}. "
            f"stderr tail:\n{tail}\nCommand: {' '.join(cmd)}"
        )

    # Success path
    logger.debug("py3dtiles stdout:\n%s", stdout)
    if stderr:
        logger.debug("py3dtiles stderr:\n%s", stderr)

    return {
        "command": cmd,
        "returncode": 0,
        "stdout_tail": stdout[-1000:],
        "out_dir": str(out_path),
    }


def build_3dtiles_from_points(
    points: np.ndarray,
    out_dir: Path,
    run_cli: bool = False,
    assume_ecef: bool = False,
) -> Path:
    """
    Write a points.xyz file from input points and optionally invoke py3dtiles to generate a tileset.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, >=3). First three columns are interpreted as (x, y, z).
        If these are lon/lat/height in degrees/meters, set assume_ecef=False to enable CRS validation warnings.
    out_dir : pathlib.Path
        Output directory to write artifacts (points.xyz and, if run_cli=True, 3D Tiles output).
    run_cli : bool, optional
        If True, attempt to execute: `py3dtiles convert points.xyz --out {out_dir}`.
    assume_ecef : bool, optional
        If False, and the first two columns appear to be lon/lat degrees ([-180..180], [-90..90]),
        a warning is logged instructing users to reproject to ECEF (EPSG:4978) before tiling.

    Returns
    -------
    pathlib.Path
        The output directory path (out_dir).

    Important
    ---------
    - py3dtiles expects XYZ coordinates in a projected or geocentric CRS with meters as units
      (e.g., EPSG:4978 ECEF). Do NOT pass lon/lat degrees directly. Use
      reproject_llh_to_ecef() to convert [lon_deg, lat_deg, height_m] to [X_m, Y_m, Z_m] first.
    - On success with run_cli=True, a tileset.json is expected inside out_dir. In Phase 6 this
      path will be served under the FastAPI static mount at /tiles.

    Examples
    --------
    >>> # Prepare points from a small anomaly grid
    >>> import numpy as np
    >>> from pathlib import Path
    >>> grid = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> pts = anomalies_to_points(grid, (-10.0, 50.0, -9.0, 51.0), z_scale=500.0)
    >>> # Convert to ECEF before tiling:
    >>> # ecef = reproject_llh_to_ecef(pts[:, :3])
    >>> out = build_3dtiles_from_points(pts, Path("data/outputs/tilesets/example"), run_cli=False, assume_ecef=False)
    >>> out.exists()
    True
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be a 2D numpy array with at least 3 columns (x, y, z)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = out_dir / "points.xyz"
    arr = points[:, :3].astype(float, copy=False)

    # Heuristic CRS check: if values look like lon/lat degrees, warn unless overridden.
    if not assume_ecef:
        lon_like = arr[:, 0]
        lat_like = arr[:, 1]
        try:
            lon_in_deg_range = np.nanmin(lon_like) >= -180.0 and np.nanmax(lon_like) <= 180.0
            lat_in_deg_range = np.nanmin(lat_like) >= -90.0 and np.nanmax(lat_like) <= 90.0
        except Exception:
            lon_in_deg_range = False
            lat_in_deg_range = False
        if lon_in_deg_range and lat_in_deg_range:
            logger.warning(
                "Input coordinates appear to be lon/lat degrees. py3dtiles expects XYZ in meters "
                "(projected/ECEF; e.g., EPSG:4978). Reproject via reproject_llh_to_ecef() before tiling."
            )

    # Write XYZ file (space-delimited, sufficient precision)
    np.savetxt(xyz_path, arr, fmt="%.6f", delimiter=" ")
    logger.info("Wrote XYZ points: %s", xyz_path)

    if run_cli:
        cmd = ["py3dtiles", "convert", "points.xyz", "--out", str(out_dir)]
        logger.info("Running py3dtiles CLI: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(out_dir),
        )
        if proc.returncode != 0:
            logger.error("py3dtiles failed (code %s). stdout:\n%s\nstderr:\n%s", proc.returncode, proc.stdout, proc.stderr)
            raise RuntimeError(
                f"py3dtiles convert failed with code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        logger.info("py3dtiles succeeded. A tileset.json should be present under: %s", out_dir)

    return out_dir