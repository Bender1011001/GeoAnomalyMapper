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
from typing import Tuple

import logging
import numpy as np
import subprocess

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