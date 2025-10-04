"""Geodesy utilities for coordinate reference system transformations and calculations."""

import pyproj
import numpy as np
from typing import Any, Tuple

from gam.core.exceptions import ConfigError


def ensure_crs(crs_like: Any) -> pyproj.CRS:
    """
    Takes various inputs (EPSG code, WKT string, Proj string) and returns a pyproj.CRS object.

    Raises:
        ConfigError: If the input cannot be converted to a valid CRS.
    """
    try:
        return pyproj.CRS(crs_like)
    except Exception as e:
        raise ConfigError(f"Invalid CRS: {crs_like}") from e


def build_transformer(src_crs: pyproj.CRS, dst_crs: pyproj.CRS) -> pyproj.Transformer:
    """
    A simple wrapper around pyproj.Transformer.from_crs.

    Args:
        src_crs: Source coordinate reference system.
        dst_crs: Destination coordinate reference system.

    Returns:
        pyproj.Transformer: Transformer object for coordinate transformations.
    """
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def geodetic_to_projected(lon: np.ndarray, lat: np.ndarray, dst_crs: pyproj.CRS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms longitude and latitude arrays from WGS84 (EPSG:4326) to a specified projected CRS.

    Args:
        lon: Array of longitudes in degrees.
        lat: Array of latitudes in degrees.
        dst_crs: Destination projected CRS.

    Returns:
        Tuple of arrays: (x, y) coordinates in the destination CRS.
    """
    src_crs = pyproj.CRS("EPSG:4326")
    transformer = build_transformer(src_crs, dst_crs)
    x, y = transformer.transform(lon, lat)
    return x, y


def bbox_extent_meters(bbox: Tuple[float, float, float, float], dst_crs: pyproj.CRS) -> Tuple[float, float]:
    """
    Calculates the width and height of a bounding box (min_lon, min_lat, max_lon, max_lat) in meters
    by projecting its corners to the target CRS.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        dst_crs: Target projected CRS.

    Returns:
        Tuple of floats: (width, height) in meters.

    Raises:
        ValueError: If the bounding box is invalid (e.g., min >= max).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bounding box: min values must be less than max values")

    # Define corners of the bounding box
    corners_lon = np.array([min_lon, max_lon, max_lon, min_lon])
    corners_lat = np.array([min_lat, min_lat, max_lat, max_lat])

    x, y = geodetic_to_projected(corners_lon, corners_lat, dst_crs)

    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    return width, height