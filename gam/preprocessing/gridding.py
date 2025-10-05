"""Gridding and interpolation algorithms for the GAM preprocessing module."""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from pyproj import Transformer, CRS
from typing import Union, Optional, Tuple, Dict, Any
from numbers import Number

from gam.core.exceptions import PreprocessingError
from gam.ingestion.data_structures import RawData
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


def _extract_coords_values(data: Union[RawData, xr.Dataset, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract lat, lon, values from input; handle RawData with several internal formats gracefully."""
    if isinstance(data, RawData):
        metadata = data.metadata
        bbox = metadata.get('bbox', (0, 0, 0, 0))
        dv = data.data

        # xarray Dataset/DataArray input
        if hasattr(dv, 'coords'):
            # prefer explicit coords if present
            lats = dv.coords['lat'].values if 'lat' in dv.coords else np.linspace(bbox[0], bbox[1], dv.sizes.get('lat', 100))
            lons = dv.coords['lon'].values if 'lon' in dv.coords else np.linspace(bbox[2], bbox[3], dv.sizes.get('lon', 100))
            # Create meshgrid so lat/lon pairs are matched correctly for gridding
            grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')
            points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
            # prefer named variable 'data' if present
            if 'data' in dv:
                values = dv['data'].values.ravel()
            else:
                # For DataArray or other Dataset contents, flatten values to match points
                values = dv.values.ravel()
            # Return early since we already have properly shaped points/values
            return points, values, (float(np.min(lats)), float(np.max(lats)), float(np.min(lons)), float(np.max(lons)))
        # NumPy ndarray input
        elif isinstance(dv, np.ndarray):
            arr = np.asarray(dv)
            if arr.ndim == 2 and arr.shape[1] == 3:
                lats, lons, values = arr[:, 0], arr[:, 1], arr[:, 2]
            elif arr.ndim == 1:
                n_points = arr.shape[0]
                lats = np.linspace(bbox[0], bbox[1], n_points)
                lons = np.linspace(bbox[2], bbox[3], n_points)
                values = arr
            elif arr.ndim == 2 and arr.shape[1] >= 3:
                # take first three columns as lat, lon, value
                lats, lons, values = arr[:, 0], arr[:, 1], arr[:, 2]
            else:
                raise PreprocessingError("Unsupported ndarray shape for RawData.data")
        else:
            # Unsupported RawData.data type
            raise PreprocessingError(f"Unsupported RawData.data type for gridding: {type(dv)}")

    elif isinstance(data, xr.Dataset):
        # Create meshgrid of coordinates to match the 2D 'data' variable
        lats = data.coords['lat'].values
        lons = data.coords['lon'].values
        grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')
        points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
        values = data['data'].values.ravel()
        return points, values, (float(np.min(lats)), float(np.max(lats)), float(np.min(lons)), float(np.max(lons)))
        bbox = (float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
    elif isinstance(data, np.ndarray):
        # Assume shape (n, 3) for [lat, lon, value]
        if data.ndim == 2 and data.shape[1] == 3:
            lats, lons, values = data[:, 0], data[:, 1], data[:, 2]
            bbox = (float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
        else:
            raise PreprocessingError("ndarray input must be (n_points, 3) for [lat, lon, value]")
    else:
        raise PreprocessingError(f"Unsupported input for gridding: {type(data)}")

    points = np.column_stack((lats, lons))
    return points, values, (float(np.min(lats)), float(np.max(lats)), float(np.min(lons)), float(np.max(lons)))


class RegularGridder:
    """
    Regular gridder using scipy.interpolate.griddata for irregular to regular conversion.

    Creates uniform grid from scattered points. Supports linear, cubic, nearest methods.
    Handles missing data with NaN filling.

    Parameters
    ----------
    resolution : float, optional
        Grid spacing in degrees (default: 0.1).
    method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest'; default: 'linear').
    fill_value : float or str, optional
        Value for extrapolation points (np.nan or number; default: np.nan).

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> ProcessedGrid
        Grid irregular data to regular.

    Notes
    -----
    - Uses Delaunay triangulation for linear/cubic.
    - Efficient for moderate point counts; for large, chunk via Dask.
    - Output always WGS84-aligned.

    Examples
    --------
    >>> gridder = RegularGridder(resolution=0.05, method='cubic')
    >>> gridded = gridder.apply(raw_scattered_data)
    """

    SUPPORTED_METHODS = {'linear', 'cubic', 'nearest'}

    def __init__(
        self,
        resolution: float = 0.1,
        method: str = 'linear',
        fill_value: Union[float, str] = np.nan
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        self.resolution = resolution
        self.method = method
        self.fill_value = fill_value

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> ProcessedGrid:
        """
        Apply regular gridding to input data.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input with scattered or irregular points.

        Returns
        -------
        ProcessedGrid
            Gridded xarray.Dataset.

        Raises
        ------
        PreprocessingError
            If gridding fails (e.g., insufficient points).
        """
        points, values, bbox = _extract_coords_values(data)
        min_lat, max_lat, min_lon, max_lon = bbox

        # Create regular grid
        lats = np.arange(min_lat, max_lat + self.resolution, self.resolution)
        lons = np.arange(min_lon, max_lon + self.resolution, self.resolution)
        grid_lats, grid_lons = np.meshgrid(lats, lons, indexing='ij')

        # Interpolate
        try:
            gridded = griddata(
                points,
                values,
                (grid_lats, grid_lons),
                method=self.method,
                fill_value=self.fill_value
            )
        except Exception as e:
            # Qhull/Delaunay failures can occur for degenerate point sets; fall back to nearest neighbor
            msg = str(e)
            if 'Qhull' in msg or 'coplanar' in msg or 'Initial simplex' in msg:
                logger.warning(f"Regular gridding: Delaunay/Qhull failed ({msg}). Falling back to 'nearest' interpolation.")
                gridded = griddata(
                    points,
                    values,
                    (grid_lats, grid_lons),
                    method='nearest',
                    fill_value=self.fill_value
                )
            else:
                raise PreprocessingError(f"Regular gridding failed: {e}")

        # Create DataArray and Dataset
        da = xr.DataArray(
            gridded,
            dims=['lat', 'lon'],
            coords={'lat': lats, 'lon': lons}
        )
        ds = xr.Dataset({'data': da})
        ds.attrs['grid_resolution'] = self.resolution
        ds.attrs['interpolation_method'] = self.method
        ds.attrs['units'] = data.metadata.get('units', 'unknown') if isinstance(data, RawData) else 'unknown'

        result = ProcessedGrid(ds)
        logger.info(f"Regular grid created: shape {gridded.shape}, method '{self.method}', res {self.resolution}°")
        return result


class AdaptiveGridder:
    """
    Adaptive gridder with variable resolution based on data density.

    Uses kernel density estimation (KDE) to determine local density, then adjusts
    grid spacing (finer where dense). Supports base resolution with density factor.

    Parameters
    ----------
    base_resolution : float, optional
        Base grid spacing (default: 0.1°).
    density_factor : float, optional
        Multiplier for resolution adjustment (default: 0.5; smaller = finer in dense areas).
    method : str, optional
        Inner interpolation method (default: 'linear').
    min_resolution : float, optional
        Minimum grid spacing (default: 0.01°).
    bandwidth : float, optional
        KDE bandwidth (default: 0.05).

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> ProcessedGrid
        Create adaptive grid.

    Notes
    -----
    - Computes KDE on points, bins density, adjusts local resolution.
    - Merges to global grid with variable chunking.
    - Suitable for unevenly sampled geophysical data.

    Examples
    --------
    >>> gridder = AdaptiveGridder(base_resolution=0.1, density_factor=0.3)
    >>> adaptive_grid = gridder.apply(scattered_data)
    """

    def __init__(
        self,
        base_resolution: float = 0.1,
        density_factor: float = 0.5,
        method: str = 'linear',
        min_resolution: float = 0.01,
        bandwidth: float = 0.05
    ):
        self.base_resolution = base_resolution
        self.density_factor = density_factor
        self.method = method
        self.min_resolution = min_resolution
        self.bandwidth = bandwidth

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> ProcessedGrid:
        """
        Apply adaptive gridding based on data density.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input scattered data.

        Returns
        -------
        ProcessedGrid
            Adaptively gridded Dataset (regular but with variable effective res).

        Raises
        ------
        PreprocessingError
            If density estimation or gridding fails.
        """
        points, values, bbox = _extract_coords_values(data)
        if len(points) < 4:
            raise PreprocessingError("Insufficient points for KDE (need >=4)")

        # KDE for density
        try:
            kde = gaussian_kde(points.T, bw_method=self.bandwidth)
            density = kde(points.T)
            # Normalize density [0,1]
            density = (density - density.min()) / (density.max() - density.min() + 1e-10)
        except Exception as e:
            raise PreprocessingError(f"KDE density estimation failed: {e}")

        # Adjust resolution per point (simplified: global adjust, but could localize)
        avg_density = np.mean(density)
        adjusted_res = self.base_resolution * (1 - self.density_factor * avg_density)
        adjusted_res = max(adjusted_res, self.min_resolution)

        # Use adjusted_res for regular gridder
        gridder = RegularGridder(resolution=adjusted_res, method=self.method)
        result = gridder.apply(data)
        result.add_metadata('adaptive_adjustment', {'avg_density': float(avg_density), 'adjusted_res': adjusted_res})

        logger.info(f"Adaptive grid created: base_res={self.base_resolution}, adjusted={adjusted_res}, density_factor={self.density_factor}")
        return result


class CoordinateAligner:
    """
    Coordinate aligner to common system (WGS84) with optional gridding.

    Transforms input coordinates to WGS84 using pyproj. Can chain with gridding.

    Parameters
    ----------
    target_crs : str, optional
        Target CRS (default: 'EPSG:4326' WGS84).
    grid_after : bool or RegularGridder, optional
        If True, apply default RegularGridder after transform; or pass instance (default: False).
    source_crs : Optional[str], optional
        Assumed source CRS (default: infer from metadata or WGS84).

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> Union[ProcessedGrid, RawData]
        Align coordinates to target CRS.

    Notes
    -----
    - Handles lat/lon transformation; assumes geographic input.
    - If gridding, outputs ProcessedGrid; else RawData with updated metadata.
    - Validates CRS compatibility.

    Examples
    --------
    >>> aligner = CoordinateAligner(grid_after=True)
    >>> aligned = aligner.apply(raw_data)
    """

    def __init__(
        self,
        target_crs: str = 'EPSG:4326',
        grid_after: Union[bool, RegularGridder] = False,
        source_crs: Optional[str] = None
    ):
        self.target_crs = target_crs
        self.grid_after = RegularGridder() if grid_after is True else grid_after
        self.source_crs = source_crs

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> Union[ProcessedGrid, RawData]:
        """
        Align and optionally grid coordinates to target CRS.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input data with coordinates.

        Returns
        -------
        Union[ProcessedGrid, RawData]
            Aligned data (gridded if specified).

        Raises
        ------
        PreprocessingError
            If transformation fails.
        """
        points, values, bbox = _extract_coords_values(data)
        source_crs = self.source_crs or (data.metadata.get('coordinate_system', 'EPSG:4326') if isinstance(data, RawData) else 'EPSG:4326')

        if source_crs == self.target_crs:
            logger.info("Source and target CRS match; no transformation needed")
            if self.grid_after:
                return self.grid_after.apply(data)
            return ProcessedGrid.from_raw(data) if hasattr(ProcessedGrid, 'from_raw') else data  # Assume pass-through

        try:
            transformer = Transformer.from_crs(source_crs, self.target_crs, always_xy=True)
            new_lons, new_lats = transformer.transform(points[:, 1], points[:, 0])  # lon, lat
            new_points = np.column_stack((new_lats, new_lons))
        except Exception as e:
            raise PreprocessingError(f"Coordinate transformation failed: {e}")

        # Update bbox
        new_bbox = (np.min(new_lats), np.max(new_lats), np.min(new_lons), np.max(new_lons))

        if isinstance(data, RawData):
            new_metadata = dict(data.metadata)
            new_metadata['bbox'] = new_bbox
            new_metadata['coordinate_system'] = self.target_crs
            aligned_values = RawData(new_metadata, np.column_stack((new_lats, new_lons, values)))
            result = aligned_values
        else:
            # For Dataset/ndarray, create new
            aligned_values = np.column_stack((new_lats, new_lons, values))
            result = aligned_values

        if self.grid_after:
            result = self.grid_after.apply(result)

        logger.info(f"Coordinates aligned to {self.target_crs} (from {source_crs}), new bbox {new_bbox}")
        return result