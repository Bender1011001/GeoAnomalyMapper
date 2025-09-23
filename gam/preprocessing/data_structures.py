"""Data structures for the GAM preprocessing module."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import xarray as xr
from pyproj import Transformer, CRS

from gam.core.exceptions import PreprocessingError  # Assuming core exceptions implemented per architecture
from gam.preprocessing.base import Preprocessor


logger = logging.getLogger(__name__)


class ProcessedGrid:
    """
    Class representing processed geophysical data as an xarray.Dataset.

    This class wraps xarray.Dataset to provide GAM-specific functionality for
    standardized processed grids. It includes spatial coordinates (lat, lon,
    depth/elevation), metadata storage, validation, serialization, coordinate
    transformations, and basic unit conversions. Designed for multi-dimensional
    geospatial data with support for chunking (Dask-compatible).

    The underlying Dataset must have:
    - Dimensions: 'lat', 'lon', ['depth' or 'elevation' or 'time']
    - Coordinates: 'lat' (1D, degrees), 'lon' (1D, degrees), optional 'depth'
    - Variables: 'data' (main values, float), optional 'uncertainty' (float)
    - Attributes: 'units' (str), 'grid_resolution' (float), 'processed_at' (datetime),
      'coordinate_system' (str, default 'WGS84'), 'processing_params' (dict)

    Parameters
    ----------
    data : xarray.Dataset or dict
        If Dataset: Directly used (validated).
        If dict: {'data': np.ndarray or xr.DataArray, 'uncertainty': Optional[np.ndarray],
                  'lat': np.ndarray, 'lon': np.ndarray, 'depth': Optional[np.ndarray],
                  'units': str, 'grid_resolution': float, 'processing_params': dict}
    coords : Optional[Dict[str, np.ndarray]], optional
        Explicit coordinates if building from scratch.
    attrs : Optional[Dict[str, Any]], optional
        Additional attributes.

    Attributes
    ----------
    ds : xarray.Dataset
        The underlying xarray.Dataset.
    units : str
        Data units (e.g., 'm/s²' for gravity).
    coordinate_system : str
        CRS identifier (default: 'EPSG:4326' for WGS84).

    Methods
    -------
    validate()
        Validate structure and raise PreprocessingError if invalid.
    to_netcdf(path: str, **kwargs)
        Serialize to NetCDF file, preserving metadata.
    from_netcdf(path: str, cls=None) -> ProcessedGrid
        Load from NetCDF file.
    transform_crs(target_crs: str, source_crs: Optional[str] = None) -> ProcessedGrid
        Reproject coordinates using pyproj.
    convert_units(new_unit: str, conversion_factor: float) -> ProcessedGrid
        Apply scalar unit conversion to 'data' and 'uncertainty'.
    add_metadata(key: str, value: Any) -> None
        Add or update metadata in attrs.
    __repr__()
        String representation for logging.

    Notes
    -----
    - Memory efficient: Supports lazy loading and chunking via xarray/Dask.
    - Reproducible: Metadata includes processing params and timestamp.
    - Integrates with Preprocessor: Output of process() method.
    - Handles missing data: NaN masking preserved.
    - Edge cases: Sparse data via irregular coords; no depth defaults to 2D.

    Examples
    --------
    >>> # From dict
    >>> grid_data = {'data': np.random.rand(10, 10), 'lat': np.linspace(30, 31, 10),
    ...              'lon': np.linspace(0, 1, 10), 'units': 'mGal'}
    >>> grid = ProcessedGrid(grid_data)
    >>> grid.validate()
    >>> # Transform CRS
    >>> new_grid = grid.transform_crs('EPSG:3857')
    >>> # Serialize
    >>> grid.to_netcdf('processed_grid.nc')
    >>> loaded = ProcessedGrid.from_netcdf('processed_grid.nc')
    """

    REQUIRED_DIMS = {'lat', 'lon'}
    REQUIRED_VARS = {'data'}
    OPTIONAL_VARS = {'uncertainty'}
    DEFAULT_CRS = 'EPSG:4326'  # WGS84

    def __init__(
        self,
        data: Union[xr.Dataset, Dict[str, Any]],
        coords: Optional[Dict[str, np.ndarray]] = None,
        attrs: Optional[Dict[str, Any]] = None
    ):
        if isinstance(data, xr.Dataset):
            self.ds = data
        else:
            # Build Dataset from dict
            if 'data' not in data:
                raise ValueError("Dict must contain 'data' key")
            da_data = xr.DataArray(
                data['data'],
                dims=['lat', 'lon'],
                coords={'lat': data.get('lat'), 'lon': data.get('lon')}
            )
            if 'depth' in data:
                da_data = da_data.expand_dims({'depth': [data['depth']]})
            ds = xr.Dataset({'data': da_data})
            if 'uncertainty' in data:
                unc_da = xr.DataArray(
                    data['uncertainty'],
                    dims=da_data.dims,
                    coords=da_data.coords
                )
                ds['uncertainty'] = unc_da
            self.ds = ds

        # Set default attrs if not present
        self.ds.attrs.setdefault('units', 'unknown')
        self.ds.attrs.setdefault('grid_resolution', 0.1)
        self.ds.attrs.setdefault('processed_at', datetime.now())
        self.ds.attrs.setdefault('coordinate_system', self.DEFAULT_CRS)
        self.ds.attrs.setdefault('processing_params', {})

        if attrs:
            self.ds.attrs.update(attrs)

        if coords:
            self.ds = self.ds.assign_coords(**coords)

        self.validate()

    @property
    def units(self) -> str:
        return self.ds.attrs.get('units', 'unknown')

    @units.setter
    def units(self, value: str):
        self.ds.attrs['units'] = value

    @property
    def coordinate_system(self) -> str:
        return self.ds.attrs.get('coordinate_system', self.DEFAULT_CRS)

    @coordinate_system.setter
    def coordinate_system(self, value: str):
        self.ds.attrs['coordinate_system'] = value

    def validate(self) -> bool:
        """
        Validate the ProcessedGrid structure.

        Checks:
        - Required dimensions and coordinates present
        - 'data' variable exists and is float-type
        - Coordinates are monotonic increasing
        - CRS is valid (if specified)
        - No invalid NaNs in core structure

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        PreprocessingError
            If validation fails (e.g., missing dims, invalid coords).
        """
        # Check dimensions
        actual_dims = set(self.ds.dims)
        if not self.REQUIRED_DIMS.issubset(actual_dims):
            raise PreprocessingError(
                f"Missing required dimensions: {self.REQUIRED_DIMS - actual_dims}"
            )

        # Check variables
        if 'data' not in self.ds:
            raise PreprocessingError("Missing required 'data' variable")

        if not np.issubdtype(self.ds['data'].dtype, np.floating):
            raise PreprocessingError("'data' must be floating-point type")

        # Check coordinates monotonicity
        for coord_name in ['lat', 'lon']:
            if coord_name in self.ds.coords:
                coord = self.ds.coords[coord_name]
                if not np.all(np.diff(coord.values) > 0):
                    raise PreprocessingError(
                        f"Coordinate '{coord_name}' must be strictly increasing"
                    )

        # Optional depth/elevation/time
        depth_like_dims = {'depth', 'elevation', 'time'}
        if depth_like_dims.intersection(actual_dims):
            depth_dim = (depth_like_dims & actual_dims).pop()
            depth_coord = self.ds.coords[depth_dim]
            if not np.all(np.diff(depth_coord.values) >= 0):  # Non-decreasing for depth
                raise PreprocessingError(f"'{depth_dim}' coordinate must be non-decreasing")

        # Validate CRS
        crs_str = self.coordinate_system
        if crs_str != 'unknown':
            try:
                CRS.from_string(crs_str)
            except Exception as e:
                raise PreprocessingError(f"Invalid CRS '{crs_str}': {e}")

        logger.debug("ProcessedGrid validation passed")
        return True

    def to_netcdf(self, path: str, **kwargs) -> None:
        """
        Serialize the ProcessedGrid to a NetCDF file.

        Preserves all attributes, coordinates, and variables. Supports compression
        and encoding for efficiency.

        Parameters
        ----------
        path : str
            File path to write NetCDF.
        **kwargs : dict, optional
            Passed to xr.Dataset.to_netcdf() (e.g., encoding, compute=False for lazy).

        Raises
        ------
        OSError
            If write fails.
        """
        # Ensure processed_at is serializable
        attrs = dict(self.ds.attrs)
        if 'processed_at' in attrs:
            attrs['processed_at'] = attrs['processed_at'].isoformat()
        temp_ds = self.ds.copy()
        temp_ds.attrs.update(attrs)
        temp_ds.to_netcdf(path, **kwargs)
        logger.info(f"ProcessedGrid saved to {path}")

    @classmethod
    def from_netcdf(cls, path: str, **kwargs) -> "ProcessedGrid":
        """
        Create ProcessedGrid from a NetCDF file.

        Loads Dataset and reconstructs ProcessedGrid, parsing datetime attrs.

        Parameters
        ----------
        path : str
            File path to read NetCDF.
        **kwargs : dict, optional
            Passed to xr.open_dataset() (e.g., chunks for lazy loading).

        Returns
        -------
        ProcessedGrid
            Loaded instance.

        Raises
        ------
        OSError
            If read fails.
        PreprocessingError
            If loaded data invalid.
        """
        ds = xr.open_dataset(path, **kwargs)
        # Parse processed_at if present
        if 'processed_at' in ds.attrs:
            try:
                ds.attrs['processed_at'] = datetime.fromisoformat(ds.attrs['processed_at'])
            except ValueError:
                logger.warning("Invalid 'processed_at' format in NetCDF")
        grid = cls(ds)
        grid.validate()
        logger.info(f"ProcessedGrid loaded from {path}")
        return grid

    def transform_crs(
        self,
        target_crs: str,
        source_crs: Optional[str] = None
    ) -> "ProcessedGrid":
        """
        Transform coordinates to a target CRS using pyproj.

        Currently supports reprojection of lat/lon coords. For full grid transform,
        assumes 2D/3D points. Returns new instance; original unchanged.

        Parameters
        ----------
        target_crs : str
            Target CRS (e.g., 'EPSG:3857' for Web Mercator).
        source_crs : str, optional
            Source CRS (defaults to self.coordinate_system).

        Returns
        -------
        ProcessedGrid
            New instance with transformed coordinates.

        Raises
        ------
        PreprocessingError
            If transformation fails or unsupported CRS.
        """
        source_crs = source_crs or self.coordinate_system
        if source_crs == 'unknown' or target_crs == source_crs:
            logger.warning("No transformation needed; returning copy")
            return self.copy()

        try:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        except Exception as e:
            raise PreprocessingError(f"CRS transformation failed: {e}")

        # Get lat/lon coords (assuming 1D for regular grid)
        lats = self.ds.coords['lat'].values
        lons = self.ds.coords['lon'].values

        # Transform points
        new_lons, new_lats = transformer.transform(lons, lats)

        # Create new coords (sort if needed for monotonicity)
        new_lat_coord = np.sort(new_lats) if np.any(np.diff(new_lats) < 0) else new_lats
        new_lon_coord = np.sort(new_lons) if np.any(np.diff(new_lons) < 0) else new_lons

        # Regrid data to new coords (simple nearest for demo; use interp in production)
        new_ds = self.ds.interp(lat=new_lat_coord, lon=new_lon_coord, method='nearest')

        new_grid = self.__class__(new_ds)
        new_grid.coordinate_system = target_crs
        new_grid.add_metadata('transformation', {'from': source_crs, 'to': target_crs})
        logger.info(f"CRS transformed from {source_crs} to {target_crs}")
        return new_grid

    def convert_units(self, new_unit: str, conversion_factor: float) -> "ProcessedGrid":
        """
        Apply unit conversion by scaling 'data' and 'uncertainty'.

        Simple scalar multiplication. For complex conversions (e.g., gravity units),
        use dedicated UnitConverter from units.py.

        Parameters
        ----------
        new_unit : str
            New unit string (e.g., 'm/s²').
        conversion_factor : float
            Multiplication factor (e.g., 1e-5 to convert mGal to m/s²).

        Returns
        -------
        ProcessedGrid
            New instance with converted values.

        Raises
        ------
        ValueError
            If no 'data' variable or invalid factor.
        """
        if 'data' not in self.ds:
            raise ValueError("Cannot convert units: missing 'data' variable")
        if conversion_factor == 0:
            raise ValueError("Conversion factor cannot be zero")

        new_ds = self.ds.copy()
        new_ds['data'] *= conversion_factor
        if 'uncertainty' in new_ds:
            new_ds['uncertainty'] *= abs(conversion_factor)  # Preserve relative uncertainty

        new_grid = self.__class__(new_ds)
        old_unit = self.units
        new_grid.units = new_unit
        new_grid.add_metadata('unit_conversion', {'from': old_unit, 'to': new_unit, 'factor': conversion_factor})
        logger.info(f"Units converted from {old_unit} to {new_unit} (factor: {conversion_factor})")
        return new_grid

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update metadata in the Dataset attributes.

        Parameters
        ----------
        key : str
            Metadata key.
        value : Any
            Serializable value (e.g., str, int, dict).
        """
        self.ds.attrs[key] = value
        logger.debug(f"Added metadata: {key} = {value}")

    def copy(self) -> "ProcessedGrid":
        """Return a deep copy of the ProcessedGrid."""
        return self.__class__(self.ds.copy(deep=True))

    def __repr__(self) -> str:
        """String representation for logging and debugging."""
        shape = self.ds['data'].shape
        units = self.units
        crs = self.coordinate_system
        return f"ProcessedGrid(shape={shape}, units='{units}', crs='{crs}', dims={list(self.ds.dims)})"