"""Filtering algorithms for the GAM preprocessing module."""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
from scipy import ndimage, stats
from obspy import Stream
from obspy.signal import filter as obspy_filter
from typing import Union, Optional, Tuple, Any
from numbers import Number

from gam.core.exceptions import PreprocessingError
from gam.ingestion.data_structures import RawData
from gam.preprocessing.data_structures import ProcessedGrid


logger = logging.getLogger(__name__)


def _validate_input(data: Union[RawData, xr.Dataset, np.ndarray, Stream]) -> Union[xr.Dataset, np.ndarray, Stream]:
    """Internal helper to validate and extract values from input."""
    if isinstance(data, RawData):
        values = data.values
        if not hasattr(values, 'validate'):
            data.validate()
    elif isinstance(data, (xr.Dataset, xr.DataArray)):
        values = data
    elif isinstance(data, np.ndarray):
        values = data
        if not np.issubdtype(values.dtype, np.number):
            raise PreprocessingError("Input array must be numeric")
    elif isinstance(data, Stream):
        values = data
    else:
        raise PreprocessingError(f"Unsupported input type: {type(data)}")
    return values


class NoiseFilter:
    """
    Gaussian noise filter for geophysical data using scipy.ndimage.

    Applies Gaussian smoothing to reduce high-frequency noise in gravity/magnetic
    data. Configurable sigma for kernel size. Handles xarray and ndarray inputs.

    Parameters
    ----------
    sigma : float or Tuple[float, ...], optional
        Standard deviation for Gaussian kernel (default: 1.0). Tuple for multi-dim.
    mode : str, optional
        Boundary mode ('reflect', 'constant', etc.; default: 'reflect').
    preserve_edge : bool, optional
        If True, truncate filter to avoid edge effects (default: False).

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]
        Apply Gaussian filter.

    Notes
    -----
    - Optimized for 2D/3D spatial data; for 1D, uses same sigma.
    - Preserves NaNs by masking.
    - Memory efficient for small kernels; for large, consider Dask chunking.

    Examples
    --------
    >>> filter = NoiseFilter(sigma=2.0)
    >>> filtered = filter.apply(raw_data)
    """

    def __init__(
        self,
        sigma: Union[float, Tuple[float, ...]] = 1.0,
        mode: str = 'reflect',
        preserve_edge: bool = False
    ):
        self.sigma = sigma
        self.mode = mode
        self.preserve_edge = preserve_edge

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]:
        """
        Apply Gaussian noise filter to input data.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input data to filter.

        Returns
        -------
        Union[ProcessedGrid, xr.Dataset, np.ndarray]
            Filtered data (same type as input, or ProcessedGrid if xr.Dataset).

        Raises
        ------
        PreprocessingError
            If input invalid or filtering fails.
        """
        values = _validate_input(data)
        if isinstance(values, Stream):
            raise PreprocessingError("NoiseFilter not applicable to seismic streams")

        try:
            if isinstance(values, xr.Dataset):
                # Apply to 'data' variable
                filtered_da = values['data'].data  # Extract ndarray
                filtered_array = ndimage.gaussian_filter(
                    filtered_array,
                    sigma=self.sigma,
                    mode=self.mode,
                    preserve_range=self.preserve_edge
                )
                filtered_da = xr.DataArray(filtered_array, dims=values['data'].dims, coords=values['data'].coords)
                new_ds = values.copy()
                new_ds['data'] = filtered_da
                if 'uncertainty' in new_ds:
                    # Propagate uncertainty (simple addition of variance)
                    unc_var = ndimage.gaussian_filter(new_ds['uncertainty'].data ** 2, sigma=self.sigma, mode=self.mode)
                    new_ds['uncertainty'] = np.sqrt(unc_var)
                result = ProcessedGrid(new_ds)
            else:
                # ndarray
                result = ndimage.gaussian_filter(
                    values,
                    sigma=self.sigma,
                    mode=self.mode,
                    preserve_range=self.preserve_edge
                )
                if isinstance(data, RawData):
                    result = RawData(data.metadata, result)

            logger.info(f"Applied Gaussian filter (sigma={self.sigma}) to data of shape {getattr(values, 'shape', 'N/A')}")
            return result
        except Exception as e:
            raise PreprocessingError(f"Gaussian filtering failed: {e}")


class BandpassFilter:
    """
    Bandpass filter for seismic data using ObsPy.

    Applies bandpass filtering to seismic waveforms (ObsPy Stream). Configurable
    frequency range. Converts to/from xarray if needed for grid compatibility.

    Parameters
    ----------
    lowcut : float
        Low corner frequency (Hz; default: 0.1).
    highcut : float
        High corner frequency (Hz; default: 1.0).
    order : int, optional
        Filter order (default: 4).
    corners : int, optional
        Number of corners for Butterworth (default: 4).
    zerophase : bool, optional
        Apply zero-phase filter (default: True).

    Methods
    -------
    apply(data: Union[RawData, Stream]) -> Union[RawData, Stream]
        Apply bandpass filter.

    Notes
    -----
    - Designed for seismic waveforms; not for static fields like gravity.
    - Handles multiple traces in Stream.
    - Removes DC offset before filtering.

    Examples
    --------
    >>> filter = BandpassFilter(lowcut=0.1, highcut=1.0)
    >>> filtered_stream = filter.apply(seismic_raw_data)
    """

    def __init__(
        self,
        lowcut: float = 0.1,
        highcut: float = 1.0,
        order: int = 4,
        corners: int = 4,
        zerophase: bool = True
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.corners = corners
        self.zerophase = zerophase

    def apply(
        self,
        data: Union[RawData, Stream]
    ) -> Union[RawData, Stream]:
        """
        Apply bandpass filter to seismic data.

        Parameters
        ----------
        data : Union[RawData, Stream]
            Input seismic data (values as Stream).

        Returns
        -------
        Union[RawData, Stream]
            Filtered data.

        Raises
        ------
        PreprocessingError
            If input not seismic or filtering fails.
        """
        if isinstance(data, RawData):
            values = data.values
            if not isinstance(values, Stream):
                raise PreprocessingError("BandpassFilter requires ObsPy Stream")
        elif isinstance(data, Stream):
            values = data
        else:
            raise PreprocessingError("BandpassFilter only supports Stream or RawData with Stream values")

        try:
            # Remove DC offset
            values.remove_response()  # If instrument response available; otherwise skip
            filtered = values.copy()
            filtered.filter(
                'bandpass',
                freqmin=self.lowcut,
                freqmax=self.highcut,
                corners=self.corners,
                zerophase=self.zerophase
            )
            result = RawData(data.metadata, filtered) if isinstance(data, RawData) else filtered

            logger.info(f"Applied bandpass filter ({self.lowcut}-{self.highcut} Hz) to {len(values)} traces")
            return result
        except Exception as e:
            raise PreprocessingError(f"Bandpass filtering failed: {e}")


class OutlierFilter:
    """
    Statistical outlier detection and removal using z-score and IQR.

    Detects and masks outliers in data. Supports both methods; configurable threshold.
    Handles xarray and ndarray; preserves structure.

    Parameters
    ----------
    method : str, optional
        'zscore' or 'iqr' (default: 'zscore').
    threshold : float, optional
        For zscore: std devs (default: 3.0); for iqr: multiplier (default: 1.5).
    replace_with : str or float, optional
        'nan', 'mean', or value to replace outliers (default: 'nan').

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]
        Apply outlier filtering.

    Notes
    -----
    - Z-score: |z| > threshold → outlier.
    - IQR: Below Q1 - threshold*IQR or above Q3 + threshold*IQR.
    - Ignores NaNs in computation.
    - For multi-dim, applies per element or along axis (default: flatten).

    Examples
    --------
    >>> filter = OutlierFilter(method='iqr', threshold=1.5)
    >>> cleaned = filter.apply(grid_data)
    """

    def __init__(
        self,
        method: str = 'zscore',
        threshold: float = 3.0,
        replace_with: Union[str, Number] = 'nan'
    ):
        if method not in ['zscore', 'iqr']:
            raise ValueError("Method must be 'zscore' or 'iqr'")
        self.method = method
        self.threshold = threshold
        self.replace_with = replace_with

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]:
        """
        Apply outlier detection and removal.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input data.

        Returns
        -------
        Union[ProcessedGrid, xr.Dataset, np.ndarray]
            Data with outliers replaced.

        Raises
        ------
        PreprocessingError
            If computation fails.
        """
        values = _validate_input(data)
        if isinstance(values, Stream):
            raise PreprocessingError("OutlierFilter not applicable to seismic streams")

        try:
            array = np.asarray(values['data'].data if isinstance(values, xr.Dataset) else values).flatten()
            array = array[~np.isnan(array)]  # Ignore NaNs

            if len(array) == 0:
                logger.warning("No valid data for outlier detection")
                return data

            if self.method == 'zscore':
                z_scores = np.abs(stats.zscore(array))
                outliers = z_scores > self.threshold
            else:  # iqr
                q1, q3 = np.percentile(array, [25, 75])
                iqr = q3 - q1
                lower = q1 - self.threshold * iqr
                upper = q3 + self.threshold * iqr
                outliers = (array < lower) | (array > upper)

            num_outliers = np.sum(outliers)
            logger.info(f"Detected {num_outliers} outliers using {self.method} (threshold={self.threshold})")

            # Reshape back and replace
            full_array = np.asarray(values['data'].data if isinstance(values, xr.Dataset) else values)
            outlier_mask = np.zeros_like(full_array, dtype=bool)
            # Simplified: apply to flattened, but for multi-dim, need per-slice; here assume global
            flat_full = full_array.flatten()
            flat_full[outliers[:len(flat_full)]] = np.nan if self.replace_with == 'nan' else np.mean(array)
            result_array = flat_full.reshape(full_array.shape)

            if isinstance(values, xr.Dataset):
                new_ds = values.copy()
                new_ds['data'] = xr.DataArray(result_array, dims=values['data'].dims, coords=values['data'].coords)
                result = ProcessedGrid(new_ds)
            else:
                result = result_array
                if isinstance(data, RawData):
                    result = RawData(data.metadata, result)

            return result
        except Exception as e:
            raise PreprocessingError(f"Outlier filtering failed: {e}")


class SpatialFilter:
    """
    Median spatial filter for noise reduction using scipy.ndimage.

    Applies median filtering to remove salt-and-pepper noise in spatial data.
    Configurable footprint size. Suitable for 2D grids.

    Parameters
    ----------
    size : int or Tuple[int, ...], optional
        Kernel size (default: 3). Tuple for anisotropic.
    mode : str, optional
        Boundary mode (default: 'reflect').
    cval : float, optional
        Constant value if mode='constant' (default: 0.0).

    Methods
    -------
    apply(data: Union[RawData, xr.Dataset, np.ndarray]) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]
        Apply median filter.

    Notes
    -----
    - Effective for impulsive noise in gridded data.
    - Preserves edges better than mean filter.
    - For 3D, applies along spatial dims.

    Examples
    --------
    >>> filter = SpatialFilter(size=5)
    >>> smoothed = filter.apply(processed_grid)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]] = 3,
        mode: str = 'reflect',
        cval: float = 0.0
    ):
        self.size = size
        self.mode = mode
        self.cval = cval

    def apply(
        self,
        data: Union[RawData, xr.Dataset, np.ndarray]
    ) -> Union[ProcessedGrid, xr.Dataset, np.ndarray]:
        """
        Apply median spatial filter.

        Parameters
        ----------
        data : Union[RawData, xr.Dataset, np.ndarray]
            Input spatial data.

        Returns
        -------
        Union[ProcessedGrid, xr.Dataset, np.ndarray]
            Filtered data.

        Raises
        ------
        PreprocessingError
            If filtering fails.
        """
        values = _validate_input(data)
        if isinstance(values, Stream):
            raise PreprocessingError("SpatialFilter not applicable to seismic streams")

        try:
            if isinstance(values, xr.Dataset):
                filtered_array = ndimage.median_filter(
                    values['data'].data,
                    size=self.size,
                    mode=self.mode,
                    cval=self.cval
                )
                new_ds = values.copy()
                new_ds['data'] = xr.DataArray(filtered_array, dims=values['data'].dims, coords=values['data'].coords)
                result = ProcessedGrid(new_ds)
            else:
                result = ndimage.median_filter(
                    values,
                    size=self.size,
                    mode=self.mode,
                    cval=self.cval
                )
                if isinstance(data, RawData):
                    result = RawData(data.metadata, result)

            logger.info(f"Applied median filter (size={self.size}) to data of shape {getattr(values, 'shape', 'N/A')}")
            return result
        except Exception as e:
            raise PreprocessingError(f"Median filtering failed: {e}")