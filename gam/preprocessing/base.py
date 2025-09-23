"""Abstract base class for the GAM preprocessing module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from gam.ingestion.data_structures import RawData


class ProcessedGrid:  # Forward reference placeholder; full impl in data_structures.py
    """ProcessedGrid xarray.Dataset representation."""


class Preprocessor(ABC):
    """
    Abstract base class for geophysical data preprocessors.

    This class defines the interface for cleaning, filtering, gridding, and
    transforming raw geophysical data into a standardized processed grid format.
    Subclasses must implement the `process` method to handle modality-specific
    preprocessing pipelines.

    All preprocessors must ensure:
    - Memory efficiency for large datasets (e.g., via chunking)
    - Reproducibility (deterministic operations)
    - Proper error handling and logging
    - Output as ProcessedGrid (xarray.Dataset) with WGS84 coordinates

    Parameters
    ----------
    None
        Instantiated without parameters; configuration via **kwargs in process().

    Attributes
    ----------
    None

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid
        Main processing method.

    Notes
    -----
    - Follows the GAM architecture: Integrates with RawData from ingestion.
    - Supports parallel execution via Dask when wrapped in parallel.py.
    - Docstrings follow NumPy style for consistency.

    Examples
    --------
    >>> from gam.preprocessing.processors import GravityPreprocessor
    >>> preprocessor = GravityPreprocessor()
    >>> processed = preprocessor.process(raw_data, grid_resolution=0.05, apply_filters=True)
    """

    @abstractmethod
    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process raw geophysical data into a standardized grid.

        This method performs modality-specific preprocessing including:
        - Data cleaning and validation
        - Filtering (noise, outliers)
        - Unit conversion
        - Gridding and interpolation
        - Coordinate alignment to WGS84

        Parameters
        ----------
        data : RawData
            Raw data from ingestion module, containing metadata and values.
        **kwargs : dict, optional
            Processing parameters, e.g.:
            - grid_resolution: float, grid spacing in degrees (default: 0.1)
            - apply_filters: bool or list[str], filters to apply (default: True)
            - config: dict, additional modality-specific settings
            - parallel: bool, enable Dask parallel processing (default: False)

        Returns
        -------
        ProcessedGrid
            xarray.Dataset with dimensions (lat, lon, depth/time), variables
            ('data', 'uncertainty'), coordinates (lat, lon, depth), and attrs
            (units, processed_at, etc.).

        Raises
        ------
        PreprocessingError
            If processing fails (e.g., invalid data, convergence issues).
        ValueError
            If required kwargs are missing or data validation fails.

        Notes
        -----
        - Validates input RawData using its built-in validate() method.
        - Logs processing steps and parameters for reproducibility.
        - Handles missing values via interpolation or masking.
        - Ensures output is chunked for large datasets if parallel=True.

        Examples
        --------
        >>> kwargs = {'grid_resolution': 0.05, 'filters': ['noise', 'outlier']}
        >>> grid = preprocessor.process(raw_data, **kwargs)
        >>> print(grid.data.shape)  # e.g., (lat: 100, lon: 100)
        """
        pass