"""Abstract base classes for the GAM data ingestion module."""

from abc import ABC, abstractmethod
from typing import Tuple, Any


class DataSource(ABC):
    r"""
    Abstract base class for geophysical data sources.

    This class defines the interface that all concrete data source implementations
    must follow. Subclasses are responsible for fetching raw geophysical data from
    specific public APIs or sources.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Fetch raw data for the specified bounding box.

    Notes
    -----
    Implementations must handle authentication, rate limiting, and data parsing
    specific to their source. All data should be returned as RawData objects
    for standardization.

    Examples
    --------
    >>> class MySource(DataSource):
    ...     def fetch_data(self, bbox, **kwargs):
    ...         # Implementation
    ...         pass
    """

    @abstractmethod
    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> 'RawData':
        r"""
        Fetch raw geophysical data for the given bounding box.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            Bounding box in the format (min_lat, max_lat, min_lon, max_lon).
            Coordinates in decimal degrees (WGS84).
        **kwargs : dict, optional
            Additional modality-specific parameters, such as date range,
            resolution, or authentication tokens.

        Returns
        -------
        RawData
            A RawData object containing metadata and values from the source.

        Raises
        ------
        IngestionError
            If the data fetch operation fails due to network issues, API errors,
            or invalid parameters.
        IngestionError
            If the API request times out.
        ValueError
            If the bounding box is invalid (e.g., min > max).

        Notes
        -----
        The bounding box should be validated for geographic reasonableness
        (e.g., latitudes between -90 and 90). Implementations should respect
        API rate limits and include appropriate retry logic.

        Examples
        --------
        >>> source = MySource()
        >>> data = source.fetch_data((30.0, 32.0, -120.0, -118.0), start_date='2023-01-01')
        """
        pass