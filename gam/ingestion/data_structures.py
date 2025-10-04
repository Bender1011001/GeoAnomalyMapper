"""Data structures for the GAM data ingestion module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Union
import numpy as np
import logging
from gam.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class RawData:
    """
    Dataclass representing raw geophysical data.

    This structure standardizes the representation of raw data fetched from ingestion sources.
    It contains the core data array, associated metadata, and coordinate reference system information.

    Parameters
    ----------
    data : np.ndarray
        The raw data array. Must be a NumPy array and not empty.
    metadata : Dict[str, Any]
        Dictionary containing metadata about the data (e.g., source, timestamp, units).
    crs : Union[str, int]
        Coordinate reference system identifier (e.g., 'EPSG:4326' or integer EPSG code).

    Attributes
    ----------
    data : np.ndarray
        Raw data values.
    metadata : Dict[str, Any]
        Associated metadata.
    crs : Union[str, int]
        CRS identifier.

    Methods
    -------
    validate()
        Validate the data structure and raise DataValidationError if invalid.
    to_dict()
        Convert to dictionary for JSON serialization.
    from_dict(cls, data)
        Classmethod to create instance from dictionary.

    Notes
    -----
    This class uses DataValidationError from gam.core.exceptions for validation failures.
    Serialization handles NumPy arrays by converting to lists.

    Examples
    --------
    >>> data = np.array([1.2, 3.4, 5.6])
    >>> metadata = {'source': 'USGS', 'units': 'mGal'}
    >>> raw = RawData(data, metadata, 'EPSG:4326')
    >>> raw.validate()
    >>> raw_dict = raw.to_dict()
    """
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    crs: Union[str, int] = "EPSG:4326"

    def validate(self) -> None:
        """
        Validate the RawData instance.

        Checks:
        - data is a NumPy array and not empty
        - metadata is a dictionary
        - crs is a non-empty string or integer

        Raises
        ------
        DataValidationError
            If any validation fails.
        """
        if not isinstance(self.data, np.ndarray):
            raise DataValidationError("data must be a NumPy array")
        if self.data.size == 0:
            raise DataValidationError("data array must not be empty")
        if not isinstance(self.metadata, dict):
            raise DataValidationError("metadata must be a dictionary")
        if not isinstance(self.crs, (str, int)) or (isinstance(self.crs, str) and not self.crs.strip()):
            raise DataValidationError("crs must be a non-empty string or integer")
        logger.debug("RawData validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RawData to dictionary for JSON serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'data', 'metadata', and 'crs' keys.
            NumPy array is converted to list.
        """
        return {
            'data': self.data.tolist(),
            'metadata': self.metadata,
            'crs': self.crs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawData":
        """
        Create RawData instance from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with 'data', 'metadata', and 'crs' keys.
            'data' should be a list convertible to NumPy array.

        Returns
        -------
        RawData
            New instance.

        Raises
        ------
        DataValidationError
            If dictionary structure is invalid.
        """
        try:
            data_array = np.array(data['data'])
            metadata = data.get('metadata', {})
            crs = data.get('crs', 'EPSG:4326')
            instance = cls(data_array, metadata, crs)
            instance.validate()
            return instance
        except KeyError as e:
            raise DataValidationError(f"Missing required key in dictionary: {e}")
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Invalid data format: {e}")