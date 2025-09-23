"""Data structures for the GAM data ingestion module."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple, Optional
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


@dataclass
class RawData:
    r"""
    Dataclass representing raw geophysical data fetched from a source.

    This structure standardizes the representation of raw data across different
    modalities (gravity, seismic, etc.). It includes metadata for traceability
    and context, and flexible values for modality-specific data formats.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Dictionary containing metadata about the data.
        Required keys:
        - 'source': str, the data source (e.g., 'USGS Gravity')
        - 'timestamp': datetime, when the data was fetched
        - 'bbox': Tuple[float, float, float, float], (min_lat, max_lat, min_lon, max_lon)
        - 'parameters': Dict[str, Any], fetch parameters (e.g., date range)
        Optional keys: 'units', 'resolution', 'count', etc.
    values : Any
        The actual data values. Modality-specific format:
        - Gravity/Magnetic: np.ndarray of shape (n_points,) or (n_points, features)
        - Seismic: obspy.Stream or list of traces
        - InSAR: xarray.Dataset or raster array
        Must be serializable for caching.

    Attributes
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary.
    values : Any
        Raw data values.

    Methods
    -------
    validate()
        Validate the data structure and raise ValueError if invalid.
    to_dict()
        Convert to dictionary for serialization.
    from_dict(cls, data)
        Classmethod to create instance from dictionary.
    __str__()
        String representation for logging/printing.
    __repr__()
        Official representation.

    Notes
    -----
    Validation occurs in __post_init__ for basic checks. Full validation via
    validate() method. This class is immutable post-validation for thread-safety.

    Examples
    --------
    >>> metadata = {
    ...     'source': 'USGS Gravity',
    ...     'timestamp': datetime.now(),
    ...     'bbox': (29.0, 31.0, 30.0, 32.0),
    ...     'parameters': {'date': '2023-01-01'}
    ... }
    >>> data = RawData(metadata, values=np.array([1.2, 3.4]))
    >>> data.validate()
    >>> print(data)
    RawData(source='USGS Gravity', bbox=(29.0, 31.0, 30.0, 32.0), values=array([1.2, 3.4]))
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    values: Any = None

    def __post_init__(self):
        """Perform basic validation after initialization."""
        if self.values is None:
            raise ValueError("RawData requires non-None values")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        # Basic metadata checks; full validation in validate()
        if 'source' not in self.metadata:
            logger.warning("Metadata missing 'source' key")
        if 'timestamp' not in self.metadata or not isinstance(self.metadata['timestamp'], datetime):
            logger.warning("Metadata missing or invalid 'timestamp'")
        if 'bbox' not in self.metadata or len(self.metadata['bbox']) != 4:
            raise ValueError("Metadata 'bbox' must be a 4-tuple")

    def validate(self) -> bool:
        r"""
        Validate the RawData instance.

        Checks:
        - bbox: 4 floats, min_lat < max_lat (-90 <= min_lat < max_lat <= 90),
                min_lon < max_lon (-180 <= min_lon < max_lon <= 180)
        - timestamp: datetime object
        - source: non-empty str
        - parameters: dict
        - values: not None, basic type check (e.g., array-like or Dataset)

        Returns
        -------
        bool
            True if valid, raises ValueError if invalid.

        Raises
        ------
        ValueError
            If any validation fails.

        Notes
        -----
        This method should be called after construction to ensure data integrity
        before processing or caching.

        Examples
        --------
        >>> data.validate()  # Raises ValueError if invalid
        True
        """
        # Validate bbox
        bbox = self.metadata.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must be a 4-element tuple/list of floats")
        min_lat, max_lat, min_lon, max_lon = map(float, bbox)
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("Invalid latitude range in bbox")
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("Invalid longitude range in bbox")

        # Validate timestamp
        timestamp = self.metadata.get('timestamp')
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")

        # Validate source
        source = self.metadata.get('source')
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        # Validate parameters
        params = self.metadata.get('parameters', {})
        if not isinstance(params, dict):
            raise ValueError("parameters must be a dictionary")

        # Basic values check (extend for specific modalities if needed)
        if self.values is None:
            raise ValueError("values cannot be None")

        logger.debug(f"RawData validated successfully for source '{source}'")
        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RawData:
        r"""
        Create RawData instance from a dictionary (e.g., from JSON or cache).

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with 'metadata' and 'values' keys.
            'values' will be reconstructed if serializable.

        Returns
        -------
        RawData
            New instance.

        Raises
        ------
        ValueError
            If dictionary structure is invalid.
        TypeError
            If values cannot be deserialized.

        Notes
        -----
        Assumes values are JSON-serializable or use custom deserializer.
        Timestamp string will be parsed to datetime.

        Examples
        --------
        >>> data_dict = {'metadata': {...}, 'values': [1.2, 3.4]}
        >>> data = RawData.from_dict(data_dict)
        """
        metadata = data.get('metadata', {})
        # Parse timestamp if string
        if 'timestamp' in metadata and isinstance(metadata['timestamp'], str):
            metadata['timestamp'] = datetime.fromisoformat(metadata['timestamp'])
        values = data.get('values')
        instance = cls(metadata, values)
        instance.validate()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Convert RawData to dictionary for serialization (e.g., JSON, cache).

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'metadata' and 'values'.
            Timestamp converted to ISO string for JSON compatibility.

        Notes
        -----
        Values must be serializable; complex objects (e.g., xarray) may need
        custom handling (e.g., to_netcdf path).

        Examples
        --------
        >>> data_dict = data.to_dict()
        >>> # Can be json.dumps(data_dict)
        """
        metadata = dict(self.metadata)
        if 'timestamp' in metadata:
            metadata['timestamp'] = metadata['timestamp'].isoformat()
        return {
            'metadata': metadata,
            'values': self.values  # Assume serializable
        }

    def __str__(self) -> str:
        """String representation for logging and printing."""
        source = self.metadata.get('source', 'Unknown')
        bbox = self.metadata.get('bbox', 'Unknown')
        count = len(self.values) if hasattr(self.values, '__len__') else 'N/A'
        return f"RawData(source='{source}', bbox={bbox}, count={count}, values_type={type(self.values).__name__})"

    def __repr__(self) -> str:
        """Official representation including all fields."""
        return f"RawData(metadata={self.metadata}, values={self.values})"