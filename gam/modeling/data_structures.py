"""Data structures for the GAM modeling/inversion module."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from dataclasses import dataclass, asdict, field
from typing_extensions import Self

from gam.core.exceptions import DataValidationError


logger = logging.getLogger(__name__)


@dataclass
class ProcessedGrid:
    """
    Dataclass representing processed geophysical grid data.

    This structure standardizes gridded data after preprocessing, using xarray.DataArray
    as the core container for spatial data with coordinates and metadata.

    Parameters
    ----------
    data : xr.DataArray
        The processed grid data as an xarray DataArray.

    Attributes
    ----------
    data : xr.DataArray
        Processed grid data.

    Methods
    -------
    validate()
        Validate the data structure and raise DataValidationError if invalid.
    save_netcdf(path)
        Save to NetCDF file, preserving CRS.
    load_netcdf(path)
        Classmethod to load from NetCDF file.

    Notes
    -----
    Uses DataValidationError from gam.core.exceptions for validation failures.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 10), coords={'lat': range(10), 'lon': range(10)}, attrs={'crs': 'EPSG:4326'})
    >>> grid = ProcessedGrid(da)
    >>> grid.validate()
    >>> grid.save_netcdf('grid.nc')
    """
    data: xr.DataArray

    def validate(self) -> None:
        """
        Validate the ProcessedGrid instance.

        Checks:
        - data is an xarray.DataArray
        - 'lat' and 'lon' coordinates exist
        - 'crs' attribute is present and non-null in attrs

        Raises
        ------
        DataValidationError
            If any validation fails.
        """
        if not isinstance(self.data, xr.DataArray):
            raise DataValidationError("data must be an xarray.DataArray")
        if 'lat' not in self.data.coords:
            raise DataValidationError("lat coordinate must exist")
        if 'lon' not in self.data.coords:
            raise DataValidationError("lon coordinate must exist")
        if 'crs' not in self.data.attrs or self.data.attrs['crs'] is None:
            raise DataValidationError("crs attribute must be present and non-null in attrs")
        logger.debug("ProcessedGrid validation passed")

    def save_netcdf(self, path: str) -> None:
        """
        Save the ProcessedGrid to a NetCDF file.

        Preserves CRS and coordinates.

        Parameters
        ----------
        path : str
            File path to save to.
        """
        self.data.to_netcdf(path)
        logger.info(f"ProcessedGrid saved to {path}")

    @classmethod
    def load_netcdf(cls, path: str) -> "ProcessedGrid":
        """
        Load ProcessedGrid from a NetCDF file.

        Parameters
        ----------
        path : str
            File path to load from.

        Returns
        -------
        ProcessedGrid
            Loaded instance.
        """
        data = xr.open_dataarray(path)
        instance = cls(data)
        instance.validate()
        return instance


@dataclass
class InversionResults:
    """
    Dataclass representing results from a geophysical inversion.

    This structure encapsulates the output of an inversion algorithm, providing
    a standardized interface for subsurface models, uncertainties, and metadata.
    Uses xarray.DataArray for model and uncertainty to maintain spatial coordinates.

    Parameters
    ----------
    model : xr.DataArray
        Inverted subsurface model as xarray DataArray.
    uncertainty : xr.DataArray
        Uncertainty estimates matching model shape and coordinates.
    metadata : Dict[str, Any]
        Processing metadata (e.g., convergence status, units, algorithm).

    Attributes
    ----------
    model : xr.DataArray
        Inverted subsurface model.
    uncertainty : xr.DataArray
        Model uncertainties.
    metadata : Dict[str, Any]
        Associated metadata.

    Methods
    -------
    validate()
        Validate structure and raise DataValidationError if invalid.
    to_dict()
        Convert to dictionary for JSON serialization.
    from_dict(cls, data)
        Classmethod to create instance from dictionary.

    Notes
    -----
    Uses DataValidationError from gam.core.exceptions for validation failures.
    Serialization handles xarray objects appropriately.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> model_da = xr.DataArray(np.random.rand(10, 10), coords={'lat': range(10), 'lon': range(10)})
    >>> uncertainty_da = xr.DataArray(np.random.rand(10, 10) * 0.1, coords=model_da.coords)
    >>> results = InversionResults(model_da, uncertainty_da, {'units': 'kg/m³'})
    >>> results.validate()
    >>> results.to_dict()
    """

    model: xr.DataArray
    uncertainty: xr.DataArray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the InversionResults instance.

        Checks:
        - model and uncertainty are xarray.DataArray objects
        - Their shapes and coordinates match exactly

        Raises
        ------
        DataValidationError
            If validation fails.
        """
        if not isinstance(self.model, xr.DataArray):
            raise DataValidationError("model must be an xarray.DataArray")
        if not isinstance(self.uncertainty, xr.DataArray):
            raise DataValidationError("uncertainty must be an xarray.DataArray")
        if self.model.shape != self.uncertainty.shape:
            raise DataValidationError(f"Shape mismatch: model {self.model.shape}, uncertainty {self.uncertainty.shape}")
        if not self.model.coords.equals(self.uncertainty.coords):
            raise DataValidationError("model and uncertainty coordinates must match exactly")
        logger.debug("InversionResults validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Handles xarray objects by converting to dict representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'model', 'uncertainty', and 'metadata'.
        """
        return {
            'model': self.model.to_dict(),
            'uncertainty': self.uncertainty.to_dict(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InversionResults":
        """
        Create InversionResults instance from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with 'model', 'uncertainty', and 'metadata' keys.

        Returns
        -------
        InversionResults
            New instance.

        Raises
        ------
        DataValidationError
            If dictionary structure is invalid.
        """
        try:
            model = xr.DataArray.from_dict(data['model'])
            uncertainty = xr.DataArray.from_dict(data['uncertainty'])
            metadata = data.get('metadata', {})
            instance = cls(model, uncertainty, metadata)
            instance.validate()
            return instance
        except KeyError as e:
            raise DataValidationError(f"Missing required key in dictionary: {e}")
        except Exception as e:
            raise DataValidationError(f"Invalid data format: {e}")

    def __repr__(self) -> str:
        shape = self.model.shape
        return f"InversionResults(shape={shape})"


class AnomalyOutput(pd.DataFrame):
    """
    Pandas DataFrame extension for anomaly detection outputs.

    This class extends pd.DataFrame to represent detected subsurface anomalies
    with standardized columns and validation. It includes methods for confidence
    scoring, serialization (JSON/CSV), and basic filtering/clustering support.
    Designed for export to visualization and database storage.

    Required columns:
    - 'lat': float, latitude in degrees
    - 'lon': float, longitude in degrees
    - 'depth': float, depth in meters (positive down)
    - 'confidence': float, 0-1 anomaly probability
    - 'anomaly_type': str, e.g., 'void', 'density_contrast', 'fault'
    - 'strength': float, anomaly magnitude (modality-specific units)

    Optional columns:
    - 'uncertainty': float, position/strength uncertainty
    - 'modality_contributions': Dict[str, float], per-modality weights
    - 'cluster_id': int, for grouped anomalies
    - 'timestamp': datetime, detection time

    Parameters
    ----------
    data : dict or pd.DataFrame
        Input data with required columns.
    **kwargs : dict, optional
        Additional DataFrame kwargs.

    Attributes
    ----------
    Inherits all pd.DataFrame attributes.

    Methods
    -------
    validate()
        Ensure required columns and data types.
    to_json(path: str, **kwargs)
        Export to JSON (orient='records').
    to_csv(path: str, **kwargs)
        Export to CSV.
    filter_confidence(min_conf: float) -> Self
        Filter rows by minimum confidence.
    add_cluster_labels(n_clusters: int) -> Self
        Add clustering using sklearn (e.g., KMeans on lat/lon/depth).
    compute_confidence(method: str = 'zscore') -> Self
        Update confidence based on strength (z-score or percentile).

    Notes
    -----
    - **Validation**: Enforces required columns, numeric types, confidence [0,1].
    - **Serialization**: Handles dict columns (modality_contributions) as JSON strings in CSV.
    - **Clustering**: Uses sklearn.cluster; requires installation.
    - **Performance**: Vectorized operations; suitable for large anomaly sets.
    - **Integration**: Output of AnomalyDetector; input to Visualizer.
    - **Edge Cases**: Empty DataFrame allowed; warns on invalid types.

    Examples
    --------
    >>> data = {
    ...     'lat': [40.0], 'lon': [-100.0], 'depth': [500.0],
    ...     'confidence': [0.85], 'anomaly_type': ['void'], 'strength': [-0.5]
    ... }
    >>> anomalies = AnomalyOutput(data)
    >>> anomalies.validate()
    >>> filtered = anomalies.filter_confidence(0.8)
    >>> anomalies.to_csv('anomalies.csv')
    """

    REQUIRED_COLUMNS = {'lat', 'lon', 'depth', 'confidence', 'anomaly_type', 'strength'}
    REQUIRED_DTYPES = {
        'lat': 'float64', 'lon': 'float64', 'depth': 'float64',
        'confidence': 'float64', 'strength': 'float64', 'anomaly_type': 'object'
    }

    def __init__(self, data: Optional[Union[Dict[str, Any], pd.DataFrame]] = None, **kwargs):
        if data is None:
            data = pd.DataFrame()
        super().__init__(data, **kwargs)
        self.validate()
        # Set default timestamp if missing
        if 'timestamp' not in self.columns:
            self['timestamp'] = datetime.now()

    def validate(self) -> bool:
        """
        Validate the AnomalyOutput structure.

        Checks:
        - All required columns present
        - Data types match (numeric for coords/strength/confidence)
        - Confidence in [0,1]; depth > 0
        - No duplicate anomalies (lat/lon/depth)

        Returns
        -------
        bool
            True if valid.

        Raises
        -------
        DataValidationError
            If validation fails.
        """
        missing_cols = self.REQUIRED_COLUMNS - set(self.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        for col, dtype in self.REQUIRED_DTYPES.items():
            if col in self.columns:
                if not pd.api.types.is_dtype_equal(self[col].dtype, dtype):
                    self[col] = self[col].astype(dtype)

        if self['confidence'].min() < 0 or self['confidence'].max() > 1:
            raise DataValidationError("Confidence must be in [0, 1]")
        if self['depth'].min() < 0:
            raise DataValidationError("Depth must be positive (positive down)")
        if self.duplicated(subset=['lat', 'lon', 'depth']).any():
            raise DataValidationError("Duplicate anomalies detected")

        # Handle dict columns (serialize to JSON string for DataFrame compatibility)
        if 'modality_contributions' in self.columns:
            self['modality_contributions'] = self['modality_contributions'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

        logger.debug("AnomalyOutput validation passed")
        return True

    def to_json(self, path: str, **kwargs) -> None:
        """
        Serialize to JSON file.

        Uses orient='records' for list of dicts.

        Parameters
        ----------
        path : str
            Output JSON path.
        **kwargs : dict, optional
            Passed to df.to_json.
        """
        # Deserialize dict columns for JSON
        if 'modality_contributions' in self.columns:
            self_copy = self.copy()
            self_copy['modality_contributions'] = self_copy['modality_contributions'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        else:
            self_copy = self
        self_copy.to_json(path, orient='records', **kwargs)
        logger.info(f"AnomalyOutput saved to {path}")

    def to_csv(self, path: str, **kwargs) -> None:
        """
        Serialize to CSV.

        Dict columns saved as JSON strings.

        Parameters
        ----------
        path : str
            Output CSV path.
        **kwargs : dict, optional
            Passed to df.to_csv.
        """
        # Ensure dict columns are strings
        df_to_save = self.copy()
        if 'modality_contributions' in df_to_save.columns:
            df_to_save['modality_contributions'] = df_to_save['modality_contributions'].astype(str)
        df_to_save.to_csv(path, **kwargs)
        logger.info(f"AnomalyOutput exported to {path}")

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> Self:
        """Load from CSV, parsing JSON strings in dict columns."""
        df = pd.read_csv(path, **kwargs)
        # Parse modality_contributions if present
        if 'modality_contributions' in df.columns:
            df['modality_contributions'] = df['modality_contributions'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith('{') else {}
            )
        # Parse timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return cls(df)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> Self:
        """Load from JSON."""
        df = pd.read_json(path, **kwargs)
        return cls(df)

    def filter_confidence(self, min_conf: float) -> Self:
        """
        Filter anomalies by minimum confidence.

        Parameters
        ----------
        min_conf : float
            Minimum confidence threshold (0-1).

        Returns
        -------
        AnomalyOutput
            Filtered instance.
        """
        if min_conf < 0 or min_conf > 1:
            raise ValueError("min_conf must be in [0, 1]")
        filtered = self[self['confidence'] >= min_conf].copy()
        return self.__class__(filtered)

    def add_cluster_labels(self, n_clusters: int = 3) -> Self:
        """
        Add cluster labels using KMeans on spatial coordinates.

        Requires sklearn.cluster.KMeans.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (default: 3).

        Returns
        -------
        AnomalyOutput
            Copy with 'cluster_id' column.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("sklearn required for clustering; install with 'pip install scikit-learn'")
        
        if len(self) < n_clusters:
            logger.warning("Fewer anomalies than clusters; using 1 cluster")
            n_clusters = 1
        
        coords = self[['lat', 'lon', 'depth']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self['cluster_id'] = kmeans.fit_predict(coords)
        return self

    def compute_confidence(self, method: str = 'zscore') -> Self:
        """
        Update confidence scores based on strength.

        Methods:
        - 'zscore': Normalize strength to z-scores, map to [0,1]
        - 'percentile': Use empirical CDF for [0,1] mapping

        Parameters
        ----------
        method : str, optional
            Scoring method (default: 'zscore').

        Returns
        -------
        AnomalyOutput
            Copy with updated 'confidence'.
        """
        if method == 'zscore':
            mean = self['strength'].mean()
            std = self['strength'].std()
            if std == 0:
                self['confidence'] = 0.5
            else:
                z = (self['strength'] - mean) / std
                self['confidence'] = (1 + np.tanh(z / 2)) / 2  # Map to [0,1]
        elif method == 'percentile':
            self['confidence'] = self['strength'].rank(pct=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self['confidence'] = np.clip(self['confidence'], 0, 1)
        return self

    def __repr__(self) -> str:
        n_anoms = len(self)
        conf_mean = self['confidence'].mean() if n_anoms > 0 else 0
        types = self['anomaly_type'].unique() if n_anoms > 0 else []
        return f"AnomalyOutput(n_anomalies={n_anoms}, mean_confidence={conf_mean:.3f}, types={list(types)})"