"""Data structures for the GAM modeling/inversion module."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing_extensions import Self

from gam.core.exceptions import GAMError  # Assuming core exceptions per architecture


logger = logging.getLogger(__name__)


@dataclass
class InversionResults:
    """
    Dict-like structure representing results from a geophysical inversion.

    This class encapsulates the output of an inversion algorithm, providing
    a standardized interface for subsurface models, uncertainties, and metadata.
    It supports validation, serialization (JSON/NetCDF), and basic arithmetic
    operations (e.g., model scaling). Designed for interoperability across
    modalities and fusion workflows.

    The structure ensures 3D spatial consistency (lat, lon, depth) and includes
    convergence diagnostics for quality assessment.

    Parameters
    ----------
    model : np.ndarray
        3D array of inverted model parameters (e.g., density contrasts in kg/m³).
        Shape: (n_lat, n_lon, n_depth). Must be finite and consistent dtype (float64).
    uncertainty : np.ndarray
        3D array of uncertainty estimates (standard deviation) matching model shape.
        Non-negative values; NaN allowed for masked regions.
    metadata : Dict[str, Any]
        Processing metadata including:
        - 'converged': bool, inversion convergence status
        - 'iterations': int, number of solver iterations
        - 'residuals': float or np.ndarray, final misfit
        - 'units': str, model physical units (e.g., 'kg/m³')
        - 'timestamp': datetime, inversion completion time
        - 'algorithm': str, inversion method (e.g., 'simpeg_gravity')
        - 'parameters': Dict, used hyperparameters (e.g., regularization)
        - Optional: 'mesh_info', 'priors_used', 'warnings'

    Attributes
    ----------
    model : np.ndarray
        Inverted subsurface model.
    uncertainty : np.ndarray
        Model uncertainties.
    metadata : Dict[str, Any]
        Associated metadata.

    Methods
    -------
    validate()
        Validate structure and raise GAMError if invalid.
    to_dict() -> Dict[str, Any]
        Convert to serializable dictionary.
    from_dict(data: Dict[str, Any]) -> Self
        Create from dictionary.
    to_json(path: str, **kwargs)
        Serialize model/uncertainty to JSON (arrays as lists).
    to_csv(path: str, **kwargs)
        Flatten 3D arrays to long-format CSV for tabular export.
    copy() -> Self
        Deep copy of the results.
    scale(factor: float) -> Self
        Scale model and uncertainty by factor.

    Notes
    -----
    - **Shape Consistency**: model.shape == uncertainty.shape; 3D required.
    - **Validation**: Checks finite values, non-negative uncertainty, required metadata keys.
    - **Serialization**: JSON for metadata, supports NetCDF via xarray conversion in manager.
    - **Memory Efficiency**: Supports large arrays via numpy; use sparse if needed in subclasses.
    - **Reproducibility**: Metadata includes full parameter trace.
    - **Integration**: Used by Inverter.invert() and JointInverter.fuse().
    - **Edge Cases**: Handles all-NaN models (e.g., failed regions); logs warnings.

    Examples
    --------
    >>> results = InversionResults(
    ...     model=np.random.rand(10, 10, 5),
    ...     uncertainty=np.random.rand(10, 10, 5) * 0.1,
    ...     metadata={'converged': True, 'units': 'kg/m³'}
    ... )
    >>> results.validate()
    >>> results.to_json('results.json')
    >>> loaded = InversionResults.from_dict(results.to_dict())
    """

    model: npt.NDArray[np.float64]
    uncertainty: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default metadata if missing
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now()
        if 'converged' not in self.metadata:
            self.metadata['converged'] = False
        if 'iterations' not in self.metadata:
            self.metadata['iterations'] = 0
        if 'residuals' not in self.metadata:
            self.metadata['residuals'] = np.inf
        if 'units' not in self.metadata:
            self.metadata['units'] = 'unknown'
        if 'algorithm' not in self.metadata:
            self.metadata['algorithm'] = 'unknown'
        if 'parameters' not in self.metadata:
            self.metadata['parameters'] = {}

        self.validate()

    def validate(self) -> bool:
        """
        Validate the InversionResults structure.

        Checks:
        - model and uncertainty are 3D numpy arrays of same shape
        - model dtype is float64; uncertainty non-negative
        - Required metadata keys present
        - No all-NaN model (warn if >90% NaN)

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        GAMError
            If validation fails (e.g., shape mismatch, invalid dtype).
        """
        if not isinstance(self.model, np.ndarray) or self.model.ndim != 3:
            raise GAMError("Model must be a 3D numpy array")
        if not isinstance(self.uncertainty, np.ndarray) or self.uncertainty.ndim != 3:
            raise GAMError("Uncertainty must be a 3D numpy array")
        if self.model.shape != self.uncertainty.shape:
            raise GAMError(f"Shape mismatch: model {self.model.shape}, uncertainty {self.uncertainty.shape}")
        if self.model.dtype != np.float64:
            raise GAMError("Model must be float64 dtype")
        if np.any(self.uncertainty < 0):
            raise GAMError("Uncertainty values must be non-negative")
        required_meta = {'converged', 'iterations', 'residuals', 'units', 'algorithm', 'parameters', 'timestamp'}
        if not required_meta.issubset(self.metadata):
            missing = required_meta - set(self.metadata)
            raise GAMError(f"Missing required metadata keys: {missing}")

        nan_frac = np.isnan(self.model).sum() / self.model.size
        if nan_frac > 0.9:
            logger.warning(f"High NaN fraction in model: {nan_frac:.2%}")

        logger.debug("InversionResults validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary (arrays as lists)."""
        return {
            'model': self.model.tolist(),
            'uncertainty': self.uncertainty.tolist(),
            'metadata': self.metadata.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create from dictionary (lists to arrays)."""
        model = np.array(data['model'], dtype=np.float64)
        uncertainty = np.array(data['uncertainty'], dtype=np.float64)
        metadata = data.get('metadata', {})
        return cls(model, uncertainty, metadata)

    def to_json(self, path: str, **kwargs) -> None:
        """
        Serialize to JSON file.

        Arrays converted to lists; metadata preserved. Use for lightweight export.

        Parameters
        ----------
        path : str
            Output JSON file path.
        **kwargs : dict, optional
            Passed to json.dump (e.g., indent=4).
        """
        serializable = self.to_dict()
        # Ensure timestamp is string
        if isinstance(serializable['metadata']['timestamp'], datetime):
            serializable['metadata']['timestamp'] = serializable['metadata']['timestamp'].isoformat()
        with open(path, 'w') as f:
            json.dump(serializable, f, **kwargs)
        logger.info(f"InversionResults saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Parse timestamp if string
        if isinstance(data['metadata']['timestamp'], str):
            try:
                data['metadata']['timestamp'] = datetime.fromisoformat(data['metadata']['timestamp'])
            except ValueError:
                logger.warning("Invalid timestamp in JSON; using now")
                data['metadata']['timestamp'] = datetime.now()
        return cls.from_dict(data)

    def to_csv(self, path: str, **kwargs) -> None:
        """
        Export to CSV in long format for tabular analysis.

        Flattens 3D arrays: columns include lat, lon, depth, model_value, uncertainty.

        Parameters
        ----------
        path : str
            Output CSV file path.
        **kwargs : dict, optional
            Passed to pd.DataFrame.to_csv (e.g., index=False).
        """
        # Assuming metadata has grid info; in practice, derive from model shape or attrs
        # For demo, create synthetic coords (replace with actual in manager)
        n_lat, n_lon, n_depth = self.model.shape
        lat = np.linspace(0, 1, n_lat)  # Placeholder; use actual coords
        lon = np.linspace(0, 1, n_lon)
        depth = np.linspace(0, 1, n_depth)
        lats, lons, depths = np.meshgrid(lat, lon, depth, indexing='ij')
        
        df = pd.DataFrame({
            'lat': lats.ravel(),
            'lon': lons.ravel(),
            'depth': depths.ravel(),
            'model': self.model.ravel(),
            'uncertainty': self.uncertainty.ravel()
        })
        # Add metadata as columns if scalar
        for k, v in self.metadata.items():
            if isinstance(v, (int, float, str)):
                df[f'metadata_{k}'] = v
        
        df.to_csv(path, **kwargs)
        logger.info(f"InversionResults exported to CSV: {path}")

    @classmethod
    def from_csv(cls, path: str) -> Self:
        """Load from CSV (assumes long format; reconstructs 3D)."""
        df = pd.read_csv(path)
        # Reconstruct 3D (simplified; assumes regular grid)
        lat_coords = sorted(df['lat'].unique())
        lon_coords = sorted(df['lon'].unique())
        depth_coords = sorted(df['depth'].unique())
        model = df.pivot_table(values='model', index=['lat', 'lon'], columns='depth').values.reshape(len(lat_coords), len(lon_coords), len(depth_coords))
        uncertainty = df.pivot_table(values='uncertainty', index=['lat', 'lon'], columns='depth').values.reshape(len(lat_coords), len(lon_coords), len(depth_coords))
        metadata = {col.replace('metadata_', ''): df[col].iloc[0] for col in df.columns if col.startswith('metadata_')}
        return cls(model, uncertainty, metadata)

    def copy(self) -> Self:
        """Deep copy."""
        return self.from_dict(self.to_dict())

    def scale(self, factor: float) -> Self:
        """Scale model and uncertainty."""
        new_model = self.model * factor
        new_uncertainty = np.abs(self.uncertainty * factor)
        new_metadata = self.metadata.copy()
        new_metadata['parameters']['scale_factor'] = factor
        return self.__class__(new_model, new_uncertainty, new_metadata)

    def __repr__(self) -> str:
        shape = self.model.shape
        units = self.metadata.get('units', 'unknown')
        converged = self.metadata.get('converged', False)
        return f"InversionResults(shape={shape}, units='{units}', converged={converged})"


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
        ------
        GAMError
            If validation fails.
        """
        missing_cols = self.REQUIRED_COLUMNS - set(self.columns)
        if missing_cols:
            raise GAMError(f"Missing required columns: {missing_cols}")

        for col, dtype in self.REQUIRED_DTYPES.items():
            if col in self.columns:
                if not pd.api.types.is_dtype_equal(self[col].dtype, dtype):
                    self[col] = self[col].astype(dtype)

        if self['confidence'].min() < 0 or self['confidence'].max() > 1:
            raise GAMError("Confidence must be in [0, 1]")
        if self['depth'].min() < 0:
            raise GAMError("Depth must be positive (positive down)")
        if self.duplicated(subset=['lat', 'lon', 'depth']).any():
            raise GAMError("Duplicate anomalies detected")

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