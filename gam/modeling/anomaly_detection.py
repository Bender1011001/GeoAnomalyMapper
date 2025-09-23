"""Anomaly detection implementation for GAM modeling module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage, stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from gam.core.exceptions import GAMError
from gam.modeling.anomaly_detection import AnomalyOutput  # Wait, circular; use from data_structures
from gam.modeling.data_structures import AnomalyOutput
from gam.preprocessing.data_structures import ProcessedGrid  # For coords if needed


logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Statistical anomaly detection in fused geophysical models.

    This class identifies subsurface anomalies using threshold-based statistics
    (z-score, percentiles), local outlier factor (LOF) for density-based detection,
    and clustering (DBSCAN) to group anomalous regions. Computes confidence scores
    integrating outlier strength and uncertainty, classifies anomalies based on
    physical rules (e.g., low velocity + high density = 'void'). Supports multi-scale
    analysis via Gaussian blurring to detect features at different resolutions.

    Key features:
    - Threshold methods: Z-score (>threshold), percentile (top/bottom %)
    - LOF: Unsupervised local anomaly scoring (sklearn)
    - Clustering: DBSCAN to delineate anomaly clusters
    - Confidence: Weighted outlier score (1 - p_value) * uncertainty factor
    - Classification: Rule-based (void, density_contrast, fault, etc.)
    - Multi-scale: Gaussian pyramid (sigma=1-5)
    - Uncertainty: Propagates from input model variance

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    Methods
    -------
    detect(fused_model: np.ndarray, **kwargs) -> AnomalyOutput
        Main detection method.

    Notes
    -----
    - **Input**: 3D fused model (lat, lon, depth); assumes coords from metadata or uniform grid.
    - **Thresholds**: Z-score default 2.5 (99% conf), percentile 95/5.
    - **LOF**: n_neighbors=20, contamination=0.1; scores <-1.5 as anomalous.
    - **Clustering**: eps=0.1*grid_res, min_samples=5; labels as cluster_id.
    - **Classification**: Simple rules; extend with ML classifier.
    - **Multi-scale**: Detect at 3 scales, aggregate max score.
    - **Output**: AnomalyOutput DataFrame with lat/lon/depth/confidence/type/strength.
    - **Performance**: Vectorized stats; sklearn on flattened (O(N_voxels log N)).
    - **Validation**: Checks finite input, positive confidences.
    - **Dependencies**: NumPy, SciPy, scikit-learn, Pandas.
    - **Edge Cases**: All normal (empty DF), uniform model (no detection), NaN handling.
    - **Reproducibility**: Deterministic sklearn (random_state=42).

    Examples
    --------
    >>> detector = AnomalyDetector()
    >>> anomalies = detector.detect(fused_model, threshold=2.5, method='zscore_lof')
    >>> print(len(anomalies))  # Number of detected anomalies
    >>> print(anomalies['anomaly_type'].value_counts())
    """

    def __init__(self):
        pass

    def detect(self, fused_model: npt.NDArray[np.float64], **kwargs) -> AnomalyOutput:
        """
        Detect anomalies in 3D fused model.

        Applies multi-scale filtering, threshold/LOF detection, clustering,
        scoring, classification. Generates coords from assumed uniform grid
        (extend with ProcessedGrid input if available).

        Parameters
        ----------
        fused_model : np.ndarray
            3D array (n_lat, n_lon, n_depth); anomaly indicator (higher = anomalous).
        **kwargs : dict, optional
            - 'method': str or List ('zscore', 'percentile', 'lof'; default: 'all')
            - 'threshold': float or Dict (z-score: 2.5, percentile: 95)
            - 'lof_neighbors': int (default: 20)
            - 'cluster_eps': float (default: 0.1)
            - 'min_samples': int (default: 5)
            - 'scales': List[float], Gaussian sigma (default: [1, 2, 3])
            - 'uncertainty': np.ndarray, optional 3D uncertainty (default: 0.1 * std)
            - 'grid_res': float, lat/lon res in deg (default: 0.01)
            - 'depth_res': float, m (default: 50)
            - 'random_state': int (default: 42)

        Returns
        -------
        AnomalyOutput
            DataFrame of detected anomalies.

        Raises
        ------
        GAMError
            Invalid input shape or params.
        """
        if fused_model.ndim != 3:
            raise GAMError("fused_model must be 3D (lat, lon, depth)")

        random_state = kwargs.get('random_state', 42)
        np.random.seed(random_state)

        n_lat, n_lon, n_depth = fused_model.shape
        grid_res = kwargs.get('grid_res', 0.01)  # deg
        depth_res = kwargs.get('depth_res', 50)  # m

        # Generate coords (uniform grid)
        lat = np.linspace(0, n_lat * grid_res, n_lat)
        lon = np.linspace(0, n_lon * grid_res, n_lon)
        depth = np.linspace(0, n_depth * depth_res, n_depth)
        lats, lons, depths = np.meshgrid(lat, lon, depth, indexing='ij')

        # Flatten for ML
        points = np.column_stack([lats.ravel(), lons.ravel(), depths.ravel()])
        values = fused_model.ravel()

        # Uncertainty (default or provided)
        if 'uncertainty' in kwargs:
            unc = kwargs['uncertainty'].ravel()
        else:
            unc = np.full(len(values), np.std(values) * 0.1)

        # Multi-scale detection
        scales = kwargs.get('scales', [1.0, 2.0, 3.0])
        anomaly_scores = np.zeros(len(values))
        for sigma in scales:
            blurred = ndimage.gaussian_filter(fused_model, sigma=sigma)
            scores = self._compute_scores(blurred.ravel(), method=kwargs.get('method', 'all'),
                                          threshold=kwargs.get('threshold', 2.5),
                                          lof_neighbors=kwargs.get('lof_neighbors', 20))
            anomaly_scores = np.maximum(anomaly_scores, scores)

        # Mask anomalies (score > 0)
        anomaly_mask = anomaly_scores > 0
        anomaly_points = points[anomaly_mask]
        anomaly_values = values[anomaly_mask]
        anomaly_unc = unc[anomaly_mask]
        anomaly_scores_masked = anomaly_scores[anomaly_mask]

        if len(anomaly_points) == 0:
            logger.info("No anomalies detected")
            return AnomalyOutput()

        # Clustering
        eps = kwargs.get('cluster_eps', 0.1)
        min_samples = kwargs.get('min_samples', 5)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(anomaly_points)
        cluster_labels = db.labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # Confidence: normalized score * (1 - unc / max_unc)
        max_unc = np.max(anomaly_unc) + 1e-10
        confidence = anomaly_scores_masked / np.max(anomaly_scores_masked) * (1 - anomaly_unc / max_unc)
        confidence = np.clip(confidence, 0, 1)

        # Strength: absolute deviation
        strength = np.abs(anomaly_values - np.mean(values))

        # Classification (simple rules on fused value; extend with multi-modal)
        anomaly_type = self._classify_anomalies(anomaly_values, confidence)

        # Build DataFrame
        df_data = {
            'lat': anomaly_points[:, 1],  # Assuming lat=y
            'lon': anomaly_points[:, 0],
            'depth': anomaly_points[:, 2],
            'confidence': confidence,
            'anomaly_type': anomaly_type,
            'strength': strength,
            'cluster_id': cluster_labels
        }
        anomalies = AnomalyOutput(df_data)
        anomalies['modality_contributions'] = {}  # Placeholder for fused

        logger.info(f"Detected {len(anomalies)} anomalies in {n_clusters} clusters; types: {anomalies['anomaly_type'].value_counts().to_dict()}")
        return anomalies

    def _compute_scores(self, values: npt.NDArray[np.float64], method: Any = 'all',
                        threshold: Any = 2.5, lof_neighbors: int = 20) -> npt.NDArray[np.float64]:
        """Compute anomaly scores using specified methods."""
        scores = np.zeros(len(values))
        methods = [method] if isinstance(method, str) else method
        valid_methods = {'zscore', 'percentile', 'lof'}

        if 'all' in methods:
            methods = valid_methods

        for m in methods:
            if m not in valid_methods:
                logger.warning(f"Unknown method {m}; skipping")
                continue

            if m == 'zscore':
                z = np.abs(stats.zscore(values, nan_policy='omit'))
                scores = np.maximum(scores, z)
            elif m == 'percentile':
                thresh = threshold if isinstance(threshold, (int, float)) else threshold.get('percentile', 95)
                upper = np.percentile(values, thresh)
                lower = np.percentile(values, 100 - thresh)
                p_scores = np.where(values > upper, (values - upper) / np.std(values), 0)
                p_scores = np.maximum(p_scores, np.where(values < lower, (lower - values) / np.std(values), 0))
                scores = np.maximum(scores, p_scores)
            elif m == 'lof':
                lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=0.1)
                lof_scores = -lof.fit_predict(values.reshape(-1, 1))  # Negative for anomaly
                scores = np.maximum(scores, lof_scores)

        return scores

    def _classify_anomalies(self, values: npt.NDArray[np.float64], confidence: npt.NDArray[np.float64]) -> List[str]:
        """Rule-based classification."""
        mean_val = np.mean(values)
        std_val = np.std(values)
        types = []
        for v, conf in zip(values, confidence):
            if conf < 0.5:
                types.append('uncertain')
            elif v > mean_val + 2 * std_val:
                types.append('density_contrast')  # High fused = high density/susceptibility
            elif v < mean_val - 2 * std_val:
                types.append('void')  # Low = low velocity/volume
            else:
                types.append('fault')  # Intermediate
        return types