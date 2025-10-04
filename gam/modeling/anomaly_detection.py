"""Anomaly detection implementation for GAM modeling module."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import xarray as xr

from gam.core.exceptions import GAMError
from gam.modeling.data_structures import InversionResults

logger = logging.getLogger(__name__)


def detect_anomalies(inversion_results: InversionResults, method: str = "percentile", **kwargs) -> pd.DataFrame:
    """
    Detect anomalies in inversion results using percentile-based thresholding.

    This function implements a robust, configurable anomaly detection method that identifies
    grid cells where the model values exceed a specified percentile threshold. The method
    is statistically defensible and suitable for geophysical data analysis.

    Parameters
    ----------
    inversion_results : InversionResults
        The inversion results containing the model DataArray and metadata.
    method : str, optional
        Detection method. Currently only "percentile" is supported (default: "percentile").
    **kwargs : dict
        Additional parameters:
        - p : float
            Percentile threshold (0-100). Required for percentile method.
            Values above this percentile are considered anomalies.

    Returns
    -------
    pd.DataFrame
        DataFrame containing detected anomalies with columns:
        - lat : float64, latitude in degrees
        - lon : float64, longitude in degrees
        - anomaly_score : float64, the model value at the anomalous location

    Raises
    ------
    GAMError
        If input validation fails or required parameters are missing/invalid.
    ValueError
        If method is not supported or percentile is invalid.

    Notes
    -----
    - The percentile method calculates the threshold as model.quantile(p / 100.0).
    - Only grid cells where model > threshold are included in the output.
    - NaN values in the model are handled by excluding them from the results.
    - The calculation is deterministic and reproducible.
    - Assumes the model DataArray has 'lat' and 'lon' coordinates.

    Examples
    --------
    >>> results = InversionResults(model_da, uncertainty_da, metadata)
    >>> anomalies = detect_anomalies(results, p=95)
    >>> print(f"Detected {len(anomalies)} anomalies")
    """
    # Input validation
    if not isinstance(inversion_results, InversionResults):
        raise GAMError("inversion_results must be an InversionResults instance")
    inversion_results.validate()

    if method != "percentile":
        raise ValueError(f"Unsupported method: {method}. Only 'percentile' is currently supported.")

    # Extract required parameters
    if 'p' not in kwargs:
        raise GAMError("Percentile threshold 'p' is required for percentile method")
    p = kwargs['p']
    if not (0 < p < 100):
        raise ValueError("Percentile 'p' must be between 0 and 100 (exclusive)")

    # Extract model DataArray
    model = inversion_results.model

    # Calculate threshold
    threshold = model.quantile(p / 100.0).values
    logger.debug(f"Calculated threshold at {p}th percentile: {threshold}")

    # Identify anomalies: cells where model > threshold
    anomalies_da = model.where(model > threshold, drop=True)

    # Handle NaN values gracefully by dropping them
    anomalies_da = anomalies_da.dropna('lat', how='all').dropna('lon', how='all')

    # Convert to DataFrame
    if anomalies_da.size == 0:
        # Return empty DataFrame with correct dtypes
        df = pd.DataFrame(columns=['lat', 'lon', 'anomaly_score']).astype({
            'lat': 'float64',
            'lon': 'float64',
            'anomaly_score': 'float64'
        })
        logger.info("No anomalies detected above the threshold")
        return df

    # Convert xarray to DataFrame
    df = anomalies_da.to_dataframe(name='anomaly_score').reset_index()

    # Ensure correct dtypes
    df = df.astype({
        'lat': 'float64',
        'lon': 'float64',
        'anomaly_score': 'float64'
    })

    logger.info(f"Detected {len(df)} anomalies using {p}th percentile threshold")
    return df