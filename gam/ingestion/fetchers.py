"""Specific data fetchers for geophysical modalities in GAM ingestion."""

import json
import logging
import os
from datetime import datetime
from typing import Tuple, Any, Dict
import numpy as np
import requests
from requests.exceptions import Timeout, RequestException

from obspy.clients.fdsn import Client as FDSNClient
from obspy import UTCDateTime

from .base import DataSource
from .data_structures import RawData
from .exceptions import (
    retry_fetch, rate_limit, DataFetchError, APITimeoutError
)


logger = logging.getLogger(__name__)


class GravityFetcher(DataSource):
    r"""
    Fetcher for USGS gravity data using the Gravity API.

    Queries the USGS MRData Gravity service for gravity anomaly measurements
    within a bounding box. Extracts gravity values in mGal units from GeoJSON.

    Parameters
    ----------
    None

    Attributes
    ----------
    api_url : str
        Base URL for USGS Gravity API.
    timeout : float
        Request timeout in seconds (30s).

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Fetch gravity data for bbox.

    Notes
    -----
    API: https://mrdata.usgs.gov/services/gravity?minlat={min_lat}&maxlat={max_lat}&minlon={min_lon}&maxlon={max_lon}&format=geojson
    Returns np.ndarray of gravity anomalies. Supports kwargs like 'mindepth', 'maxdepth'.
    Rate limited to 60/min; retries on failure/timeout.

    Examples
    --------
    >>> fetcher = GravityFetcher()
    >>> data = fetcher.fetch_data((29.0, 31.0, 30.0, 32.0))
    """

    def __init__(self):
        self.api_url = "https://mrdata.usgs.gov/services/gravity"
        self.timeout = 30.0

    @retry_fetch()
    @rate_limit(60)
    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Fetch gravity data for the given bounding box.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon)
        **kwargs : dict
            Additional params (e.g., 'mindepth', 'maxdepth')

        Returns
        -------
        RawData
            With values as np.ndarray of gravity anomalies (mGal).

        Raises
        ------
        DataFetchError
            On API error or invalid response.
        APITimeoutError
            On request timeout.
        ValueError
            On invalid bbox.
        """
        min_lat, max_lat, min_lon, max_lon = bbox
        if not (min_lat < max_lat and min_lon < max_lon):
            raise ValueError("Invalid bbox: min < max required")

        params = {
            'minlat': min_lat,
            'maxlat': max_lat,
            'minlon': min_lon,
            'maxlon': max_lon,
            'format': 'geojson',
            **kwargs
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data_json = response.json()
            features = data_json.get('features', [])
            if not features:
                raise DataFetchError("USGS Gravity", "No data found for bbox")
            # Extract gravity values from properties (assume 'gravity' field; adjust per API docs)
            gravity_values = np.array([feat['properties'].get('gravity', np.nan) for feat in features])
            gravity_values = gravity_values[~np.isnan(gravity_values)]  # Clean NaNs
            if len(gravity_values) == 0:
                raise DataFetchError("USGS Gravity", "No valid gravity values")
            metadata = {
                'source': 'USGS Gravity',
                'timestamp': datetime.now(),
                'bbox': bbox,
                'parameters': kwargs
            }
            raw_data = RawData(metadata, gravity_values)
            raw_data.validate()
            logger.info(f"Fetched {len(gravity_values)} gravity measurements for bbox {bbox}")
            return raw_data
        except Timeout:
            raise APITimeoutError("USGS Gravity", self.timeout)
        except RequestException as e:
            raise DataFetchError("USGS Gravity", f"Request failed: {str(e)}")


class MagneticFetcher(DataSource):
    r"""
    Fetcher for USGS magnetic data using the Magnetic API.

    Similar to GravityFetcher but for aeromagnetic surveys. Extracts magnetic
    field anomaly values in nT.

    Parameters
    ----------
    None

    Attributes
    ----------
    api_url : str
        Base URL for USGS Magnetic API.
    timeout : float
        Request timeout (30s).

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Fetch magnetic data.

    Notes
    -----
    API: https://mrdata.usgs.gov/services/magnetic?minlat={min_lat}&maxlat={max_lat}&minlon={min_lon}&maxlon={max_lon}&format=geojson
    Returns np.ndarray of magnetic anomalies (nT). Kwargs for filters like 'survey'.

    Examples
    --------
    >>> fetcher = MagneticFetcher()
    >>> data = fetcher.fetch_data((29.0, 31.0, 30.0, 32.0))
    """

    def __init__(self):
        self.api_url = "https://mrdata.usgs.gov/services/magnetic"
        self.timeout = 30.0

    @retry_fetch()
    @rate_limit(60)
    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Fetch magnetic data for the given bounding box.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon)
        **kwargs : dict
            Additional params (e.g., 'survey')

        Returns
        -------
        RawData
            With values as np.ndarray of magnetic anomalies (nT).

        Raises
        ------
        DataFetchError, APITimeoutError, ValueError
            As in GravityFetcher.
        """
        min_lat, max_lat, min_lon, max_lon = bbox
        if not (min_lat < max_lat and min_lon < max_lon):
            raise ValueError("Invalid bbox: min < max required")

        params = {
            'minlat': min_lat,
            'maxlat': max_lat,
            'minlon': min_lon,
            'maxlon': max_lon,
            'format': 'geojson',
            **kwargs
        }
        try:
            response = requests.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data_json = response.json()
            features = data_json.get('features', [])
            if not features:
                raise DataFetchError("USGS Magnetic", "No data found for bbox")
            # Extract magnetic values (assume 'magnetic' field)
            magnetic_values = np.array([feat['properties'].get('magnetic', np.nan) for feat in features])
            magnetic_values = magnetic_values[~np.isnan(magnetic_values)]
            if len(magnetic_values) == 0:
                raise DataFetchError("USGS Magnetic", "No valid magnetic values")
            metadata = {
                'source': 'USGS Magnetic',
                'timestamp': datetime.now(),
                'bbox': bbox,
                'parameters': kwargs
            }
            raw_data = RawData(metadata, magnetic_values)
            raw_data.validate()
            logger.info(f"Fetched {len(magnetic_values)} magnetic measurements for bbox {bbox}")
            return raw_data
        except Timeout:
            raise APITimeoutError("USGS Magnetic", self.timeout)
        except RequestException as e:
            raise DataFetchError("USGS Magnetic", f"Request failed: {str(e)}")


class SeismicFetcher(DataSource):
    r"""
    Fetcher for seismic data using ObsPy FDSN client to IRIS.

    Queries seismic stations and waveforms within bbox. Returns ObsPy Stream
    object for further processing.

    Parameters
    ----------
    None

    Attributes
    ----------
    client : FDSNClient
        Connected to IRIS FDSN service.
    timeout : float
        Query timeout (30s).

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Fetch seismic data.

    Notes
    -----
    Uses IRIS FDSN: https://service.iris.edu/fdsnws/station/1/query
    Requires starttime/endtime in kwargs (UTCDateTime or str). Returns Stream
    of available waveforms. For events, use get_events if needed.
    Rate limited; retries.

    Setup
    -----
    pip install obspy

    Examples
    --------
    >>> fetcher = SeismicFetcher()
    >>> data = fetcher.fetch_data((29.0, 31.0, 30.0, 32.0), starttime='2023-01-01', endtime='2023-01-02')
    """

    def __init__(self):
        self.client = FDSNClient("IRIS")
        self.timeout = 30.0

    @retry_fetch()
    @rate_limit(60)
    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Fetch seismic waveforms for the given bounding box and time range.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon)
        **kwargs : dict
            Required: 'starttime', 'endtime' (str or UTCDateTime).
            Optional: 'network', 'station', 'channel', 'location'.

        Returns
        -------
        RawData
            With values as obspy.Stream.

        Raises
        ------
        DataFetchError
            On query failure or no data.
        APITimeoutError
            On timeout.
        ValueError
            On invalid bbox or missing time params.
        """
        min_lat, max_lat, min_lon, max_lon = bbox
        if not (min_lat < max_lat and min_lon < max_lon):
            raise ValueError("Invalid bbox: min < max required")
        starttime = kwargs.get('starttime')
        endtime = kwargs.get('endtime')
        if not starttime or not endtime:
            raise ValueError("starttime and endtime required for seismic fetch")
        starttime = UTCDateTime(starttime) if isinstance(starttime, str) else starttime
        endtime = UTCDateTime(endtime) if isinstance(endtime, str) else endtime

        try:
            # Get waveforms (broad query; filter post-fetch if needed)
            st = self.client.get_waveforms(
                network=kwargs.get('network', '*'),
                station=kwargs.get('station', '*'),
                location=kwargs.get('location', '*'),
                channel=kwargs.get('channel', '*'),
                starttime=starttime,
                endtime=endtime,
                minlatitude=min_lat,
                maxlatitude=max_lat,
                minlongitude=min_lon,
                maxlongitude=max_lon,
                timeout=self.timeout
            )
            if len(st) == 0:
                raise DataFetchError("IRIS Seismic", "No waveforms found for query")
            metadata = {
                'source': 'IRIS Seismic',
                'timestamp': datetime.now(),
                'bbox': bbox,
                'parameters': kwargs
            }
            raw_data = RawData(metadata, st)
            raw_data.validate()
            logger.info(f"Fetched seismic Stream with {len(st)} traces for bbox {bbox}")
            return raw_data
        except Timeout:
            raise APITimeoutError("IRIS Seismic", self.timeout)
        except Exception as e:  # FDSN errors
            raise DataFetchError("IRIS Seismic", f"Query failed: {str(e)}")


class InSARFetcher(DataSource):
    r"""
    Placeholder fetcher for ESA Sentinel-1 InSAR data via SciHub API.

    This is a placeholder implementation. Full integration requires ESA API key
    and handling of SAR product downloads (GRD/SLC). Once implemented, will
    query for interferograms or DEMs within bbox.

    Parameters
    ----------
    None

    Attributes
    ----------
    api_url : str
        Base URL for Copernicus SciHub (DHuS API).
    api_key : str or None
        ESA API key from environment.

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Placeholder; raises NotImplementedError.

    Notes
    -----
    API: https://scihub.copernicus.eu/dhus/search (OpenSearch format)
    Requires authentication: Basic auth with username/password or API key.
    Products are large; download and process with GDAL/snaphu.
    Kwargs: 'start_date', 'end_date', 'product_type' (e.g., 'SLC').
    Rate limited to API quotas.

    Setup Instructions
    -----------------
    1. Register at https://scihub.copernicus.eu/ for account.
    2. Set environment variable: export ESA_API_KEY='your_username:your_password'
    3. Install: pip install requests sentinelsat (alternative client)
    4. Full impl: Use sentinelsat to query/download, extract phase/displacement to xarray.

    Examples
    --------
    >>> fetcher = InSARFetcher()
    >>> # data = fetcher.fetch_data((29.0, 31.0, 30.0, 32.0), start_date='2023-01-01')
    """

    def __init__(self):
        self.api_url = "https://scihub.copernicus.eu/dhus"
        self.api_key = os.getenv('ESA_API_KEY')
        if not self.api_key:
            logger.warning("ESA_API_KEY not set; InSAR fetcher will not work")

    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Placeholder for fetching InSAR data.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            (min_lat, max_lat, min_lon, max_lon)
        **kwargs : dict
            Query params (e.g., dates, product_type)

        Returns
        -------
        RawData
            Not implemented.

        Raises
        ------
        NotImplementedError
            Always raised; implement per setup instructions.
        """
        raise NotImplementedError(
            "InSARFetcher is a placeholder. Set ESA_API_KEY and implement "
            "query/download using sentinelsat or requests to SciHub API. "
            "Expected values: xarray.Dataset of displacement/phase. "
            "See class docstring for setup."
        )