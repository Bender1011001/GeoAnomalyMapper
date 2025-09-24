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
from sentinelsat import SentinelAPI, make_path_filter
from shapely.geometry import box
import zipfile
import tempfile
import shutil
import hashlib
from datetime import timedelta, date
import rasterio
from rasterio.windows import Window

from .base import DataSource
from .data_structures import RawData
from .exceptions import (
    retry_fetch, rate_limit, DataFetchError, APITimeoutError
)
from .cache_manager import HDF5CacheManager


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
    Fetcher for ESA Sentinel-1 InSAR data using Copernicus SciHub API.

    Queries and downloads Sentinel-1 SLC products, extracts wrapped LOS displacement
    from phase data as a basic proxy. Full interferometric processing (pair selection,
    unwrapping) deferred to modeling stage. Integrates caching to avoid redundant
    downloads. Supports bbox, date range filtering, VV polarization.

    Parameters
    ----------
    None

    Attributes
    ----------
    api_url : str
        Base URL for Copernicus API Hub.
    cache : HDF5CacheManager
        Cache instance for storing processed data.
    source : str
        Data source name.
    timeout : float
        Download timeout in seconds (300s for large files).
    wavelength : float
        Sentinel-1 C-band wavelength in meters.

    Methods
    -------
    fetch_data(bbox, **kwargs)
        Fetch InSAR displacement data.

    Notes
    -----
    API: https://apihub.copernicus.eu/apihub/ (SentinelAPI client)
    Requires free Copernicus account (username/password via kwargs or env: ESA_USERNAME, ESA_PASSWORD).
    Downloads SLC (~1-10GB); subsamples for efficiency. Returns wrapped LOS displacement
    in meters (phase * lambda / 4pi); not unwrapped/true displacement.
    Coords: Approximate WGS84 from raster transform (valid for small bbox).
    Rate limited; retries on failure. Products filtered by bbox intersection, date, SLC/VV.
    For GRD (amplitude only), set product_type='GRD' in kwargs (no phase).

    Setup
    -----
    1. Register at https://scihub.copernicus.eu/
    2. Set env: export ESA_USERNAME='user' ESA_PASSWORD='pass'
    3. Install: pip install .[geophysics] (includes sentinelsat, rasterio)
    4. Usage: fetcher.fetch_data(bbox, start_date='2024-01-01', username='user', password='pass')

    Examples
    --------
    >>> fetcher = InSARFetcher()
    >>> data = fetcher.fetch_data((29.0, 29.5, 31.0, 31.5), start_date='2024-01-01')
    """

    def __init__(self):
        self.api_url = "https://apihub.copernicus.eu/apihub/"
        self.cache = HDF5CacheManager()
        self.source = "ESA Sentinel-1"
        self.timeout = 300.0
        self.wavelength = 0.05546576  # C-band in meters

    @retry_fetch(max_attempts=3)
    @rate_limit(10)  # ESA quota ~10/min
    def fetch_data(self, bbox: Tuple[float, float, float, float], **kwargs) -> RawData:
        r"""
        Fetch InSAR displacement data from ESA Sentinel-1.

        Args:
            bbox: (lat_min, lat_max, lon_min, lon_max)
            start_date: ISO format date string (default: 30 days ago)
            end_date: ISO format date string (default: today)
            username: ESA Copernicus Hub username
            password: ESA Copernicus Hub password
            product_type: 'SLC' or 'GRD' (default: 'SLC' for phase)

        Returns:
            RawData object with InSAR displacement measurements (wrapped LOS in m)

        Raises:
            ValueError: Invalid bbox or missing credentials
            DataFetchError: No products, download failure, processing error
            APITimeoutError: Query/download timeout
        """
        min_lat, max_lat, min_lon, max_lon = bbox
        if not (min_lat < max_lat and min_lon < max_lon):
            raise ValueError("Invalid bbox: min < max required")

        # Date handling
        today = date.today()
        start_date = kwargs.get('start_date', (today - timedelta(days=30)).isoformat())
        end_date = kwargs.get('end_date', today.isoformat())
        product_type = kwargs.get('product_type', 'SLC')

        # Credentials
        username = kwargs.get('username') or os.getenv('ESA_USERNAME')
        password = kwargs.get('password') or os.getenv('ESA_PASSWORD')
        if not username or not password:
            raise ValueError("ESA credentials required: provide username/password or set ESA_USERNAME/ESA_PASSWORD env vars")

        # Cache key
        key_str = f"{self.source}_{min_lat:.6f}_{max_lat:.6f}_{min_lon:.6f}_{max_lon:.6f}_{start_date}_{end_date}_{product_type}"
        key = hashlib.md5(key_str.encode('utf-8')).hexdigest()

        # Check cache
        cached_data = self.cache.load_data(key)
        if cached_data:
            logger.info(f"Loaded cached InSAR data for key: {key}")
            return cached_data

        # Query API
        api = SentinelAPI(username, password, self.api_url)
        footprint = box(min_lon, min_lat, max_lon, max_lat).wkt
        try:
            products = api.query(
                footprint,
                date=(start_date, end_date),
                platformname='Sentinel-1',
                producttype=product_type,
                polarisationmode='VV'
            )
            if not products:
                raise DataFetchError(self.source, f"No {product_type} products found for bbox/date range")
            # Select most recent
            product_id = max(products, key=lambda k: products[k]['beginposition'])
            product = products[product_id]
        except Exception as e:
            raise DataFetchError(self.source, f"Query failed: {str(e)}")

        # Download
        temp_dir = tempfile.mkdtemp()
        try:
            api.download(
                [product_id],
                directory_path=temp_dir,
                checksum=True,
                timeout=self.timeout
            )
            zip_path = os.path.join(temp_dir, f"{product_id}.zip")
            if not os.path.exists(zip_path):
                raise DataFetchError(self.source, "Download zip not found")

            # Extract
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

            # Find VV TIFF in measurement
            tiff_path = None
            for root, dirs, files in os.walk(extract_dir):
                if 'measurement' in root:
                    for file in files:
                        if file.endswith('.tiff') and 'vv' in file.lower():
                            tiff_path = os.path.join(root, file)
                            break
                    if tiff_path:
                        break
            if not tiff_path:
                raise DataFetchError(self.source, "VV TIFF not found in SAFE archive")

            # Process with rasterio
            with rasterio.open(tiff_path) as src:
                # Subsample for efficiency (1/20 size)
                width = src.width // 20
                height = src.height // 20
                window = Window(0, 0, width, height)
                data = src.read(1, window=window, masked=True)
                if product_type == 'SLC':
                    # Complex to phase, then LOS displacement (wrapped)
                    phase = np.angle(data)
                    disp = (phase * self.wavelength / (4 * np.pi)).filled(0)
                else:  # GRD: amplitude as proxy (normalize to 0-1, scale to mm for consistency)
                    amp = data.filled(0)
                    disp = (amp / np.max(amp) * 1000).astype(np.float32)  # mm proxy

                # Generate coords
                rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                lons, lats = rasterio.transform.xy(src.window_transform(window), rows.flatten(), cols.flatten())

                # Filter to bbox
                mask = (
                    (np.array(lats) >= min_lat) & (np.array(lats) <= max_lat) &
                    (np.array(lons) >= min_lon) & (np.array(lons) <= max_lon)
                )
                if not np.any(mask):
                    raise DataFetchError(self.source, "No data points within bbox after filtering")

                values = disp.flatten()[mask]
                lats_filtered = np.array(lats)[mask]
                lons_filtered = np.array(lons)[mask]

            # Metadata
            metadata = {
                'source': self.source,
                'timestamp': datetime.now(),
                'bbox': bbox,
                'parameters': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'product_type': product_type,
                    'product_id': product_id
                },
                'lat': lats_filtered.tolist(),
                'lon': lons_filtered.tolist(),
                'acquisition_dates': [product['beginposition'], product['endposition']],
                'processing_level': product_type,
                'pass_direction': product['orbitdirection'],
                'units': 'm' if product_type == 'SLC' else 'mm (amplitude proxy)',
                'crs': 'EPSG:4326',
                'note': 'Wrapped LOS displacement from single SLC phase (SLC) or amplitude proxy (GRD); full interferometry required for absolute displacement'
            }

            raw_data = RawData(metadata, values)
            raw_data.validate()
            self.cache.save_data(raw_data)
            logger.info(f"Fetched InSAR data: {len(values)} points for bbox {bbox}, product {product_id}")
            return raw_data

        except requests.exceptions.Timeout:
            raise APITimeoutError(self.source, self.timeout)
        except Exception as e:
            raise DataFetchError(self.source, f"Processing failed: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if 'extract_dir' in locals():
                shutil.rmtree(extract_dir, ignore_errors=True)