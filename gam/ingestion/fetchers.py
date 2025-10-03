"""
Specific data fetchers for geophysical modalities in GAM ingestion.
"""

import logging
import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)

# -------------------------------
# Optional pyshtools
# -------------------------------
try:
    import pyshtools
    SHGravCoeffs = pyshtools.SHGravCoeffs
    _HARMONIC_AVAILABLE = True
except Exception as e:
    logger.warning(f"pyshtools not available; harmonic gravity fetching disabled. ({e})")
    _HARMONIC_AVAILABLE = False

# -------------------------------
# Base Fetcher
# -------------------------------
class BaseFetcher:
    def fetch(self, *args, **kwargs):
        raise NotImplementedError

# -------------------------------
# Gravity (USGS)
# -------------------------------
class USGSGravityFetcher(BaseFetcher):
    def fetch(self, bbox, **kwargs):
        url = f"https://mrdata.usgs.gov/services/gravity?bbox={bbox}&format=geojson"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (RequestException, Timeout) as e:
            logger.error(f"USGSGravityFetcher failed: {e}")
            return None

# -------------------------------
# Magnetic (USGS)
# -------------------------------
class USGSMagneticFetcher(BaseFetcher):
    def fetch(self, bbox, **kwargs):
        url = f"https://mrdata.usgs.gov/services/mag?bbox={bbox}&format=geojson"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (RequestException, Timeout) as e:
            logger.error(f"USGSMagneticFetcher failed: {e}")
            return None

# -------------------------------
# ESA InSAR fetcher (Sentinel-1, Copernicus Open Access Hub)
# -------------------------------
class ESAInSARFetcher:
    """Fetcher for Sentinel-1 InSAR scenes from ESA Copernicus SciHub."""

    def __init__(self, username=None, password=None):
        try:
            from sentinelsat import SentinelAPI
            self.SentinelAPI = SentinelAPI
        except ImportError:
            self.SentinelAPI = None
            logger.warning("sentinelsat not available; InSAR fetching disabled. Install with: pip install sentinelsat")

        self.username = username
        self.password = password

    def fetch(self, bbox=None, start_date="NOW-30DAYS", end_date="NOW", product_type="SLC"):
        if not self.SentinelAPI:
            raise RuntimeError("SentinelAPI not available. Install with: pip install sentinelsat")
        if not self.username or not self.password:
            raise RuntimeError("ESAInSARFetcher requires ESA SciHub credentials (ESA_USERNAME/ESA_PASSWORD)")

        # Copernicus SciHub endpoint
        api = self.SentinelAPI(self.username, self.password, "https://scihub.copernicus.eu/dhus")

        # Query Sentinel-1 scenes
        products = api.query(
            bbox=bbox,
            date=(start_date, end_date),
            platformname="Sentinel-1",
            producttype=product_type
        )

        return products

# -------------------------------
# Harmonic Gravity (pyshtools)
# -------------------------------
if _HARMONIC_AVAILABLE:
    class HarmonicGravityFetcher(BaseFetcher):
        def fetch(self, lmax=60, model="demo"):
            try:
                coeffs = SHGravCoeffs.from_random(lmax)
                return {"coeffs": coeffs, "model": model}
            except Exception as e:
                logger.error(f"HarmonicGravityFetcher failed: {e}")
                return None
else:
    class HarmonicGravityFetcher(BaseFetcher):
        def fetch(self, *args, **kwargs):
            logger.error("HarmonicGravityFetcher unavailable (pyshtools missing).")
            return None

# -------------------------------
# Seismic (IRIS FDSN via ObsPy)
# -------------------------------
class SeismicFetcher(BaseFetcher):
    """Fetcher for seismic waveform/metadata via IRIS FDSN."""
    def __init__(self, *args, **kwargs):
        try:
            from obspy.clients.fdsn import Client as FDSNClient
            self.client = FDSNClient("IRIS")
        except ImportError:
            self.client = None

    def fetch(self, network="IU", station="ANMO", location="00", channel="BHZ", starttime=None, endtime=None):
        if not self.client:
            raise RuntimeError("ObsPy not available; install with: pip install obspy")
        from obspy import UTCDateTime
        st = self.client.get_waveforms(network, station, location, channel,
                                       UTCDateTime(starttime), UTCDateTime(endtime))
        return st

# -------------------------------
# ScienceBase (USGS)
# -------------------------------
class ScienceBaseFetcher:
    """Fetcher for datasets hosted on USGS ScienceBase."""

    def __init__(self, username=None, password=None):
        try:
            from sciencebasepy import SbSession
            self.SbSession = SbSession
        except ImportError:
            self.SbSession = None
            logger.warning("sciencebasepy not available; ScienceBase fetching disabled. Install with: pip install sciencebasepy")

        self.username = username
        self.password = password
        self.session = None

    def connect(self):
        if not self.SbSession:
            raise RuntimeError("sciencebasepy not available. Install with: pip install sciencebasepy")

        self.session = self.SbSession()
        if self.username and self.password:
            self.session.login(self.username, self.password)

    def fetch_item(self, item_id):
        """Download metadata or files for a ScienceBase item."""
        if not self.session:
            self.connect()
        return self.session.get_item(item_id)

    def download_file(self, item_id, dest_dir="."):
        """Download files for a ScienceBase item into dest_dir."""
        if not self.session:
            self.connect()
        return self.session.get_item_files(item_id, dest_dir)
