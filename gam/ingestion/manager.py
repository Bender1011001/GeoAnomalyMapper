"""
IngestionManager orchestrates data fetching across all supported modalities.
"""

import logging
from . import (
    USGSGravityFetcher,
    USGSMagneticFetcher,
    IRISSeismicFetcher,
    ESAInSARFetcher,
    ScienceBaseFetcher,
    HarmonicGravityFetcher,
)

logger = logging.getLogger(__name__)


class IngestionManager:
    def __init__(self):
        self.fetchers = {}

        # Core fetchers always available
        self.register("gravity", USGSGravityFetcher())
        self.register("magnetic", USGSMagneticFetcher())
        self.register("seismic", IRISSeismicFetcher())
        self.register("insar", ESAInSARFetcher())
        self.register("sciencebase", ScienceBaseFetcher())

        # Optional harmonic gravity (pyshtools/pygmt)
        if HarmonicGravityFetcher is not None:
            try:
                self.register("harmonic_gravity", HarmonicGravityFetcher())
            except Exception as e:
                logger.warning(f"HarmonicGravityFetcher unavailable: {e}")
        else:
            logger.warning("HarmonicGravityFetcher skipped (pyshtools/pygmt not installed).")

    def register(self, name, fetcher):
        self.fetchers[name] = fetcher

    def fetch_modality(self, name, *args, **kwargs):
        if name not in self.fetchers:
            raise ValueError(f"Fetcher '{name}' not registered.")
        return self.fetchers[name].fetch(*args, **kwargs)

    def list_modalities(self):
        """Return list of available fetchers."""
        return list(self.fetchers.keys())
