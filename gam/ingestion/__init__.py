"""
Ingestion module initialization.

This file wires up available fetchers and exposes them for manager use.
Some fetchers (like HarmonicGravityFetcher using pyshtools/pygmt) are optional.
"""

from .fetchers import (
    USGSGravityFetcher,
    USGSMagneticFetcher,
    SeismicFetcher,
    ESAInSARFetcher,
    ScienceBaseFetcher,
)

# Backwards compatibility alias
IRISSeismicFetcher = SeismicFetcher

# Optional harmonic fetcher guarded to avoid GMT.dll / pyshtools issues
try:
    from .fetchers import HarmonicGravityFetcher
except Exception:
    HarmonicGravityFetcher = None

__all__ = [
    "USGSGravityFetcher",
    "USGSMagneticFetcher",
    "SeismicFetcher",
    "IRISSeismicFetcher",   # alias still exported
    "ESAInSARFetcher",
    "ScienceBaseFetcher",
]

if HarmonicGravityFetcher is not None:
    __all__.append("HarmonicGravityFetcher")
