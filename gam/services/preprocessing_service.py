"""
Preprocessing service interface and a deterministic mock implementation.

Import-safe: no I/O or heavy computation at import time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import xarray as xr

from gam.core.data_contracts import RawData, ProcessedGrid


class PreprocessingServiceInterface(ABC):
    """
    Stateless preprocessing service interface.

    Implementations should accept RawData and return a ProcessedGrid.
    """

    @abstractmethod
    def process(self, raw: RawData) -> ProcessedGrid:
        """
        Process raw ingestion payload into a standardized grid.

        Args:
            raw: RawData produced by an ingestion service.

        Returns:
            ProcessedGrid wrapping an xarray.Dataset with 'lat' and 'lon' coords.
        """
        raise NotImplementedError


class MockPreprocessingService(PreprocessingServiceInterface):
    """
    Deterministic mock preprocessing that converts RawData.data["values"]
    into a tiny xr.Dataset with coords lat=[30.0, 31.0], lon=[-120.0, -119.0].

    This mock avoids I/O and heavy computation and is safe for unit tests
    and orchestrator smoke tests.
    """

    def process(self, raw: RawData) -> ProcessedGrid:
        # Extract values deterministically; fallback to zeros if missing/invalid
        values = raw.data.get("values")
        arr = None
        try:
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 2:
                # Expect a 2D array for the simple mock
                arr = np.atleast_2d(arr)
        except Exception:
            arr = np.zeros((2, 2), dtype=float)

        # Ensure shape (2,2)
        if arr.shape != (2, 2):
            # If different shape, pad or trim deterministically
            tmp = np.zeros((2, 2), dtype=float)
            min_rows = min(2, arr.shape[0])
            min_cols = min(2, arr.shape[1] if arr.ndim > 1 else 1)
            tmp[:min_rows, :min_cols] = arr[:min_rows, :min_cols]
            arr = tmp

        lats = np.array([30.0, 31.0], dtype=float)
        lons = np.array([-120.0, -119.0], dtype=float)

        ds = xr.Dataset(
            data_vars={"data": (("lat", "lon"), arr)},
            coords={"lat": lats, "lon": lons},
            attrs={"source": raw.source, "units": raw.metadata.get("units", "")},
        )

        return ProcessedGrid(grid=ds)