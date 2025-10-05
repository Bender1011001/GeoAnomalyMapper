"""
Ingestion service interface and a deterministic mock implementation.

This module is import-safe and has no side-effects at import time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

from gam.core.data_contracts import RawData


class IngestionServiceInterface(ABC):
    """
    Stateless ingestion service interface.

    Implementations must be side-effect free on import and stateless.
    """

    @abstractmethod
    def fetch(self, modality: str, bbox: Tuple[float, float, float, float]) -> RawData:
        """
        Fetch raw data for the given modality over bbox.

        Args:
            modality: modality name (e.g. "gravity", "magnetic")
            bbox: tuple(min_lon, min_lat, max_lon, max_lat)

        Returns:
            RawData instance validated by pydantic contract.
        """
        raise NotImplementedError


class MockIngestionService(IngestionServiceInterface):
    """
    Deterministic mock ingestion that returns a tiny fixed dataset.

    This implementation intentionally avoids I/O and heavy computation, and is
    suitable for local unit tests and orchestrator smoke tests.
    """

    def fetch(self, modality: str, bbox: Tuple[float, float, float, float]) -> RawData:
        # Build a tiny deterministic payload that conforms to RawData
        payload: Dict[str, Any] = {
            "values": [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        }
        metadata: Dict[str, Any] = {"units": "mGal", "resolution_deg": 0.1}
        return RawData(source=modality, bbox=bbox, data=payload, metadata=metadata)