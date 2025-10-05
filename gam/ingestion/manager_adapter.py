"""
Adapter stub for legacy ingestion manager.

This module is intentionally a lightweight stub for Phase 1:
- It subclasses the ingestion service interface so it can be wired into the
  new PipelineOrchestrator without changing existing code.
- It does NOT import or reference legacy manager modules at import time to
  avoid side effects. When implementing a real adapter, import legacy modules
  inside the method body.
"""
from __future__ import annotations

from typing import Tuple, Any, Dict

from gam.services.ingestion_service import IngestionServiceInterface
from gam.core.data_contracts import RawData


class IngestionManagerAdapter(IngestionServiceInterface):
    """
    Adapter that will map the new IngestionServiceInterface to the legacy
    ingestion manager API.

    Current behavior: stub that raises NotImplementedError.

    Implementation notes (future phases):
      - Import legacy managers inside fetch() to avoid import-time side effects:
          from gam.ingestion.manager import LegacyIngestionManager
      - Map parameters and return a gam.core.data_contracts.RawData instance.
      - Keep the adapter stateless; avoid global singletons.
    """

    def fetch(self, modality: str, bbox: Tuple[float, float, float, float]) -> RawData:
        """
        Fetch raw data using the legacy ingestion manager.

        Args:
            modality: modality name (e.g., "gravity")
            bbox: tuple(min_lon, min_lat, max_lon, max_lat)

        Returns:
            RawData validated by Pydantic contract.

        Raises:
            NotImplementedError: stub for Phase 1.
        """
        raise NotImplementedError(
            "IngestionManagerAdapter.fetch is not implemented. "
            "Implement mapping to legacy ingestion manager here and import legacy "
            "modules inside this method to avoid import-time side effects."
        )