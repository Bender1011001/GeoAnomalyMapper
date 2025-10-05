"""
Adapter stub for legacy preprocessing manager.

Phase 1: lightweight stub that can be imported safely. Do NOT import legacy
preprocessing.manager at module import time; import inside methods when implementing.
"""
from __future__ import annotations

from typing import Any
from gam.services.preprocessing_service import PreprocessingServiceInterface
from gam.core.data_contracts import RawData, ProcessedGrid


class PreprocessingManagerAdapter(PreprocessingServiceInterface):
    """
    Adapter placeholder mapping new PreprocessingServiceInterface to legacy manager.

    Implementation notes:
      - Import legacy manager inside process() to avoid import-time side effects:
            from gam.preprocessing.manager import LegacyPreprocessingManager
      - Translate RawData -> legacy inputs and map legacy outputs into ProcessedGrid.
      - Keep the adapter stateless; avoid module-level singletons.
    """

    def process(self, raw: RawData) -> ProcessedGrid:
        """
        Process raw data using legacy preprocessing manager.

        Raises:
            NotImplementedError: stub for Phase 1.
        """
        raise NotImplementedError(
            "PreprocessingManagerAdapter.process is not implemented. "
            "Implement mapping to legacy preprocessing manager inside this method "
            "and ensure imports happen within the method body to avoid side effects."
        )