"""
Adapter stubs for legacy modeling and anomaly detector managers.

These stubs subclass the new service interfaces so they can be wired into the
PipelineOrchestrator without modifying legacy code. They intentionally avoid
importing legacy manager modules at import time to prevent side-effects. When
implementing real adapters in later phases, import legacy modules inside the
method bodies.
"""
from __future__ import annotations

from typing import List, Tuple

from gam.services.modeling_service import ModelingServiceInterface
from gam.services.anomaly_service import AnomalyServiceInterface
from gam.core.data_contracts import ProcessedGrid, InversionResult, Anomaly


class ModelingManagerAdapter(ModelingServiceInterface):
    """
    Adapter placeholder mapping ModelingServiceInterface to the legacy modeling manager.

    Implementation notes (future phases):
      - Import legacy managers inside invert(), for example:
            from gam.modeling.manager import LegacyModelingManager
      - Translate ProcessedGrid -> legacy inputs and map legacy outputs into
        an InversionResult instance validated by Pydantic.
      - Keep the adapter stateless; avoid module-level singletons.
    """

    def invert(self, grid: ProcessedGrid, modality: str) -> InversionResult:
        """
        Perform inversion using legacy modeling manager.

        Raises:
            NotImplementedError: stub for Phase 1.
        """
        raise NotImplementedError(
            "ModelingManagerAdapter.invert is not implemented. "
            "Implement mapping to legacy modeling.manager inside this method and "
            "ensure any legacy imports occur within the method body to avoid "
            "import-time side effects."
        )


class AnomalyDetectorAdapter(AnomalyServiceInterface):
    """
    Adapter placeholder mapping AnomalyServiceInterface to a legacy anomaly detector.

    Implementation notes:
      - Import legacy anomaly detector inside detect() when implementing.
      - Map legacy outputs into a list of gam.core.data_contracts.Anomaly objects.
      - Keep deterministic behavior and avoid global state.
    """

    def detect(self, inv: InversionResult) -> List[Anomaly]:
        """
        Detect anomalies using legacy anomaly detector.

        Raises:
            NotImplementedError: stub for Phase 1.
        """
        raise NotImplementedError(
            "AnomalyDetectorAdapter.detect is not implemented. "
            "Implement mapping to legacy anomaly detector here and import legacy "
            "modules inside the method body to avoid side effects."
        )