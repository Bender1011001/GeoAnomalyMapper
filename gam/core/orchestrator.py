"""
Pipeline orchestrator for Phase 1 scaffolding.

This orchestrator is constructor-injected with service implementations and is
fully import-safe (no side effects on import).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

from gam.services import (
    IngestionServiceInterface,
    PreprocessingServiceInterface,
    ModelingServiceInterface,
    AnomalyServiceInterface,
)
from gam.core.data_contracts import Anomaly


@dataclass
class PipelineOrchestrator:
    """
    Orchestrates a simple end-to-end pipeline using injected services.

    Note: This class does not instantiate services itself; the caller provides
    concrete implementations (e.g., mocks or production adapters).
    """
    ingestion: IngestionServiceInterface
    preprocessing: PreprocessingServiceInterface
    modeling: ModelingServiceInterface
    anomaly: AnomalyServiceInterface

    def run(self, modality: str, bbox: Tuple[float, float, float, float]) -> List[Anomaly]:
        """
        Execute the pipeline for a single modality and spatial bbox.

        Args:
            modality: modality name, e.g., "gravity" or "magnetic".
            bbox: tuple(min_lon, min_lat, max_lon, max_lat)

        Returns:
            A list of detected Anomaly instances.
        """
        # Ingestion
        raw = self.ingestion.fetch(modality, bbox)

        # Preprocessing
        grid = self.preprocessing.process(raw)

        # Modeling / Inversion
        inv = self.modeling.invert(grid, modality)

        # Anomaly detection
        anomalies = self.anomaly.detect(inv)

        return anomalies