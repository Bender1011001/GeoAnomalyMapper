"""
Anomaly detection service interface and a deterministic mock.

Import-safe: no I/O or heavy computation at import time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from gam.core.data_contracts import InversionResult, Anomaly


class AnomalyServiceInterface(ABC):
    """
    Stateless anomaly detection interface.

    Implementations should accept an InversionResult and return a list of Anomaly
    objects describing detected features.
    """

    @abstractmethod
    def detect(self, inv: InversionResult) -> List[Anomaly]:
        """
        Detect anomalies from inversion results.

        Args:
            inv: InversionResult produced by a modeling service.

        Returns:
            A list of Anomaly dataclass instances.
        """
        raise NotImplementedError


class MockAnomalyService(AnomalyServiceInterface):
    """
    Deterministic mock anomaly detector.

    This returns one fixed anomaly suitable for orchestrator smoke tests and
    unit tests. It avoids any I/O or nondeterminism.
    """

    def detect(self, inv: InversionResult) -> List[Anomaly]:
        # Return a single deterministic anomaly as specified in Phase 1 scaffolding.
        anomaly = Anomaly(
            latitude=30.5,
            longitude=-119.5,
            depth_meters=100.0,
            confidence=0.9,
            anomaly_type="mock",
        )
        return [anomaly]