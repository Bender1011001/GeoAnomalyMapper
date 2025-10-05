from pathlib import Path

from GeoAnomalyMapper.gam.core.orchestrator import PipelineOrchestrator
from GeoAnomalyMapper.gam.core.data_contracts import Anomaly
from GeoAnomalyMapper.gam.services import (
    MockIngestionService,
    MockPreprocessingService,
    MockModelingService,
    MockAnomalyService,
)


def test_pipeline_with_mocks_returns_anomaly():
    """
    Smoke-test the DI-only PipelineOrchestrator using the package-provided
    deterministic mock services. The mocks are import-safe and should return
    a small, deterministic list containing one Anomaly.
    """
    orch = PipelineOrchestrator(
        ingestion=MockIngestionService(),
        preprocessing=MockPreprocessingService(),
        modeling=MockModelingService(),
        anomaly=MockAnomalyService(),
    )

    bbox = (-120.0, 30.0, -119.0, 31.0)
    results = orch.run("gravity", bbox)

    assert isinstance(results, list)
    assert len(results) == 1
    first = results[0]
    assert isinstance(first, Anomaly)
    # Verify confidence bounds are respected
    assert 0.0 <= first.confidence <= 1.0