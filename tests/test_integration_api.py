"""
Integration tests for GAM API and dashboard communication.

Tests full workflow: Start job → Poll status → Get results, with different analysis scenarios and presets.
Tests API + Dashboard patterns, error scenarios/recovery, mocking external data sources/services.
Uses TestClient for API, mocks background tasks and pipeline for end-to-end simulation without real execution.
Achieves >85% coverage for integrated workflows.

Run with: pytest tests/test_integration_api.py -v --cov=gam/api --cov=dashboard/app
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

from gam.api.main import app, jobs, run_analysis_job
from dashboard.app import start_analysis_job, get_job_status, get_job_results, extract_anomalies
from dashboard.presets import get_all_presets
from gam.core.pipeline import GAMPipeline
from gam.core.config import GAMConfig
from gam.core.exceptions import IngestionError, PreprocessingError, ModelingError, VisualizationError, PipelineError


@pytest.fixture
def test_client():
    """TestClient for API integration."""
    return TestClient(app)


@pytest.fixture
def mock_background_task():
    """Mock run_analysis_job to simulate lifecycle without real execution."""
    with patch('gam.api.main.run_analysis_job') as mock:
        yield mock


@pytest.fixture
def mock_pipeline():
    """Mock GAMPipeline for integration."""
    mock_pipeline = MagicMock()
    mock_pipeline.ingestion.fetch_multiple.return_value = {"gravity": MagicMock()}
    mock_pipeline.preprocessing.process.return_value = MagicMock()
    mock_pipeline.modeling.invert.return_value = MagicMock()
    mock_pipeline.modeling.detect_anomalies