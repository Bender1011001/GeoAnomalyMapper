"""End-to-end system integration tests for GeoAnomalyMapper (GAM).
These tests validate the complete pipeline: ingestion -> preprocessing -> modeling -> visualization.
Run with: pytest tests/system/test_full_system.py -v -m integration

Requires Docker for containerized testing and synthetic test data in tests/data/.
"""

import os
import pytest
import subprocess
import time
import pandas as pd
import rasterio
from pathlib import Path
from hashlib import md5

from gam.core.pipeline import GAMPipeline
from gam.core.config import load_config
from gam.core.cli import cli
from gam.tests.conftest import synthetic_data_dir, test_region_geojson

# Mark as integration tests
pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def test_config_path(tmp_path_factory):
    """Fixture for test configuration."""
    config_dir = tmp_path_factory.mktemp("test_config")
    test_config = config_dir / "test_config.yaml"
    test_config.write_text("""
global:
  environment: test
  debug: true
  timeout_seconds: 300

database:
  url: sqlite:///:memory:  # In-memory for tests

ingestion:
  cache_ttl_seconds: 60
  sources:
    synthetic: true  # Use test data only

preprocessing:
  parallel_workers: 2
  grid:
    resolution_deg: 0.1

modeling:
  fusion:
    method: simple
  anomaly_detection:
    contamination: 0.1

visualization:
  export_formats: ['csv', 'geotiff']
  interactive: false

performance:
  batch_size: 100
""")
    return test_config

@pytest.fixture(scope="module")
def gam_pipeline(test_config_path):
    """Fixture for GAM pipeline with test config."""
    config = load_config(test_config_path)
    return GAMPipeline(config=config)

class TestFullSystem:
    """Full system integration tests."""

    def test_complete_pipeline_run(self, gam_pipeline, synthetic_data_dir, test_region_geojson, tmp_path):
        """Test end-to-end pipeline execution with synthetic data."""
        start_time = time.time()
        
        # Set input/output paths
        input_dir = synthetic_data_dir
        bbox = (29.0, 31.0, 30.0, 32.0)  # Small test region
        output_dir = tmp_path / "gam_output"
        output_dir.mkdir()
        
        # Run full pipeline
        results = gam_pipeline.run_analysis(
            bbox=bbox,
            modalities=['gravity', 'magnetic'],
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            region_file=str(test_region_geojson)
        )
        
        execution_time = time.time() - start_time
        assert execution_time < 60, f"Pipeline took too long: {execution_time}s"
        
        # Verify outputs
        anomalies_csv = output_dir / "anomalies.csv"
        assert anomalies_csv.exists(), "Anomalies CSV not generated"
        
        # Validate CSV content
        df = pd.read_csv(anomalies_csv)
        assert not df.empty, "Anomalies CSV is empty"
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'anomaly_value' in df.columns
        assert len(df) > 0, "No anomalies detected"
        
        # Check visualization outputs
        geotiff = output_dir / "anomaly_map.geotiff"
        assert geotiff.exists(), "GeoTIFF not generated"
        
        # Validate GeoTIFF
        with rasterio.open(geotiff) as src:
            assert src.crs is not None, "GeoTIFF missing CRS"
            assert src.width > 0 and src.height > 0, "GeoTIFF empty"
            assert src.count == 1, "GeoTIFF should have 1 band"
        
        # Verify report
        report = output_dir / "analysis_report.pdf"
        assert report.exists(), "Report PDF not generated"
        
        # Check checksum of expected output (for regression testing)
        expected_checksum = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # Example for empty
        with open(anomalies_csv, 'rb') as f:
            checksum = md5(f.read()).hexdigest()
        assert checksum == expected_checksum, f"Output checksum mismatch: {checksum}"
        
        logger.info(f"Full pipeline test passed in {execution_time:.2f}s")

    def test_cli_full_workflow(self, test_config_path, synthetic_data_dir, tmp_path):
        """Test full workflow via CLI."""
        output_dir = tmp_path / "cli_output"
        output_dir.mkdir()
        
        # Run via CLI subprocess
        cmd = [
            sys.executable, "-m", "gam.core.cli",
            "run",
            "--config", str(test_config_path),
            "--bbox", "29,31,30,32",
            "--modalities", "gravity,magnetic",
            "--input-dir", str(synthetic_data_dir),
            "--output-dir", str(output_dir),
            "--region-file", str(test_region_geojson)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Pipeline completed successfully" in result.stdout
        
        # Verify outputs exist
        anomalies_csv = output_dir / "anomalies.csv"
        assert anomalies_csv.exists()
        df = pd.read_csv(anomalies_csv)
        assert len(df) > 0

    def test_error_handling_in_pipeline(self, gam_pipeline, tmp_path):
        """Test pipeline error handling with invalid input."""
        with pytest.raises(ValueError, match="Invalid bbox format"):
            gam_pipeline.run_analysis(bbox="invalid", output_dir=str(tmp_path))

        # Test with missing modality
        with pytest.raises(KeyError, match="Unknown modality"):
            gam_pipeline.run_analysis(bbox=(0,0,1,1), modalities=['unknown'], output_dir=str(tmp_path))

        logger.info("Error handling tests passed")

    def test_config_validation(self, test_config_path, gam_pipeline):
        """Test production config loading and validation."""
        config = load_config(test_config_path)
        assert config['global']['environment'] == 'test'
        assert config['database']['url'].startswith('sqlite')
        assert config['ingestion']['sources']['synthetic'] is True
        
        # Validate pipeline accepts config
        gam_pipeline.config = config
        gam_pipeline.validate_config()
        
        logger.info("Config validation passed")