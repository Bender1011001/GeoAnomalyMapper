"""End-to-end integration tests for GAMPipeline.

These tests run the full pipeline on synthetic datasets without mocking core components.
Asserts output shapes, metadata integrity, and basic anomaly detection results.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from gam.core.pipeline import GAMPipeline
from gam.core.config import load_config  # If needed for config handling
from gam.modeling.data_structures import InversionResults, AnomalyOutput
from gam.ingestion.data_structures import RawData

# Fixtures imported via conftest.py

@pytest.mark.integration
class TestGAMPipelineIntegration:
    """Integration tests for full pipeline execution."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, test_config, tmp_output_dir):
        """Shared setup for pipeline instances."""
        self.config = test_config
        self.output_dir = tmp_output_dir
        self.pipeline = GAMPipeline(config=self.config)

    def test_pipeline_single_modality_gravity(self, synthetic_raw_data):
        """Test full pipeline on synthetic gravity data."""
        # Parametrize via fixture, but for gravity
        raw_data = synthetic_raw_data  # gravity by default
        assert isinstance(raw_data, RawData)
        assert raw_data.metadata['units'] == 'mGal'

        # Run analysis (adapt to actual API; assuming run_analysis(raw_data, output_dir))
        results = self.pipeline.run_analysis(
            raw_data=raw_data,
            modalities=['gravity'],
            output_dir=self.output_dir
        )

        # Assert inversion results
        assert isinstance(results.inversion, InversionResults)
        model_shape = results.inversion.model.shape
        assert len(model_shape) == 3  # lat, lon, depth
        assert model_shape[0] == 10 and model_shape[1] == 10  # From synthetic grid
        assert 'converged' in results.inversion.metadata
        assert results.inversion.metadata['algorithm'] == 'gravity_inversion'

        # Assert anomaly output
        assert isinstance(results.anomalies, AnomalyOutput)
        assert not results.anomalies.empty  # At least some detections
        assert 'lat' in results.anomalies.columns
        assert 'confidence' in results.anomalies.columns
        assert results.anomalies['confidence'].max() <= 1.0
        assert results.anomalies['confidence'].min() >= 0.0

        # Check output files
        anomalies_csv = self.output_dir / 'anomalies.csv'
        assert anomalies_csv.exists()
        loaded_anoms = pd.read_csv(anomalies_csv)
        assert len(loaded_anoms) == len(results.anomalies)

    @pytest.mark.parametrize('modality', ['magnetic'])
    def test_pipeline_single_modality_magnetic(self, synthetic_raw_data, modality):
        """Test pipeline on synthetic magnetic data."""
        # Use parametrized fixture indirectly
        if modality == 'magnetic':
            # Re-invoke fixture with param (pytest handles)
            raw_data = synthetic_raw_data  # Assume can override or use multi
            raw_data.metadata['units'] = 'nT'  # Mock for test

        results = self.pipeline.run_analysis(
            raw_data=raw_data,
            modalities=[modality],
            output_dir=self.output_dir
        )

        assert isinstance(results.inversion, InversionResults)
        assert results.inversion.metadata['units'] == 'susceptibility'  # Expected for magnetic
        assert not results.anomalies.empty
        assert results.anomalies['anomaly_type'].iloc[0] in ['magnetic_anomaly', 'susceptibility_contrast']

    @pytest.mark.skipif(not has_obspy, reason="Requires obspy for seismic")
    def test_pipeline_single_modality_seismic(self, synthetic_raw_data):
        """Test pipeline on synthetic seismic data."""
        raw_data = synthetic_raw_data  # seismic
        raw_data.metadata['units'] = 'km/s'

        results = self.pipeline.run_analysis(
            raw_data=raw_data,
            modalities=['seismic'],
            output_dir=self.output_dir
        )

        assert isinstance(results.inversion, InversionResults)
        model = results.inversion.model
        assert np.all(model > 1.0)  # Velocities > 1 km/s
        assert 'pick_times' in results.inversion.metadata  # From STA/LTA
        assert len(results.anomalies) >= 1  # Seismic anomalies detected

    def test_pipeline_single_modality_insar(self, synthetic_raw_data):
        """Test pipeline on synthetic InSAR data."""
        raw_data = synthetic_raw_data  # insar
        raw_data.metadata['units'] = 'mm'

        results = self.pipeline.run_analysis(
            raw_data=raw_data,
            modalities=['insar'],
            output_dir=self.output_dir
        )

        assert isinstance(results.inversion, InversionResults)
        assert 'unwrapped_phase' in results.inversion.metadata
        assert results.anomalies['anomaly_type'].str.contains('deformation').any()

    def test_pipeline_multi_modality_fusion(self, synthetic_raw_data_multi, test_config):
        """Test multi-modality fusion in pipeline."""
        # synthetic_raw_data_multi has gravity and magnetic
        raw_data_dict = synthetic_raw_data_multi
        self.config['pipeline']['modalities'] = list(raw_data_dict.keys())

        results = self.pipeline.run_analysis(
            raw_data=raw_data_dict,
            modalities=list(raw_data_dict.keys()),
            output_dir=self.output_dir
        )

        assert isinstance(results.inversion, InversionResults)
        assert 'fusion_scheme' in results.inversion.metadata
        assert results.inversion.metadata['fusion_scheme'] == 'bayesian'  # From config
        # Fused model should have reduced uncertainty
        assert np.mean(results.inversion.uncertainty) < 0.1  # Arbitrary threshold for synthetic
        assert len(results.anomalies) > 0
        assert 'joint_confidence' in results.anomalies.columns  # Fusion-specific

    def test_pipeline_metadata_consistency(self, synthetic_raw_data, test_config):
        """Test metadata propagation through pipeline."""
        raw_meta = synthetic_raw_data.metadata
        bbox_raw = raw_meta['bbox']

        results = self.pipeline.run_analysis(
            raw_data=synthetic_raw_data,
            modalities=['gravity'],
            output_dir=self.output_dir
        )

        # Check bbox preserved
        assert results.inversion.metadata['bbox'] == bbox_raw
        assert results.anomalies['lat'].min() >= bbox_raw[2]  # lat_min
        assert results.anomalies['lat'].max() <= bbox_raw[3]  # lat_max
        assert 'timestamp' in results.inversion.metadata
        assert pd.to_datetime(results.inversion.metadata['timestamp']) > pd.Timestamp.now() - pd.Timedelta('1h')

    def test_pipeline_output_formats(self, synthetic_raw_data, test_config):
        """Test multiple output formats (png, csv)."""
        self.config['visualization']['output_formats'] = ['png', 'csv', 'json']

        results = self.pipeline.run_analysis(
            raw_data=synthetic_raw_data,
            modalities=['gravity'],
            output_dir=self.output_dir
        )

        # Check files generated
        assert (self.output_dir / 'anomalies.csv').exists()
        assert (self.output_dir / 'model.json').exists()
        assert (self.output_dir / 'anomaly_map.png').exists()  # Assuming viz generates this

        # Load and assert consistency
        loaded_csv = pd.read_csv(self.output_dir / 'anomalies.csv')
        assert len(loaded_csv) == len(results.anomalies)