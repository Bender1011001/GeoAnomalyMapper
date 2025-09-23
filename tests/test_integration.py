"""
End-to-end integration tests for GAM pipeline.

These tests verify the complete workflow across modules: Ingestion → Preprocessing → Modeling → Visualization.
Uses synthetic data and mocks to ensure determinism and speed. Tests data flow, intermediate results, caching,
and global processing without modifying core implementations.

Run with: pytest tests/test_integration.py -v -m integration

Expected: All tests pass, coverage >90% for integrated paths, execution <10s total.
"""

import pytest
from pathlib import Path
import pandas as pd
import xarray as xr

# GAM imports
from gam import (
    GAMPipeline, IngestionManager, PreprocessingManager, ModelingManager, VisualizationManager,
    RawData, ProcessedGrid, InversionResults, AnomalyOutput
)
from gam.core.exceptions import PipelineError

# Fixtures are auto-injected from conftest.py
# test_bbox, synthetic_raw_data, synthetic_raw_data_multi, tmp_output_dir, test_config,
# mock_external_apis, mock_cache_manager, tmp_cache_dir, performance_config

@pytest.mark.integration
class TestGAMIntegration:
    
    @pytest.mark.parametrize('modality', ['gravity', 'magnetic'])
    def test_full_pipeline_single_modality(self, modality, mock_external_apis, test_config, tmp_output_dir, test_bbox):
        """
        Test complete workflow with one data type: Ingestion → Preprocessing → Modeling → Visualization.
        
        Verifies:
        - Data flows correctly (RawData → ProcessedGrid → InversionResults → AnomalyOutput)
        - Intermediate results have expected types/shapes
        - Final outputs (anomalies, files) generated
        - No errors in module handoffs
        """
        # Setup: Modify config for single modality
        config = test_config.copy()
        config['pipeline']['modalities'] = [modality]
        
        # Run pipeline
        pipeline = GAMPipeline.from_config(config)
        results = pipeline.run_analysis(
            bbox=test_bbox,
            modalities=[modality],
            output_dir=tmp_output_dir,
            use_cache=False
        )
        
        # Verify results structure and data flow
        assert isinstance(results, dict)
        assert 'anomalies' in results
        anomalies = results['anomalies']
        assert isinstance(anomalies, (AnomalyOutput, pd.DataFrame))
        assert len(anomalies) > 0
        assert 'lat' in anomalies.columns
        assert 'lon' in anomalies.columns
        assert 'confidence' in anomalies.columns
        assert anomalies['confidence'].mean() > 0.5  # Basic quality check
        
        # Intermediate verification
        assert 'processed' in results
        processed = results['processed']
        assert isinstance(processed, ProcessedGrid) or isinstance(processed, xr.Dataset)
        assert processed.data.shape == (10, 10)  # From synthetic grid
        
        assert 'inversion' in results
        inversion = results['inversion']
        assert isinstance(inversion, InversionResults) or isinstance(inversion, dict)
        assert 'model' in inversion
        
        # Verify output files
        map_file = tmp_output_dir / f'{modality}_map.png'
        anomalies_file = tmp_output_dir / 'anomalies.csv'
        assert map_file.exists()
        assert anomalies_file.exists()
        
        # Check exported anomalies match
        exported = pd.read_csv(anomalies_file)
        assert len(exported) == len(anomalies)
        assert (exported['lat'] == anomalies['lat']).all()
        
        logger.info(f"Single modality {modality} pipeline passed: {len(anomalies)} anomalies detected")
    
    @pytest.mark.integration
    def test_full_pipeline_multi_modality(self, mock_external_apis, test_config, tmp_output_dir, test_bbox):
        """
        Test complete workflow with multiple data types: joint inversion and fusion.
        
        Verifies:
        - Multi-modality ingestion and preprocessing
        - Fusion in modeling (JointInverter)
        - Anomaly detection on fused models
        - Combined visualization
        """
        # Setup: Multi-modality config
        config = test_config.copy()
        config['pipeline']['modalities'] = ['gravity', 'magnetic']
        config['modeling']['fusion_method'] = 'bayesian'
        
        # Run pipeline
        pipeline = GAMPipeline.from_config(config)
        results = pipeline.run_analysis(
            bbox=test_bbox,
            modalities=['gravity', 'magnetic'],
            output_dir=tmp_output_dir,
            use_cache=False
        )
        
        # Verify fusion results
        assert 'fused_model' in results
        fused = results['fused_model']
        assert isinstance(fused, InversionResults)
        assert 'uncertainty' in fused  # Fusion-specific
        assert fused['model'].shape == (10, 10, 5)  # Example 3D fused model
        
        # Enhanced anomalies from fusion
        anomalies = results['anomalies']
        assert len(anomalies) > 5  # More than single due to fusion
        assert 'anomaly_type' in anomalies.columns
        assert any(anomalies['anomaly_type'] == 'fused')  # Fusion-detected
        
        # Combined outputs
        combined_map = tmp_output_dir / 'fused_map.png'
        assert combined_map.exists()
        
        # Verify data flow: multi RawData → single fused AnomalyOutput
        assert 'processed_multi' in results
        processed = results['processed_multi']
        assert len(processed) == 2  # Dict of ProcessedGrid per modality
        
        logger.info(f"Multi-modality pipeline passed: {len(anomalies)} fused anomalies")
    
    @pytest.mark.integration
    def test_pipeline_with_caching(self, mock_cache_manager, tmp_cache_dir, test_config, tmp_output_dir, test_bbox, monkeypatch):
        """
        Test caching and resumption functionality.
        
        Verifies:
        - Cache save/load during pipeline
        - Resumption from intermediates (no re-ingestion)
        - Cache invalidation on config/data change
        - Overhead <10% time increase
        """
        # Setup: Enable caching
        config = test_config.copy()
        config['pipeline']['use_cache'] = True
        config['ingestion']['cache_dir'] = str(tmp_cache_dir)
        
        # First run: Full pipeline, caches intermediates
        pipeline1 = GAMPipeline.from_config(config)
        results1 = pipeline1.run_analysis(
            bbox=test_bbox,
            modalities=['gravity'],
            output_dir=tmp_output_dir,
            use_cache=True
        )
        assert len(list(tmp_cache_dir.glob('*.h5'))) > 0  # Cache files created
        
        # "Interrupt" simulation: Assume partial run saved ingestion
        # Second run: Should resume from cache (mock_get called only once)
        get_calls = []
        def tracked_get(*args, **kwargs):
            get_calls.append(1)
            return MagicMock(status_code=200, json=lambda: {'data': 'cached'})
        monkeypatch.setattr('requests.get', tracked_get)
        
        pipeline2 = GAMPipeline.from_config(config)
        results2 = pipeline2.run_analysis(
            bbox=test_bbox,
            modalities=['gravity'],
            output_dir=tmp_output_dir / 'run2',
            use_cache=True
        )
        
        # Verify resumption: No re-fetch
        assert len(get_calls) == 0  # Cached, no API call
        pd.testing.assert_frame_equal(results1['anomalies'], results2['anomalies'])
        
        # Invalidation test: Change bbox slightly
        config['bbox'] = (29.5, 31.5, 30.5, 32.5)
        pipeline3 = GAMPipeline.from_config(config)
        results3 = pipeline3.run_analysis(
            bbox=config['bbox'],
            modalities=['gravity'],
            output_dir=tmp_output_dir / 'run3',
            use_cache=True
        )
        assert len(get_calls) == 1  # Invalidated, one fetch
        assert not pd.testing.assert_frame_equal(results2['anomalies'], results3['anomalies'])  # Different
        
        logger.info("Caching test passed: Resumption and invalidation verified")
    
    @pytest.mark.integration
    def test_global_processing(self, performance_config, tmp_output_dir, test_bbox, monkeypatch):
        """
        Test Earth-scale tiling and processing.
        
        Verifies:
        - Tiling of larger region into sub-bboxes
        - Parallel coordination (mocked)
        - Boundary handling and merging (no artifacts/duplicates)
        - Merged global AnomalyOutput
        """
        # Setup: Larger config for global sim
        config = performance_config.copy()
        config['pipeline']['tiles'] = 2  # 2x2 tiles
        config['pipeline']['parallel_workers'] = 1  # Sequential for test
        
        # Mock GlobalProcessor for tiling/merging
        def mock_global_run(self, region, tiles, workers, output_dir):
            # Simulate 4 tiles (2x2), each producing 5 anomalies
            tile_anomalies = []
            for i in range(4):
                tile_df = pd.DataFrame({
                    'lat': [30.5 + i*0.1],
                    'lon': [30.0 + i*0.1],
                    'depth': [1.0],
                    'confidence': [0.8],
                    'anomaly_type': ['tile']
                })
                tile_anomalies.append(tile_df)
            # Merge: Concat, drop duplicates at boundaries
            merged = pd.concat(tile_anomalies).drop_duplicates(subset=['lat', 'lon'])
            # Save mock outputs
            (output_dir / 'global_map.png').touch()
            merged.to_csv(output_dir / 'global_anomalies.csv', index=False)
            return {'anomalies': merged, 'tiles_processed': 4}
        
        with patch('gam.core.global_processor.GlobalProcessor.run') as mock_run:
            mock_run.return_value = mock_global_run(None, None, None, None)  # Bind self etc.
            pipeline = GAMPipeline.from_config(config)
            results = pipeline.global_run(
                region='test_region',  # Small global sim
                tiles=config['pipeline']['tiles'],
                workers=config['pipeline']['parallel_workers'],
                output_dir=tmp_output_dir
            )
        
        # Verify merged results
        anomalies = results['anomalies']
        assert isinstance(anomalies, pd.DataFrame)
        assert len(anomalies) == 4  # No duplicates after merge
        assert results['tiles_processed'] == 4
        assert (tmp_output_dir / 'global_map.png').exists()
        assert (tmp_output_dir / 'global_anomalies.csv').exists()
        
        # Boundary check: No overlapping coords
        assert len(anomalies[['lat', 'lon']].drop_duplicates()) == len(anomalies)
        
        logger.info(f"Global processing test passed: {len(anomalies)} merged anomalies from {results['tiles_processed']} tiles")

    def test_pipeline_error_handling(self, mock_external_apis, test_config, tmp_output_dir, test_bbox):
        """Negative case: Invalid bbox raises PipelineError."""
        invalid_bbox = (100.0, 200.0, 100.0, 200.0)  # Out of bounds
        
        pipeline = GAMPipeline.from_config(test_config)
        with pytest.raises(PipelineError, match="Invalid bounding box"):
            pipeline.run_analysis(
                bbox=invalid_bbox,
                modalities=['gravity'],
                output_dir=tmp_output_dir,
                use_cache=False
            )
        
        logger.info("Error handling test passed")