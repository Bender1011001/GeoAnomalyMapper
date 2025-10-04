"""
Data validation tests for GAM pipeline.

These tests verify data format consistency across modules, coordinate system handling,
and error propagation. Ensures RawData → ProcessedGrid → InversionResults → AnomalyOutput
flow preserves metadata, units, and coordinates. Uses synthetic data for determinism.

Run with: pytest tests/test_data_validation.py -v -m validation

Validations:
- Format: Types, shapes, metadata preservation
- Coordinates: Projections, bbox calculations
- Errors: Graceful handling and reporting

Dependencies: pytest, pyproj, xarray, pandas.
"""

import pytest
from pyproj import Transformer, CRS
import numpy as np
import xarray as xr
import pandas as pd

# GAM imports
from gam import (
    IngestionManager, PreprocessingManager, ModelingManager, VisualizationManager,
    RawData, ProcessedGrid, InversionResults, AnomalyOutput
)
from gam.core.exceptions import PipelineError, GAMError, IngestionError

# Fixtures from conftest.py
# synthetic_raw_data, test_config, mock_external_apis, test_bbox

@pytest.mark.validation
class TestDataValidation:
    
    def test_data_format_consistency(self, synthetic_raw_data, test_config, mock_external_apis):
        """
        Verify data formats between modules: RawData → ProcessedGrid → InversionResults → AnomalyOutput.
        
        Checks:
        - Type consistency (dataclass/Dataset/DataFrame)
        - Shape preservation (grid dims)
        - Coordinate system (WGS84 lat/lon)
        - Units and metadata propagation
        """
        modality = 'gravity'  # From param or default
        
        # Stage 1: Ingestion → RawData
        ingestion = IngestionManager.from_config(test_config['ingestion'])
        raw_data = ingestion.fetch_modality(modality, bbox=test_bbox)
        assert isinstance(raw_data, RawData)
        assert isinstance(raw_data.values, xr.Dataset)
        assert 'lat' in raw_data.values.dims
        assert 'lon' in raw_data.values.dims
        assert raw_data.values.data.shape == (10, 10)  # Synthetic size
        initial_bbox = raw_data.metadata['bbox']
        initial_units = raw_data.metadata['units']
        assert initial_bbox == test_bbox
        assert initial_units == 'mGal'
        
        # Stage 2: Preprocessing → ProcessedGrid
        preprocessing = PreprocessingManager.from_config(test_config['preprocessing'])
        processed = preprocessing.process(raw_data, modality)
        assert isinstance(processed, ProcessedGrid) or isinstance(processed, xr.Dataset)
        assert processed.data.shape == (10, 10)  # Same grid
        assert 'lat' in processed.coords
        assert 'lon' in processed.coords
        assert processed.attrs.get('units') == initial_units  # Preserved
        assert processed.attrs.get('bbox') == initial_bbox
        
        # Stage 3: Modeling → InversionResults
        modeling = ModelingManager.from_config(test_config['modeling'])
        inversion = modeling.run_inversion(processed, modality)
        assert isinstance(inversion, InversionResults) or isinstance(inversion, dict)
        assert 'model' in inversion
        model = inversion['model']
        assert isinstance(model, xr.Dataset) or isinstance(model, np.ndarray)
        if isinstance(model, xr.Dataset):
            assert model.dims == ('lat', 'lon')  # Consistent dims
        assert inversion.get('uncertainty') is not None
        assert inversion['metadata']['bbox'] == initial_bbox  # Propagated
        
        # Stage 4: Visualization → AnomalyOutput
        visualization = VisualizationManager.from_config(test_config['visualization'])
        anomalies = visualization.generate(inversion)
        assert isinstance(anomalies, AnomalyOutput) or isinstance(anomalies, pd.DataFrame)
        assert len(anomalies) > 0
        assert 'lat' in anomalies.columns
        assert 'lon' in anomalies.columns
        assert anomalies['lat'].between(30.0, 32.0).all()  # Within bbox
        assert anomalies['lon'].between(29.0, 31.0).all()
        
        # Overall consistency
        assert processed.attrs.get('crs') == 'EPSG:4326'  # WGS84
        assert anomalies.attrs.get('source_units') == initial_units if hasattr(anomalies, 'attrs') else True
        
        logger.info("Data format consistency test passed: Metadata preserved across stages")
    
    @pytest.mark.parametrize('projection', ['EPSG:4326', 'EPSG:32636'])  # WGS84, UTM 36N
    def test_coordinate_transformations(self, projection, synthetic_raw_data, test_config, mock_external_apis):
        """
        Test geographic coordinate handling and transformations.
        
        Verifies:
        - Different projections (WGS84, UTM)
        - Bounding box calculations (min/max, area)
        - Round-trip accuracy (transform back <1m error)
        """
        # Define transformer
        wgs84 = CRS.from_epsg(4326)
        target_crs = CRS.from_string(projection)
        transformer = Transformer.from_crs(wgs84, target_crs, always_xy=True)
        
        # Sample points from bbox
        lon_min, lon_max, lat_min, lat_max = test_bbox
        sample_points = [(lon_min, lat_min), (lon_max, lat_max)]
        
        # Transform forward
        transformed = [transformer.transform(lon, lat) for lon, lat in sample_points]
        trans_bbox = (min(x for x, y in transformed), max(x for x, y in transformed),
                      min(y for x, y in transformed), max(y for x, y in transformed))
        
        # Run pipeline with transformed input (mock preprocess to use trans)
        config_copy = test_config.copy()
        config_copy['preprocessing']['projection'] = projection
        preprocessing = PreprocessingManager.from_config(config_copy['preprocessing'])
        processed = preprocessing.process(synthetic_raw_data, 'gravity')
        
        # Verify output bbox matches transformed
        output_bbox = processed.attrs.get('bbox')
        if output_bbox:
            assert np.allclose(output_bbox, trans_bbox, atol=1e-6)  # Precision
        
        # Round-trip: Transform back to WGS84
        back_transformer = Transformer.from_crs(target_crs, wgs84, always_xy=True)
        back_points = [back_transformer.transform(x, y) for x, y in transformed]
        for orig, back in zip(sample_points, back_points):
            assert np.allclose(orig, back, atol=1e-6)  # <1m error approx
        
        # Bbox area consistency (simplified)
        wgs84_area = (lon_max - lon_min) * (lat_max - lat_min)  # Deg approx
        processed_area = processed.attrs.get('area', 0)
        assert processed_area > 0  # Valid calculation
        
        logger.info(f"Coordinate transformation test passed for {projection}")
    
    def test_error_propagation(self, mock_external_apis, test_config, tmp_output_dir, test_bbox, monkeypatch):
        """
        Test how errors propagate through the pipeline.
        
        Simulates:
        - Ingestion failure (IngestionError)
        - Preprocessing NaN/invalid data
        - Modeling failure (e.g., singular matrix)
        - Graceful degradation and error reporting
        """
        # Case 1: Ingestion error
        def mock_fetch_error(modality, bbox):
            raise IngestionError("Mock API failure")
        monkeypatch.setattr(IngestionManager, 'fetch_modality', mock_fetch_error)
        
        pipeline = GAMPipeline.from_config(test_config)
        with pytest.raises(PipelineError, match="Ingestion failed.*IngestionError"):
            pipeline.run_analysis(bbox=test_bbox, modalities=['gravity'], output_dir=tmp_output_dir, use_cache=False)
        
        # Case 2: Preprocessing invalid data (NaN grid)
        def mock_process_invalid(raw_data, modality):
            da = xr.DataArray(np.full((10, 10), np.nan), coords=raw_data.values.coords, dims=['lat', 'lon'])
            processed = da.to_dataset(name='data')
            processed.attrs['valid'] = False
            return ProcessedGrid(processed)  # Or xr.Dataset
        
        monkeypatch.setattr(PreprocessingManager, 'process', mock_process_invalid)
        
        # Run full, expect modeling to handle or skip
        results = pipeline.run_analysis(bbox=test_bbox, modalities=['gravity'], output_dir=tmp_output_dir, use_cache=False)
        assert 'errors' in results or 'warnings' in results  # Reported
        assert len(results.get('anomalies', pd.DataFrame())) == 0  # No anomalies from invalid
        assert "Invalid data in preprocessing" in str(results.get('log', ''))  # Logged
        
        # Case 3: Modeling error (e.g., timeout or singular)
        def mock_inversion_error(data, modality):
            raise GAMError("Mock inversion failure: singular matrix")
        monkeypatch.setattr(ModelingManager, 'run_inversion', mock_inversion_error)
        
        with pytest.raises(GAMError, match="singular matrix"):
            # Partial run or full with catch
            inversion = ModelingManager.from_config(test_config['modeling']).run_inversion(processed, 'gravity')
        
        # Graceful: Pipeline continues with partial results
        config_partial = test_config.copy()
        config_partial['pipeline']['graceful_degrade'] = True
        pipeline_partial = GAMPipeline.from_config(config_partial)
        partial_results = pipeline_partial.run_analysis(bbox=test_bbox, modalities=['gravity'], output_dir=tmp_output_dir, use_cache=False)
        assert 'partial' in partial_results
        assert len(partial_results['errors']) == 1  # Captured
        
        logger.info("Error propagation test passed: Graceful handling verified")