"""Scientific validation tests for GAM.

These tests validate pipeline outputs against benchmark synthetic data,
ensuring scientific correctness (RMSE < 0.1) and reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

from gam.core.pipeline import GAMPipeline
from gam.ingestion.data_structures import RawData

# Fixtures from conftest.py

@pytest.mark.validation
class TestScientificValidation:
    """Validation tests using benchmark datasets."""

    @pytest.fixture(autouse=True)
    def setup(self, benchmark_dataset, test_config, tmp_path):
        """Shared setup."""
        self.benchmark = benchmark_dataset
        self.config = test_config
        self.tmp_base = tmp_path
        # Convert raw_data to RawData if needed
        raw_json = self.benchmark['raw_data']
        # Assume json is dict for RawData; simplify
        values = np.array(raw_json.get('values', np.random.rand(10,10)))  # Fallback
        metadata = raw_json.get('metadata', {})
        self.raw_data = RawData(values=values, metadata=metadata)
        self.expected_anoms = self.benchmark['expected_anomalies']
        self.tolerance = self.benchmark['tolerance']

    def test_benchmark_accuracy(self):
        """Run GAM on benchmark, assert RMSE < 0.1 on anomalies."""
        output_dir1 = self.tmp_base / "run1"
        output_dir1.mkdir()

        pipeline = GAMPipeline(config=self.config)
        results = pipeline.run_analysis(
            raw_data=self.raw_data,
            modalities=['gravity'],  # Benchmark modality
            output_dir=output_dir1
        )

        # Load generated anomalies.csv
        gen_csv = output_dir1 / 'anomalies.csv'
        assert gen_csv.exists()
        generated_anoms = pd.read_csv(gen_csv)

        # Align columns for RMSE (assume same structure)
        common_cols = ['lat', 'lon', 'depth', 'confidence', 'strength']  # Exclude type if categorical
        if 'anomaly_type' in generated_anoms.columns:
            generated_anoms = generated_anoms.drop('anomaly_type', axis=1)
        if 'anomaly_type' in self.expected_anoms.columns:
            self.expected_anoms = self.expected_anoms.drop('anomaly_type', axis=1)

        # Match by position or nearest; simplify to same order
        rmse = np.sqrt(mean_squared_error(
            self.expected_anoms[common_cols],
            generated_anoms[common_cols].head(len(self.expected_anoms))
        ))
        assert rmse < self.tolerance, f"RMSE {rmse} exceeds tolerance {self.tolerance}"

    def test_reproducibility(self):
        """Run pipeline twice, assert identical outputs."""
        output_dir1 = self.tmp_base / "run1"
        output_dir2 = self.tmp_base / "run2"
        output_dir1.mkdir()
        output_dir2.mkdir()

        pipeline1 = GAMPipeline(config=self.config)
        results1 = pipeline1.run_analysis(
            raw_data=self.raw_data,
            modalities=['gravity'],
            output_dir=output_dir1
        )

        pipeline2 = GAMPipeline(config=self.config)
        results2 = pipeline2.run_analysis(
            raw_data=self.raw_data,
            modalities=['gravity'],
            output_dir=output_dir2
        )

        # Assert identical anomalies DataFrames
        gen_csv1 = output_dir1 / 'anomalies.csv'
        gen_csv2 = output_dir2 / 'anomalies.csv'
        df1 = pd.read_csv(gen_csv1)
        df2 = pd.read_csv(gen_csv2)
        pd.testing.assert_frame_equal(df1, df2)

        # Assert identical models (within float precision)
        np.testing.assert_allclose(results1.inversion.model, results2.inversion.model, atol=1e-6)
        np.testing.assert_allclose(results1.inversion.uncertainty, results2.inversion.uncertainty, atol=1e-6)

    def test_validation_multi_modality(self):
        """Validate multi-modality benchmark if available."""
        # Extend benchmark if multi; here use single for simplicity
        # Assume benchmark has multi; skip if not
        if len(self.benchmark['raw_data']) > 1:  # Dict of modalities
            self.config['pipeline']['modalities'] = list(self.benchmark['raw_data'].keys())
            # Similar to above, run and assert RMSE
            output_dir = self.tmp_base / "multi"
            output_dir.mkdir()
            pipeline = GAMPipeline(config=self.config)
            results = pipeline.run_analysis(
                raw_data=self.benchmark['raw_data'],  # Dict
                modalities=self.config['pipeline']['modalities'],
                output_dir=output_dir
            )
            gen_csv = output_dir / 'anomalies.csv'
            generated = pd.read_csv(gen_csv)
            rmse = np.sqrt(mean_squared_error(self.expected_anoms, generated.head(len(self.expected_anoms))))
            assert rmse < self.tolerance * 1.2  # Slightly looser for fusion