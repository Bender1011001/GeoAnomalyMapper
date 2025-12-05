#!/usr/bin/env python3
"""
Dry-run test for GeoAnomalyMapper v2.0 full pipeline.
Mocks input rasters in temp RAW_DIR, verifies all 7 key outputs generated.
Ensures end-to-end execution without crashes, data flow, and reasonable outputs.
"""

import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
from pathlib import Path

# Fix PROJ_LIB for conda environment
if 'PROJ_LIB' not in os.environ:
    os.environ['PROJ_LIB'] = os.path.join(sys.prefix, 'Library', 'share', 'proj')

import rasterio
from unittest.mock import patch
import workflow


class TestPipelineV2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="geoanomaly_test_")
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.raw_dir.mkdir()

        # Create directory structure mimicking RAW_DIR
        (self.raw_dir / "gravity").mkdir()
        insar_dir = self.raw_dir / "insar" / "sentinel1"
        insar_dir.mkdir(parents=True)
        (self.raw_dir / "emag2").mkdir()
        (self.raw_dir / "dem").mkdir()

        # Test parameters: small region for 100x100 grid
        self.bounds = (-105.0, 32.0, -104.0, 33.0)
        self.resolution = 0.01
        self.shape = (100, 100)  # (height, width)

        # Create mock input rasters with valid metadata and plausible data ranges
        # Gravity: Bouguer-like values
        self._create_mock_raster(
            self.raw_dir / "gravity" / "mock_gravity.tif",
            data=np.random.normal(50, 10, self.shape).astype(np.float32)
        )
        # Magnetic: EMAG2 nT values
        self._create_mock_raster(
            self.raw_dir / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif",
            data=np.random.normal(0, 100, self.shape).astype(np.float32)
        )
        # InSAR coherence stack: 2 files, 0-1 range
        for i in range(2):
            coh_data = np.random.uniform(0.1, 0.9, self.shape).astype(np.float32)
            self._create_mock_raster(
                insar_dir / f"coh_{i+1:03d}.tif",
                data=coh_data
            )
        # DEM: elevation-like
        dem_data = np.random.uniform(0, 1500, self.shape).astype(np.float32)
        self._create_mock_raster(
            self.raw_dir / "dem" / "mock_dem.tif",
            data=dem_data
        )

        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)

    def _create_mock_raster(self, path: Path, data: np.ndarray = None):
        """
        Create a GeoTIFF with valid CRS, transform matching test bounds/resolution,
        and provided data.
        """
        if data is None:
            data = np.random.normal(50, 10, self.shape).astype(np.float32)

        west, south, east, north = self.bounds
        transform = rasterio.transform.from_bounds(
            west, south, east, north, self.shape[1], self.shape[0]
        )

        profile = {
            "driver": "GTiff",
            "height": self.shape[0],
            "width": self.shape[1],
            "count": 1,
            "dtype": data.dtype,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

    def test_end_to_end(self):
        """Test full pipeline execution with mocks: no crashes, all outputs exist."""
        region = self.bounds
        resolution = self.resolution
        output_prefix = str(self.output_dir / "test_run")

        # Monkeypatch RAW_DIR to use temp mocks
        with patch("project_paths.RAW_DIR", self.raw_dir):
            results = workflow.run_full_workflow(
                region=region,
                resolution=resolution,
                output_prefix=output_prefix,
                skip_visuals=True,
            )

        # Verify pipeline steps succeeded (expect most/all)
        success_count = sum(results.values())
        self.assertGreaterEqual(
            success_count, 6,
            f"Expected >=6/6 steps to succeed, got {success_count}: {results}"
        )

        # Verify all 7 expected v2 output files exist with correct shape/dtype
        expected_suffixes = [
            "_gravity_residual.tif",
            "_gravity_tdr.tif",
            "_structural_artificiality.tif",
            "_poisson_correlation.tif",
            "_gravity_prior_highres.tif",
            "_fused_belief_reinforced.tif",
            "_dumb_probability_v2.tif",
        ]
        for suffix in expected_suffixes:
            out_path = Path(f"{output_prefix}{suffix}")
            self.assertTrue(
                out_path.exists(),
                f"Expected v2 output missing: {out_path}"
            )
            with rasterio.open(out_path) as src:
                self.assertEqual(src.shape, self.shape)
                self.assertEqual(src.count, 1)
                self.assertEqual(src.dtypes[0], "float32")
                # Basic sanity: not all NaN
                data = src.read(1, masked=True)
                self.assertLess(
                    np.ma.count_masked(data) / data.size,
                    0.99,
                    f"Output {out_path} is almost entirely NaN/masked"
                )

    def tearDown(self):
        """Clean up temp files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()