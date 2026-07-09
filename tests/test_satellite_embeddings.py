import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import satellite_embeddings


def _write_embedding_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(500000.0, 4100000.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": 64,
        "dtype": "float32",
        "crs": "EPSG:32610",
        "transform": transform,
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        for index, band_name in enumerate(satellite_embeddings.EMBEDDING_BANDS, start=1):
            dst.set_band_description(index, band_name)


class SatelliteEmbeddingMathTests(unittest.TestCase):
    def test_similarity_and_change_score_for_unit_vectors(self):
        before = np.zeros((64, 1, 3), dtype=np.float32)
        after = np.zeros((64, 1, 3), dtype=np.float32)

        before[0, :, :] = 1.0
        after[0, 0, 0] = 1.0
        after[1, 0, 1] = 1.0
        after[0, 0, 2] = -1.0

        similarity, change = satellite_embeddings.compute_embedding_similarity(before, after)

        np.testing.assert_allclose(similarity, [[1.0, 0.0, -1.0]])
        np.testing.assert_allclose(change, [[0.0, 1.0, 2.0]])


class SatelliteEmbeddingRasterTests(unittest.TestCase):
    def test_compare_embedding_rasters_writes_real_change_geotiff_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            before = np.zeros((64, 4, 4), dtype=np.float32)
            after = np.zeros((64, 4, 4), dtype=np.float32)
            before[0, :, :] = 1.0
            after[0, :, :] = 1.0
            after[:, 1, 1] = 0.0
            after[1, 1, 1] = 1.0
            after[:, 2, 2] = 0.0
            after[0, 2, 2] = -1.0

            before_path = tmp_path / "before.tif"
            after_path = tmp_path / "after.tif"
            change_path = tmp_path / "change.tif"
            similarity_path = tmp_path / "similarity.tif"
            summary_path = tmp_path / "summary.json"
            _write_embedding_raster(before_path, before)
            _write_embedding_raster(after_path, after)

            summary = satellite_embeddings.compare_embedding_rasters(
                before_path,
                after_path,
                change_out=change_path,
                similarity_out=similarity_path,
                summary_out=summary_path,
                change_thresholds=(0.5, 1.5),
            )

            self.assertEqual(summary["valid_pixels"], 16)
            self.assertEqual(summary["change_threshold_pixel_counts"]["0.5"], 2)
            self.assertEqual(summary["change_threshold_pixel_counts"]["1.5"], 1)
            self.assertTrue(summary_path.exists())

            with rasterio.open(change_path) as src:
                change = src.read(1)
                self.assertEqual(src.descriptions[0], "alphaearth_change_score")
            self.assertAlmostEqual(float(change[0, 0]), 0.0)
            self.assertAlmostEqual(float(change[1, 1]), 1.0)
            self.assertAlmostEqual(float(change[2, 2]), 2.0)

            with rasterio.open(similarity_path) as src:
                self.assertEqual(src.descriptions[0], "alphaearth_dot_product_similarity")

    def test_inspect_embedding_raster_reports_unit_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            raster_path = Path(tmp) / "embedding.tif"
            data = np.zeros((64, 3, 3), dtype=np.float32)
            data[0, :, :] = 1.0
            _write_embedding_raster(raster_path, data)

            summary = satellite_embeddings.inspect_embedding_raster(raster_path)

            self.assertEqual(summary["band_count"], 64)
            self.assertEqual(summary["unit_length_check"]["count"], 9)
            self.assertAlmostEqual(summary["unit_length_check"]["mean"], 1.0)


class EarthEngineExportScriptTests(unittest.TestCase):
    def test_export_script_uses_official_collection_and_all_embedding_bands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_js = Path(tmp) / "export.js"

            satellite_embeddings.write_earth_engine_export_script(
                lat=38.3512,
                lon=-121.986,
                buffer_deg=0.02,
                years=[2023, 2024],
                output_js=out_js,
                description_prefix="vacaville_alphaearth",
            )

            script = out_js.read_text(encoding="utf-8")
            self.assertIn("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", script)
            self.assertIn(".filterDate('2023-01-01', '2024-01-01')", script)
            self.assertIn(".filterDate('2024-01-01', '2025-01-01')", script)
            self.assertIn("'A00'", script)
            self.assertIn("'A63'", script)
            self.assertIn("Export.image.toDrive", script)


if __name__ == "__main__":
    unittest.main()
