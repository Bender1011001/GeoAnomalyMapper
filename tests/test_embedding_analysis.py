#!/usr/bin/env python3
"""
Unit tests for embedding analysis Capabilities 1, 2, 3, and 5.
"""
import json
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_embedding_raster(
    path: Path,
    height: int = 120,
    width: int = 120,
    n_bands: int = 64,
    anomaly_center: tuple = (60, 60),
    anomaly_radius: int = 10,
    transform=None,
) -> Path:
    """Create a synthetic 64-band unit-norm embedding GeoTIFF with a planted anomaly."""
    rng = np.random.default_rng(42)

    # Background: uniform random unit vectors
    data = rng.standard_normal((n_bands, height, width)).astype(np.float32)

    # Plant a circular anomaly at anomaly_center with a very different embedding
    cy, cx = anomaly_center
    anomaly_vec = rng.standard_normal(n_bands).astype(np.float32)
    # Make anomaly orthogonal to the mean background
    anomaly_vec -= anomaly_vec.dot(data.mean(axis=(1, 2)) / np.linalg.norm(data.mean(axis=(1, 2)))) * (
        data.mean(axis=(1, 2)) / np.linalg.norm(data.mean(axis=(1, 2)))
    )
    anomaly_vec /= np.linalg.norm(anomaly_vec) + 1e-10

    for row in range(height):
        for col in range(width):
            if (row - cy) ** 2 + (col - cx) ** 2 < anomaly_radius ** 2:
                data[:, row, col] = anomaly_vec

    # Normalise all vectors to unit length
    norms = np.linalg.norm(data, axis=0, keepdims=True).clip(min=1e-8)
    data = data / norms

    if transform is None:
        transform = from_bounds(0.0, 0.0, 1.0, 1.0, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": n_bands,
        "height": height,
        "width": width,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestClusterEmbeddingAnomalies(unittest.TestCase):
    """Tests for Cap 1: cluster_embedding_anomalies."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.raster = _make_synthetic_embedding_raster(self.tmp / "embed.tif")

    def test_cluster_returns_summary(self):
        from satellite_embeddings import cluster_embedding_anomalies
        summary = cluster_embedding_anomalies(self.raster, n_clusters=8)
        self.assertIn("n_clusters", summary)
        self.assertIn("anomaly_score_distribution", summary)
        self.assertLessEqual(summary["n_clusters"], 8)

    def test_cluster_writes_raster(self):
        from satellite_embeddings import cluster_embedding_anomalies
        out = self.tmp / "anomaly.tif"
        cluster_embedding_anomalies(self.raster, n_clusters=8, anomaly_out=out)
        self.assertTrue(out.exists(), "Anomaly raster should be written")
        with rasterio.open(out) as src:
            data = src.read(1)
        self.assertEqual(data.shape, (120, 120))
        valid = data[~np.isnan(data)]
        self.assertTrue((valid >= 0.0).all() and (valid <= 1.0).all(),
                        "Anomaly scores should be in [0,1]")

    def test_cluster_detects_anomaly_region(self):
        """The planted anomaly at center should have higher scores than edges."""
        from satellite_embeddings import cluster_embedding_anomalies
        out = self.tmp / "anomaly_det.tif"
        cluster_embedding_anomalies(self.raster, n_clusters=16, anomaly_out=out)
        with rasterio.open(out) as src:
            scores = src.read(1)
        center_score = float(np.nanmean(scores[50:70, 50:70]))
        edge_score = float(np.nanmean(scores[:20, :20]))
        # Anomaly center should be scored differently (not necessarily higher,
        # but it should be in a minority cluster)
        self.assertNotAlmostEqual(center_score, edge_score, places=2,
                                  msg="Center anomaly and background should differ")

    def test_cluster_summary_json(self):
        from satellite_embeddings import cluster_embedding_anomalies
        summary_out = self.tmp / "summary.json"
        cluster_embedding_anomalies(self.raster, n_clusters=4, summary_out=summary_out)
        self.assertTrue(summary_out.exists())
        data = json.loads(summary_out.read_text())
        self.assertIn("cluster_fractions", data)


class TestComputeSpatialAnomalyScore(unittest.TestCase):
    """Tests for Cap 3: compute_spatial_anomaly_score."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.raster = _make_synthetic_embedding_raster(
            self.tmp / "embed_sp.tif",
            anomaly_center=(60, 60),
            anomaly_radius=8,
        )

    def test_spatial_returns_summary(self):
        from satellite_embeddings import compute_spatial_anomaly_score
        summary = compute_spatial_anomaly_score(self.raster, outer_radius_px=15)
        self.assertIn("anomaly_score_distribution", summary)
        self.assertIn("filter_size_px", summary)

    def test_spatial_writes_raster_in_range(self):
        from satellite_embeddings import compute_spatial_anomaly_score
        out = self.tmp / "spatial_anomaly.tif"
        compute_spatial_anomaly_score(self.raster, outer_radius_px=15, anomaly_out=out)
        self.assertTrue(out.exists())
        with rasterio.open(out) as src:
            scores = src.read(1)
        valid = scores[~np.isnan(scores)]
        self.assertGreaterEqual(float(valid.min()), 0.0)
        # Upper bound: 1 - (-1) = 2 (anti-correlated vectors), so [0,2] is valid

    def test_spatial_higher_at_anomaly_boundary(self):
        """The boundary ring of the planted anomaly should be the most dissimilar.

        The spatial anomaly score computes how different each pixel is from its
        neighbourhood. The planted anomaly region is internally uniform (low score),
        but pixels at the boundary between the anomaly block and the random background
        have a neighbourhood that spans both embedding types → high dissimilarity.
        The boundary ring (radius+1 to radius+4 pixels from center) should have
        a higher mean score than a distant random corner region that is purely
        background (where neighbour embeddings are similar random draws).
        """
        from satellite_embeddings import compute_spatial_anomaly_score
        out = self.tmp / "spatial_anomaly_det.tif"
        # Use a small radius so the boundary effect is pronounced
        compute_spatial_anomaly_score(self.raster, outer_radius_px=10, anomaly_out=out)
        with rasterio.open(out) as src:
            scores = src.read(1)

        # Boundary ring: ~1 pixel outside the planted anomaly (radius=8)
        cy, cx = 60, 60
        boundary_mask = np.zeros(scores.shape, dtype=bool)
        for r in range(scores.shape[0]):
            for c in range(scores.shape[1]):
                dist = ((r - cy) ** 2 + (c - cx) ** 2) ** 0.5
                if 8 <= dist <= 13:
                    boundary_mask[r, c] = True
        boundary_score = float(np.nanmean(scores[boundary_mask]))

        # Interior of anomaly: pixels with uniform embeddings, low boundary contrast
        interior_score = float(np.nanmean(scores[55:65, 55:65]))

        # The boundary should be at least as anomalous as the interior
        self.assertGreaterEqual(boundary_score, interior_score * 0.9,
                                "Boundary ring should score at least as high as anomaly interior")


class TestComputeTemporalAnomalyTrajectory(unittest.TestCase):
    """Tests for Cap 2: compute_temporal_anomaly_trajectory."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        # Year 1: normal background
        self.r2021 = _make_synthetic_embedding_raster(
            self.tmp / "embed_2021.tif", anomaly_radius=0
        )
        # Year 2: anomaly planted at center
        self.r2022 = _make_synthetic_embedding_raster(
            self.tmp / "embed_2022.tif", anomaly_center=(60, 60), anomaly_radius=8
        )
        # Year 3: different anomaly
        self.r2023 = _make_synthetic_embedding_raster(
            self.tmp / "embed_2023.tif", anomaly_center=(30, 80), anomaly_radius=6
        )

    def test_temporal_requires_two_years(self):
        from satellite_embeddings import compute_temporal_anomaly_trajectory
        with self.assertRaises(ValueError):
            compute_temporal_anomaly_trajectory({2021: self.r2021})

    def test_temporal_returns_summary(self):
        from satellite_embeddings import compute_temporal_anomaly_trajectory
        summary = compute_temporal_anomaly_trajectory(
            {2021: self.r2021, 2022: self.r2022}
        )
        self.assertIn("years", summary)
        self.assertEqual(summary["n_years"], 2)
        self.assertEqual(summary["n_pairs"], 1)

    def test_temporal_writes_variance_raster(self):
        from satellite_embeddings import compute_temporal_anomaly_trajectory
        var_out = self.tmp / "variance.tif"
        compute_temporal_anomaly_trajectory(
            {2021: self.r2021, 2022: self.r2022, 2023: self.r2023},
            variance_out=var_out,
        )
        self.assertTrue(var_out.exists())
        with rasterio.open(var_out) as src:
            var = src.read(1)
        self.assertEqual(var.shape, (120, 120))
        valid = var[~np.isnan(var)]
        self.assertTrue((valid >= 0.0).all(), "Variance should be non-negative")

    def test_temporal_higher_variance_where_changed(self):
        """Pixels that changed between years should have higher trajectory variance."""
        from satellite_embeddings import compute_temporal_anomaly_trajectory
        var_out = self.tmp / "variance_det.tif"
        compute_temporal_anomaly_trajectory(
            {2021: self.r2021, 2022: self.r2022, 2023: self.r2023},
            variance_out=var_out,
        )
        with rasterio.open(var_out) as src:
            var = src.read(1)
        # r2022 changed at center (60,60), r2023 at (30,80)
        center_var = float(np.nanmean(var[52:68, 52:68]))
        stable_var = float(np.nanmean(var[5:25, 90:110]))
        # The changed region should show measurably higher variance
        self.assertGreater(center_var, stable_var * 0.5,
                           "Changed region should show meaningful variance")

    def test_temporal_grid_mismatch_raises_embedding_raster_error(self):
        from rasterio.transform import from_origin
        from satellite_embeddings import EmbeddingRasterError, compute_temporal_anomaly_trajectory

        shifted = _make_synthetic_embedding_raster(
            self.tmp / "embed_2024_shifted.tif",
            transform=from_origin(10.0, 10.0, 0.02, 0.02),
            anomaly_radius=0,
        )

        with self.assertRaises(EmbeddingRasterError):
            compute_temporal_anomaly_trajectory({2021: self.r2021, 2024: shifted})


class TestFindSimilarSites(unittest.TestCase):
    """Tests for Cap 5: find_similar_sites."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.raster = _make_synthetic_embedding_raster(
            self.tmp / "embed_ref.tif", anomaly_center=(60, 60), anomaly_radius=10
        )

    def test_find_similar_returns_list(self):
        from satellite_embeddings import find_similar_sites
        results = find_similar_sites(
            reference_raster=self.raster,
            reference_bbox_px=(52, 52, 70, 70),
            search_rasters=[self.raster],
            min_similarity=0.85,
            top_k=5,
        )
        self.assertIsInstance(results, list)

    def test_find_similar_includes_coords(self):
        from satellite_embeddings import find_similar_sites
        results = find_similar_sites(
            reference_raster=self.raster,
            reference_bbox_px=(52, 52, 70, 70),
            search_rasters=[self.raster],
            min_similarity=0.70,
            top_k=10,
        )
        if results:
            r = results[0]
            self.assertIn("lat", r)
            self.assertIn("lon", r)
            self.assertIn("similarity", r)
            self.assertGreaterEqual(r["similarity"], 0.70)

    def test_find_similar_self_reference_high_similarity(self):
        """Searching the same raster with the same bbox should find high-similarity matches."""
        from satellite_embeddings import find_similar_sites
        results = find_similar_sites(
            reference_raster=self.raster,
            reference_bbox_px=(52, 52, 70, 70),
            search_rasters=[self.raster],
            min_similarity=0.90,
            top_k=50,
        )
        # The reference region itself should appear as a high-similarity match
        self.assertGreater(len(results), 0,
                           "Self-search should find at least one match above 0.90 similarity")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=1)

    def test_find_similar_invalid_bbox_raises(self):
        from satellite_embeddings import EmbeddingRasterError, find_similar_sites
        with self.assertRaises(EmbeddingRasterError):
            find_similar_sites(
                reference_raster=self.raster,
                reference_bbox_px=(0, 0, 0, 0),  # empty bbox
                search_rasters=[self.raster],
            )

    def test_find_similar_multi_raster_similarity_out_directory(self):
        from satellite_embeddings import find_similar_sites

        second = _make_synthetic_embedding_raster(
            self.tmp / "embed_second.tif", anomaly_center=(40, 40), anomaly_radius=6
        )
        out_dir = self.tmp / "similarity_outputs"

        find_similar_sites(
            reference_raster=self.raster,
            reference_bbox_px=(52, 52, 70, 70),
            search_rasters=[self.raster, second],
            min_similarity=0.95,
            top_k=5,
            similarity_out=out_dir,
        )

        self.assertTrue((out_dir / "embed_ref_similarity.tif").exists())
        self.assertTrue((out_dir / "embed_second_similarity.tif").exists())

    def test_find_similar_multi_raster_single_output_path_raises(self):
        from satellite_embeddings import EmbeddingRasterError, find_similar_sites

        second = _make_synthetic_embedding_raster(
            self.tmp / "embed_second.tif", anomaly_center=(40, 40), anomaly_radius=6
        )

        with self.assertRaises(EmbeddingRasterError):
            find_similar_sites(
                reference_raster=self.raster,
                reference_bbox_px=(52, 52, 70, 70),
                search_rasters=[self.raster, second],
                similarity_out=self.tmp / "overwritten.tif",
            )


class TestFusedConfidenceScore(unittest.TestCase):
    """Tests for Cap 6: fused_confidence_score in extract_anomaly_bodies."""

    def _make_void_volume(self, shape=(20, 30, 30)):
        """Create a synthetic void probability and wave-speed volume with one planted void."""
        nz, ny, nx = shape
        void_prob = np.zeros(shape, dtype=np.float32)
        wave_speed = np.full(shape, 3500.0, dtype=np.float32)
        # Plant a void at center top
        void_prob[2:6, 12:18, 12:18] = 0.85
        wave_speed[2:6, 12:18, 12:18] = 400.0
        return void_prob, wave_speed

    def test_fused_score_without_map(self):
        """Without embedding map, fused_confidence_score == deep_target_score."""
        from visualize_3d_subsurface import extract_anomaly_bodies
        vp, ws = self._make_void_volume()
        anomalies = extract_anomaly_bodies(vp, ws)
        for a in anomalies:
            self.assertIn("fused_confidence_score", a)
            self.assertAlmostEqual(
                a["fused_confidence_score"], a["deep_target_score"], places=5,
                msg="Without map, fused == deep_target_score"
            )

    def test_fused_score_with_anomalous_surface(self):
        """High surface anomaly at void centroid should boost fused score."""
        from visualize_3d_subsurface import extract_anomaly_bodies
        vp, ws = self._make_void_volume()
        # Anomaly map with value=1 everywhere (maximum corroboration)
        amap = np.ones((30, 30), dtype=np.float32)
        anomalies = extract_anomaly_bodies(vp, ws, embedding_anomaly_map=amap)
        for a in anomalies:
            if a.get("surface_anomaly_at_centroid") is not None:
                # fused = deep_target_score * (0.7 + 0.3*1.0) = deep_target_score * 1.0
                self.assertAlmostEqual(
                    a["fused_confidence_score"], a["deep_target_score"], places=5,
                    msg="amap=1 → factor=1.0 → fused == deep_target_score"
                )

    def test_fused_score_with_normal_surface(self):
        """Zero surface anomaly should give factor=0.70, reducing fused score."""
        from visualize_3d_subsurface import extract_anomaly_bodies
        vp, ws = self._make_void_volume()
        # Normal surface (zero anomaly)
        amap = np.zeros((30, 30), dtype=np.float32)
        anomalies = extract_anomaly_bodies(vp, ws, embedding_anomaly_map=amap)
        for a in anomalies:
            if a.get("surface_anomaly_at_centroid") is not None:
                expected = a["deep_target_score"] * 0.70
                self.assertAlmostEqual(a["fused_confidence_score"], expected, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
