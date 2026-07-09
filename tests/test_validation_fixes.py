import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sar_vibrometry
import visualize_3d_subsurface
from json_utils import dumps_strict_json


class SyntheticVibrationFixtureTests(unittest.TestCase):
    def test_synthetic_ground_truth_is_bounded_deterministic_and_metadata_rich(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "synthetic_first.npy"
            second_path = Path(tmp) / "synthetic_second.npy"

            sar_vibrometry.generate_synthetic_vibration_test(
                str(first_path), grid_size=32, num_anomalies=3, noise_level=0.0
            )
            sar_vibrometry.generate_synthetic_vibration_test(
                str(second_path), grid_size=32, num_anomalies=3, noise_level=0.0
            )

            first_mask = np.load(str(first_path).replace(".npy", "_ground_truth.npy"))
            second_mask = np.load(str(second_path).replace(".npy", "_ground_truth.npy"))

            self.assertEqual(first_mask.shape, (32, 32))
            self.assertTrue(np.array_equal(first_mask, second_mask))
            self.assertGreater(float(first_mask.max()), 0.0)
            self.assertLessEqual(float(first_mask.max()), 1.0)
            self.assertLess(float((first_mask > 0.05).mean()), 0.35)

            metadata_path = str(first_path).replace(".npy", "_ground_truth_metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["num_anomalies"], 3)
            self.assertIn("active_coverage_fraction", metadata)
            self.assertIn("core_coverage_fraction", metadata)
            self.assertEqual(metadata["synthetic_doppler_model"], "sub_aperture_phase_evolution")
            self.assertEqual(metadata["num_sub_apertures"], 5)
            self.assertIn("phase_step_max_rad", metadata)
            self.assertLess(metadata["active_coverage_fraction"], 0.35)
            for anomaly in metadata["anomalies"]:
                self.assertIn("center_px", anomaly)
                self.assertIn("radius_px", anomaly)
                self.assertIn("depth_m", anomaly)
                self.assertIn("coverage_fraction", anomaly)


class PinnLossHistoryObservabilityTests(unittest.TestCase):
    def test_loss_history_append_records_surface_prior_activity_and_zero_values(self):
        import pinn_vibro_inversion

        cfg = {
            "physics_weight": 1.0,
            "data_weight": 20.0,
            "sparsity_weight": 0.01,
            "regularization_weight": 0.1,
            "deep_prior_weight": 0.1,
            "surface_prior_weight": 0.25,
        }
        history = pinn_vibro_inversion._initialize_loss_history(
            cfg,
            surface_prior_map_supplied=True,
            surface_prior_enabled=True,
        )

        pinn_vibro_inversion._append_loss_history(
            history,
            epoch=0,
            raw_losses={
                "total": 1.0,
                "physics": 0.0,
                "data": 0.5,
                "sparse": 0.0,
                "reg": 0.1,
                "deep": 0.0,
                "surface_prior": 0.0,
                "sommerfeld": 0.0,
            },
            weights={
                "physics": 1.0,
                "data": 20.0,
                "sparse": 0.01,
                "reg": 0.1,
                "deep": 0.1,
                "surface_prior": 0.25,
                "sommerfeld": 1.0,
            },
            active={
                "physics": True,
                "data": True,
                "sparse": True,
                "reg": True,
                "deep": True,
                "surface_prior": True,
                "sommerfeld": True,
            },
        )

        self.assertEqual(history["schema_version"], 2)
        self.assertEqual(history["epoch"], [0])
        self.assertEqual(history["surface_prior"], [0.0])
        self.assertEqual(history["deep"], [0.0])
        self.assertEqual(history["active"]["surface_prior"], [True])
        self.assertEqual(history["active"]["deep"], [True])
        self.assertEqual(history["weights"]["surface_prior"], [0.25])
        self.assertEqual(history["weighted"]["surface_prior"], [0.0])
        self.assertTrue(history["configured"]["surface_prior"])
        self.assertTrue(history["metadata"]["surface_prior_enabled"])

    def test_loss_history_append_distinguishes_inactive_surface_prior(self):
        import pinn_vibro_inversion

        cfg = {
            "physics_weight": 1.0,
            "data_weight": 20.0,
            "sparsity_weight": 0.01,
            "regularization_weight": 0.1,
            "deep_prior_weight": 0.1,
            "surface_prior_weight": 0.25,
        }
        history = pinn_vibro_inversion._initialize_loss_history(
            cfg,
            surface_prior_map_supplied=False,
            surface_prior_enabled=False,
        )

        pinn_vibro_inversion._append_loss_history(
            history,
            epoch=0,
            raw_losses={"total": 0.2, "surface_prior": 0.0, "deep": 0.0, "sommerfeld": 0.0},
            weights={"surface_prior": 0.25, "deep": 0.1, "sommerfeld": 1.0},
            active={"surface_prior": False, "deep": True, "sommerfeld": True},
        )

        self.assertTrue(history["configured"]["surface_prior"])
        self.assertFalse(history["metadata"]["surface_prior_map_supplied"])
        self.assertFalse(history["active"]["surface_prior"][0])
        self.assertTrue(history["active"]["deep"][0])
        self.assertEqual(history["surface_prior"][0], 0.0)
        self.assertEqual(history["deep"][0], 0.0)


class VisualizerAnomalyReportingTests(unittest.TestCase):
    def test_thin_deep_body_survives_default_extraction(self):
        wave_speed = np.full((16, 10, 10), 3500.0, dtype=np.float32)
        void_probability = np.zeros((16, 10, 10), dtype=np.float32)

        void_probability[10:15, 5, 5] = 0.9
        wave_speed[void_probability > 0.0] = 450.0

        anomalies = visualize_3d_subsurface.extract_anomaly_bodies(
            void_probability,
            wave_speed,
            config={
                "void_threshold": 0.35,
                "min_anomaly_voxels": 1,
                "max_depth_m": 160.0,
                "domain_width_m": 100.0,
            },
        )

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["voxel_count"], 5)
        self.assertGreater(anomalies[0]["depth_m"], 100.0)
        self.assertEqual(anomalies[0]["deep_target_rank"], 1)

    def test_centroid_depth_uses_voxel_centers(self):
        wave_speed = np.full((4, 4, 4), 3500.0, dtype=np.float32)
        void_probability = np.zeros((4, 4, 4), dtype=np.float32)
        void_probability[0, 1, 1] = 0.9

        anomalies = visualize_3d_subsurface.extract_anomaly_bodies(
            void_probability,
            wave_speed,
            config={
                "void_threshold": 0.35,
                "min_anomaly_voxels": 1,
                "max_depth_m": 40.0,
                "domain_width_m": 40.0,
            },
        )

        self.assertEqual(len(anomalies), 1)
        self.assertAlmostEqual(anomalies[0]["depth_m"], 5.0)

    def test_anomaly_extraction_ranks_deeper_targets_when_evidence_matches(self):
        wave_speed = np.full((12, 8, 8), 3500.0, dtype=np.float32)
        void_probability = np.zeros((12, 8, 8), dtype=np.float32)

        void_probability[1:4, 1:4, 1:4] = 0.8
        void_probability[8:11, 4:7, 4:7] = 0.8
        wave_speed[void_probability > 0.0] = 500.0

        anomalies = visualize_3d_subsurface.extract_anomaly_bodies(
            void_probability,
            wave_speed,
            config={
                "void_threshold": 0.35,
                "min_anomaly_voxels": 1,
                "max_depth_m": 120.0,
                "domain_width_m": 80.0,
            },
        )

        self.assertEqual(len(anomalies), 2)
        self.assertEqual([a["deep_target_rank"] for a in anomalies], [1, 2])
        self.assertGreater(anomalies[0]["depth_m"], anomalies[1]["depth_m"])
        self.assertGreater(anomalies[0]["deep_target_score"], anomalies[1]["deep_target_score"])
        self.assertIn("void_evidence_score", anomalies[0])

    def test_pipeline_returns_explicit_anomaly_list_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wave_speed = np.full((8, 8, 8), 3500.0, dtype=np.float32)
            void_probability = np.zeros((8, 8, 8), dtype=np.float32)
            void_probability[2:5, 2:5, 2:5] = 0.8

            wave_speed_path = tmp_path / "wave_speed.npy"
            void_probability_path = tmp_path / "void_probability.npy"
            np.save(wave_speed_path, wave_speed)
            np.save(void_probability_path, void_probability)

            with mock.patch.object(
                visualize_3d_subsurface,
                "render_3d_subsurface",
                return_value={"anomaly_report": str(tmp_path / "anomaly_report.txt")},
            ):
                outputs = visualize_3d_subsurface.run_visualization_pipeline(
                    str(wave_speed_path),
                    output_dir=str(tmp_path),
                    void_probability_path=str(void_probability_path),
                    config={
                        "void_threshold": 0.35,
                        "min_anomaly_voxels": 1,
                        "max_depth_m": 80.0,
                        "domain_width_m": 80.0,
                    },
                    interactive=False,
                )

            self.assertIn("anomaly_list", outputs)
            self.assertIn("anomaly_count", outputs)
            self.assertEqual(outputs["anomaly_count"], len(outputs["anomaly_list"]))
            self.assertEqual(outputs["anomaly_count"], 1)
            self.assertIn("anomalies_csv", outputs)

    def test_pipeline_removes_stale_positive_artifacts_on_no_anomaly_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wave_speed = np.full((4, 4, 4), 3500.0, dtype=np.float32)
            void_probability = np.zeros((4, 4, 4), dtype=np.float32)

            wave_speed_path = tmp_path / "wave_speed.npy"
            void_probability_path = tmp_path / "void_probability.npy"
            np.save(wave_speed_path, wave_speed)
            np.save(void_probability_path, void_probability)

            stale_csv = tmp_path / "detected_anomalies.csv"
            stale_stl = tmp_path / "void_surface.stl"
            stale_csv.write_text("id,mean_void_probability\n1,0.9\n", encoding="utf-8")
            stale_stl.write_text("solid stale\nendsolid stale\n", encoding="utf-8")

            with mock.patch.object(
                visualize_3d_subsurface,
                "render_3d_subsurface",
                return_value={
                    "anomaly_report": str(tmp_path / "anomaly_report.txt"),
                    "void_contour_points": 0,
                    "void_contour_cells": 0,
                },
            ):
                outputs = visualize_3d_subsurface.run_visualization_pipeline(
                    str(wave_speed_path),
                    output_dir=str(tmp_path),
                    void_probability_path=str(void_probability_path),
                    config={
                        "void_threshold": 0.35,
                        "min_anomaly_voxels": 1,
                        "max_depth_m": 40.0,
                        "domain_width_m": 40.0,
                    },
                    interactive=False,
                )

            self.assertEqual(outputs["anomaly_count"], 0)
            self.assertFalse(stale_csv.exists())
            self.assertFalse(stale_stl.exists())
            self.assertIn(str(stale_csv), outputs["stale_artifacts_removed"])
            self.assertIn(str(stale_stl), outputs["stale_artifacts_removed"])

    def test_audit_manifest_records_threshold_counts_and_basic_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wave_speed = np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 3000.0
            void_probability = np.array(
                [0.0, 0.1, 0.2, 0.35, 0.36, 0.4, 0.5, 0.01],
                dtype=np.float32,
            ).reshape(2, 2, 2)

            wave_speed_path = tmp_path / "wave_speed.npy"
            void_probability_path = tmp_path / "void_probability.npy"
            np.save(wave_speed_path, wave_speed)
            np.save(void_probability_path, void_probability)

            with mock.patch.object(
                visualize_3d_subsurface,
                "render_3d_subsurface",
                return_value={"void_contour_points": 0, "void_contour_cells": 0},
            ):
                outputs = visualize_3d_subsurface.run_visualization_pipeline(
                    str(wave_speed_path),
                    output_dir=str(tmp_path),
                    void_probability_path=str(void_probability_path),
                    config={
                        "void_threshold": 0.35,
                        "min_anomaly_voxels": 99,
                        "max_depth_m": 20.0,
                        "domain_width_m": 20.0,
                    },
                    interactive=False,
                )

            manifest_path = Path(outputs["audit_manifest"])
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            void_stats = manifest["arrays"]["void_probability"]
            self.assertEqual(void_stats["shape"], [2, 2, 2])
            self.assertEqual(void_stats["finite_count"], 8)
            self.assertEqual(void_stats["threshold_counts"]["threshold"], 0.35)
            self.assertEqual(void_stats["threshold_counts"]["voxels_above_threshold"], 3)
            self.assertIn("quantiles", void_stats)
            self.assertAlmostEqual(void_stats["quantiles"]["p50"], 0.275, places=6)
            self.assertAlmostEqual(void_stats["max"], 0.5, places=6)
            self.assertEqual(manifest["counts"]["anomaly_count"], 0)
            self.assertEqual(manifest["counts"]["voxels_crossing_void_threshold"], 3)
            self.assertTrue(manifest["diagnostics"]["threshold_crossing"]["any_voxel_crosses_threshold"])
            self.assertFalse(manifest["diagnostics"]["threshold_crossing"]["any_component_meets_min_voxels"])
            self.assertIn("sha256", manifest["inputs"]["wave_speed_volume"])

    def test_compute_void_probability_low_speed_maps_high(self):
        wave_speed = np.array([3500.0, 2450.0, 450.0], dtype=np.float32)

        void_probability = visualize_3d_subsurface.compute_void_probability_from_wave_speed(
            wave_speed,
            background_wave_speed=3500.0,
        )

        self.assertLess(float(void_probability[0]), 0.01)
        self.assertAlmostEqual(float(void_probability[1]), 0.5, places=6)
        self.assertGreater(float(void_probability[2]), 0.99)


class ControlledVoidRegressionTests(unittest.TestCase):
    def test_strict_json_helper_nulls_nonfinite_values(self):
        payload = {
            "nan_value": float("nan"),
            "positive_inf": float("inf"),
            "negative_inf": float("-inf"),
            "array": np.array([1.0, np.nan, np.inf], dtype=np.float32),
        }

        encoded = dumps_strict_json(payload, sort_keys=True)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        decoded = json.loads(encoded)
        self.assertIsNone(decoded["nan_value"])
        self.assertIsNone(decoded["positive_inf"])
        self.assertIsNone(decoded["negative_inf"])
        self.assertEqual(decoded["array"], [1.0, None, None])

    def test_absent_surface_anomaly_serializes_as_null(self):
        wave_speed = np.full((4, 4, 4), 3500.0, dtype=np.float32)
        void_probability = np.zeros((4, 4, 4), dtype=np.float32)
        void_probability[1:3, 1:3, 1:3] = 0.9

        anomalies = visualize_3d_subsurface.extract_anomaly_bodies(
            void_probability,
            wave_speed,
            config={
                "void_threshold": 0.35,
                "min_anomaly_voxels": 1,
                "max_depth_m": 40.0,
                "domain_width_m": 40.0,
            },
        )

        self.assertGreaterEqual(len(anomalies), 1)
        self.assertIsNone(anomalies[0]["surface_anomaly_at_centroid"])
        encoded = dumps_strict_json({"anomalies": anomalies}, indent=2)
        self.assertNotIn("NaN", encoded)
        decoded = json.loads(encoded)
        self.assertIsNone(decoded["anomalies"][0]["surface_anomaly_at_centroid"])

    def test_controlled_void_regression_detects_known_injected_void(self):
        import controlled_void_regression

        with tempfile.TemporaryDirectory() as tmp:
            summary = controlled_void_regression.run_controlled_void_regression(
                output_dir=Path(tmp) / "control",
                clean=True,
                enable_pyvista=False,
            )

            self.assertTrue(summary["controlled_void_detected"])
            self.assertGreaterEqual(summary["anomaly_count"], 1)
            observed = summary["observed"]
            expected = summary["expected"]
            self.assertLessEqual(
                observed["centroid_error_m"],
                expected["max_allowed_centroid_error_m"],
            )
            self.assertGreaterEqual(
                observed["best_anomaly"]["voxel_count"],
                expected["min_anomaly_voxels"],
            )
            self.assertTrue(observed["threshold_diagnostics"]["any_component_meets_min_voxels"])

            for key in [
                "wave_speed_volume",
                "computed_void_probability_volume",
                "anomaly_report",
                "anomaly_catalog",
                "audit_manifest",
                "cross_sections",
                "summary",
            ]:
                self.assertTrue(Path(summary["artifacts"][key]).exists(), key)

            summary_path = Path(summary["artifacts"]["summary"])
            raw_summary = summary_path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw_summary)
            self.assertNotIn("Infinity", raw_summary)
            decoded_summary = json.loads(raw_summary)
            self.assertIsNone(
                decoded_summary["observed"]["best_anomaly"].get("surface_anomaly_at_centroid")
            )


class CliDefaultConsistencyTests(unittest.TestCase):
    @staticmethod
    def _arg_default_expr(source_path, option_name):
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if node.args[0].value != option_name:
                continue
            for keyword in node.keywords:
                if keyword.arg == "default":
                    return ast.unparse(keyword.value)
        raise AssertionError(f"No default found for {option_name}")

    def test_visualizer_cli_threshold_uses_internal_default(self):
        self.assertEqual(visualize_3d_subsurface.DEFAULT_VIZ_CONFIG["void_threshold"], 0.35)
        default_expr = self._arg_default_expr("visualize_3d_subsurface.py", "--threshold")
        self.assertEqual(default_expr, "DEFAULT_VIZ_CONFIG['void_threshold']")

    def test_pinn_cli_depth_uses_internal_default(self):
        source = Path("pinn_vibro_inversion.py").read_text(encoding="utf-8")
        self.assertIn('"max_depth_m": 1000.0', source)
        default_expr = self._arg_default_expr("pinn_vibro_inversion.py", "--depth")
        self.assertEqual(default_expr, "DEFAULT_INVERSION_CONFIG['max_depth_m']")

    def test_orchestrator_exposes_real_deep_profile(self):
        source = Path("run_biondi_exploration.py").read_text(encoding="utf-8")
        self.assertIn('"deep": {', source)
        self.assertIn('"max_depth_m": 5000.0', source)
        self.assertIn('"excitation_frequency_hz": 0.25', source)
        self.assertIn('choices=["quick", "standard", "high", "deep"]', source)
        self.assertIn('"depth_slices": select_depth_slices(adaptive_max_depth)', source)


class EmbeddingPriorOrchestratorTests(unittest.TestCase):
    def test_orchestrator_passes_surface_prior_weight_only_when_enabled(self):
        import run_biondi_exploration

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "prior.tif"
            with mock.patch.object(run_biondi_exploration, "EXPLORE_DIR", tmp_path), \
                 mock.patch.object(run_biondi_exploration.slc_data_fetcher, "_build_search_bbox", return_value=(0, 0, 1, 1)), \
                 mock.patch.object(run_biondi_exploration.sar_vibrometry, "generate_synthetic_vibration_test", side_effect=lambda output_path, **_: np.save(output_path, np.ones((4, 4), dtype=np.float32)) or output_path), \
                 mock.patch.object(run_biondi_exploration.sar_vibrometry, "run_vibrometry_pipeline", return_value={"vibration_amplitude_npy": str(tmp_path / "vib.npy")}), \
                 mock.patch.object(run_biondi_exploration.visualize_3d_subsurface, "run_visualization_pipeline", return_value={"anomaly_count": 0}), \
                 mock.patch("rasterio.open") as raster_open, \
                 mock.patch.object(run_biondi_exploration.pinn_vibro_inversion, "train_vibro_pinn", return_value={"wave_speed_volume": str(tmp_path / "ws.npy"), "void_probability_volume": str(tmp_path / "vp.npy")}) as train_mock:
                np.save(tmp_path / "vib.npy", np.ones((4, 4), dtype=np.float32))
                raster_path.write_text("placeholder", encoding="utf-8")
                raster_open.return_value.__enter__.return_value.read.return_value = np.ones((4, 4), dtype=np.float32)

                target = {"name": "Prior Target", "lat": 0.0, "lon": 0.0, "buffer_deg": 0.1, "description": "test", "expected_depth_m": 100}
                result = run_biondi_exploration.execute_biondi_pipeline_for_target(
                    target,
                    credentials={},
                    use_synthetic_fallback=True,
                    resolution="quick",
                    use_embeddings=True,
                    embedding_rasters=[raster_path],
                    surface_prior_weight=0.25,
                )

                self.assertTrue(result["embedding_prior_used"])
                self.assertEqual(result["anomaly_count"], 0)
                self.assertEqual(result["anomalies_detected"], 0)
                self.assertEqual(train_mock.call_args.kwargs["config"]["surface_prior_weight"], 0.25)

    def test_orchestrator_does_not_claim_prior_when_weight_zero(self):
        import run_biondi_exploration

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raster_path = tmp_path / "prior.tif"
            with mock.patch.object(run_biondi_exploration, "EXPLORE_DIR", tmp_path), \
                 mock.patch.object(run_biondi_exploration.slc_data_fetcher, "_build_search_bbox", return_value=(0, 0, 1, 1)), \
                 mock.patch.object(run_biondi_exploration.sar_vibrometry, "generate_synthetic_vibration_test", side_effect=lambda output_path, **_: np.save(output_path, np.ones((4, 4), dtype=np.float32)) or output_path), \
                 mock.patch.object(run_biondi_exploration.sar_vibrometry, "run_vibrometry_pipeline", return_value={"vibration_amplitude_npy": str(tmp_path / "vib.npy")}), \
                 mock.patch.object(run_biondi_exploration.visualize_3d_subsurface, "run_visualization_pipeline", return_value={"anomaly_count": 0}), \
                 mock.patch("rasterio.open") as raster_open, \
                 mock.patch.object(run_biondi_exploration.pinn_vibro_inversion, "train_vibro_pinn", return_value={"wave_speed_volume": str(tmp_path / "ws.npy"), "void_probability_volume": str(tmp_path / "vp.npy")}) as train_mock:
                np.save(tmp_path / "vib.npy", np.ones((4, 4), dtype=np.float32))
                raster_path.write_text("placeholder", encoding="utf-8")
                raster_open.return_value.__enter__.return_value.read.return_value = np.ones((4, 4), dtype=np.float32)

                target = {"name": "Prior Target", "lat": 0.0, "lon": 0.0, "buffer_deg": 0.1, "description": "test", "expected_depth_m": 100}
                result = run_biondi_exploration.execute_biondi_pipeline_for_target(
                    target,
                    credentials={},
                    use_synthetic_fallback=True,
                    resolution="quick",
                    use_embeddings=True,
                    embedding_rasters=[raster_path],
                    surface_prior_weight=0.0,
                )

                self.assertFalse(result["embedding_prior_used"])
                self.assertEqual(result["anomaly_count"], 0)
                self.assertEqual(result["anomalies_detected"], 0)
                self.assertEqual(train_mock.call_args.kwargs["config"]["surface_prior_weight"], 0.0)

    def test_synthetic_positive_control_diagnostics_classifies_pinn_collapse(self):
        import run_biondi_exploration

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            slc_path = tmp_path / "synthetic.npy"
            gt = np.zeros((4, 4), dtype=np.float32)
            gt[1:3, 1:3] = 1.0
            np.save(str(slc_path).replace(".npy", "_ground_truth.npy"), gt)

            vib = np.ones((4, 4), dtype=np.float32)
            vib[1:3, 1:3] = 2.0
            vib_path = tmp_path / "vib.npy"
            np.save(vib_path, vib)

            wave_speed_path = tmp_path / "wave_speed.npy"
            np.save(wave_speed_path, np.full((2, 4, 4), 3300.0, dtype=np.float32))

            void_prob_path = tmp_path / "void_prob.npy"
            np.save(void_prob_path, np.full((2, 4, 4), 0.01, dtype=np.float32))

            diag_path = run_biondi_exploration.write_synthetic_positive_control_diagnostics(
                slc_path=str(slc_path),
                vib_amp_path=str(vib_path),
                wave_speed_path=str(wave_speed_path),
                void_prob_path=str(void_prob_path),
                output_dir=tmp_path,
                anomaly_count=0,
                void_threshold=0.35,
                min_anomaly_voxels=3,
                background_wave_speed=3500.0,
            )

            self.assertIsNotNone(diag_path)
            with open(diag_path, "r", encoding="utf-8") as f:
                diagnostics = json.load(f)

            self.assertEqual(diagnostics["primary_blocker"], "pinn_collapse_no_low_velocity_volume")
            self.assertEqual(diagnostics["anomaly_count"], diagnostics["anomalies_detected"])
            self.assertGreater(diagnostics["vibrometry"]["active_to_inactive_ratio"], 1.10)
            self.assertEqual(diagnostics["void_probability"]["voxels_above_threshold"], 0)


class ResultSchemaCompatibilityTests(unittest.TestCase):
    def test_scan_consumers_accept_current_and_legacy_anomaly_keys(self):
        import run_full_scan
        import run_full_scan_remote
        import run_real_data

        self.assertEqual(run_full_scan._get_anomaly_count({"anomaly_count": "3.0"}), 3)
        self.assertEqual(run_full_scan._get_anomaly_count({"anomalies_detected": "4"}), 4)
        self.assertEqual(
            run_full_scan._get_anomaly_count({"anomaly_count": "not numeric", "anomalies_detected": "5"}),
            5,
        )
        self.assertEqual(run_full_scan_remote._get_anomaly_count({"anomaly_count": "6.0"}), 6)
        self.assertEqual(run_full_scan_remote._get_anomaly_count({"anomalies_detected": "7"}), 7)
        remote_record = {"anomaly_count": "not numeric", "anomalies_detected": "8"}
        self.assertEqual(run_full_scan_remote._normalize_result_anomaly_counts(remote_record), 8)
        self.assertEqual(remote_record["anomaly_count"], 8)
        self.assertEqual(remote_record["anomalies_detected"], 8)
        self.assertEqual(run_real_data._get_anomaly_count({"anomalies_detected": 2}), 2)

    def test_full_scan_formats_visualizer_top_target_fields(self):
        import run_full_scan

        summary = run_full_scan._format_top_target_summary({
            "deep_target_rank": 1,
            "depth_m": "125.5",
            "deep_target_score": "0.8123",
            "shape_classification": "CYLINDRICAL",
        })

        self.assertIn("Rank 1", summary)
        self.assertIn("depth=126m", summary)
        self.assertIn("score=0.812", summary)
        self.assertIn("shape=CYLINDRICAL", summary)


class ParserHardeningTests(unittest.TestCase):
    def test_parse_results_handles_missing_and_nonnumeric_fields(self):
        import parse_results

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps({
                "targets_completed": "not numeric",
                "results": [{
                    "target_name": "Test Target",
                    "status": "success",
                    "anomalies_detected": "not numeric",
                    "top_deep_targets": [{
                        "deep_target_rank": 1,
                        "depth_m": None,
                        "volume_m3": "bad",
                        "fused_confidence_score": "0.7",
                        "artificiality_score": "bad",
                    }],
                }],
            }), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = parse_results.main([str(path)])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Total targets scanned so far: 1", output)
            self.assertIn("Anomalies: 0", output)
            self.assertIn("Depth: 0m", output)

    def test_parse_results_reports_json_errors(self):
        import parse_results

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = parse_results.main([str(path)])

            self.assertEqual(exit_code, 1)
            self.assertIn("Could not parse JSON", stderr.getvalue())

    def test_check_handles_legacy_count_and_bad_numeric_fields(self):
        import check

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps({
                "targets_completed": "1",
                "results": [{
                    "target_name": "Legacy Target",
                    "anomalies_detected": "2",
                    "top_deep_targets": [{"depth_m": "bad", "volume_m3": "123.4"}],
                }],
            }), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = check.main([str(path)])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Processed: 1", output)
            self.assertIn("Legacy Target: 2 anomalies", output)
            self.assertIn("Depths: [0]", output)


class EmbeddingTargetDiscoveryTests(unittest.TestCase):
    def test_discover_targets_accepts_output_dir_none(self):
        import embedding_target_discovery

        with tempfile.TemporaryDirectory() as tmp:
            raster = Path(tmp) / "embedding.tif"
            raster.write_text("placeholder", encoding="utf-8")
            fake_scores = np.zeros((5, 5), dtype=np.float32)
            fake_scores[2, 2] = 1.0

            class FakeSrc:
                height = 5
                width = 5
                crs = "EPSG:4326"
                transform = mock.Mock()

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

                def profile(self):
                    return {}

            FakeSrc.transform.__mul__ = lambda self, xy: (xy[0], xy[1])

            def write_score(*_, anomaly_out=None, **__):
                Path(anomaly_out).write_text("score", encoding="utf-8")
                return {}

            with mock.patch.object(embedding_target_discovery, "cluster_embedding_anomalies", side_effect=write_score), \
                 mock.patch.object(embedding_target_discovery, "compute_spatial_anomaly_score", side_effect=write_score), \
                 mock.patch.object(embedding_target_discovery, "_load_single_band", return_value=(fake_scores, FakeSrc.transform, "EPSG:4326")), \
                 mock.patch.object(embedding_target_discovery, "_find_local_maxima", return_value=[(2, 2, 1.0)]), \
                 mock.patch.object(embedding_target_discovery.rasterio, "open", return_value=FakeSrc()):
                targets = embedding_target_discovery.discover_targets(
                    raster,
                    region_name="test_region",
                    output_dir=None,
                    max_targets=1,
                )

            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0]["name"], "test_region_auto_001")


if __name__ == "__main__":
    unittest.main()
