import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import run_biondi_exploration
import visualize_3d_subsurface


class SyntheticEndToEndPositiveControlTests(unittest.TestCase):
    def test_synthetic_sar_vibrometry_pinn_contract_and_visualization_detect_void(self):
        """Fast positive control for the runnable underground-void path.

        This deliberately runs the real synthetic SLC generator, real Doppler
        vibrometry, and real visualization/anomaly extraction.  The expensive
        neural PINN optimizer is replaced with a deterministic PINN-contract
        stand-in that consumes the vibrometry map and emits physically plausible
        low-speed / high-void-probability volumes.  That boundary keeps this
        regression fast and stable while still exercising the file/interface
        handoff across SAR -> vibrometry -> PINN outputs -> visualization.
        """

        def deterministic_pinn_positive_control(
            vibration_map,
            output_dir,
            config=None,
            frequency_map=None,
            surface_anomaly_map=None,
        ):
            cfg = dict(config or {})
            self.assertEqual(cfg["excitation_frequency_hz"], 2.0)
            self.assertGreater(float(np.nanmax(vibration_map)), float(np.nanmean(vibration_map)))
            self.assertIsNotNone(frequency_map)
            self.assertEqual(frequency_map.shape, vibration_map.shape)

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            nz = int(cfg.get("grid_nz", 8))
            ny = int(cfg.get("grid_ny", 8))
            nx = int(cfg.get("grid_nx", 8))
            background = float(cfg.get("background_wave_speed", 3500.0))

            wave_speed = np.full((nz, ny, nx), background, dtype=np.float32)
            void_probability = np.zeros((nz, ny, nx), dtype=np.float32)

            peak_y, peak_x = np.unravel_index(np.nanargmax(vibration_map), vibration_map.shape)
            cy = int(round(peak_y / max(vibration_map.shape[0] - 1, 1) * (ny - 1)))
            cx = int(round(peak_x / max(vibration_map.shape[1] - 1, 1) * (nx - 1)))
            cz = max(1, nz // 2)
            ys = slice(max(cy - 1, 0), min(cy + 2, ny))
            xs = slice(max(cx - 1, 0), min(cx + 2, nx))
            zs = slice(cz, min(cz + 3, nz))

            wave_speed[zs, ys, xs] = 450.0
            void_probability[zs, ys, xs] = 0.92

            wave_speed_path = out_dir / "wave_speed_volume.npy"
            void_probability_path = out_dir / "void_probability_volume.npy"
            np.save(wave_speed_path, wave_speed)
            np.save(void_probability_path, void_probability)

            metadata_path = out_dir / "inversion_metadata.txt"
            metadata_path.write_text(
                "max_depth_m: {0}\n"
                "domain_width_m: {1}\n"
                "background_wave_speed_ms: {2}\n"
                "excitation_freq_hz: {3}\n".format(
                    cfg.get("max_depth_m", 200.0),
                    cfg.get("domain_width_m", 80.0),
                    background,
                    cfg["excitation_frequency_hz"],
                ),
                encoding="utf-8",
            )

            return {
                "wave_speed_volume": str(wave_speed_path),
                "void_probability_volume": str(void_probability_path),
                "metadata": str(metadata_path),
            }

        tiny_quick_profile = dict(run_biondi_exploration.RESOLUTION_PROFILES["quick"])
        tiny_quick_profile.update(
            {
                "epochs": 1,
                "grid_nx": 8,
                "grid_ny": 8,
                "grid_nz": 8,
                "domain_width_m": 80.0,
                "max_depth_m": 200.0,
                "excitation_frequency_hz": 2.0,
                "synthetic_grid_size": 32,
                "num_sub_apertures": 3,
            }
        )

        target = {
            "name": "Synthetic Positive Control Void",
            "lat": 0.0,
            "lon": 0.0,
            "buffer_deg": 0.01,
            "description": "deterministic synthetic underground void target",
            "expected_depth_m": 100,
            "expected_void_type": "natural_cave",
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            old_enable_pyvista = visualize_3d_subsurface.DEFAULT_VIZ_CONFIG["enable_pyvista"]
            visualize_3d_subsurface.DEFAULT_VIZ_CONFIG["enable_pyvista"] = False
            try:
                with mock.patch.object(run_biondi_exploration, "EXPLORE_DIR", tmp_path), \
                     mock.patch.dict(
                         run_biondi_exploration.RESOLUTION_PROFILES,
                         {"quick": tiny_quick_profile},
                         clear=False,
                     ), \
                     mock.patch.object(
                         run_biondi_exploration.pinn_vibro_inversion,
                         "train_vibro_pinn",
                         side_effect=deterministic_pinn_positive_control,
                     ) as pinn_mock:
                    result = run_biondi_exploration.execute_biondi_pipeline_for_target(
                        target=target,
                        credentials={},
                        use_synthetic_fallback=True,
                        resolution="quick",
                    )
            finally:
                visualize_3d_subsurface.DEFAULT_VIZ_CONFIG["enable_pyvista"] = old_enable_pyvista

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["data_source"], "synthetic")
            self.assertGreaterEqual(result["anomaly_count"], 1)
            self.assertEqual(result["anomalies_detected"], result["anomaly_count"])
            self.assertTrue(result["top_deep_targets"])
            self.assertEqual(
                pinn_mock.call_args.kwargs["config"]["excitation_frequency_hz"],
                tiny_quick_profile["excitation_frequency_hz"],
            )

            for key in [
                "vibration_amplitude",
                "wave_speed_volume",
                "void_probability_volume",
                "visualization_dir",
                "anomaly_report",
                "anomaly_catalog",
                "audit_manifest",
                "synthetic_positive_control_diagnostics",
            ]:
                self.assertTrue(Path(result["outputs"][key]).exists(), key)

            with open(result["outputs"]["synthetic_positive_control_diagnostics"], "r", encoding="utf-8") as f:
                diagnostics = json.load(f)
            self.assertEqual(diagnostics["primary_blocker"], "none_detected")
            self.assertGreater(diagnostics["vibrometry"]["active_to_inactive_ratio"], 1.0)
            self.assertGreaterEqual(diagnostics["void_probability"]["voxels_above_threshold"], 3)


if __name__ == "__main__":
    unittest.main()
