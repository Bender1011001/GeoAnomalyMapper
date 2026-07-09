import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import run_biondi_exploration
import slc_data_fetcher


class TargetLocalSlcChipTests(unittest.TestCase):
    def test_geolocated_target_local_chips_from_same_source_are_distinct(self):
        slc = (
            np.arange(100, dtype=np.float32).reshape(10, 10)
            + 1j * np.zeros((10, 10), dtype=np.float32)
        ).astype(np.complex64)
        geolocation_points = [
            {"line": 0, "pixel": 0, "latitude": 10.0, "longitude": 20.0},
            {"line": 0, "pixel": 9, "latitude": 10.0, "longitude": 29.0},
            {"line": 9, "pixel": 0, "latitude": 19.0, "longitude": 20.0},
            {"line": 9, "pixel": 9, "latitude": 19.0, "longitude": 29.0},
            {"line": 2, "pixel": 2, "latitude": 12.0, "longitude": 22.0},
            {"line": 7, "pixel": 7, "latitude": 17.0, "longitude": 27.0},
        ]

        chip_a, meta_a = slc_data_fetcher.extract_target_local_slc_chip(
            slc,
            target_lat=12.0,
            target_lon=22.0,
            geolocation_points=geolocation_points,
            chip_shape=(4, 4),
        )
        chip_b, meta_b = slc_data_fetcher.extract_target_local_slc_chip(
            slc,
            target_lat=17.0,
            target_lon=27.0,
            geolocation_points=geolocation_points,
            chip_shape=(4, 4),
        )

        self.assertEqual(chip_a.shape, (4, 4))
        self.assertEqual(chip_b.shape, (4, 4))
        self.assertFalse(np.array_equal(chip_a, chip_b))
        self.assertNotEqual(meta_a["window"], meta_b["window"])
        self.assertEqual(meta_a["method"], "geolocation_grid_nearest")
        self.assertEqual(meta_b["method"], "geolocation_grid_nearest")

    def test_target_local_chip_refuses_missing_geolocation(self):
        slc = np.ones((10, 10), dtype=np.complex64)

        with self.assertRaisesRegex(ValueError, "target-local SLC chip requires geolocation"):
            slc_data_fetcher.extract_target_local_slc_chip(
                slc,
                target_lat=12.0,
                target_lon=22.0,
                geolocation_points=[],
                chip_shape=(4, 4),
            )

    def test_target_local_chip_refuses_full_source_window(self):
        slc = np.ones((10, 10), dtype=np.complex64)

        with self.assertRaisesRegex(ValueError, "target-local SLC chip would cover the full source"):
            slc_data_fetcher.extract_target_local_slc_chip(
                slc,
                target_lat=12.0,
                target_lon=22.0,
                geolocation_points=[
                    {"line": 5, "pixel": 5, "latitude": 12.0, "longitude": 22.0},
                ],
                chip_shape=(20, 20),
            )

    def test_locked_product_pipeline_passes_target_coordinates_into_extraction(self):
        target = {
            "name": "Target A",
            "lat": 12.0,
            "lon": 22.0,
            "buffer_deg": 0.1,
            "description": "test",
            "expected_depth_m": 100,
        }
        tiny_profile = dict(run_biondi_exploration.RESOLUTION_PROFILES["quick"])
        tiny_profile.update({"target_slc_chip_pixels": 128, "epochs": 1})

        with mock.patch.object(run_biondi_exploration, "EXPLORE_DIR", Path("unused-test-output")), \
             mock.patch.dict(run_biondi_exploration.RESOLUTION_PROFILES, {"quick": tiny_profile}), \
             mock.patch.object(run_biondi_exploration.slc_data_fetcher, "_build_search_bbox", return_value=(21.9, 11.9, 22.1, 12.1)), \
             mock.patch.object(run_biondi_exploration.slc_data_fetcher, "resolve_earthdata_auth", return_value={"mode": "token", "token": "x"}), \
             mock.patch.object(run_biondi_exploration.slc_data_fetcher, "download_slc_product", return_value=Path("locked.SAFE")), \
             mock.patch.object(run_biondi_exploration.slc_data_fetcher, "extract_slc_burst", return_value=[] ) as extract_mock:
            result = run_biondi_exploration.execute_biondi_pipeline_for_target(
                target,
                credentials={"EARTHDATA_TOKEN": "x"},
                use_synthetic_fallback=False,
                resolution="quick",
                locked_sentinel1_products=[{"product_id": "LOCKED", "granule_name": "LOCKED"}],
                require_locked_sentinel1=True,
            )

        self.assertEqual(result["status"], "failed")
        extract_mock.assert_called_once()
        self.assertEqual(extract_mock.call_args.kwargs["target_lat"], 12.0)
        self.assertEqual(extract_mock.call_args.kwargs["target_lon"], 22.0)
        self.assertEqual(extract_mock.call_args.kwargs["target_bbox"], (21.9, 11.9, 22.1, 12.1))
        self.assertEqual(extract_mock.call_args.kwargs["target_chip_shape"], (128, 128))


if __name__ == "__main__":
    unittest.main()
