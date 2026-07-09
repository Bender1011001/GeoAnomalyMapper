import importlib.util
import unittest
from pathlib import Path


def load_incremental_module():
    module_path = Path("data/kill_test/run_incremental_kill_test.py")
    spec = importlib.util.spec_from_file_location("run_incremental_kill_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IncrementalKillTestDefinitionTests(unittest.TestCase):
    def test_preregistration_declares_two_small_groups_and_locked_score_policy(self):
        module = load_incremental_module()

        payload = module.build_preregistration_payload(generated_at_utc="LOCKED_TIME")

        self.assertEqual(payload["schema"], "incremental_known_positive_control_preregistration_v1")
        self.assertEqual(payload["generated_at_utc"], "LOCKED_TIME")
        self.assertFalse(payload["downloads_attempted_by_plan"])
        self.assertEqual(len(payload["groups"]), 2)
        for group in payload["groups"]:
            positives = [region for region in group["regions"] if "known_positive" in region["role"]]
            controls = [region for region in group["regions"] if "control" in region["role"]]
            self.assertEqual(len(positives), 1)
            self.assertGreaterEqual(len(controls), 1)
            self.assertLessEqual(len(controls), 3)
            self.assertIn("public_evidence_note", positives[0])
            for control in controls:
                self.assertIn("matching_note", control)
        self.assertEqual(payload["score_path"]["vibrometry_config"], module.minimal.VIBROMETRY_CONFIG)
        self.assertEqual(payload["score_path"]["scoring_config"], module.minimal.SCORING_CONFIG)
        self.assertEqual(payload["score_path"]["score_units"], "mm/s")

    def test_group_summary_ranks_positive_against_controls_without_extra_metrics(self):
        module = load_incremental_module()
        group = {
            "group_id": "demo_group",
            "positive_region_id": "positive_site",
            "regions": [
                {"region_id": "positive_site", "role": "known_positive_public_void_bearing_region"},
                {"region_id": "control_site", "role": "matched_public_control_region"},
            ],
        }
        rows = [
            {
                "region_id": "control_site",
                "max_anomaly_score": 2.0,
                "candidate_count": 4,
                "total_candidate_surface_pixels": 40,
                "total_candidate_volume_m3": None,
                "mean_void_probability": None,
                "top_void_probability": None,
            },
            {
                "region_id": "positive_site",
                "max_anomaly_score": 3.0,
                "candidate_count": 5,
                "total_candidate_surface_pixels": 50,
                "total_candidate_volume_m3": None,
                "mean_void_probability": None,
                "top_void_probability": None,
            },
        ]

        summary = module.summarize_group_result(group, rows)

        self.assertEqual(summary["group_id"], "demo_group")
        self.assertEqual(summary["positive_region_id"], "positive_site")
        self.assertEqual(summary["positive_rank"], 1)
        self.assertEqual(summary["positive_score"], 3.0)
        self.assertEqual(summary["best_control_score"], 2.0)
        self.assertEqual(summary["pass_fail"], "PASS_POSITIVE_RANK_1")
        self.assertIsNone(summary["positive_total_candidate_volume_m3"])
        self.assertIsNone(summary["positive_top_void_probability"])


if __name__ == "__main__":
    unittest.main()
