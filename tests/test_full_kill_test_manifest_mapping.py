import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_full_runner():
    module_path = ROOT / "data" / "kill_test" / "run_full_kill_test.py"
    spec = importlib.util.spec_from_file_location("full_kill_test_runner_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FullKillTestManifestMappingTest(unittest.TestCase):
    def test_downloaded_manifest_groups_are_wired_to_product_zips(self):
        runner = load_full_runner()
        manifest_path = ROOT / "data" / "slc" / "sentinel1_slc_selection_manifest.json"
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = []
        for row in rows:
            # The resolver manifest only covers the eight groups that were
            # missing from the original full runner: manifest group 1 maps to
            # full-run group 3, manifest group 8 maps to full-run group 10.
            full_group = runner.GROUPS[int(row["group_id"]) + 1]
            if not full_group.get("product_zip"):
                missing.append(full_group["group_id"])

        self.assertEqual(
            [],
            missing,
            "Downloaded manifest groups must have product_zip values before the full kill test can score them.",
        )


if __name__ == "__main__":
    unittest.main()
