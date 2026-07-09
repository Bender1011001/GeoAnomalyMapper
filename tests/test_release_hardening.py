import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geoanomaly
import blind_validation


class ReleaseHardeningHealthTests(unittest.TestCase):
    def test_health_report_is_secret_safe_for_local_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "EARTHDATA_TOKEN=super-secret-token\nEARTHDATA_PASSWORD=super-secret-password\n",
                encoding="utf-8",
            )
            (root / ".env.example").write_text("EARTHDATA_TOKEN=\n", encoding="utf-8")

            report = geoanomaly.build_health_report(root, probe_gpu=False)
            serialized = json.dumps(report, sort_keys=True)

            self.assertTrue(report["secrets"]["dotenv"]["present"])
            self.assertTrue(report["secrets"]["dotenv"]["expected_keys_present"]["EARTHDATA_TOKEN"])
            self.assertTrue(report["secrets"]["dotenv"]["expected_keys_present"]["EARTHDATA_PASSWORD"])
            self.assertFalse(report["secrets"]["dotenv"]["values_printed"])
            self.assertNotIn("super-secret-token", serialized)
            self.assertNotIn("super-secret-password", serialized)
            self.assertTrue(report["no_downloads_attempted"])
            self.assertTrue(report["no_training_attempted"])

    def test_health_json_cli_outputs_valid_no_download_report(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = geoanomaly.main(["health", "--json", "--skip-gpu"])

        self.assertEqual(rc, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema_version"], "geoanomaly-health-v1")
        self.assertTrue(report["python"]["ok"])
        self.assertTrue(report["no_downloads_attempted"])
        self.assertTrue(report["no_training_attempted"])
        self.assertFalse(report["secrets"]["dotenv"]["values_printed"])
        self.assertTrue(any(command["name"] == "unit_tests" for command in report["commands"]))

    def test_commands_include_reproducible_fixture_chain_without_real_execution(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = geoanomaly.main(["commands"])

        output = stdout.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("python geoanomaly.py health --json --skip-gpu", output)
        self.assertIn("validate-public --manifest validation_examples/public_manifest_fixture.json", output)
        self.assertIn("baseline-report data/blind_validation/fixture_score.json", output)
        self.assertIn("package-report --public-manifest validation_examples/public_manifest_fixture.json", output)
        self.assertNotIn("--execute-real", output)

    def test_top_level_validation_wrapper_delegates_to_blind_validation(self):
        with mock.patch.object(geoanomaly.blind_validation, "main", return_value=7) as delegated:
            rc = geoanomaly.main(["validation", "validate-public", "--manifest", "validation_examples/public_manifest_fixture.json"])

        self.assertEqual(rc, 7)
        delegated.assert_called_once_with(
            ["validate-public", "--manifest", "validation_examples/public_manifest_fixture.json"]
        )

    def test_blind_validation_namespace_redaction_is_json_safe(self):
        namespace = mock.Mock()
        namespace.func = lambda args: 0
        namespace.env_path = Path(".env")
        namespace.EARTHDATA_TOKEN = "should-not-print"

        safe = blind_validation._namespace_to_safe_dict(namespace)
        encoded = json.dumps(safe, sort_keys=True)

        self.assertIn("<callable:<lambda>>", encoded)
        self.assertIn('"env_path": ".env"', encoded)
        self.assertNotIn("should-not-print", encoded)

    def test_packaging_entry_points_and_exclusions_are_declared(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn('geoanomaly = "geoanomaly:main"', pyproject)
        self.assertIn('geoanomaly-validation = "blind_validation:main"', pyproject)
        self.assertIn('blind-validation = "blind_validation:main"', pyproject)
        self.assertIn('"numpy>=1.24.0"', pyproject)
        self.assertIn('"scikit-gstat>=1.0.0"', pyproject)
        self.assertIn('"json_utils"', pyproject)
        self.assertIn('"pandas>=2.0.0"', pyproject)
        self.assertIn("prune data", manifest)
        self.assertIn("prune results_extracted", manifest)
        self.assertIn("global-exclude .env", manifest)

    def test_clean_environment_commands_are_documented_and_no_download(self):
        docs = Path("docs/VALIDATION_FIRST_WORKFLOW.md").read_text(encoding="utf-8")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            rc = geoanomaly.main(["commands"])

        self.assertEqual(rc, 0)
        commands_text = output.getvalue()
        self.assertIn("Clean environment smoke commands", commands_text)
        self.assertIn("python -m venv .venv-clean", commands_text)
        self.assertIn("pip install -e .", commands_text)
        self.assertNotIn("--execute-real", commands_text)
        self.assertIn("scikit-gstat", docs)
        self.assertIn("skgstat", docs)

    def test_real_run_and_field_verification_docs_are_explicit(self):
        docs = Path("docs/VALIDATION_FIRST_WORKFLOW.md").read_text(encoding="utf-8")
        field_template = Path("validation_examples/field_verification_template.json").read_text(encoding="utf-8")

        self.assertIn("--confirm-real-downloads-and-training", docs)
        self.assertIn("estimated_download_size_mb", docs)
        self.assertIn("private withheld labels", docs.lower())
        self.assertIn("independent field evidence", docs.lower())
        self.assertIn("field_verification_template", docs)
        self.assertIn("must_not_be_committed", field_template)
        self.assertIn("independent_field_evidence", field_template)


if __name__ == "__main__":
    unittest.main()
