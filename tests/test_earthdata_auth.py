import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import slc_data_fetcher


class EarthdataAuthSelectionTests(unittest.TestCase):
    def test_token_env_is_preferred_and_bearer_prefix_is_removed(self):
        fake_token = "dummy-earthdata-token-for-unit-test"
        auth = slc_data_fetcher.resolve_earthdata_auth(
            environ={
                "EARTHDATA_TOKEN": f"Bearer {fake_token}",
                "EARTHDATA_USERNAME": "dummy-user",
                "EARTHDATA_PASSWORD": "dummy-password",
            }
        )

        self.assertEqual(auth["mode"], "token")
        self.assertEqual(auth["token"], fake_token)
        self.assertEqual(auth["token_source"], "EARTHDATA_TOKEN")

        summary = slc_data_fetcher.describe_earthdata_auth(auth)
        self.assertNotIn(fake_token, summary)
        self.assertIn(f"length={len(fake_token)}", summary)
        self.assertIn("masked=<redacted>", summary)

    def test_username_password_remains_supported_without_token(self):
        auth = slc_data_fetcher.resolve_earthdata_auth(
            environ={
                "EARTHDATA_USERNAME": "dummy-user",
                "EARTHDATA_PASSWORD": "dummy-password",
            }
        )

        self.assertEqual(auth["mode"], "credentials")
        summary = slc_data_fetcher.describe_earthdata_auth(auth)
        self.assertIn("username/password", summary)
        self.assertNotIn("dummy-user", summary)
        self.assertNotIn("dummy-password", summary)
        self.assertIn("masked=<redacted>", summary)


class EarthdataNoSecretLoggingTests(unittest.TestCase):
    def test_download_uses_token_auth_without_logging_token_contents(self):
        fake_token = "dummy-token-never-log-me"

        class FakeSession:
            instances = []

            def __init__(self):
                self.calls = []
                FakeSession.instances.append(self)

            def auth_with_token(self, token):
                self.calls.append(("token", token))
                return self

            def auth_with_creds(self, username, password):
                self.calls.append(("credentials", username, password))
                return self

        class FakeResult:
            def __init__(self, granule):
                self.granule = granule

            def download(self, output_dir, session=None):
                Path(output_dir, f"{self.granule}.zip").write_bytes(b"fake zip")

        fake_asf = types.SimpleNamespace(
            ASFSession=FakeSession,
            granule_search=lambda names: [FakeResult(names[0])],
        )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"asf_search": fake_asf}):
                with self.assertLogs("slc_data_fetcher", level="INFO") as logs:
                    downloaded = slc_data_fetcher.download_slc_product(
                        {"granule_name": "TEST_GRANULE", "size_mb": 1},
                        output_dir=Path(tmp),
                        earthdata_token=fake_token,
                    )
                self.assertTrue(downloaded.exists())

        self.assertEqual(FakeSession.instances[0].calls, [("token", fake_token)])

        log_text = "\n".join(logs.output)
        self.assertNotIn(fake_token, log_text)
        self.assertIn("token", log_text.lower())
        self.assertIn(f"length={len(fake_token)}", log_text)
        self.assertIn("masked=<redacted>", log_text)

    def test_env_file_loader_does_not_print_secret_values(self):
        fake_token = "dummy-env-token-never-log-me"
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "EARTHDATA_TOKEN=Bearer " + fake_token + "\n"
                "EARTHDATA_USERNAME=dummy-user\n"
                "PLAIN_VALUE=visible\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with mock.patch.dict(os.environ, {}, clear=False):
                with contextlib.redirect_stdout(stdout):
                    with self.assertLogs(level="INFO") as logs:
                        loaded = slc_data_fetcher.load_env_file(env_path)

        self.assertTrue(loaded)
        combined = stdout.getvalue() + "\n".join(logs.output)
        self.assertNotIn(fake_token, combined)
        self.assertNotIn("dummy-user", combined)


class ASFSearchRobustnessTests(unittest.TestCase):
    def test_search_retries_timeout_and_sets_configured_timeout(self):
        calls = []

        class FakeProduct:
            properties = {
                "sceneName": "TEST_SCENE",
                "url": "https://example.invalid/TEST_SCENE.zip",
                "startTime": "2024-01-01T00:00:00Z",
                "bytes": 1024 * 1024,
            }

        def fake_search(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TimeoutError("CMR took too long to respond")
            return [FakeProduct()]

        fake_internal = types.SimpleNamespace(CMR_TIMEOUT=30)
        fake_asf = types.SimpleNamespace(
            PLATFORM=types.SimpleNamespace(SENTINEL1="Sentinel-1"),
            PRODUCT_TYPE=types.SimpleNamespace(SLC="SLC"),
            BEAMMODE=types.SimpleNamespace(IW="IW"),
            FLIGHT_DIRECTION=types.SimpleNamespace(ASCENDING="ASCENDING", DESCENDING="DESCENDING"),
            search=fake_search,
        )
        fake_constants = types.SimpleNamespace(INTERNAL=fake_internal)

        with mock.patch.dict(
            sys.modules,
            {"asf_search": fake_asf, "asf_search.constants": fake_constants},
        ):
            with mock.patch("slc_data_fetcher.time.sleep", return_value=None):
                products = slc_data_fetcher.search_sentinel1_slc(
                    (-122.0, 38.0, -121.9, 38.1),
                    max_results=1,
                    search_timeout_seconds=77,
                    search_max_retries=2,
                    search_retry_backoff_seconds=0,
                )

        self.assertEqual(len(calls), 2)
        self.assertEqual(fake_internal.CMR_TIMEOUT, 77)
        self.assertEqual(products[0]["granule_name"], "TEST_SCENE")

    def test_safe_exception_summary_redacts_local_secrets(self):
        with mock.patch.dict(
            os.environ,
            {
                "EARTHDATA_USERNAME": "dummy-user-never-log",
                "EARTHDATA_PASSWORD": "dummy-password-never-log",
                "EARTHDATA_TOKEN": "Bearer dummy-token-never-log",
            },
        ):
            summary = slc_data_fetcher.safe_exception_summary(
                RuntimeError(
                    "bad dummy-user-never-log dummy-password-never-log dummy-token-never-log"
                )
            )

        self.assertNotIn("dummy-user-never-log", summary)
        self.assertNotIn("dummy-password-never-log", summary)
        self.assertNotIn("dummy-token-never-log", summary)
        self.assertIn("<redacted>", summary)


if __name__ == "__main__":
    unittest.main()
