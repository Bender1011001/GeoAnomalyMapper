import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import blind_validation
import geoanomaly


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_candidates(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "centroid_m",
        "depth_m",
        "mean_void_probability",
        "deep_target_score",
        "fused_confidence_score",
        "deep_target_rank",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_parameter_set(path, validation_id="unit_blind_validation", approved=False, updates=None):
    data = blind_validation._default_parameter_set_template(validation_id, "unit_test_parameters")
    data["template_only"] = False
    if updates:
        data.pop("parameter_set_hash", None)
        data.pop("parameter_set_id", None)
        _deep_update(data, updates)
    if approved:
        data["approval"] = {
            "status": "approved_for_holdout",
            "approved_by": "unit_test",
            "approved_at_utc": "2026-01-01T00:00:00Z",
            "approval_scope": "unit_test_holdout",
        }
        data["approval_status"] = "approved_for_holdout"
        data["approved_for_holdout"] = True
    data["parameter_set_hash"] = blind_validation.parameter_set_hash(data)
    data["parameter_set_id"] = blind_validation.parameter_set_id_from_hash(data["parameter_set_hash"])
    _write_json(path, data)
    return path


def _write_campaign_registry(
    path,
    public_manifest_path,
    parameter_set_path,
    validation_id="unit_blind_validation",
    status="draft",
):
    data = blind_validation._default_campaign_registry_template(
        validation_id,
        "unit_campaign_registry",
        public_manifest_path,
        parameter_set_path,
        campaign_name="unit_test_campaign",
        status=status,
        approved_by="unit_test" if status in {"approved", "locked"} else None,
        approved_at_utc="2026-01-01T00:00:00Z" if status in {"approved", "locked"} else None,
        locked_by="unit_test" if status == "locked" else None,
        locked_at_utc="2026-01-01T00:00:00Z" if status == "locked" else None,
    )
    data["template_only"] = False
    data["registry_hash"] = blind_validation.campaign_registry_hash(data)
    data["registry_id"] = blind_validation.campaign_registry_id_from_hash(data["registry_hash"])
    _write_json(path, data)
    return path


def _write_product_lock(path, public_manifest_path, target_ids, parameter_set_path=None, registry_path=None):
    parameter_record = None
    if parameter_set_path is not None:
        parameter_record = blind_validation._parameter_set_reference_record(
            parameter_set_path,
            blind_validation.load_parameter_set(parameter_set_path, require_approved=False),
        )
    registry_record = None
    if registry_path is not None:
        registry_record = blind_validation._campaign_registry_reference_record(
            registry_path,
            blind_validation.load_campaign_registry(registry_path, require_locked=True),
        )
    targets = []
    for target_id in target_ids:
        product = {
            "product_id": f"LOCKED_{target_id}",
            "product_name": f"LOCKED_{target_id}",
            "provider_id": "sentinel1_asf",
            "provider": "ASF Sentinel-1",
            "comparison_arm": "sentinel1_cband_iw",
            "processing_level": "SLC",
            "size_mb": 10.0,
        }
        targets.append(
            {
                "target_id": target_id,
                "status": "success",
                "provider": "sentinel1_asf",
                "provider_id": "sentinel1_asf",
                "provider_label": "ASF Sentinel-1",
                "comparison_arm": "sentinel1_cband_iw",
                "selection_count": 1,
                "selected_product_ids": [product["product_id"]],
                "selected_products": [product],
                "estimated_download_size_mb": 10.0,
                "estimated_download_size_gb": blind_validation._mb_to_gb(10.0),
            }
        )
    lock = {
        "schema_version": blind_validation.PRODUCT_LOCK_SCHEMA_VERSION,
        "validation_id": "unit_blind_validation",
        "public_manifest": str(public_manifest_path),
        "public_manifest_sha256": blind_validation._sha256_file(public_manifest_path),
        "campaign_registry": registry_record,
        "campaign_registry_id": registry_record.get("registry_id") if registry_record else None,
        "campaign_registry_hash": registry_record.get("registry_hash") if registry_record else None,
        "parameter_set": parameter_record,
        "selection_policy": blind_validation.PRODUCT_SELECTION_POLICY,
        "search_only": True,
        "no_download": True,
        "estimated_download_size_mb": 10.0 * len(targets),
        "estimated_download_size_gb": blind_validation._mb_to_gb(10.0 * len(targets)),
        "targets": targets,
    }
    _write_json(path, lock)
    return path


def _robustness_variants():
    return [
        {
            "variant_id": "baseline_locked_parameters",
            "family": "baseline_reference",
            "scope": "calibration_only",
            "arm": "baseline",
            "parameters": {},
        },
        {
            "variant_id": "void_threshold_low",
            "family": "void_threshold_sweep",
            "scope": "calibration_only",
            "arm": "threshold_sensitivity",
            "parameters": {"void_probability_threshold": 0.25},
        },
        {
            "variant_id": "min_voxels_high",
            "family": "min_anomaly_voxel_threshold",
            "scope": "calibration_only",
            "arm": "connected_component_inflation",
            "parameters": {"min_anomaly_voxels": 9},
        },
        {
            "variant_id": "morphology_26_connected",
            "family": "segmentation_topology",
            "scope": "calibration_only",
            "arm": "morphology_topology",
            "parameters": {"morphology_iterations": 2, "connected_component_connectivity": 26},
        },
        {
            "variant_id": "pinn_prior_low",
            "family": "depth_prior_pinn_regularization",
            "scope": "calibration_only",
            "arm": "pinn_prior",
            "parameters": {"deep_prior_weight": 0.5, "regularization_weight": 0.05},
        },
        {
            "variant_id": "sar_quality_strict",
            "family": "sar_preprocessing_quality_filter",
            "scope": "calibration_only",
            "arm": "sar_preprocessing",
            "parameters": {"coherence_min": 0.35, "speckle_filter": "lee"},
        },
        {
            "variant_id": "topk_10",
            "family": "top_k_candidate_cutoff",
            "scope": "calibration_only",
            "arm": "candidate_volume",
            "parameters": {"top_k": 10},
        },
        {
            "variant_id": "null_random_spatial",
            "family": "null_region_random_spatial_baseline",
            "scope": "calibration_only",
            "arm": "null_baseline",
            "null_baseline": True,
            "parameters": {"random_seed": 17, "spatial_jitter_m": 500},
        },
        {
            "variant_id": "repeat_product_stability",
            "family": "repeat_product_date_stability",
            "scope": "calibration_only",
            "arm": "repeat_product",
            "stability_group": "same_target_repeat_dates",
            "parameters": {"date_grouping": "month"},
        },
    ]


def _write_robustness_plan(
    path,
    validation_id="unit_blind_validation",
    campaign_id="unit_campaign_registry",
    variants=None,
    source_artifacts=None,
    template_only=False,
):
    data = {
        "schema_version": "blind-validation-robustness-plan-v1",
        "template_only": template_only,
        "validation_id": validation_id,
        "campaign_id": campaign_id,
        "robustness_plan_name": "unit_robustness_ablation_plan",
        "preregistration": {
            "status": "draft_calibration_only",
            "record": "unit-test-public-no-label-plan",
            "holdout_unblinding_status": "not_unblinded",
        },
        "policy": {
            "labels_withheld_from_runner": True,
            "no_downloads_or_training": True,
            "default_execution_mode": "dry_run_plan_only",
            "ablation_use_rule": "calibration_only_unless_preregistered_before_holdout_unblinding",
            "parameter_changes_after_holdout_unblinding": "forbidden",
        },
        "source_artifacts": source_artifacts or {},
        "reviewer_attack_coverage": [
            "threshold_sensitivity",
            "segmentation_topology",
            "connected_component_inflation",
            "pinn_prior_sensitivity",
            "sar_preprocessing_variants",
            "null_region_false_positives",
            "repeat_product_stability",
            "candidate_volume_topk",
        ],
        "variants": variants or _robustness_variants(),
    }
    _write_json(path, data)
    return path


class BlindValidationHarnessTests(unittest.TestCase):
    def _public_manifest(self, tmp, targets):
        path = Path(tmp) / "public.json"
        _write_json(
            path,
            {
                "schema_version": blind_validation.PUBLIC_SCHEMA_VERSION,
                "validation_id": "unit_blind_validation",
                "targets": targets,
            },
        )
        return path

    def _labels(self, tmp, labels):
        path = Path(tmp) / "withheld.json"
        _write_json(
            path,
            {
                "schema_version": blind_validation.WITHHELD_LABEL_SCHEMA_VERSION,
                "validation_id": "unit_blind_validation",
                "labels": labels,
            },
        )
        return path

    def test_runner_does_not_load_withheld_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(csv_path, [])
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "site_a",
                        "name": "Site A",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "buffer_deg": 0.01,
                        "domain_width_m": 800.0,
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            with mock.patch.object(
                blind_validation,
                "load_withheld_labels",
                side_effect=AssertionError("runner must not load labels"),
            ):
                manifest = blind_validation.run_blind_validation(
                    public_path,
                    tmp_path / "run",
                    dry_run=True,
                )
            self.assertFalse(manifest["withheld_labels_loaded"])
            self.assertEqual(manifest["targets"][0]["candidate_count"], 0)

    def test_scorer_matches_candidate_to_withheld_known_void(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(
                csv_path,
                [
                    {
                        "id": "hit",
                        "centroid_m": "[10.0, 5.0, 120.0]",
                        "depth_m": "120.0",
                        "mean_void_probability": "0.91",
                        "deep_target_score": "0.80",
                        "fused_confidence_score": "0.85",
                        "deep_target_rank": "1",
                    }
                ],
            )
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "positive_site",
                        "name": "Positive Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "buffer_deg": 0.01,
                        "area_km2": 0.5,
                        "domain_width_m": 800.0,
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            run_manifest = blind_validation.run_blind_validation(public_path, tmp_path / "run", dry_run=True)
            labels_path = self._labels(
                tmp,
                [
                    {
                        "target_id": "positive_site",
                        "site_class": "positive",
                        "known_voids": [
                            {
                                "void_id": "known_void",
                                "offset_m": [12.0, 7.0],
                                "depth_m": 125.0,
                                "horizontal_tolerance_m": 20.0,
                                "depth_tolerance_m": 20.0,
                            }
                        ],
                    }
                ],
            )
            report = blind_validation.score_blind_validation(
                tmp_path / "run" / "run_manifest.json",
                labels_path,
            )
            target = report["targets"][0]
            self.assertTrue(target["positive_site_hit"])
            self.assertFalse(target["positive_site_miss"])
            self.assertEqual(target["rank_of_first_hit"], 1)
            self.assertAlmostEqual(target["first_hit_distance_error_m"], 2.828427, places=5)
            self.assertAlmostEqual(target["first_hit_depth_error_m"], 5.0, places=5)
            self.assertEqual(report["summary"]["positive_site_hits"], 1)
            self.assertEqual(report["summary"]["site_recall_like"], 1.0)
            self.assertEqual(run_manifest["targets"][0]["candidate_count"], 1)

    def test_scorer_reports_false_positive_on_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "negative_detected_anomalies.csv"
            _write_candidates(
                csv_path,
                [
                    {
                        "id": "fp",
                        "centroid_m": "[0.0, 0.0, 50.0]",
                        "depth_m": "50.0",
                        "mean_void_probability": "0.70",
                        "deep_target_score": "0.50",
                        "fused_confidence_score": "0.50",
                        "deep_target_rank": "1",
                    }
                ],
            )
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "negative_site",
                        "name": "Negative Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 36.0, "lon": -118.0},
                        "buffer_deg": 0.01,
                        "area_km2": 1.0,
                        "domain_width_m": 800.0,
                        "existing_candidates_csv": "negative_detected_anomalies.csv",
                    }
                ],
            )
            blind_validation.run_blind_validation(public_path, tmp_path / "run", dry_run=True)
            labels_path = self._labels(
                tmp,
                [
                    {
                        "target_id": "negative_site",
                        "site_class": "negative",
                        "known_voids": [],
                    }
                ],
            )
            report = blind_validation.score_blind_validation(tmp_path / "run" / "run_manifest.json", labels_path)
            self.assertTrue(report["targets"][0]["negative_site_false_positive"])
            self.assertEqual(report["summary"]["negative_site_false_positive_sites"], 1)
            self.assertEqual(report["summary"]["site_precision_like"], 0.0)

    def test_score_summary_is_deterministic_and_malformed_public_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(
                csv_path,
                [
                    {
                        "id": "hit",
                        "centroid_m": "[0.0, 0.0, 100.0]",
                        "depth_m": "100.0",
                        "mean_void_probability": "0.90",
                        "deep_target_score": "0.80",
                        "fused_confidence_score": "0.80",
                        "deep_target_rank": "1",
                    }
                ],
            )
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "site_a",
                        "name": "Site A",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "buffer_deg": 0.01,
                        "area_km2": 1.0,
                        "domain_width_m": 800.0,
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            blind_validation.run_blind_validation(public_path, tmp_path / "run", dry_run=True)
            labels_path = self._labels(
                tmp,
                [
                    {
                        "target_id": "site_a",
                        "site_class": "positive",
                        "known_voids": [
                            {
                                "void_id": "known_void",
                                "offset_m": [0.0, 0.0],
                                "depth_m": 100.0,
                                "horizontal_tolerance_m": 10.0,
                                "depth_tolerance_m": 10.0,
                            }
                        ],
                    }
                ],
            )
            run_path = tmp_path / "run" / "run_manifest.json"
            first = blind_validation.score_blind_validation(run_path, labels_path)
            second = blind_validation.score_blind_validation(run_path, labels_path)
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["targets"], second["targets"])

            malformed_path = tmp_path / "malformed_public.json"
            _write_json(
                malformed_path,
                {
                    "schema_version": blind_validation.PUBLIC_SCHEMA_VERSION,
                    "validation_id": "bad",
                    "targets": [
                        {
                            "target_id": "leaky",
                            "name": "Leaky",
                            "center": {"lat": 0.0, "lon": 0.0},
                            "known_voids": [],
                        }
                    ],
                },
            )
            with self.assertRaises(blind_validation.ManifestValidationError):
                blind_validation.load_public_manifest(malformed_path)

    def test_inventory_uses_public_manifest_only_and_no_download_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "inventory_site",
                        "name": "Inventory Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 37.0, "lon": -86.0},
                        "buffer_deg": 0.02,
                        "sar_search": {
                            "start_date": "2022-01-01T00:00:00Z",
                            "end_date": "2022-02-01T00:00:00Z",
                            "max_results": 5,
                            "selection_count": 1,
                        },
                    }
                ],
            )

            observed_search_kwargs = []

            def mock_searcher(**kwargs):
                observed_search_kwargs.append(kwargs)
                return [
                    {
                        "granule_name": "S1_PUBLIC_ONLY",
                        "file_id": "S1_PUBLIC_ONLY_ID",
                        "start_time": "2022-01-15T00:00:00Z",
                        "size_mb": 123.0,
                        "beam_mode": "IW",
                        "processing_level": "SLC",
                    }
                ]

            with mock.patch.object(
                blind_validation,
                "load_withheld_labels",
                side_effect=AssertionError("inventory must not load labels"),
            ):
                inventory = blind_validation.build_sar_inventory(
                    public_path,
                    tmp_path / "inventory.json",
                    searcher=mock_searcher,
                    load_dotenv=False,
                )

            self.assertTrue(inventory["search_only"])
            self.assertTrue(inventory["no_download"])
            self.assertFalse(inventory["downloads_attempted"])
            self.assertFalse(inventory["withheld_labels_loaded"])
            self.assertEqual(len(observed_search_kwargs), 1)
            self.assertNotIn("labels", observed_search_kwargs[0])
            self.assertNotIn("withheld", json.dumps(observed_search_kwargs[0]).lower())
            self.assertEqual(inventory["targets"][0]["products_found"], 1)
            self.assertEqual(inventory["targets"][0]["selected_product_ids"], ["S1_PUBLIC_ONLY_ID"])

    def test_inventory_product_selection_and_lock_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "deterministic_site",
                        "name": "Deterministic Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 32.0, "lon": -104.0},
                        "buffer_deg": 0.03,
                        "sar_search": {
                            "start_date": "2022-01-01T00:00:00Z",
                            "end_date": "2023-01-01T00:00:00Z",
                            "max_results": 3,
                            "selection_count": 2,
                        },
                    }
                ],
            )

            mocked_products = [
                {
                    "granule_name": "PRODUCT_B",
                    "file_id": "PRODUCT_B_ID",
                    "start_time": "2022-09-01T00:00:00Z",
                    "size_mb": 200.0,
                    "provider": "ASF Sentinel-1",
                    "beam_mode": "IW",
                    "processing_level": "SLC",
                },
                {
                    "granule_name": "PRODUCT_A",
                    "file_id": "PRODUCT_A_ID",
                    "start_time": "2022-09-01T00:00:00Z",
                    "size_mb": 100.0,
                    "provider": "ASF Sentinel-1",
                    "beam_mode": "IW",
                    "processing_level": "SLC",
                },
                {
                    "granule_name": "PRODUCT_OLD",
                    "file_id": "PRODUCT_OLD_ID",
                    "start_time": "2022-01-01T00:00:00Z",
                    "size_mb": 300.0,
                    "provider": "ASF Sentinel-1",
                    "beam_mode": "IW",
                    "processing_level": "SLC",
                },
            ]

            def mock_searcher(**kwargs):
                return list(mocked_products)

            lock_path = tmp_path / "product_lock.json"
            inventory = blind_validation.build_sar_inventory(
                public_path,
                tmp_path / "inventory.json",
                lock_output_path=lock_path,
                searcher=mock_searcher,
                load_dotenv=False,
            )
            self.assertTrue(lock_path.exists())
            selected_ids = inventory["targets"][0]["selected_product_ids"]
            self.assertEqual(selected_ids, ["PRODUCT_A_ID", "PRODUCT_B_ID"])

            first_lock = blind_validation.build_product_lock(inventory)
            second_lock = blind_validation.build_product_lock(inventory)
            self.assertEqual(first_lock["targets"], second_lock["targets"])
            self.assertEqual(first_lock["selection_policy"], blind_validation.PRODUCT_SELECTION_POLICY)

            with open(lock_path, "r", encoding="utf-8") as f:
                written_lock = json.load(f)
            self.assertEqual(written_lock["targets"][0]["selected_product_ids"], selected_ids)
            self.assertEqual(written_lock["targets"][0]["estimated_download_size_mb"], 300.0)
            self.assertEqual(written_lock["estimated_download_size_mb"], 300.0)
            self.assertTrue(written_lock["no_download"])
            self.assertFalse(written_lock["withheld_labels_loaded"])

    def test_parameter_set_is_deterministic_hashable_and_label_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            params_path = _write_parameter_set(tmp_path / "parameters.json")
            approved_path = _write_parameter_set(tmp_path / "parameters_approved.json", approved=True)

            with mock.patch.object(
                blind_validation,
                "load_withheld_labels",
                side_effect=AssertionError("parameter validation must not load labels"),
            ):
                first = blind_validation.load_parameter_set(params_path)
                second = blind_validation.load_parameter_set(params_path)
                approved = blind_validation.load_parameter_set(approved_path, require_approved=True)

            self.assertEqual(first["parameter_set_hash"], second["parameter_set_hash"])
            self.assertEqual(first["parameter_set_id"], second["parameter_set_id"])
            self.assertTrue(first["parameter_set_id"].startswith(blind_validation.PARAMETER_SET_ID_PREFIX))
            self.assertFalse(first["withheld_labels_loaded"])
            self.assertTrue(approved["approved_for_holdout"])
            self.assertEqual(first["parameter_set_hash"], approved["parameter_set_hash"])

            comparison = blind_validation.compare_parameter_sets(params_path, approved_path)
            self.assertTrue(comparison["matching_hash"])
            self.assertFalse(comparison["withheld_labels_loaded"])

            malformed_path = tmp_path / "parameters_with_labels.json"
            with open(params_path, "r", encoding="utf-8") as f:
                malformed = json.load(f)
            malformed["labels"] = []
            malformed.pop("parameter_set_hash", None)
            malformed.pop("parameter_set_id", None)
            _write_json(malformed_path, malformed)
            with self.assertRaises(blind_validation.ManifestValidationError):
                blind_validation.load_parameter_set(malformed_path)

    def test_campaign_registry_is_deterministic_hashable_and_template_validates(self):
        template_path = Path("validation_examples/campaign_registry_template.json")
        registry = blind_validation.load_campaign_registry(template_path, allow_templates=True)
        second = blind_validation.load_campaign_registry(template_path, allow_templates=True)
        self.assertTrue(registry["template_only"])
        self.assertEqual(registry["schema_version"], blind_validation.CAMPAIGN_REGISTRY_SCHEMA_VERSION)
        self.assertEqual(registry["registry_hash"], second["registry_hash"])
        self.assertEqual(registry["registry_id"], second["registry_id"])
        self.assertTrue(registry["registry_id"].startswith(blind_validation.CAMPAIGN_REGISTRY_ID_PREFIX))
        self.assertFalse(registry["withheld_labels_loaded"])
        registry_text = json.dumps(registry, sort_keys=True).lower()
        for private_key in (
            "site_class",
            "known_voids",
            "withheld_labels_path",
            "private_label_path",
            "field_evidence_path",
            "earthdata_token",
        ):
            self.assertNotIn(private_key, registry_text)
        with self.assertRaisesRegex(blind_validation.ManifestValidationError, "template_only"):
            blind_validation.load_campaign_registry(template_path)

    def test_campaign_registry_detects_manifest_and_parameter_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "registry_site",
                        "name": "Registry Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "split": "holdout",
                        "sar_search": {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                    }
                ],
            )
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )

            registry = blind_validation.load_campaign_registry(registry_path, require_locked=True)
            self.assertEqual(registry["status"], "locked")
            self.assertTrue(registry["parameter_set"]["approved_for_holdout"])

            comparison = blind_validation.compare_campaign_registry(
                registry_path,
                public_manifest_path=public_path,
                parameter_set_path=params_path,
            )
            self.assertFalse(comparison["drift_detected"])
            self.assertTrue(comparison["matching_registry"])

            changed_public_path = tmp_path / "public_changed.json"
            public_data = json.loads(public_path.read_text(encoding="utf-8"))
            public_data["targets"][0]["buffer_deg"] = 0.123
            _write_json(changed_public_path, public_data)
            changed_comparison = blind_validation.compare_campaign_registry(
                registry_path,
                public_manifest_path=changed_public_path,
                parameter_set_path=params_path,
            )
            self.assertTrue(changed_comparison["drift_detected"])
            self.assertTrue(any(diff["path"] == "public_manifest.file_sha256" for diff in changed_comparison["differences"]))
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "drift"):
                blind_validation.verify_campaign_registry_for_inputs(
                    registry_path,
                    changed_public_path,
                    parameter_set_path=params_path,
                    require_locked=True,
                )

            changed_params = _write_parameter_set(
                tmp_path / "changed_parameters.json",
                approved=True,
                updates={"thresholds": {"void_probability_threshold": 0.51}},
            )
            parameter_drift = blind_validation.compare_campaign_registry(
                registry_path,
                public_manifest_path=public_path,
                parameter_set_path=changed_params,
            )
            self.assertTrue(parameter_drift["drift_detected"])
            self.assertTrue(any(diff["path"] == "parameter_set.parameter_set_hash" for diff in parameter_drift["differences"]))

    def test_campaign_registry_rejects_label_leakage_and_withheld_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "registry_leak_site",
                        "name": "Registry Leak Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 35.0, "lon": -117.0},
                    }
                ],
            )
            params_path = _write_parameter_set(tmp_path / "parameters.json", approved=True)
            registry_path = _write_campaign_registry(tmp_path / "campaign_registry.json", public_path, params_path)
            base = json.loads(registry_path.read_text(encoding="utf-8"))

            for key, value in (
                ("withheld_labels_path", "private/withheld_labels.json"),
                ("labels", []),
                ("field_evidence_path", "private/evidence.json"),
            ):
                malformed = dict(base)
                malformed["protocol"] = dict(base["protocol"])
                malformed["protocol"][key] = value
                malformed_path = tmp_path / f"leaky_{key}.json"
                _write_json(malformed_path, malformed)
                with self.assertRaises(blind_validation.ManifestValidationError, msg=key):
                    blind_validation.load_campaign_registry(malformed_path)

            path_leak = dict(base)
            path_leak["protocol"] = dict(base["protocol"])
            path_leak["protocol"]["review_note"] = "do not include private/withheld_labels.json in public registry"
            path_leak_path = tmp_path / "leaky_path_value.json"
            _write_json(path_leak_path, path_leak)
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "withheld"):
                blind_validation.load_campaign_registry(path_leak_path)

    def test_campaign_registry_metadata_propagates_to_inventory_lock_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "registry_inventory_site",
                        "name": "Registry Inventory Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "sar_search": {"selection_count": 1},
                    }
                ],
            )
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )
            registry = blind_validation.load_campaign_registry(registry_path)

            def mock_searcher(**kwargs):
                return [
                    {
                        "granule_name": "REGISTRY_PRODUCT",
                        "file_id": "REGISTRY_PRODUCT_ID",
                        "start_time": "2022-01-15T00:00:00Z",
                        "size_mb": 11.0,
                    }
                ]

            lock_path = tmp_path / "product_lock.json"
            inventory = blind_validation.build_sar_inventory(
                public_path,
                tmp_path / "inventory.json",
                lock_output_path=lock_path,
                parameter_set_path=params_path,
                campaign_registry_path=registry_path,
                searcher=mock_searcher,
                load_dotenv=False,
            )
            self.assertEqual(inventory["campaign_registry_hash"], registry["registry_hash"])
            self.assertEqual(inventory["campaign_registry"]["registry_id"], registry["registry_id"])
            with open(lock_path, "r", encoding="utf-8") as f:
                lock = json.load(f)
            self.assertEqual(lock["campaign_registry_hash"], registry["registry_hash"])
            self.assertEqual(lock["campaign_registry"]["registry_id"], registry["registry_id"])
            self.assertTrue(lock["no_download"])

            run_manifest = blind_validation.run_blind_validation(
                public_path,
                tmp_path / "run",
                dry_run=True,
                parameter_set_path=params_path,
                require_approved_parameters=True,
                campaign_registry_path=registry_path,
                require_locked_campaign_registry=True,
            )
            self.assertEqual(run_manifest["campaign_registry_hash"], registry["registry_hash"])
            self.assertEqual(run_manifest["campaign_registry_status"], "locked")
            self.assertEqual(run_manifest["reproducibility"]["campaign_registry_hash"], registry["registry_hash"])

    def test_score_requires_approved_matching_parameter_set_for_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(csv_path, [])
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "holdout_site",
                        "name": "Holdout Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            approved_params = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            run_manifest = blind_validation.run_blind_validation(
                public_path,
                tmp_path / "run",
                dry_run=True,
                parameter_set_path=approved_params,
                require_approved_parameters=True,
            )
            self.assertEqual(
                run_manifest["parameter_set_hash"],
                blind_validation.load_parameter_set(approved_params)["parameter_set_hash"],
            )
            labels_path = self._labels(
                tmp,
                [{"target_id": "holdout_site", "site_class": "negative", "known_voids": []}],
            )
            report = blind_validation.score_blind_validation(
                tmp_path / "run" / "run_manifest.json",
                labels_path,
                parameter_set_path=approved_params,
                require_approved_parameters=True,
            )
            self.assertEqual(report["parameter_set_hash"], run_manifest["parameter_set_hash"])
            self.assertTrue(report["parameter_set_verification"]["approved_for_holdout"])
            self.assertTrue(report["parameter_set_verification"]["matching_hash"])

            changed_params = _write_parameter_set(
                tmp_path / "changed_parameters.json",
                approved=True,
                updates={"thresholds": {"void_probability_threshold": 0.5}},
            )
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "hash does not match"):
                blind_validation.score_blind_validation(
                    tmp_path / "run" / "run_manifest.json",
                    labels_path,
                    parameter_set_path=changed_params,
                    require_approved_parameters=True,
                )

            draft_params = _write_parameter_set(tmp_path / "draft_parameters.json", approved=False)
            draft_run = blind_validation.run_blind_validation(
                public_path,
                tmp_path / "draft_run",
                dry_run=True,
                parameter_set_path=draft_params,
            )
            self.assertFalse(draft_run["parameter_set_approved_for_holdout"])
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "not approved"):
                blind_validation.score_blind_validation(
                    tmp_path / "draft_run" / "run_manifest.json",
                    labels_path,
                    require_approved_parameters=True,
                )

    def test_provider_comparison_arm_inventory_and_lock_metadata_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "provider_site",
                        "name": "Provider Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 37.0, "lon": -86.0},
                        "buffer_deg": 0.02,
                        "comparison_arms": [
                            {
                                "provider": "sentinel1_asf",
                                "comparison_arm": "sentinel1_cband_iw",
                                "selection_count": 1,
                            },
                            {
                                "provider": "umbra_open_data",
                                "comparison_arm": "umbra_xband_spotlight_placeholder",
                                "resolution_m": 0.5,
                                "notes": "placeholder only; no download in validation preflight",
                            },
                        ],
                    }
                ],
            )

            observed_search_kwargs = []

            def mock_searcher(**kwargs):
                observed_search_kwargs.append(kwargs)
                return [
                    {
                        "granule_name": "S1_ARM_PRODUCT",
                        "file_id": "S1_ARM_PRODUCT_ID",
                        "start_time": "2022-01-15T00:00:00Z",
                        "size_mb": 123.0,
                    }
                ]

            lock_path = tmp_path / "product_lock.json"
            inventory = blind_validation.build_sar_inventory(
                public_path,
                tmp_path / "inventory.json",
                lock_output_path=lock_path,
                searcher=mock_searcher,
                load_dotenv=False,
            )
            self.assertEqual(len(observed_search_kwargs), 1)
            self.assertEqual(len(inventory["targets"]), 2)
            by_arm = {target["comparison_arm"]: target for target in inventory["targets"]}
            self.assertEqual(by_arm["sentinel1_cband_iw"]["provider"], "sentinel1_asf")
            self.assertEqual(by_arm["sentinel1_cband_iw"]["selected_product_ids"], ["S1_ARM_PRODUCT_ID"])
            self.assertEqual(
                by_arm["umbra_xband_spotlight_placeholder"]["status"],
                "placeholder_not_searched",
            )
            self.assertTrue(by_arm["umbra_xband_spotlight_placeholder"]["placeholder_only"])
            self.assertFalse(by_arm["umbra_xband_spotlight_placeholder"]["real_lock_supported"])

            with open(lock_path, "r", encoding="utf-8") as f:
                lock = json.load(f)
            lock_by_arm = {target["comparison_arm"]: target for target in lock["targets"]}
            self.assertEqual(lock_by_arm["sentinel1_cband_iw"]["provider"], "sentinel1_asf")
            self.assertEqual(lock_by_arm["umbra_xband_spotlight_placeholder"]["provider"], "umbra_open_data")
            self.assertTrue(lock_by_arm["umbra_xband_spotlight_placeholder"]["placeholder_only"])

    def test_public_manifest_rejects_label_like_site_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for forbidden_key in ("site_class", "label_class", "expected_site_class", "is_positive"):
                malformed_path = Path(tmp) / f"malformed_{forbidden_key}.json"
                _write_json(
                    malformed_path,
                    {
                        "schema_version": blind_validation.PUBLIC_SCHEMA_VERSION,
                        "validation_id": f"bad_{forbidden_key}",
                        "targets": [
                            {
                                "target_id": "leaky",
                                "name": "Leaky",
                                "center": {"lat": 0.0, "lon": 0.0},
                                forbidden_key: "positive",
                            }
                        ],
                    },
                )
                with self.assertRaises(blind_validation.ManifestValidationError):
                    blind_validation.load_public_manifest(malformed_path)

    def test_campaign_public_metadata_is_accepted_and_preserved_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(csv_path, [])
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "campaign_pair_001_a",
                        "name": "Campaign Pair 001 A",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "existing_candidates_csv": "detected_anomalies.csv",
                        "split": "holdout",
                        "split_designation": "primary_holdout_candidate",
                        "pair_id": "campaign_pair_001",
                        "group_id": "campaign_group_public_strata_a",
                        "public_site_category": "broad_public_context_region",
                        "public_strata": {
                            "terrain": "plateau",
                            "land_cover": "mixed",
                            "lithology": "public_carbonate_context",
                        },
                        "acquisition_stratum": "sentinel1_cband_iw_temperate",
                        "terrain_descriptor": "moderate relief public descriptor",
                        "land_cover_descriptor": "mixed public descriptor",
                        "lithology_descriptor": "regional public descriptor",
                        "campaign_tier": "primary_holdout_candidate",
                        "prior_run_status": "new_public_campaign_target",
                        "audit_only": False,
                        "prior_run_audit_only": False,
                        "public_scoring_status": "eligible_after_registry_lock_and_private_label_custody",
                        "sar_search": {
                            "provider": "sentinel1_asf",
                            "comparison_arm": "sentinel1_cband_iw",
                            "selection_count": 1,
                        },
                    }
                ],
            )
            manifest = blind_validation.load_public_manifest(public_path)
            target = manifest["targets"][0]
            self.assertEqual(target["split_designation"], "primary_holdout_candidate")
            self.assertEqual(target["pair_id"], "campaign_pair_001")
            self.assertEqual(target["public_strata"]["lithology"], "public_carbonate_context")
            self.assertFalse(target["audit_only"])
            self.assertNotIn("site_class", json.dumps(manifest))

            run_manifest = blind_validation.run_blind_validation(public_path, tmp_path / "run", dry_run=True)
            run_target = run_manifest["targets"][0]
            self.assertEqual(run_target["split_designation"], "primary_holdout_candidate")
            self.assertEqual(run_target["public_campaign_metadata"]["pair_id"], "campaign_pair_001")

            labels_path = self._labels(
                tmp,
                [{"target_id": "campaign_pair_001_a", "site_class": "negative", "known_voids": []}],
            )
            report = blind_validation.score_blind_validation(tmp_path / "run" / "run_manifest.json", labels_path)
            scored = report["targets"][0]
            self.assertEqual(scored["split"], "holdout")
            self.assertEqual(scored["pair_id"], "campaign_pair_001")
            self.assertEqual(scored["public_site_category"], "broad_public_context_region")
            self.assertEqual(report["summary"]["metadata_counts"]["campaign_tier"], {"primary_holdout_candidate": 1})
            self.assertEqual(report["summary"]["by_split"]["holdout"]["target_count"], 1)

    def test_public_manifest_rejects_forbidden_leakage_keys_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            leakage_cases = [
                ("truthGeometry", {"type": "Point"}),
                ("knownCaveGeometry", {"coordinates": []}),
                ("exact_void_coordinates", [1.0, 2.0]),
                ("privateLabelPath", "private/labels.json"),
                ("expectedDepthM", 100.0),
                ("positiveClass", True),
                ("custody_record", {"id": "private"}),
            ]
            for forbidden_key, forbidden_value in leakage_cases:
                malformed_path = Path(tmp) / f"recursive_{forbidden_key}.json"
                _write_json(
                    malformed_path,
                    {
                        "schema_version": blind_validation.PUBLIC_SCHEMA_VERSION,
                        "validation_id": f"bad_recursive_{forbidden_key}",
                        "targets": [
                            {
                                "target_id": "leaky",
                                "name": "Leaky",
                                "center": {"lat": 0.0, "lon": 0.0},
                                "public_strata": {
                                    "safe_context": "public",
                                    "nested": {forbidden_key: forbidden_value},
                                },
                            }
                        ],
                    },
                )
                with self.assertRaises(blind_validation.ManifestValidationError, msg=forbidden_key):
                    blind_validation.load_public_manifest(malformed_path, allow_templates=True)

    def test_campaign_templates_validate_with_allow_templates_and_private_fields_stay_private(self):
        campaign_path = Path("validation_examples/public_manifest_campaign_scaffold.json")
        withheld_template_path = Path("validation_examples/withheld_labels_template.json")
        campaign = blind_validation.load_public_manifest(campaign_path, allow_templates=True)
        self.assertTrue(campaign["template_only"])
        campaign_text = json.dumps(campaign, sort_keys=True)
        for private_key in (
            "site_class",
            "known_voids",
            "label_provenance",
            "custody_record",
            "scoring_eligibility",
            "release_metadata",
            "withheld_labels_path",
        ):
            self.assertNotIn(private_key, campaign_text)

        with self.assertRaisesRegex(blind_validation.ManifestValidationError, "template_only"):
            blind_validation.load_public_manifest(campaign_path)
        labels_template = blind_validation.load_withheld_labels(withheld_template_path, allow_templates=True)
        self.assertTrue(labels_template["template_only"])
        self.assertIn("custody_metadata", labels_template)
        self.assertIn("site_subclass", labels_template["labels"][0])
        self.assertIn("label_provenance", labels_template["labels"][0])
        self.assertIn("custody_record", labels_template["labels"][0])
        with self.assertRaisesRegex(blind_validation.ManifestValidationError, "template_only"):
            blind_validation.load_withheld_labels(withheld_template_path)

        with tempfile.TemporaryDirectory() as leak_tmp:
            public_with_private_path = Path(leak_tmp) / "public_with_private.json"
            public_copy = dict(campaign)
            public_copy["template_only"] = True
            public_copy["targets"] = [dict(campaign["targets"][0])]
            public_copy["targets"][0]["scoring_eligibility"] = labels_template["labels"][0]["scoring_eligibility"]
            _write_json(public_with_private_path, public_copy)
            with self.assertRaises(blind_validation.ManifestValidationError):
                blind_validation.load_public_manifest(public_with_private_path, allow_templates=True)

    def test_mammoth_audit_only_split_does_not_imply_primary_holdout_scoring(self):
        campaign = blind_validation.load_public_manifest(
            Path("validation_examples/public_manifest_campaign_scaffold.json"),
            allow_templates=True,
        )
        mammoth_targets = [target for target in campaign["targets"] if "mammoth" in target["target_id"]]
        self.assertEqual(len(mammoth_targets), 1)
        mammoth = mammoth_targets[0]
        self.assertEqual(mammoth["split"], "audit")
        self.assertEqual(mammoth["split_designation"], "audit_only_prior_run")
        self.assertEqual(mammoth["campaign_tier"], "audit_calibration_only")
        self.assertEqual(mammoth["prior_run_status"], "prior_run_public_target_audit_only")
        self.assertTrue(mammoth["audit_only"])
        self.assertTrue(mammoth["prior_run_audit_only"])
        self.assertEqual(mammoth["public_scoring_status"], "not_primary_holdout_scoring_site")
        self.assertTrue(mammoth["public_scoring_status"].startswith("not_"))

        primary_holdout_targets = [
            target
            for target in campaign["targets"]
            if target.get("split") == "holdout" and not target.get("audit_only")
        ]
        self.assertTrue(primary_holdout_targets)
        self.assertNotIn(mammoth["target_id"], {target["target_id"] for target in primary_holdout_targets})

    def test_required_product_lock_verifies_and_passes_locked_product_to_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "locked_site",
                        "name": "Locked Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "buffer_deg": 0.01,
                        "sar_search": {"selection_count": 1},
                    }
                ],
            )
            public_manifest = blind_validation.load_public_manifest(public_path)
            locked_product = {
                "product_id": "LOCKED_PRODUCT_ID",
                "product_name": "LOCKED_PRODUCT_NAME",
                "granule_name": "LOCKED_PRODUCT_NAME",
                "start_time": "2023-01-01T00:00:00Z",
                "processing_level": "SLC",
            }
            lock_path = tmp_path / "product_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": blind_validation.PRODUCT_LOCK_SCHEMA_VERSION,
                    "validation_id": "unit_blind_validation",
                    "public_manifest_sha256": blind_validation._sha256_file(public_path),
                    "selection_policy": blind_validation.PRODUCT_SELECTION_POLICY,
                    "targets": [
                        {
                            "target_id": "locked_site",
                            "status": "success",
                            "selected_product_ids": ["LOCKED_PRODUCT_ID"],
                            "selected_products": [locked_product],
                        }
                    ],
                },
            )
            verification = blind_validation.verify_product_lock_for_manifest(
                lock_path,
                public_manifest,
                public_path,
                require_single_product_per_real_target=True,
            )
            self.assertTrue(verification["public_manifest_sha256_matches"])

            observed_kwargs = {}
            out_csv = tmp_path / "pipeline_candidates.csv"
            _write_candidates(out_csv, [])

            def mock_pipeline(**kwargs):
                observed_kwargs.update(kwargs)
                return {"status": "success", "outputs": {"anomaly_catalog": str(out_csv)}}

            run_manifest = blind_validation.run_blind_validation(
                public_path,
                tmp_path / "run",
                dry_run=False,
                allow_real_downloads=True,
                product_lock_path=lock_path,
                require_product_lock=True,
                confirm_real_downloads_and_training=True,
                pipeline_executor=mock_pipeline,
            )
            self.assertTrue(observed_kwargs["require_locked_sentinel1"])
            self.assertEqual(
                observed_kwargs["locked_sentinel1_products"][0]["product_id"],
                "LOCKED_PRODUCT_ID",
            )
            self.assertEqual(run_manifest["product_lock_sha256"], blind_validation._sha256_file(lock_path))
            self.assertTrue(run_manifest["targets"][0]["product_lock_enforced"])
            self.assertEqual(run_manifest["targets"][0]["locked_product_ids"], ["LOCKED_PRODUCT_ID"])

    def test_required_product_lock_rejects_multi_product_real_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "multi_locked_site",
                        "name": "Multi Locked Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "buffer_deg": 0.01,
                    }
                ],
            )
            lock_path = tmp_path / "product_lock.json"
            _write_json(
                lock_path,
                {
                    "schema_version": blind_validation.PRODUCT_LOCK_SCHEMA_VERSION,
                    "validation_id": "unit_blind_validation",
                    "public_manifest_sha256": blind_validation._sha256_file(public_path),
                    "selection_policy": blind_validation.PRODUCT_SELECTION_POLICY,
                    "targets": [
                        {
                            "target_id": "multi_locked_site",
                            "status": "success",
                            "selected_product_ids": ["P1", "P2"],
                            "selected_products": [
                                {"product_id": "P1", "product_name": "P1"},
                                {"product_id": "P2", "product_name": "P2"},
                            ],
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "exactly one"):
                blind_validation.run_blind_validation(
                    public_path,
                    tmp_path / "run",
                    dry_run=False,
                    allow_real_downloads=True,
                    product_lock_path=lock_path,
                    require_product_lock=True,
                    confirm_real_downloads_and_training=True,
                    pipeline_executor=lambda **kwargs: {},
                )

    def test_real_execution_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "confirm_real_site",
                        "name": "Confirm Real Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "buffer_deg": 0.01,
                    }
                ],
            )

            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "confirm-real-downloads-and-training"):
                blind_validation.run_blind_validation(
                    public_path,
                    tmp_path / "run",
                    dry_run=False,
                    allow_real_downloads=True,
                    pipeline_executor=lambda **kwargs: {},
                )

    def test_campaign_plan_is_deterministic_strict_json_and_excludes_audit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "holdout_site",
                        "name": "Holdout Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "split": "holdout",
                        "public_scoring_status": "eligible_after_registry_lock_and_private_label_custody",
                        "sar_search": {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                    },
                    {
                        "target_id": "audit_site",
                        "name": "Audit Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.1, "lon": -118.1},
                        "split": "audit",
                        "audit_only": True,
                        "prior_run_audit_only": True,
                        "public_scoring_status": "not_primary_holdout_scoring_site",
                        "sar_search": {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                    },
                    {
                        "target_id": "umbra_site",
                        "name": "Umbra Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.2, "lon": -118.2},
                        "split": "holdout",
                        "comparison_arms": [
                            {"provider": "umbra_open_data", "comparison_arm": "umbra_xband_spotlight_placeholder"}
                        ],
                    },
                ],
            )
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )

            first = blind_validation.plan_campaign_execution(
                public_path,
                registry_path,
                params_path,
                tmp_path / "campaign_run",
            )
            second = blind_validation.plan_campaign_execution(
                public_path,
                registry_path,
                params_path,
                tmp_path / "campaign_run",
            )
            self.assertEqual(first, second)
            json.loads(json.dumps(first, sort_keys=True, allow_nan=False))
            self.assertEqual(first["schema_version"], blind_validation.CAMPAIGN_EXECUTION_PLAN_SCHEMA_VERSION)
            by_target = {step["target_id"]: step for step in first["steps"]}
            self.assertTrue(by_target["holdout_site"]["executable"])
            self.assertEqual(by_target["audit_site"]["status"], "skipped_audit_only")
            self.assertEqual(by_target["umbra_site"]["status"], "skipped_unsupported_provider")
            self.assertTrue(first["run_policy"]["synthetic_fallback_refused_by_default"])
            self.assertFalse(first["run_policy"]["downloads_attempted_by_plan"])
            self.assertIn("--target-id", by_target["holdout_site"]["command"])

    def test_campaign_guards_refuse_synthetic_real_without_confirmation_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "holdout_site",
                        "name": "Holdout Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "split": "holdout",
                        "sar_search": {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                    }
                ],
            )
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )
            lock_path = _write_product_lock(
                tmp_path / "product_lock.json",
                public_path,
                ["holdout_site"],
                parameter_set_path=params_path,
                registry_path=registry_path,
            )
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "synthetic fallback"):
                blind_validation.plan_campaign_execution(
                    public_path,
                    registry_path,
                    params_path,
                    tmp_path / "campaign_run",
                    allow_synthetic_fallback=True,
                )
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "confirm-real"):
                blind_validation.plan_campaign_execution(
                    public_path,
                    registry_path,
                    params_path,
                    tmp_path / "campaign_run",
                    product_lock_path=lock_path,
                    execute_real=True,
                )

            changed_public_path = tmp_path / "public_changed.json"
            changed = json.loads(public_path.read_text(encoding="utf-8"))
            changed["targets"][0]["buffer_deg"] = 0.2
            _write_json(changed_public_path, changed)
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "drift"):
                blind_validation.plan_campaign_execution(
                    changed_public_path,
                    registry_path,
                    params_path,
                    tmp_path / "campaign_run",
                )

            draft_registry_path = _write_campaign_registry(
                tmp_path / "draft_registry.json",
                public_path,
                params_path,
                status="draft",
            )
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "not approved|not locked"):
                blind_validation.plan_campaign_execution(
                    public_path,
                    draft_registry_path,
                    params_path,
                    tmp_path / "campaign_run",
                )

    def test_campaign_run_status_and_no_label_package_with_dummy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "site_a",
                        "name": "Site A",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "split": "holdout",
                        "existing_candidates_csv": "site_a.csv",
                    },
                    {
                        "target_id": "site_b",
                        "name": "Site B",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 34.1, "lon": -118.1},
                        "split": "holdout",
                        "existing_candidates_csv": "site_b.csv",
                    },
                ],
            )
            _write_candidates(tmp_path / "site_a.csv", [{"id": "a", "centroid_m": "[0,0,10]"}])
            _write_candidates(tmp_path / "site_b.csv", [])
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )
            plan = blind_validation.plan_campaign_execution(
                public_path,
                registry_path,
                params_path,
                tmp_path / "campaign_run",
            )
            plan_path = tmp_path / "campaign_plan.json"
            blind_validation._write_json(plan_path, plan)
            status = blind_validation.run_campaign_execution_plan(plan_path)
            self.assertEqual(status["schema_version"], blind_validation.CAMPAIGN_EXECUTION_STATUS_SCHEMA_VERSION)
            self.assertEqual(status["summary"]["completed_targets"], 2)
            self.assertEqual(status["summary"]["total_candidate_count"], 1)
            resumed = blind_validation.run_campaign_execution_plan(plan_path)
            self.assertTrue(all(result["status"] == "resumed_existing" for result in resumed["run_results"]))

            status_path = tmp_path / "campaign_status.json"
            blind_validation._write_json(status_path, status)
            package_path = tmp_path / "campaign_evidence.json"
            evidence = blind_validation.package_campaign_no_label_evidence(
                plan_path,
                package_path,
                status_path=status_path,
            )
            self.assertTrue(package_path.exists())
            self.assertEqual(evidence["schema_version"], blind_validation.CAMPAIGN_EVIDENCE_PACKAGE_SCHEMA_VERSION)
            self.assertFalse(evidence["labels_included"])
            self.assertFalse(evidence["withheld_labels_loaded"])
            self.assertEqual(evidence["summary"]["completed_targets"], 2)
            self.assertTrue(all("run_manifest_sha256" in target for target in evidence["targets"]))

    def test_robustness_plan_validates_hash_and_rejects_label_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = _write_robustness_plan(tmp_path / "robustness_plan.json")

            first = blind_validation.load_robustness_plan(plan_path)
            second = blind_validation.load_robustness_plan(plan_path)
            self.assertEqual(first["schema_version"], blind_validation.ROBUSTNESS_PLAN_SCHEMA_VERSION)
            self.assertEqual(first["robustness_plan_hash"], second["robustness_plan_hash"])
            self.assertEqual(first["robustness_plan_id"], second["robustness_plan_id"])
            self.assertTrue(first["robustness_plan_id"].startswith(blind_validation.ROBUSTNESS_PLAN_ID_PREFIX))
            self.assertFalse(first["withheld_labels_loaded"])
            self.assertTrue(all(variant["scope"] == "calibration_only" for variant in first["variants"]))
            families = {variant["family"] for variant in first["variants"]}
            self.assertIn("void_threshold_sweep", families)
            self.assertIn("segmentation_topology", families)
            self.assertIn("null_region_random_spatial_baseline", families)
            self.assertIn("repeat_product_date_stability", families)

            leaky_path = tmp_path / "leaky_robustness_plan.json"
            leaky = json.loads(plan_path.read_text(encoding="utf-8"))
            leaky["variants"][0]["known_voids"] = []
            _write_json(leaky_path, leaky)
            with self.assertRaises(blind_validation.ManifestValidationError):
                blind_validation.load_robustness_plan(leaky_path)

            holdout_path = tmp_path / "holdout_ablation_plan.json"
            holdout_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            holdout_plan["variants"][1]["scope"] = "holdout"
            _write_json(holdout_path, holdout_plan)
            with self.assertRaisesRegex(blind_validation.ManifestValidationError, "calibration_only"):
                blind_validation.load_robustness_plan(holdout_path)

    def test_committed_robustness_plan_template_validates_without_private_fields(self):
        template_path = Path("validation_examples/robustness_ablation_plan_template.json")
        plan = blind_validation.load_robustness_plan(template_path, allow_templates=True)
        self.assertTrue(plan["template_only"])
        self.assertEqual(plan["schema_version"], blind_validation.ROBUSTNESS_PLAN_SCHEMA_VERSION)
        self.assertTrue(plan["robustness_plan_id"].startswith(blind_validation.ROBUSTNESS_PLAN_ID_PREFIX))
        self.assertFalse(plan["withheld_labels_loaded"])
        plan_text = json.dumps(plan, sort_keys=True).lower()
        for private_key in ("site_class", "known_voids", "withheld_labels_path", "private_label_path"):
            self.assertNotIn(private_key, plan_text)
        with self.assertRaisesRegex(blind_validation.ManifestValidationError, "template_only"):
            blind_validation.load_robustness_plan(template_path)

    def test_robustness_dry_run_plan_is_deterministic_and_links_campaign_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "holdout_site",
                        "name": "Holdout Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.0, "lon": -118.0},
                        "split": "holdout",
                        "public_scoring_status": "eligible_after_registry_lock_and_private_label_custody",
                        "sar_search": {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                    },
                    {
                        "target_id": "calibration_site",
                        "name": "Calibration Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 34.1, "lon": -118.1},
                        "split": "calibration",
                        "existing_candidates_csv": "calibration.csv",
                    },
                    {
                        "target_id": "audit_site",
                        "name": "Audit Site",
                        "data_mode": "real_slc",
                        "center": {"lat": 34.2, "lon": -118.2},
                        "split": "audit",
                        "audit_only": True,
                        "public_scoring_status": "not_primary_holdout_scoring_site",
                    },
                ],
            )
            _write_candidates(tmp_path / "calibration.csv", [])
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )
            robustness_path = _write_robustness_plan(tmp_path / "robustness_plan.json")

            first = blind_validation.plan_robustness_ablations(
                robustness_path,
                public_path,
                registry_path,
                params_path,
                tmp_path / "robustness_outputs",
            )
            second = blind_validation.plan_robustness_ablations(
                robustness_path,
                public_path,
                registry_path,
                params_path,
                tmp_path / "robustness_outputs",
            )
            self.assertEqual(first, second)
            json.loads(json.dumps(first, sort_keys=True, allow_nan=False))
            self.assertEqual(first["schema_version"], blind_validation.ROBUSTNESS_EXECUTION_PLAN_SCHEMA_VERSION)
            self.assertFalse(first["run_policy"]["downloads_attempted_by_plan"])
            self.assertFalse(first["run_policy"]["training_attempted_by_plan"])
            self.assertEqual(first["campaign_registry_hash"], blind_validation.load_campaign_registry(registry_path)["registry_hash"])
            self.assertEqual(first["parameter_set_hash"], blind_validation.load_parameter_set(params_path)["parameter_set_hash"])
            self.assertEqual(first["summary"]["target_count"], 1)
            self.assertEqual(first["summary"]["variant_count"], len(_robustness_variants()))
            self.assertEqual(first["summary"]["planned_step_count"], len(_robustness_variants()))
            self.assertTrue(all(step["target_id"] == "calibration_site" for step in first["steps"]))
            self.assertTrue(all(step["execution_mode"] == "dry_run_plan_only" for step in first["steps"]))
            self.assertTrue(all("--variant-id" in step["command"] for step in first["steps"]))
            self.assertTrue(any(step["null_baseline"] for step in first["steps"]))

    def test_robustness_summary_groups_fixture_scores_by_variant_provider_arm_and_null_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            score_a = {
                "schema_version": blind_validation.SCORE_REPORT_SCHEMA_VERSION,
                "validation_id": "unit_blind_validation",
                "robustness_variant": {
                    "variant_id": "void_threshold_low",
                    "family": "void_threshold_sweep",
                    "arm": "threshold_sensitivity",
                    "null_baseline": False,
                },
                "summary": {"target_count": 1},
                "targets": [
                    {
                        "target_id": "positive_site",
                        "site_class": "positive",
                        "provider": "sentinel1_asf",
                        "comparison_arm": "sentinel1_cband_iw",
                        "split": "calibration",
                        "candidate_count": 1,
                        "matched_candidate_count": 1,
                        "known_void_count": 1,
                        "matched_known_void_count": 1,
                        "positive_site_hit": True,
                        "positive_site_miss": False,
                        "negative_site_false_positive": False,
                    }
                ],
            }
            score_b = {
                "schema_version": blind_validation.SCORE_REPORT_SCHEMA_VERSION,
                "validation_id": "unit_blind_validation",
                "ablation_variant": {
                    "variant_id": "null_random_spatial",
                    "family": "null_region_random_spatial_baseline",
                    "arm": "null_baseline",
                    "null_baseline": True,
                },
                "summary": {"target_count": 1},
                "targets": [
                    {
                        "target_id": "negative_site",
                        "site_class": "negative",
                        "provider": "umbra_open_data",
                        "comparison_arm": "umbra_xband_spotlight_placeholder",
                        "split": "calibration",
                        "candidate_count": 2,
                        "matched_candidate_count": 0,
                        "known_void_count": 0,
                        "matched_known_void_count": 0,
                        "positive_site_hit": False,
                        "positive_site_miss": False,
                        "negative_site_false_positive": True,
                    }
                ],
            }
            score_a_path = tmp_path / "score_a.json"
            score_b_path = tmp_path / "score_b.json"
            _write_json(score_a_path, score_a)
            _write_json(score_b_path, score_b)

            summary = blind_validation.build_robustness_summary([score_a_path, score_b_path])
            self.assertEqual(summary["schema_version"], blind_validation.ROBUSTNESS_SUMMARY_SCHEMA_VERSION)
            self.assertEqual(summary["summary"]["score_report_count"], 2)
            self.assertEqual(summary["summary"]["by_variant_id"]["void_threshold_low"]["positive_site_hits"], 1)
            self.assertEqual(summary["summary"]["by_provider"]["umbra_open_data"]["negative_false_positive_sites"], 1)
            self.assertEqual(summary["summary"]["by_comparison_arm"]["sentinel1_cband_iw"]["target_count"], 1)
            self.assertEqual(summary["summary"]["by_null_baseline"]["true"]["negative_false_positive_sites"], 1)
            self.assertEqual(summary["summary"]["by_family"]["void_threshold_sweep"]["positive_site_hits"], 1)

    def test_robustness_cli_commands_emit_strict_json_and_geoanomaly_delegates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "calibration_site",
                        "name": "Calibration Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 34.1, "lon": -118.1},
                        "split": "calibration",
                        "existing_candidates_csv": "calibration.csv",
                    }
                ],
            )
            _write_candidates(tmp_path / "calibration.csv", [])
            params_path = _write_parameter_set(tmp_path / "approved_parameters.json", approved=True)
            registry_path = _write_campaign_registry(
                tmp_path / "campaign_registry.json",
                public_path,
                params_path,
                status="locked",
            )
            robustness_path = _write_robustness_plan(tmp_path / "robustness_plan.json")

            validate_stdout = io.StringIO()
            with contextlib.redirect_stdout(validate_stdout):
                exit_code = blind_validation.main(["validate-robustness-plan", "--plan", str(robustness_path)])
            self.assertEqual(exit_code, 0)
            validate_payload = json.loads(validate_stdout.getvalue())
            self.assertEqual(validate_payload["schema_version"], blind_validation.ROBUSTNESS_PLAN_VALIDATION_SCHEMA_VERSION)
            self.assertTrue(validate_payload["valid"])

            plan_stdout = io.StringIO()
            with contextlib.redirect_stdout(plan_stdout):
                exit_code = blind_validation.main(
                    [
                        "robustness-plan",
                        "--plan",
                        str(robustness_path),
                        "--manifest",
                        str(public_path),
                        "--campaign-registry",
                        str(registry_path),
                        "--parameter-set",
                        str(params_path),
                        "--output-dir",
                        str(tmp_path / "robustness_outputs"),
                    ]
                )
            self.assertEqual(exit_code, 0)
            plan_payload = json.loads(plan_stdout.getvalue())
            self.assertEqual(plan_payload["schema_version"], blind_validation.ROBUSTNESS_EXECUTION_PLAN_SCHEMA_VERSION)
            self.assertFalse(plan_payload["run_policy"]["downloads_attempted_by_plan"])

            score_path = tmp_path / "score.json"
            _write_json(
                score_path,
                {
                    "schema_version": blind_validation.SCORE_REPORT_SCHEMA_VERSION,
                    "validation_id": "unit_blind_validation",
                    "robustness_variant": {"variant_id": "baseline", "family": "baseline_reference", "arm": "baseline"},
                    "targets": [
                        {
                            "target_id": "calibration_site",
                            "site_class": "negative",
                            "provider": "sentinel1_asf",
                            "comparison_arm": "sentinel1_cband_iw",
                            "candidate_count": 0,
                            "matched_candidate_count": 0,
                            "known_void_count": 0,
                            "matched_known_void_count": 0,
                            "positive_site_hit": False,
                            "positive_site_miss": False,
                            "negative_site_false_positive": False,
                        }
                    ],
                },
            )
            summary_stdout = io.StringIO()
            with contextlib.redirect_stdout(summary_stdout):
                exit_code = blind_validation.main(["robustness-summary", "--score", str(score_path)])
            self.assertEqual(exit_code, 0)
            summary_payload = json.loads(summary_stdout.getvalue())
            self.assertEqual(summary_payload["schema_version"], blind_validation.ROBUSTNESS_SUMMARY_SCHEMA_VERSION)

            commands_stdout = io.StringIO()
            with contextlib.redirect_stdout(commands_stdout):
                self.assertEqual(geoanomaly.main(["commands"]), 0)
            self.assertIn("robustness-plan", commands_stdout.getvalue())

            wrapper_stdout = io.StringIO()
            with contextlib.redirect_stdout(wrapper_stdout):
                exit_code = geoanomaly.main(["validation", "validate-robustness-plan", "--plan", str(robustness_path)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(wrapper_stdout.getvalue())["schema_version"], blind_validation.ROBUSTNESS_PLAN_VALIDATION_SCHEMA_VERSION)

            run_stdout = io.StringIO()
            with contextlib.redirect_stdout(run_stdout):
                exit_code = blind_validation.main(
                    [
                        "run",
                        "--manifest",
                        str(public_path),
                        "--output-dir",
                        str(tmp_path / "variant_run"),
                        "--parameter-set",
                        str(params_path),
                        "--require-approved-parameters",
                        "--campaign-registry",
                        str(registry_path),
                        "--require-locked-campaign-registry",
                        "--robustness-plan",
                        str(robustness_path),
                        "--variant-id",
                        "baseline_locked_parameters",
                    ]
                )
            self.assertEqual(exit_code, 0)
            variant_manifest = json.loads((tmp_path / "variant_run" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(variant_manifest["robustness_variant_id"], "baseline_locked_parameters")
            self.assertEqual(variant_manifest["robustness_variant"]["family"], "baseline_reference")

    def test_run_manifest_has_reproducibility_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(csv_path, [])
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "repro_site",
                        "name": "Repro Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            run_manifest = blind_validation.run_blind_validation(
                public_path,
                tmp_path / "run",
                dry_run=True,
                command_args={"command": "unit", "secret_token": "do_not_record"},
            )
            repro = run_manifest["reproducibility"]
            self.assertEqual(repro["public_manifest_sha256"], blind_validation._sha256_file(public_path))
            self.assertIn("blind_validation.py", repro["code_fingerprints"])
            self.assertIn("python_version", repro["runtime"])
            self.assertFalse(repro["random_seed_policy"]["deterministic_seed_set_by_runner"])
            self.assertTrue(repro["no_synthetic_fallback_status"]["synthetic_fallback_disabled"])
            self.assertEqual(repro["command_args"]["secret_token"], "<redacted>")

    def test_baseline_report_aggregates_score_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            positive_csv = tmp_path / "positive.csv"
            negative_csv = tmp_path / "negative.csv"
            _write_candidates(
                positive_csv,
                [
                    {
                        "id": "hit",
                        "centroid_m": "[0.0, 0.0, 100.0]",
                        "depth_m": "100.0",
                        "fused_confidence_score": "0.90",
                        "deep_target_rank": "1",
                    }
                ],
            )
            _write_candidates(
                negative_csv,
                [
                    {
                        "id": "fp",
                        "centroid_m": "[10.0, 0.0, 50.0]",
                        "depth_m": "50.0",
                        "fused_confidence_score": "0.20",
                        "deep_target_rank": "1",
                    }
                ],
            )
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "positive_site",
                        "name": "Positive Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "area_km2": 0.5,
                        "split": "holdout",
                        "sar_search": {
                            "provider": "sentinel1_asf",
                            "comparison_arm": "sentinel1_cband_iw",
                        },
                        "existing_candidates_csv": "positive.csv",
                    },
                    {
                        "target_id": "negative_site",
                        "name": "Negative Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 36.0, "lon": -118.0},
                        "area_km2": 2.0,
                        "split": "holdout",
                        "sar_search": {
                            "provider": "umbra_open_data",
                            "comparison_arm": "umbra_xband_spotlight_placeholder",
                        },
                        "existing_candidates_csv": "negative.csv",
                    },
                ],
            )
            blind_validation.run_blind_validation(public_path, tmp_path / "run", dry_run=True)
            labels_path = self._labels(
                tmp,
                [
                    {
                        "target_id": "positive_site",
                        "site_class": "positive",
                        "known_voids": [
                            {
                                "void_id": "known_void",
                                "offset_m": [0.0, 0.0],
                                "depth_m": 100.0,
                                "horizontal_tolerance_m": 10.0,
                                "depth_tolerance_m": 10.0,
                            }
                        ],
                    },
                    {"target_id": "negative_site", "site_class": "negative", "known_voids": []},
                ],
            )
            score_path = tmp_path / "score.json"
            blind_validation.score_blind_validation(tmp_path / "run" / "run_manifest.json", labels_path, score_path)
            report = blind_validation.build_baseline_report([score_path], confidence_bins=5)
            self.assertEqual(report["schema_version"], blind_validation.BASELINE_REPORT_SCHEMA_VERSION)
            summary = report["summary"]
            self.assertEqual(summary["score_report_count"], 1)
            self.assertEqual(summary["positive_site_hits"], 1)
            self.assertEqual(summary["negative_false_positive_sites"], 1)
            self.assertEqual(summary["by_split"]["holdout"]["target_count"], 2)
            self.assertEqual(summary["by_provider"]["sentinel1_asf"]["positive_site_hits"], 1)
            self.assertEqual(summary["by_provider"]["umbra_open_data"]["negative_false_positive_sites"], 1)
            self.assertEqual(summary["by_comparison_arm"]["sentinel1_cband_iw"]["target_count"], 1)
            self.assertEqual(
                summary["metadata_counts"]["comparison_arm"]["umbra_xband_spotlight_placeholder"],
                1,
            )
            self.assertEqual(report["targets"][0]["provider"], "sentinel1_asf")
            self.assertAlmostEqual(
                summary["false_positives_per_area"]["negative_false_positive_candidates_per_km2"],
                0.5,
            )
            self.assertEqual(summary["rank_of_first_hit_distribution"]["histogram"], {"1": 1})
            self.assertEqual(summary["localization_error_m_summary"]["median"], 0.0)
            bins = summary["confidence_calibration_bins"]
            self.assertTrue(any(bucket["candidate_count"] > 0 for bucket in bins))
            text = blind_validation.format_baseline_report_text(report)
            self.assertIn("Blind validation baseline report", text)
            self.assertIn("Hit rate by provider", text)
            self.assertIn("Hit rate by comparison arm", text)

    def test_report_package_is_deterministic_redacted_and_claim_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "detected_anomalies.csv"
            _write_candidates(
                csv_path,
                [
                    {
                        "id": "hit",
                        "centroid_m": "[0.0, 0.0, 100.0]",
                        "depth_m": "100.0",
                        "fused_confidence_score": "0.90",
                        "deep_target_rank": "1",
                    }
                ],
            )
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "package_site",
                        "name": "Package Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                        "area_km2": 0.5,
                        "domain_width_m": 800.0,
                        "existing_candidates_csv": "detected_anomalies.csv",
                    }
                ],
            )
            params_path = _write_parameter_set(tmp_path / "parameters.json", approved=True)
            run_dir = tmp_path / "run"
            blind_validation.run_blind_validation(
                public_path,
                run_dir,
                dry_run=True,
                parameter_set_path=params_path,
                require_approved_parameters=True,
            )
            labels_path = self._labels(
                tmp,
                [
                    {
                        "target_id": "package_site",
                        "site_class": "positive",
                        "known_voids": [
                            {
                                "void_id": "known_void",
                                "offset_m": [0.0, 0.0],
                                "depth_m": 100.0,
                                "horizontal_tolerance_m": 10.0,
                                "depth_tolerance_m": 10.0,
                            }
                        ],
                    }
                ],
            )
            score_path = tmp_path / "score.json"
            blind_validation.score_blind_validation(
                run_dir / "run_manifest.json",
                labels_path,
                score_path,
                parameter_set_path=params_path,
                require_approved_parameters=True,
            )
            baseline_path = tmp_path / "baseline.json"
            baseline_text_path = tmp_path / "baseline.txt"
            baseline = blind_validation.build_baseline_report([score_path])
            blind_validation._write_json(baseline_path, baseline)
            baseline_text_path.write_text(
                blind_validation.format_baseline_report_text(baseline),
                encoding="utf-8",
            )
            log_path = tmp_path / "commands.log"
            log_path.write_text(
                "EARTHDATA_TOKEN=super-secret-token\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz\n",
                encoding="utf-8",
            )

            first_dir = tmp_path / "package_a"
            second_dir = tmp_path / "package_b"
            first = blind_validation.package_validation_report(
                first_dir,
                public_manifest=public_path,
                parameter_set=params_path,
                run_manifest=run_dir / "run_manifest.json",
                score_json=score_path,
                baseline_json=baseline_path,
                baseline_text=baseline_text_path,
                command_logs=[log_path],
                notes="api_key=notes-secret-value",
            )
            exit_code = blind_validation.main(
                [
                    "package-report",
                    "--public-manifest",
                    str(public_path),
                    "--parameter-set",
                    str(params_path),
                    "--run-manifest",
                    str(run_dir / "run_manifest.json"),
                    "--score-json",
                    str(score_path),
                    "--baseline-json",
                    str(baseline_path),
                    "--baseline-text",
                    str(baseline_text_path),
                    "--command-log",
                    str(log_path),
                    "--notes",
                    "api_key=notes-secret-value",
                    "--output-dir",
                    str(second_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            for filename in ["validation_summary.json", "methods_limitations_claim_boundary.md", "file_hash_manifest.json"]:
                self.assertEqual(
                    (first_dir / filename).read_bytes(),
                    (second_dir / filename).read_bytes(),
                    filename,
                )

            summary = json.loads((first_dir / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], blind_validation.REPORT_PACKAGE_SCHEMA_VERSION)
            self.assertEqual(summary["claim_boundary"], blind_validation.REPORT_CANDIDATE_CLAIM)
            roles = {artifact["role"] for artifact in summary["artifacts"]}
            self.assertIn("public_manifest", roles)
            self.assertIn("parameter_set", roles)
            self.assertIn("run_manifest", roles)
            self.assertIn("score_json", roles)
            self.assertIn("baseline_json", roles)
            self.assertIn("command_log", roles)
            self.assertIn("notes", roles)
            self.assertIn("EVIDENCE_WITHHELD_LABEL_SCORE_OUTPUT", json.dumps(summary, sort_keys=True))
            self.assertTrue(first["validation_id_consistent"])
            self.assertEqual(summary["findings"]["score_summary"]["positive_site_hits"], 1)

            packaged_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in first_dir.rglob("*")
                if path.is_file()
            )
            self.assertNotIn("super-secret-token", packaged_text)
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz", packaged_text)
            self.assertNotIn("notes-secret-value", packaged_text)
            self.assertIn("<redacted>", packaged_text)
            self.assertIn("candidate detections unless independently field-verified", packaged_text)

            hash_manifest = json.loads((first_dir / "file_hash_manifest.json").read_text(encoding="utf-8"))
            files = {record["path"]: record for record in hash_manifest["files"]}
            self.assertIn("validation_summary.json", files)
            self.assertIn("methods_limitations_claim_boundary.md", files)
            self.assertTrue(any(path.startswith("artifacts/score_json__") for path in files))
            self.assertTrue(all("sha256" in record for record in files.values()))

    def test_top_level_cli_help_commands_and_validation_delegation(self):
        help_text = geoanomaly.build_arg_parser().format_help()
        self.assertIn("validation", help_text)
        self.assertIn("commands", help_text)

        commands_stdout = io.StringIO()
        with contextlib.redirect_stdout(commands_stdout):
            commands_exit = geoanomaly.main(["commands"])
        self.assertEqual(commands_exit, 0)
        self.assertIn("package-report", commands_stdout.getvalue())
        self.assertIn("validation-first", commands_stdout.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            public_path = self._public_manifest(
                tmp,
                [
                    {
                        "target_id": "cli_site",
                        "name": "CLI Site",
                        "data_mode": "existing_candidates",
                        "center": {"lat": 35.0, "lon": -117.0},
                    }
                ],
            )
            validation_stdout = io.StringIO()
            with contextlib.redirect_stdout(validation_stdout):
                validation_exit = geoanomaly.main(
                    ["validation", "validate-public", "--manifest", str(public_path)]
                )
            self.assertEqual(validation_exit, 0)
            self.assertIn("OK public manifest", validation_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
