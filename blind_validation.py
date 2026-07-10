#!/usr/bin/env python3
"""Blind known-void validation harness.

This module keeps blind run generation separate from withheld-label scoring:

* public manifests describe sites and processing inputs only;
* run manifests freeze candidate CSV outputs without loading labels;
* withheld-label manifests are read only by the scorer.

The harness is intentionally standard-library only so validation manifests stay
portable and auditable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from json_utils import dump_strict_json, dumps_strict_json, to_strict_jsonable


PUBLIC_SCHEMA_VERSION = "blind-validation-public-v1"
WITHHELD_LABEL_SCHEMA_VERSION = "blind-validation-labels-v1"
PARAMETER_SET_SCHEMA_VERSION = "blind-validation-parameter-set-v1"
PARAMETER_COMPARISON_SCHEMA_VERSION = "blind-validation-parameter-comparison-v1"
CAMPAIGN_REGISTRY_SCHEMA_VERSION = "blind-validation-campaign-registry-v1"
CAMPAIGN_REGISTRY_COMPARISON_SCHEMA_VERSION = "blind-validation-campaign-registry-comparison-v1"
RUN_MANIFEST_SCHEMA_VERSION = "blind-validation-run-v1"
SCORE_REPORT_SCHEMA_VERSION = "blind-validation-score-v1"
BASELINE_REPORT_SCHEMA_VERSION = "blind-validation-baseline-report-v1"
SAR_INVENTORY_SCHEMA_VERSION = "blind-validation-sar-inventory-v1"
PRODUCT_LOCK_SCHEMA_VERSION = "blind-validation-product-lock-v1"
REPORT_PACKAGE_SCHEMA_VERSION = "blind-validation-report-package-v1"
CAMPAIGN_EXECUTION_PLAN_SCHEMA_VERSION = "blind-validation-campaign-execution-plan-v1"
CAMPAIGN_EXECUTION_STATUS_SCHEMA_VERSION = "blind-validation-campaign-execution-status-v1"
CAMPAIGN_EVIDENCE_PACKAGE_SCHEMA_VERSION = "blind-validation-campaign-evidence-package-v1"
ROBUSTNESS_PLAN_SCHEMA_VERSION = "blind-validation-robustness-plan-v1"
ROBUSTNESS_PLAN_VALIDATION_SCHEMA_VERSION = "blind-validation-robustness-plan-validation-v1"
ROBUSTNESS_EXECUTION_PLAN_SCHEMA_VERSION = "blind-validation-robustness-execution-plan-v1"
ROBUSTNESS_SUMMARY_SCHEMA_VERSION = "blind-validation-robustness-summary-v1"

DEFAULT_HORIZONTAL_TOLERANCE_M = 100.0
DEFAULT_DEPTH_TOLERANCE_M = 150.0
DEFAULT_SAR_SEARCH_START_DATE = "2022-01-01T00:00:00Z"
DEFAULT_SAR_SEARCH_END_DATE = "2024-01-01T00:00:00Z"
DEFAULT_SAR_MAX_RESULTS = 10
DEFAULT_PRODUCT_SELECTION_COUNT = 2
PRODUCT_SELECTION_POLICY = "start_time_desc_product_id_asc"
PARAMETER_SET_ID_PREFIX = "params-"
PARAMETER_SET_ID_HASH_CHARS = 16
CAMPAIGN_REGISTRY_ID_PREFIX = "campaign-registry-"
CAMPAIGN_REGISTRY_ID_HASH_CHARS = 16
ROBUSTNESS_PLAN_ID_PREFIX = "robustness-plan-"
ROBUSTNESS_PLAN_ID_HASH_CHARS = 16
EMPTY_CANDIDATE_FIELDS = [
    "id",
    "centroid_m",
    "depth_m",
    "deep_target_rank",
    "fused_confidence_score",
    "deep_target_score",
    "mean_void_probability",
]

PUBLIC_FORBIDDEN_KEYS = {
    "actual_class",
    "actual_label",
    "binary_label",
    "case_control_status",
    "cave_geometry",
    "cave_map",
    "cave_passage_geometry",
    "caves",
    "class",
    "class_label",
    "classification_label",
    "contains_void",
    "control_class",
    "control_label",
    "control_type",
    "custodian",
    "custody_metadata",
    "custody_record",
    "expected_label",
    "expected_depth",
    "expected_depth_m",
    "expected_depth_range_m",
    "expected_depths",
    "expected_site_class",
    "expected_void_type",
    "evidence_tier",
    "ground_truth",
    "ground_truth_geometry",
    "ground_truth_label",
    "ground_truth_site_class",
    "has_void",
    "known_coordinates",
    "is_negative",
    "is_positive",
    "known_cave",
    "known_cave_geometry",
    "known_caves",
    "known_depth_m",
    "known_feature",
    "known_feature_geometry",
    "known_lat",
    "known_lon",
    "known_void",
    "known_void_coordinates",
    "known_void_lat",
    "known_void_lon",
    "known_voids",
    "label",
    "label_class",
    "label_path",
    "label_paths",
    "label_provenance",
    "labels",
    "labels_path",
    "negative",
    "negative_class",
    "negative_label",
    "positive",
    "positive_class",
    "positive_label",
    "private_label_path",
    "private_label_paths",
    "private_coordinates",
    "release_metadata",
    "scoring_eligibility",
    "site_class",
    "site_label",
    "site_subclass",
    "target_class",
    "true_class",
    "truth",
    "truth_class",
    "truth_coordinates",
    "truth_geometry",
    "truth_label",
    "exact_cave_coordinates",
    "exact_void_coordinates",
    "void_coordinates",
    "void_coordinate",
    "validation_label",
    "void_geometry",
    "voids",
    "withheld_label_file",
    "withheld_label_path",
    "withheld_label_paths",
    "withheld_labels_file",
    "withheld_labels",
    "withheld_labels_path",
}

PUBLIC_FORBIDDEN_KEY_ALIASES = {
    "actualclass",
    "actuallabel",
    "binarylabel",
    "casecontrolstatus",
    "cavegeometry",
    "cavemap",
    "cavepassagegeometry",
    "classlabel",
    "classificationlabel",
    "containsvoid",
    "controlclass",
    "controllabel",
    "controltype",
    "custodian",
    "custodymetadata",
    "custodyrecord",
    "evidencetier",
    "exactcavecoordinates",
    "exactvoidcoordinates",
    "expecteddepth",
    "expecteddepthm",
    "expecteddepthrangem",
    "expecteddepths",
    "expectedlabel",
    "expectedsiteclass",
    "expectedvoidtype",
    "groundtruth",
    "groundtruthgeometry",
    "groundtruthlabel",
    "groundtruthsiteclass",
    "hasvoid",
    "isnegative",
    "ispositive",
    "knowncoordinates",
    "knowncave",
    "knowncavegeometry",
    "knowncaves",
    "knowndepthm",
    "knownfeature",
    "knownfeaturegeometry",
    "knownlat",
    "knownlon",
    "knownvoid",
    "knownvoidcoordinates",
    "knownvoidlat",
    "knownvoidlon",
    "knownvoids",
    "labelclass",
    "labelpath",
    "labelpaths",
    "labelprovenance",
    "labelspath",
    "negativeclass",
    "negativelabel",
    "positiveclass",
    "positivelabel",
    "privatelabelpath",
    "privatelabelpaths",
    "privatecoordinates",
    "releasemetadata",
    "scoringeligibility",
    "siteclass",
    "sitelabel",
    "sitesubclass",
    "targetclass",
    "trueclass",
    "truthclass",
    "truthcoordinates",
    "truthgeometry",
    "truthlabel",
    "validationlabel",
    "voidcoordinate",
    "voidcoordinates",
    "voidgeometry",
    "withheldlabelfile",
    "withheldlabelpath",
    "withheldlabelpaths",
    "withheldlabels",
    "withheldlabelsfile",
    "withheldlabelspath",
}

PUBLIC_CAMPAIGN_METADATA_STRING_KEYS = {
    "acquisition_stratum",
    "campaign_id",
    "campaign_stage",
    "campaign_tier",
    "group_id",
    "land_cover_descriptor",
    "lithology_descriptor",
    "pair_id",
    "prior_run_status",
    "public_context",
    "public_geologic_context",
    "public_scoring_status",
    "public_site_category",
    "split",
    "split_designation",
    "terrain_descriptor",
}

PUBLIC_CAMPAIGN_METADATA_BOOL_KEYS = {
    "audit_only",
    "prior_run_audit_only",
}

PUBLIC_CAMPAIGN_METADATA_PAYLOAD_KEYS = {
    "land_cover_descriptors",
    "lithology_descriptors",
    "public_strata",
    "public_stratum",
    "public_tags",
    "terrain_descriptors",
}

PUBLIC_CAMPAIGN_METADATA_KEYS = tuple(
    sorted(
        PUBLIC_CAMPAIGN_METADATA_STRING_KEYS
        | PUBLIC_CAMPAIGN_METADATA_BOOL_KEYS
        | PUBLIC_CAMPAIGN_METADATA_PAYLOAD_KEYS
    )
)

SENSITIVE_KEY_MARKERS = ("PASSWORD", "TOKEN", "SECRET", "KEY")
REPORT_CANDIDATE_CLAIM = (
    "Results are candidate detections unless independently field-verified; "
    "this package does not claim confirmed voids, objects, hazards, or drill targets."
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?im)\b(Authorization\s*:\s*Bearer\s+)([^\s,;]+)"
)
_BARE_BEARER_RE = re.compile(r"(?im)\b(Bearer\s+)([A-Za-z0-9._~+/=-]{8,})")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)\b([A-Za-z0-9_.-]*(?:token|password|secret|api[_-]?key|access[_-]?key|private[_-]?key|authorization|bearer)[A-Za-z0-9_.-]*)\b(\s*[:=]\s*)([^\s,;]+)"
)

SAR_PROVIDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "sentinel1_asf": {
        "provider_label": "ASF Sentinel-1",
        "platform": "Sentinel-1",
        "band": "C",
        "processing_level": "SLC",
        "beam_mode": "IW",
        "acquisition_mode": "TOPSAR IW",
        "default_comparison_arm": "sentinel1_cband_iw",
        "resolution_class": "free_broad_coverage_c_band",
        "search_supported": True,
        "real_lock_supported": True,
        "download_supported": True,
        "requires_credentials": False,
        "limitations": [],
    },
    "umbra_open_data": {
        "provider_label": "Umbra Open Data",
        "platform": "Umbra",
        "band": "X",
        "processing_level": "SLC",
        "beam_mode": "SPOTLIGHT",
        "acquisition_mode": "Spotlight",
        "default_comparison_arm": "umbra_xband_spotlight",
        "resolution_class": "x_band_spotlight_high_resolution_placeholder",
        "search_supported": False,
        "real_lock_supported": False,
        "download_supported": False,
        "requires_credentials": False,
        "limitations": [
            "Validation inventory can record Umbra/X-band comparison arms, but locked real execution is not wired end-to-end for this provider."
        ],
    },
    "capella_commercial": {
        "provider_label": "Capella Commercial Placeholder",
        "platform": "Capella",
        "band": "X",
        "processing_level": "SLC",
        "beam_mode": "SPOTLIGHT",
        "acquisition_mode": "Spotlight",
        "default_comparison_arm": "capella_xband_spotlight_placeholder",
        "resolution_class": "commercial_x_band_placeholder",
        "search_supported": False,
        "real_lock_supported": False,
        "download_supported": False,
        "requires_credentials": True,
        "limitations": [
            "Commercial-provider credentials and ordering workflows are not recorded or invoked by blind validation scaffolds."
        ],
    },
    "iceye_commercial": {
        "provider_label": "ICEYE Commercial Placeholder",
        "platform": "ICEYE",
        "band": "X",
        "processing_level": "SLC",
        "beam_mode": "SPOTLIGHT",
        "acquisition_mode": "Spotlight",
        "default_comparison_arm": "iceye_xband_spotlight_placeholder",
        "resolution_class": "commercial_x_band_placeholder",
        "search_supported": False,
        "real_lock_supported": False,
        "download_supported": False,
        "requires_credentials": True,
        "limitations": [
            "Commercial-provider credentials and ordering workflows are not recorded or invoked by blind validation scaffolds."
        ],
    },
    "xband_spotlight_placeholder": {
        "provider_label": "Generic X-band Spotlight Placeholder",
        "platform": "Commercial X-band SAR",
        "band": "X",
        "processing_level": "SLC",
        "beam_mode": "SPOTLIGHT",
        "acquisition_mode": "Spotlight",
        "default_comparison_arm": "xband_spotlight_placeholder",
        "resolution_class": "commercial_x_band_placeholder",
        "search_supported": False,
        "real_lock_supported": False,
        "download_supported": False,
        "requires_credentials": True,
        "limitations": [
            "Generic X-band placeholders are comparison metadata only until a concrete provider catalog and lock path are implemented."
        ],
    },
}

SUPPORTED_SAR_PROVIDERS = set(SAR_PROVIDER_PROFILES)

PARAMETER_SET_REQUIRED_OBJECTS = (
    "thresholds",
    "resolution_profile",
    "pinn_settings",
    "scoring_tolerances",
    "acquisition_provider_preferences",
    "split_policy",
    "analysis_plan",
)

PARAMETER_SET_NON_HASH_KEYS = {
    "parameter_set_id",
    "parameter_set_hash",
    "computed_at_utc",
    "created_at_utc",
    "template_only",
    "approved_for_holdout",
    "approval",
    "approval_status",
    "withheld_labels_loaded",
}

CAMPAIGN_REGISTRY_STATUSES = {"draft", "approved", "locked"}

CAMPAIGN_REGISTRY_REQUIRED_OBJECTS = (
    "protocol",
    "public_manifest",
    "parameter_set",
    "approved_parameter_requirement",
    "split_policy",
    "scoring_tolerances",
    "metric_definitions",
    "target_counts",
)

CAMPAIGN_REGISTRY_ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "template_only",
    "validation_id",
    "campaign_id",
    "campaign_name",
    "status",
    "created_at_utc",
    "computed_at_utc",
    "protocol",
    "public_manifest",
    "parameter_set",
    "approved_parameter_requirement",
    "split_policy",
    "provider_arms",
    "scoring_tolerances",
    "metric_definitions",
    "target_counts",
    "immutable_artifacts",
    "approval",
    "lock",
    "registry_hash",
    "registry_id",
    "withheld_labels_loaded",
}

CAMPAIGN_REGISTRY_NON_HASH_KEYS = {
    "registry_id",
    "registry_hash",
    "computed_at_utc",
    "created_at_utc",
    "template_only",
    "status",
    "approval",
    "lock",
    "withheld_labels_loaded",
}

CAMPAIGN_REGISTRY_PRIVATE_KEY_ALIASES = {
    "apikey",
    "apisecret",
    "authorization",
    "authorizationheader",
    "bearer",
    "bearertoken",
    "clientsecret",
    "credential",
    "credentials",
    "earthdatatoken",
    "fieldevidence",
    "fieldevidencefile",
    "fieldevidencepath",
    "fieldevidence",
    "fieldevidencefile",
    "fieldevidencepath",
    "password",
    "privateevidence",
    "privateevidencepath",
    "secret",
    "secrettoken",
    "token",
}


class ManifestValidationError(ValueError):
    """Raised when a validation manifest fails schema checks."""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestValidationError(f"Manifest {path} must be a JSON object")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        dump_strict_json(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
    out = "".join(safe).strip("._-")
    return out or "target"


def _as_float(value: Any, field: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ManifestValidationError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ManifestValidationError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ManifestValidationError(f"{field} must be <= {maximum}")
    return result


def _optional_positive_float(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    return _as_float(value, field, minimum=0.0)


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _int_or_none(value: Any) -> Optional[int]:
    number = _finite_or_none(value)
    if number is None:
        return None
    return int(number)


def _parse_literal_sequence(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return None


def _parse_numeric_sequence(value: Any, *, min_len: int = 2) -> Optional[List[float]]:
    parsed = _parse_literal_sequence(value)
    if parsed is None or len(parsed) < min_len:
        return None
    out = []
    for item in parsed:
        number = _finite_or_none(item)
        if number is None:
            return None
        out.append(number)
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371008.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _latlon_to_offset_m(lat: float, lon: float, center_lat: float, center_lon: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_132.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
    return ((lon - center_lon) * meters_per_deg_lon, (lat - center_lat) * meters_per_deg_lat)


def _round_metric(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, list):
        return [_round_metric(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round_metric(v, digits) for k, v in value.items()}
    return value


def _public_key_alias(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _reject_public_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            alias = _public_key_alias(key)
            if lowered in PUBLIC_FORBIDDEN_KEYS or alias in PUBLIC_FORBIDDEN_KEY_ALIASES:
                raise ManifestValidationError(
                    f"Public manifest must not expose withheld label/truth key {path}.{key}"
                )
            _reject_public_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_public_forbidden_keys(child, f"{path}[{idx}]")


def _normalize_optional_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_public_metadata_payload(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [
            _validate_public_metadata_payload(item, f"{field}[{idx}]")
            for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ManifestValidationError(f"{field} object keys must be non-empty strings")
            normalized[key] = _validate_public_metadata_payload(child, f"{field}.{key}")
        return normalized
    raise ManifestValidationError(f"{field} must be JSON scalar, list, or object public metadata")


def _normalize_public_campaign_metadata(value: Dict[str, Any], object_path: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in PUBLIC_CAMPAIGN_METADATA_STRING_KEYS:
        if key in value:
            normalized[key] = _normalize_optional_text(value[key], f"{object_path}.{key}")
    for key in PUBLIC_CAMPAIGN_METADATA_BOOL_KEYS:
        if key in value:
            if not isinstance(value[key], bool):
                raise ManifestValidationError(f"{object_path}.{key} must be boolean")
            normalized[key] = value[key]
    for key in PUBLIC_CAMPAIGN_METADATA_PAYLOAD_KEYS:
        if key in value:
            normalized[key] = _validate_public_metadata_payload(value[key], f"{object_path}.{key}")
    return normalized


def _public_campaign_metadata_record(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value[key] for key in PUBLIC_CAMPAIGN_METADATA_KEYS if key in value}


def _validate_center(center: Any, target_path: str) -> Dict[str, float]:
    if not isinstance(center, dict):
        raise ManifestValidationError(f"{target_path}.center must be an object with lat/lon")
    lat = _as_float(center.get("lat"), f"{target_path}.center.lat", minimum=-90.0, maximum=90.0)
    lon = _as_float(center.get("lon"), f"{target_path}.center.lon", minimum=-180.0, maximum=180.0)
    return {"lat": lat, "lon": lon}


def load_public_manifest(path: Path | str, *, allow_templates: bool = False) -> Dict[str, Any]:
    """Load and validate a public blind-validation manifest.

    Public manifests are intentionally prohibited from carrying known void
    labels, expected depths, ground-truth geometry, or paths to withheld labels.
    """
    manifest_path = Path(path)
    data = _load_json(manifest_path)
    if data.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Public manifest schema_version must be {PUBLIC_SCHEMA_VERSION!r}"
        )
    if data.get("template_only") and not allow_templates:
        raise ManifestValidationError(
            "Public manifest is marked template_only; copy it and remove template_only after filling real values"
        )
    _reject_public_forbidden_keys(data)

    manifest_campaign_metadata = _normalize_public_campaign_metadata(data, "$")

    validation_id = data.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError("Public manifest validation_id must be a non-empty string")

    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ManifestValidationError("Public manifest targets must be a non-empty list")

    seen_ids = set()
    normalized_targets = []
    for idx, target in enumerate(targets):
        target_path = f"targets[{idx}]"
        if not isinstance(target, dict):
            raise ManifestValidationError(f"{target_path} must be an object")
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ManifestValidationError(f"{target_path}.target_id must be a non-empty string")
        if target_id in seen_ids:
            raise ManifestValidationError(f"Duplicate target_id {target_id!r}")
        seen_ids.add(target_id)
        name = target.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ManifestValidationError(f"{target_path}.name must be a non-empty string")
        center = _validate_center(target.get("center"), target_path)

        data_mode = target.get("data_mode", "real_slc")
        if data_mode not in {"real_slc", "existing_candidates", "template"}:
            raise ManifestValidationError(
                f"{target_path}.data_mode must be one of real_slc, existing_candidates, template"
            )
        if data_mode == "template" and not allow_templates:
            raise ManifestValidationError(
                f"{target_path}.data_mode is template; copy the template before running"
            )

        normalized = dict(target)
        normalized["target_id"] = target_id
        normalized["name"] = name
        normalized["center"] = center
        normalized["data_mode"] = data_mode
        normalized.update(_normalize_public_campaign_metadata(target, target_path))
        if "buffer_deg" in target:
            normalized["buffer_deg"] = _as_float(
                target.get("buffer_deg"), f"{target_path}.buffer_deg", minimum=0.0
            )
        else:
            normalized["buffer_deg"] = 0.02
        if "area_km2" in target:
            normalized["area_km2"] = _as_float(
                target.get("area_km2"), f"{target_path}.area_km2", minimum=0.0
            )
        if "domain_width_m" in target:
            normalized["domain_width_m"] = _as_float(
                target.get("domain_width_m"), f"{target_path}.domain_width_m", minimum=1.0
            )
        if "max_depth_m" in target:
            normalized["max_depth_m"] = _as_float(
                target.get("max_depth_m"), f"{target_path}.max_depth_m", minimum=1.0
            )
        if "resolution" in target and target.get("resolution") not in {"quick", "standard", "high", "deep"}:
            raise ManifestValidationError(f"{target_path}.resolution is not a supported profile")
        if "sar_search" in target:
            normalized["sar_search"] = _normalize_sar_search(target.get("sar_search"), target_path)
        if "comparison_arms" in target:
            normalized["comparison_arms"] = _normalize_comparison_arms(target.get("comparison_arms"), target_path)
            if "sar_search" not in normalized and normalized["comparison_arms"]:
                normalized["sar_search"] = dict(normalized["comparison_arms"][0])
        normalized_targets.append(normalized)

    out = dict(data)
    out.update(manifest_campaign_metadata)
    out["targets"] = normalized_targets
    return out


def load_withheld_labels(path: Path | str, *, allow_templates: bool = False) -> Dict[str, Any]:
    """Load and validate withheld scoring labels.

    This function must never be called by the runner stage.
    """
    labels_path = Path(path)
    data = _load_json(labels_path)
    if data.get("schema_version") != WITHHELD_LABEL_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Withheld-label schema_version must be {WITHHELD_LABEL_SCHEMA_VERSION!r}"
        )
    if data.get("template_only") and not allow_templates:
        raise ManifestValidationError(
            "Withheld-label manifest is marked template_only; copy it to a private location and remove template_only after filling real values"
        )
    validation_id = data.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError("Withheld labels validation_id must be a non-empty string")
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise ManifestValidationError("Withheld labels must contain a labels list")
    seen = set()
    normalized_labels = []
    for idx, entry in enumerate(labels):
        entry_path = f"labels[{idx}]"
        if not isinstance(entry, dict):
            raise ManifestValidationError(f"{entry_path} must be an object")
        target_id = entry.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ManifestValidationError(f"{entry_path}.target_id must be a non-empty string")
        if target_id in seen:
            raise ManifestValidationError(f"Duplicate withheld label target_id {target_id!r}")
        seen.add(target_id)
        site_class = entry.get("site_class")
        if site_class not in {"positive", "negative"}:
            raise ManifestValidationError(f"{entry_path}.site_class must be positive or negative")
        known_voids = entry.get("known_voids", [])
        if not isinstance(known_voids, list):
            raise ManifestValidationError(f"{entry_path}.known_voids must be a list")
        if site_class == "positive" and not known_voids:
            raise ManifestValidationError(f"{entry_path} is positive but has no known_voids")
        if site_class == "negative" and known_voids:
            raise ManifestValidationError(f"{entry_path} is negative but contains known_voids")

        normalized_voids = []
        for void_idx, void in enumerate(known_voids):
            void_path = f"{entry_path}.known_voids[{void_idx}]"
            if not isinstance(void, dict):
                raise ManifestValidationError(f"{void_path} must be an object")
            void_id = void.get("void_id") or void.get("label_id")
            if not isinstance(void_id, str) or not void_id.strip():
                raise ManifestValidationError(f"{void_path}.void_id must be a non-empty string")
            has_offset = _parse_numeric_sequence(void.get("offset_m"), min_len=2) is not None
            has_centroid = _parse_numeric_sequence(void.get("centroid_m"), min_len=2) is not None
            has_latlon = void.get("lat") is not None and void.get("lon") is not None
            if not (has_offset or has_centroid or has_latlon):
                raise ManifestValidationError(
                    f"{void_path} must provide offset_m, centroid_m, or lat/lon"
                )
            normalized = dict(void)
            normalized["void_id"] = void_id
            if "horizontal_tolerance_m" in void:
                normalized["horizontal_tolerance_m"] = _as_float(
                    void.get("horizontal_tolerance_m"), f"{void_path}.horizontal_tolerance_m", minimum=0.0
                )
            if "depth_tolerance_m" in void:
                normalized["depth_tolerance_m"] = _as_float(
                    void.get("depth_tolerance_m"), f"{void_path}.depth_tolerance_m", minimum=0.0
                )
            normalized_voids.append(normalized)
        normalized_entry = dict(entry)
        normalized_entry["known_voids"] = normalized_voids
        normalized_labels.append(normalized_entry)
    out = dict(data)
    out["labels"] = normalized_labels
    return out


def _manifest_relative_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _write_empty_candidate_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EMPTY_CANDIDATE_FIELDS)
        writer.writeheader()


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _copy_or_empty_candidate_csv(source: Optional[Path], destination: Path) -> Tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source and source.exists():
        shutil.copyfile(source, destination)
        return _count_csv_rows(destination), "copied_existing_candidates"
    _write_empty_candidate_csv(destination)
    return 0, "empty_no_candidates_available"


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _redacted_env_credentials() -> Dict[str, str]:
    credentials = {}
    for key in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN", "EARTHDATA_BEARER_TOKEN"):
        value = os.environ.get(key)
        if value:
            credentials[key] = value
    return credentials


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json_sha256(data: Dict[str, Any]) -> str:
    payload = dumps_strict_json(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_fingerprint(path: Path, base: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path)
    record: Dict[str, Any] = {
        "path": _relative_to_base(path, base) if base is not None else str(path),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        record.update(
            {
                "sha256": _sha256_file(path),
                "size_bytes": int(stat.st_size),
                "mtime_unix": int(stat.st_mtime),
            }
        )
    return record


def _key_module_fingerprints() -> Dict[str, Dict[str, Any]]:
    base = Path(__file__).resolve().parent
    module_names = [
        "blind_validation.py",
        "slc_data_fetcher.py",
        "deformation_intel/opera.py",
        "deformation_intel/timeseries.py",
        "deformation_intel/sources.py",
        "deformation_intel/detect.py",
    ]
    return {name: _file_fingerprint(base / name, base) for name in module_names}


def _runtime_metadata() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _json_safe_secret_redacted(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__)
        return f"<callable:{name}>"
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            if any(marker in str(key).upper() for marker in SENSITIVE_KEY_MARKERS):
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = _json_safe_secret_redacted(child)
        return output
    if isinstance(value, (list, tuple)):
        return [_json_safe_secret_redacted(item) for item in value]
    if isinstance(value, str):
        return _redact_env_secret_values(value)
    try:
        return to_strict_jsonable(value)
    except (TypeError, ValueError):
        return str(value)


def _namespace_to_safe_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return _json_safe_secret_redacted(vars(args))


def _safe_positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be an integer") from exc
    if result < minimum:
        raise ManifestValidationError(f"{field} must be >= {minimum}")
    return result


def _parameter_set_hash_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in data.items()
        if str(key) not in PARAMETER_SET_NON_HASH_KEYS
    }


def parameter_set_hash(data: Dict[str, Any]) -> str:
    """Return the deterministic hash of an approved/draft validation parameter set."""
    return _stable_json_sha256(_parameter_set_hash_payload(data))


def parameter_set_id_from_hash(hash_value: str) -> str:
    return f"{PARAMETER_SET_ID_PREFIX}{str(hash_value)[:PARAMETER_SET_ID_HASH_CHARS]}"


def _parameter_set_approval_status(data: Dict[str, Any]) -> str:
    approval = data.get("approval") if isinstance(data.get("approval"), dict) else {}
    raw_status = data.get("approval_status") or approval.get("status") or "draft"
    status = str(raw_status).strip().lower().replace(" ", "_")
    return status or "draft"


def _parameter_set_approved_for_holdout(data: Dict[str, Any]) -> bool:
    status = _parameter_set_approval_status(data)
    return bool(data.get("approved_for_holdout")) or status in {
        "approved",
        "approved_for_holdout",
        "locked",
        "locked_for_holdout",
    }


def _validate_parameter_provider_preferences(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field} must be an object")
    normalized = dict(value)
    primary = normalized.get("primary_provider", normalized.get("provider", "sentinel1_asf"))
    if not isinstance(primary, str) or primary not in SUPPORTED_SAR_PROVIDERS:
        raise ManifestValidationError(
            f"{field}.primary_provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}"
        )
    normalized["primary_provider"] = primary
    if "provider" in normalized:
        normalized["provider"] = primary

    preferences = normalized.get("provider_preferences", normalized.get("providers"))
    if preferences is None:
        preferences = [primary]
    if not isinstance(preferences, list) or not preferences:
        raise ManifestValidationError(f"{field}.provider_preferences must be a non-empty list")
    provider_preferences = []
    for idx, provider in enumerate(preferences):
        if not isinstance(provider, str) or provider not in SUPPORTED_SAR_PROVIDERS:
            raise ManifestValidationError(
                f"{field}.provider_preferences[{idx}] must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}"
            )
        provider_preferences.append(provider)
    if primary not in provider_preferences:
        provider_preferences.insert(0, primary)
    normalized["provider_preferences"] = provider_preferences

    arms = normalized.get("comparison_arms", [])
    if arms is None:
        arms = []
    if not isinstance(arms, list):
        raise ManifestValidationError(f"{field}.comparison_arms must be a list when provided")
    normalized_arms = []
    for idx, arm in enumerate(arms):
        arm_field = f"{field}.comparison_arms[{idx}]"
        if isinstance(arm, str):
            arm_provider = arm
            arm_obj: Dict[str, Any] = {"provider": arm_provider}
        elif isinstance(arm, dict):
            arm_obj = dict(arm)
            arm_provider = arm_obj.get("provider", primary)
        else:
            raise ManifestValidationError(f"{arm_field} must be a provider string or object")
        if not isinstance(arm_provider, str) or arm_provider not in SUPPORTED_SAR_PROVIDERS:
            raise ManifestValidationError(f"{arm_field}.provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}")
        profile = SAR_PROVIDER_PROFILES[arm_provider]
        arm_obj["provider"] = arm_provider
        arm_obj["comparison_arm"] = str(
            arm_obj.get("comparison_arm") or profile["default_comparison_arm"]
        )
        arm_obj["band"] = str(arm_obj.get("band") or profile["band"])
        arm_obj["search_supported"] = bool(profile["search_supported"])
        arm_obj["real_lock_supported"] = bool(profile["real_lock_supported"])
        arm_obj["placeholder_only"] = not bool(profile["search_supported"])
        normalized_arms.append(arm_obj)
    normalized["comparison_arms"] = normalized_arms
    normalized["real_downloads_default"] = bool(normalized.get("real_downloads_default", False))
    return normalized


def load_parameter_set(path: Path | str, *, allow_templates: bool = False, require_approved: bool = False) -> Dict[str, Any]:
    """Load and validate a deterministic no-label validation parameter set."""
    parameter_path = Path(path)
    data = _load_json(parameter_path)
    if data.get("schema_version") != PARAMETER_SET_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Parameter-set schema_version must be {PARAMETER_SET_SCHEMA_VERSION!r}"
        )
    if data.get("template_only") and not allow_templates:
        raise ManifestValidationError(
            "Parameter set is marked template_only; copy it and remove template_only after filling real values"
        )
    _reject_public_forbidden_keys(data)

    validation_id = data.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError("Parameter set validation_id must be a non-empty string")
    for key in PARAMETER_SET_REQUIRED_OBJECTS:
        if not isinstance(data.get(key), dict):
            raise ManifestValidationError(f"Parameter set {key} must be an object")
    if not isinstance(data.get("model_card", {}), dict):
        raise ManifestValidationError("Parameter set model_card must be an object when provided")

    thresholds = dict(data["thresholds"])
    if "void_probability_threshold" in thresholds:
        thresholds["void_probability_threshold"] = _as_float(
            thresholds["void_probability_threshold"],
            "thresholds.void_probability_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    if "candidate_confidence_threshold" in thresholds:
        thresholds["candidate_confidence_threshold"] = _as_float(
            thresholds["candidate_confidence_threshold"],
            "thresholds.candidate_confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        )
    if "min_anomaly_voxels" in thresholds:
        thresholds["min_anomaly_voxels"] = _safe_positive_int(
            thresholds["min_anomaly_voxels"], "thresholds.min_anomaly_voxels"
        )

    resolution_profile = dict(data["resolution_profile"])
    profile_name = resolution_profile.get("name", resolution_profile.get("profile", "quick"))
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ManifestValidationError("resolution_profile.name must be a non-empty string")
    resolution_profile["name"] = profile_name
    for numeric_key in ("domain_width_m", "max_depth_m"):
        if numeric_key in resolution_profile:
            resolution_profile[numeric_key] = _as_float(
                resolution_profile[numeric_key], f"resolution_profile.{numeric_key}", minimum=1.0
            )
    for integer_key in ("grid_nx", "grid_ny", "grid_nz"):
        if integer_key in resolution_profile:
            resolution_profile[integer_key] = _safe_positive_int(
                resolution_profile[integer_key], f"resolution_profile.{integer_key}"
            )

    pinn_settings = dict(data["pinn_settings"])
    if "epochs" in pinn_settings:
        pinn_settings["epochs"] = _safe_positive_int(pinn_settings["epochs"], "pinn_settings.epochs")
    for numeric_key in (
        "physics_weight",
        "data_weight",
        "sparsity_weight",
        "regularization_weight",
        "deep_prior_weight",
        "surface_prior_weight",
        "excitation_frequency_hz",
    ):
        if numeric_key in pinn_settings:
            pinn_settings[numeric_key] = _as_float(pinn_settings[numeric_key], f"pinn_settings.{numeric_key}")

    scoring_tolerances = dict(data["scoring_tolerances"])
    for numeric_key in ("default_horizontal_tolerance_m", "default_depth_tolerance_m"):
        if numeric_key in scoring_tolerances:
            scoring_tolerances[numeric_key] = _as_float(
                scoring_tolerances[numeric_key], f"scoring_tolerances.{numeric_key}", minimum=0.0
            )

    acquisition_provider_preferences = _validate_parameter_provider_preferences(
        data["acquisition_provider_preferences"], "acquisition_provider_preferences"
    )
    split_policy = dict(data["split_policy"])
    if "holdout_split" in split_policy and not isinstance(split_policy["holdout_split"], str):
        raise ManifestValidationError("split_policy.holdout_split must be a string")
    if "calibration_split" in split_policy and not isinstance(split_policy["calibration_split"], str):
        raise ManifestValidationError("split_policy.calibration_split must be a string")

    normalized = dict(data)
    normalized["thresholds"] = thresholds
    normalized["resolution_profile"] = resolution_profile
    normalized["pinn_settings"] = pinn_settings
    normalized["scoring_tolerances"] = scoring_tolerances
    normalized["acquisition_provider_preferences"] = acquisition_provider_preferences
    normalized["split_policy"] = split_policy

    computed_hash = parameter_set_hash(normalized)
    computed_id = parameter_set_id_from_hash(computed_hash)
    recorded_hash = normalized.get("parameter_set_hash")
    if recorded_hash is not None and str(recorded_hash) != computed_hash:
        raise ManifestValidationError("Parameter set parameter_set_hash does not match canonical content")
    recorded_id = normalized.get("parameter_set_id")
    if recorded_id is not None and str(recorded_id) != computed_id:
        raise ManifestValidationError("Parameter set parameter_set_id does not match canonical content hash")
    normalized["parameter_set_hash"] = computed_hash
    normalized["parameter_set_id"] = computed_id
    normalized["approval_status"] = _parameter_set_approval_status(normalized)
    normalized["approved_for_holdout"] = _parameter_set_approved_for_holdout(normalized)
    normalized["withheld_labels_loaded"] = False
    if require_approved and not normalized["approved_for_holdout"]:
        raise ManifestValidationError("Parameter set is not approved for holdout scoring")
    return normalized


def _parameter_set_reference_record(path: Path | str, data: Dict[str, Any]) -> Dict[str, Any]:
    parameter_path = Path(path)
    return {
        "path": str(parameter_path),
        "file_sha256": _sha256_file(parameter_path) if parameter_path.exists() else None,
        "schema_version": data.get("schema_version"),
        "validation_id": data.get("validation_id"),
        "parameter_set_name": data.get("parameter_set_name") or data.get("name"),
        "parameter_set_id": data.get("parameter_set_id"),
        "parameter_set_hash": data.get("parameter_set_hash"),
        "hash_algorithm": "sha256_canonical_json_excluding_approval_and_identity_metadata",
        "approval_status": data.get("approval_status"),
        "approved_for_holdout": bool(data.get("approved_for_holdout")),
        "split_policy": data.get("split_policy"),
        "acquisition_provider_preferences": data.get("acquisition_provider_preferences"),
        "withheld_labels_loaded": False,
    }


def _hex_sha256_or_none(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ManifestValidationError(f"{field} must be a 64-character sha256 hex digest")
    return text


def _campaign_registry_hash_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in data.items()
        if str(key) not in CAMPAIGN_REGISTRY_NON_HASH_KEYS
    }


def campaign_registry_hash(data: Dict[str, Any]) -> str:
    """Return the deterministic hash of a no-label campaign registry."""
    return _stable_json_sha256(_campaign_registry_hash_payload(data))


def campaign_registry_id_from_hash(hash_value: str) -> str:
    return f"{CAMPAIGN_REGISTRY_ID_PREFIX}{str(hash_value)[:CAMPAIGN_REGISTRY_ID_HASH_CHARS]}"


def _campaign_status(value: Any) -> str:
    status = str(value or "draft").strip().lower().replace(" ", "_")
    if status not in CAMPAIGN_REGISTRY_STATUSES:
        raise ManifestValidationError(
            f"Campaign registry status must be one of {sorted(CAMPAIGN_REGISTRY_STATUSES)}"
        )
    return status


def _looks_like_forbidden_private_path(value: str) -> bool:
    text = str(value).strip().lower().replace("\\", "/")
    if not text:
        return False
    withheld_markers = (
        "withheld_label",
        "withheld-label",
        "withheld labels",
        "private_label",
        "private-label",
        "truth_label",
        "ground_truth",
        "known_void",
    )
    path_markers = ("/", ".json", ".csv", ".geojson", ".gpkg", ".shp", ".kml", ".kmz")
    return any(marker in text for marker in withheld_markers) and any(marker in text for marker in path_markers)


def _reject_campaign_registry_private_keys(value: Any, path: str = "$", *, allow_key: Optional[str] = None) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            alias = _public_key_alias(key)
            if lowered in PUBLIC_FORBIDDEN_KEYS or alias in PUBLIC_FORBIDDEN_KEY_ALIASES:
                raise ManifestValidationError(
                    f"Campaign registry must not expose withheld label/truth key {path}.{key}"
                )
            if alias in CAMPAIGN_REGISTRY_PRIVATE_KEY_ALIASES:
                raise ManifestValidationError(
                    f"Campaign registry must not expose secret/private evidence key {path}.{key}"
                )
            _reject_campaign_registry_private_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _reject_campaign_registry_private_keys(child, f"{path}[{idx}]")
    elif isinstance(value, str) and _looks_like_forbidden_private_path(value):
        raise ManifestValidationError(
            f"Campaign registry must not include paths or path-like references to withheld labels at {path}"
        )


def _normalize_registry_provider_arms(value: Any, field: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(f"{field} must be a non-empty list")
    normalized_arms = []
    seen = set()
    for idx, arm in enumerate(value):
        arm_field = f"{field}[{idx}]"
        if isinstance(arm, str):
            arm_obj: Dict[str, Any] = {"provider": arm}
        elif isinstance(arm, dict):
            arm_obj = dict(arm)
        else:
            raise ManifestValidationError(f"{arm_field} must be a provider string or object")
        provider = arm_obj.get("provider", arm_obj.get("provider_id"))
        if not isinstance(provider, str) or provider not in SUPPORTED_SAR_PROVIDERS:
            raise ManifestValidationError(f"{arm_field}.provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}")
        profile = SAR_PROVIDER_PROFILES[provider]
        arm_obj["provider"] = provider
        arm_obj["provider_label"] = str(arm_obj.get("provider_label") or profile["provider_label"])
        arm_obj["comparison_arm"] = str(arm_obj.get("comparison_arm") or profile["default_comparison_arm"])
        arm_obj["band"] = str(arm_obj.get("band") or profile["band"])
        arm_obj["search_supported"] = bool(arm_obj.get("search_supported", profile["search_supported"]))
        arm_obj["real_lock_supported"] = bool(arm_obj.get("real_lock_supported", profile["real_lock_supported"]))
        arm_obj["download_supported"] = bool(arm_obj.get("download_supported", profile["download_supported"]))
        arm_obj["placeholder_only"] = bool(arm_obj.get("placeholder_only", not profile["search_supported"]))
        key = (arm_obj["provider"], arm_obj["comparison_arm"])
        if key in seen:
            raise ManifestValidationError(f"Duplicate provider/comparison arm in {field}: {key}")
        seen.add(key)
        normalized_arms.append(arm_obj)
    return normalized_arms


def _normalize_registry_manifest_record(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field} must be an object")
    normalized = dict(value)
    if "sha256" in normalized and "file_sha256" not in normalized:
        normalized["file_sha256"] = normalized["sha256"]
    if "file_sha256" not in normalized:
        raise ManifestValidationError(f"{field}.file_sha256 is required")
    normalized["file_sha256"] = _hex_sha256_or_none(normalized.get("file_sha256"), f"{field}.file_sha256")
    if normalized["file_sha256"] is None:
        raise ManifestValidationError(f"{field}.file_sha256 is required")
    validation_id = normalized.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError(f"{field}.validation_id must be a non-empty string")
    target_count = normalized.get("target_count")
    if target_count is not None:
        normalized["target_count"] = _safe_positive_int(target_count, f"{field}.target_count")
    target_ids = normalized.get("target_ids")
    if target_ids is not None:
        if not isinstance(target_ids, list) or not all(isinstance(item, str) and item.strip() for item in target_ids):
            raise ManifestValidationError(f"{field}.target_ids must be a list of non-empty strings")
        if len(target_ids) != len(set(target_ids)):
            raise ManifestValidationError(f"{field}.target_ids must not contain duplicates")
        normalized["target_ids"] = list(target_ids)
    normalized.setdefault("hash_algorithm", "sha256_file")
    return normalized


def _normalize_registry_parameter_record(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field} must be an object")
    normalized = dict(value)
    if "file_sha256" in normalized:
        normalized["file_sha256"] = _hex_sha256_or_none(normalized.get("file_sha256"), f"{field}.file_sha256")
    parameter_hash = normalized.get("parameter_set_hash")
    if parameter_hash is None:
        raise ManifestValidationError(f"{field}.parameter_set_hash is required")
    normalized["parameter_set_hash"] = _hex_sha256_or_none(parameter_hash, f"{field}.parameter_set_hash")
    parameter_id = normalized.get("parameter_set_id")
    if not isinstance(parameter_id, str) or not parameter_id.strip():
        raise ManifestValidationError(f"{field}.parameter_set_id must be a non-empty string")
    expected_id = parameter_set_id_from_hash(normalized["parameter_set_hash"])
    if parameter_id != expected_id:
        raise ManifestValidationError(f"{field}.parameter_set_id does not match parameter_set_hash")
    validation_id = normalized.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError(f"{field}.validation_id must be a non-empty string")
    normalized["approved_for_holdout"] = bool(normalized.get("approved_for_holdout", False))
    normalized["approval_status"] = str(normalized.get("approval_status") or "draft")
    normalized.setdefault("hash_algorithm", "sha256_canonical_json_excluding_approval_and_identity_metadata")
    normalized["withheld_labels_loaded"] = False
    return normalized


def _normalize_registry_artifacts(value: Any, field: str) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestValidationError(f"{field} must be a list")
    normalized_artifacts = []
    seen_roles = set()
    for idx, artifact in enumerate(value):
        artifact_field = f"{field}[{idx}]"
        if not isinstance(artifact, dict):
            raise ManifestValidationError(f"{artifact_field} must be an object")
        normalized = dict(artifact)
        role = normalized.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ManifestValidationError(f"{artifact_field}.role must be a non-empty string")
        normalized["role"] = role.strip()
        sha_value = normalized.get("sha256", normalized.get("file_sha256"))
        if sha_value is None:
            raise ManifestValidationError(f"{artifact_field}.sha256 is required")
        normalized["sha256"] = _hex_sha256_or_none(sha_value, f"{artifact_field}.sha256")
        normalized.setdefault("hash_algorithm", "sha256_file")
        role_key = (normalized["role"], str(normalized.get("path") or ""))
        if role_key in seen_roles:
            raise ManifestValidationError(f"Duplicate immutable artifact role/path in {field}: {role_key}")
        seen_roles.add(role_key)
        normalized_artifacts.append(normalized)
    return normalized_artifacts


def _validate_registry_target_counts(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError("Campaign registry target_counts must be an object")
    normalized = dict(value)
    for key in ("total_targets", "expected_total_targets", "primary_holdout_targets", "calibration_targets", "audit_only_targets"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = _safe_positive_int(normalized[key], f"target_counts.{key}", minimum=0)
    for nested_key in ("by_split", "by_campaign_tier", "by_public_scoring_status", "by_provider_arm"):
        if nested_key in normalized:
            nested = normalized[nested_key]
            if not isinstance(nested, dict):
                raise ManifestValidationError(f"target_counts.{nested_key} must be an object")
            normalized[nested_key] = {
                str(child_key): _safe_positive_int(child_value, f"target_counts.{nested_key}.{child_key}", minimum=0)
                for child_key, child_value in nested.items()
            }
    return normalized


def load_campaign_registry(
    path: Path | str,
    *,
    allow_templates: bool = False,
    require_approved: bool = False,
    require_locked: bool = False,
) -> Dict[str, Any]:
    """Load and validate a deterministic no-label blind campaign registry."""
    registry_path = Path(path)
    data = _load_json(registry_path)
    if data.get("schema_version") != CAMPAIGN_REGISTRY_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Campaign registry schema_version must be {CAMPAIGN_REGISTRY_SCHEMA_VERSION!r}"
        )
    unknown = sorted(str(key) for key in data if key not in CAMPAIGN_REGISTRY_ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ManifestValidationError(f"Campaign registry has unsupported top-level key(s): {unknown}")
    if data.get("template_only") and not allow_templates:
        raise ManifestValidationError(
            "Campaign registry is marked template_only; copy it and remove template_only after filling real values"
        )
    _reject_campaign_registry_private_keys(data)

    validation_id = data.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError("Campaign registry validation_id must be a non-empty string")
    campaign_id = data.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ManifestValidationError("Campaign registry campaign_id must be a non-empty string")
    status = _campaign_status(data.get("status", "draft"))
    for key in CAMPAIGN_REGISTRY_REQUIRED_OBJECTS:
        if not isinstance(data.get(key), dict):
            raise ManifestValidationError(f"Campaign registry {key} must be an object")

    protocol = dict(data["protocol"])
    if not protocol:
        raise ManifestValidationError("Campaign registry protocol must record preregistration/protocol references")
    public_manifest_record = _normalize_registry_manifest_record(data["public_manifest"], "public_manifest")
    parameter_set_record = _normalize_registry_parameter_record(data["parameter_set"], "parameter_set")
    if public_manifest_record.get("validation_id") != validation_id:
        raise ManifestValidationError("Campaign registry public_manifest.validation_id does not match validation_id")
    if parameter_set_record.get("validation_id") != validation_id:
        raise ManifestValidationError("Campaign registry parameter_set.validation_id does not match validation_id")

    approved_requirement = dict(data["approved_parameter_requirement"])
    approved_required = bool(approved_requirement.get("required", approved_requirement.get("require_approved_for_holdout", True)))
    require_approved_for_holdout = bool(approved_requirement.get("require_approved_for_holdout", approved_required))
    approved_requirement["required"] = approved_required
    approved_requirement["require_approved_for_holdout"] = require_approved_for_holdout
    approved_requirement.setdefault(
        "policy",
        "Holdout scoring and locked campaign execution must use this exact approved parameter-set hash.",
    )

    split_policy = dict(data["split_policy"])
    if split_policy.get("labels_withheld_from_runner") is not True:
        raise ManifestValidationError("Campaign registry split_policy.labels_withheld_from_runner must be true")
    if split_policy.get("parameter_changes_after_holdout_unblinding") not in {None, "forbidden"}:
        raise ManifestValidationError(
            "Campaign registry split_policy.parameter_changes_after_holdout_unblinding must be 'forbidden'"
        )

    scoring_tolerances = dict(data["scoring_tolerances"])
    for numeric_key in ("default_horizontal_tolerance_m", "default_depth_tolerance_m"):
        if numeric_key in scoring_tolerances:
            scoring_tolerances[numeric_key] = _as_float(
                scoring_tolerances[numeric_key], f"scoring_tolerances.{numeric_key}", minimum=0.0
            )
    metric_definitions = dict(data["metric_definitions"])
    if not metric_definitions:
        raise ManifestValidationError("Campaign registry metric_definitions must not be empty")
    target_counts = _validate_registry_target_counts(data["target_counts"])
    provider_arms = _normalize_registry_provider_arms(data.get("provider_arms"), "provider_arms")
    immutable_artifacts = _normalize_registry_artifacts(data.get("immutable_artifacts", []), "immutable_artifacts")

    artifact_by_role = {artifact.get("role"): artifact for artifact in immutable_artifacts}
    public_artifact = artifact_by_role.get("public_manifest")
    if public_artifact and public_artifact.get("sha256") != public_manifest_record.get("file_sha256"):
        raise ManifestValidationError("Campaign registry public_manifest immutable artifact hash does not match public_manifest.file_sha256")
    parameter_artifact = artifact_by_role.get("parameter_set")
    if parameter_artifact and parameter_set_record.get("file_sha256") and parameter_artifact.get("sha256") != parameter_set_record.get("file_sha256"):
        raise ManifestValidationError("Campaign registry parameter_set immutable artifact hash does not match parameter_set.file_sha256")

    if status in {"approved", "locked"} and require_approved_for_holdout and not parameter_set_record.get("approved_for_holdout"):
        raise ManifestValidationError("Approved or locked campaign registry requires an approved parameter set")
    if require_approved and status not in {"approved", "locked"}:
        raise ManifestValidationError("Campaign registry is not approved")
    if require_locked and status != "locked":
        raise ManifestValidationError("Campaign registry is not locked")

    normalized = dict(data)
    normalized["validation_id"] = validation_id.strip()
    normalized["campaign_id"] = campaign_id.strip()
    normalized["status"] = status
    normalized["protocol"] = protocol
    normalized["public_manifest"] = public_manifest_record
    normalized["parameter_set"] = parameter_set_record
    normalized["approved_parameter_requirement"] = approved_requirement
    normalized["split_policy"] = split_policy
    normalized["provider_arms"] = provider_arms
    normalized["scoring_tolerances"] = scoring_tolerances
    normalized["metric_definitions"] = metric_definitions
    normalized["target_counts"] = target_counts
    normalized["immutable_artifacts"] = immutable_artifacts
    normalized["withheld_labels_loaded"] = False

    computed_hash = campaign_registry_hash(normalized)
    computed_id = campaign_registry_id_from_hash(computed_hash)
    recorded_hash = normalized.get("registry_hash")
    if recorded_hash is not None and str(recorded_hash) != computed_hash:
        raise ManifestValidationError("Campaign registry registry_hash does not match canonical content")
    recorded_id = normalized.get("registry_id")
    if recorded_id is not None and str(recorded_id) != computed_id:
        raise ManifestValidationError("Campaign registry registry_id does not match canonical content hash")
    normalized["registry_hash"] = computed_hash
    normalized["registry_id"] = computed_id
    return normalized


def _campaign_registry_reference_record(path: Path | str, data: Dict[str, Any]) -> Dict[str, Any]:
    registry_path = Path(path)
    parameter_record = data.get("parameter_set", {}) if isinstance(data.get("parameter_set"), dict) else {}
    manifest_record = data.get("public_manifest", {}) if isinstance(data.get("public_manifest"), dict) else {}
    requirement = data.get("approved_parameter_requirement", {}) if isinstance(data.get("approved_parameter_requirement"), dict) else {}
    return {
        "path": str(registry_path),
        "file_sha256": _sha256_file(registry_path) if registry_path.exists() else None,
        "schema_version": data.get("schema_version"),
        "validation_id": data.get("validation_id"),
        "campaign_id": data.get("campaign_id"),
        "campaign_name": data.get("campaign_name"),
        "status": data.get("status"),
        "registry_id": data.get("registry_id"),
        "registry_hash": data.get("registry_hash"),
        "hash_algorithm": "sha256_canonical_json_excluding_status_approval_lock_and_identity_metadata",
        "public_manifest_sha256": manifest_record.get("file_sha256"),
        "parameter_set_id": parameter_record.get("parameter_set_id"),
        "parameter_set_hash": parameter_record.get("parameter_set_hash"),
        "approved_parameter_required": bool(requirement.get("require_approved_for_holdout", requirement.get("required", True))),
        "withheld_labels_loaded": False,
    }


def _campaign_target_counts(public_manifest: Dict[str, Any]) -> Dict[str, Any]:
    targets = [target for target in public_manifest.get("targets", []) if isinstance(target, dict)]
    by_split: Dict[str, int] = {}
    by_campaign_tier: Dict[str, int] = {}
    by_public_scoring_status: Dict[str, int] = {}
    by_provider_arm: Dict[str, int] = {}
    audit_only_count = 0
    for target in targets:
        split = str(target.get("split") or target.get("split_designation") or "unspecified")
        by_split[split] = by_split.get(split, 0) + 1
        tier = str(target.get("campaign_tier") or "unspecified")
        by_campaign_tier[tier] = by_campaign_tier.get(tier, 0) + 1
        scoring_status = str(target.get("public_scoring_status") or "unspecified")
        by_public_scoring_status[scoring_status] = by_public_scoring_status.get(scoring_status, 0) + 1
        audit_only_count += 1 if target.get("audit_only") or target.get("prior_run_audit_only") else 0
        for arm in _target_comparison_arms(target):
            arm_name = str(arm.get("comparison_arm") or arm.get("provider") or "unspecified")
            by_provider_arm[arm_name] = by_provider_arm.get(arm_name, 0) + 1
    return {
        "total_targets": len(targets),
        "primary_holdout_targets": sum(
            1 for target in targets
            if str(target.get("split") or "") == "holdout" and not (target.get("audit_only") or target.get("prior_run_audit_only"))
        ),
        "calibration_targets": sum(1 for target in targets if str(target.get("split") or "") == "calibration"),
        "extension_targets": sum(1 for target in targets if str(target.get("split") or "") == "extension"),
        "audit_only_targets": audit_only_count,
        "by_split": dict(sorted(by_split.items())),
        "by_campaign_tier": dict(sorted(by_campaign_tier.items())),
        "by_public_scoring_status": dict(sorted(by_public_scoring_status.items())),
        "by_provider_arm": dict(sorted(by_provider_arm.items())),
    }


def _campaign_provider_arms(public_manifest: Dict[str, Any], parameter_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    arms_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    preferences = parameter_set.get("acquisition_provider_preferences", {})
    for arm in preferences.get("comparison_arms", []) if isinstance(preferences, dict) else []:
        if isinstance(arm, dict):
            provider = str(arm.get("provider") or "sentinel1_asf")
            comparison_arm = str(arm.get("comparison_arm") or SAR_PROVIDER_PROFILES.get(provider, SAR_PROVIDER_PROFILES["sentinel1_asf"])["default_comparison_arm"])
            arms_by_key[(provider, comparison_arm)] = dict(arm)
    for target in public_manifest.get("targets", []):
        if not isinstance(target, dict):
            continue
        for arm in _target_comparison_arms(target):
            provider = str(arm.get("provider") or "sentinel1_asf")
            comparison_arm = str(arm.get("comparison_arm") or provider)
            arms_by_key.setdefault((provider, comparison_arm), dict(arm))
    return _normalize_registry_provider_arms(
        [arms_by_key[key] for key in sorted(arms_by_key)],
        "provider_arms",
    )


def _default_campaign_registry_template(
    validation_id: str,
    campaign_id: str,
    manifest_path: Path | str,
    parameter_set_path: Path | str,
    *,
    campaign_name: str = "blind_multi_site_campaign",
    status: str = "draft",
    allow_templates: bool = False,
    approved_by: Optional[str] = None,
    approved_at_utc: Optional[str] = None,
    locked_by: Optional[str] = None,
    locked_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    parameter_set_path = Path(parameter_set_path)
    public_manifest = load_public_manifest(manifest_path, allow_templates=allow_templates)
    parameter_set = load_parameter_set(parameter_set_path, allow_templates=allow_templates)
    if public_manifest.get("validation_id") != validation_id:
        raise ManifestValidationError("Public manifest validation_id does not match requested campaign registry validation_id")
    if parameter_set.get("validation_id") != validation_id:
        raise ManifestValidationError("Parameter set validation_id does not match requested campaign registry validation_id")
    normalized_status = _campaign_status(status)
    if normalized_status in {"approved", "locked"} and not parameter_set.get("approved_for_holdout"):
        raise ManifestValidationError("Approved or locked campaign registry requires an approved parameter set")
    manifest_record = {
        "path": str(manifest_path),
        "file_sha256": _sha256_file(manifest_path),
        "schema_version": public_manifest.get("schema_version"),
        "validation_id": public_manifest.get("validation_id"),
        "target_count": len(public_manifest.get("targets", [])),
        "target_ids": [target.get("target_id") for target in public_manifest.get("targets", [])],
        "hash_algorithm": "sha256_file",
    }
    parameter_record = _parameter_set_reference_record(parameter_set_path, parameter_set)
    target_counts = _campaign_target_counts(public_manifest)
    split_policy = dict(parameter_set.get("split_policy", {}))
    split_policy.setdefault("labels_withheld_from_runner", True)
    split_policy.setdefault("parameter_changes_after_holdout_unblinding", "forbidden")
    split_policy["observed_public_splits"] = sorted(target_counts.get("by_split", {}).keys())
    template = {
        "schema_version": CAMPAIGN_REGISTRY_SCHEMA_VERSION,
        "template_only": bool(allow_templates),
        "validation_id": validation_id,
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "status": normalized_status,
        "protocol": {
            "preregistration_reference": "replace_with_protocol_or_preregistration_record",
            "preregistration_sha256": None,
            "protocol_document": "docs/VALIDATION_FIRST_WORKFLOW.md",
            "protocol_document_sha256": _sha256_file(Path("docs/VALIDATION_FIRST_WORKFLOW.md"))
            if Path("docs/VALIDATION_FIRST_WORKFLOW.md").exists()
            else None,
            "label_release_rule": "Score only after withheld-label release and frozen run outputs.",
        },
        "public_manifest": manifest_record,
        "parameter_set": parameter_record,
        "approved_parameter_requirement": {
            "required": True,
            "require_approved_for_holdout": True,
            "locked_parameter_set_hash": parameter_set.get("parameter_set_hash"),
            "policy": "Campaign execution and scoring must use this exact approved parameter-set hash.",
        },
        "split_policy": split_policy,
        "provider_arms": _campaign_provider_arms(public_manifest, parameter_set),
        "scoring_tolerances": dict(parameter_set.get("scoring_tolerances", {})),
        "metric_definitions": {
            "primary_endpoint": parameter_set.get("analysis_plan", {}).get("primary_endpoint"),
            "secondary_endpoints": parameter_set.get("analysis_plan", {}).get("secondary_endpoints", []),
            "score_after_withheld_label_release_only": True,
            "candidate_output_freeze_required": True,
        },
        "target_counts": target_counts,
        "immutable_artifacts": [
            {
                "role": "public_manifest",
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "schema_version": public_manifest.get("schema_version"),
                "hash_algorithm": "sha256_file",
            },
            {
                "role": "parameter_set",
                "path": str(parameter_set_path),
                "sha256": _sha256_file(parameter_set_path),
                "schema_version": parameter_set.get("schema_version"),
                "parameter_set_hash": parameter_set.get("parameter_set_hash"),
                "parameter_set_id": parameter_set.get("parameter_set_id"),
                "hash_algorithm": "sha256_file",
            },
        ],
        "approval": {
            "status": normalized_status if normalized_status in {"approved", "locked"} else "draft",
            "approved_by": approved_by if normalized_status in {"approved", "locked"} else None,
            "approved_at_utc": approved_at_utc if normalized_status in {"approved", "locked"} else None,
            "approval_scope": "campaign_registry_identity_and_parameter_set_lock",
        },
        "lock": {
            "locked": normalized_status == "locked",
            "locked_by": locked_by if normalized_status == "locked" else None,
            "locked_at_utc": locked_at_utc if normalized_status == "locked" else None,
            "lock_scope": "public_manifest_parameter_set_scoring_metrics_provider_arms_and_immutable_artifact_hashes",
        },
        "withheld_labels_loaded": False,
    }
    computed_hash = campaign_registry_hash(template)
    template["registry_hash"] = computed_hash
    template["registry_id"] = campaign_registry_id_from_hash(computed_hash)
    return template


def _registry_public_manifest_drift(registry: Dict[str, Any], manifest_path: Path | str, *, allow_templates: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    public_manifest = load_public_manifest(manifest_path, allow_templates=allow_templates)
    current = {
        "path": str(manifest_path),
        "file_sha256": _sha256_file(manifest_path),
        "schema_version": public_manifest.get("schema_version"),
        "validation_id": public_manifest.get("validation_id"),
        "target_count": len(public_manifest.get("targets", [])),
        "target_ids": [target.get("target_id") for target in public_manifest.get("targets", [])],
    }
    recorded = registry.get("public_manifest", {}) if isinstance(registry.get("public_manifest"), dict) else {}
    differences = []
    for key in ("file_sha256", "validation_id", "target_count", "target_ids"):
        if recorded.get(key) != current.get(key):
            differences.append(
                {"path": f"public_manifest.{key}", "reference": recorded.get(key), "candidate": current.get(key), "difference": "drift"}
            )
    return current, differences


def _registry_parameter_set_drift(registry: Dict[str, Any], parameter_set_path: Path | str, *, allow_templates: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    parameter_set_path = Path(parameter_set_path)
    parameter_set = load_parameter_set(parameter_set_path, allow_templates=allow_templates)
    current = _parameter_set_reference_record(parameter_set_path, parameter_set)
    recorded = registry.get("parameter_set", {}) if isinstance(registry.get("parameter_set"), dict) else {}
    differences = []
    for key in ("file_sha256", "validation_id", "parameter_set_id", "parameter_set_hash", "approved_for_holdout"):
        if recorded.get(key) != current.get(key):
            differences.append(
                {"path": f"parameter_set.{key}", "reference": recorded.get(key), "candidate": current.get(key), "difference": "drift"}
            )
    return current, differences


def compare_campaign_registry(
    registry_path: Path | str,
    *,
    public_manifest_path: Optional[Path | str] = None,
    parameter_set_path: Optional[Path | str] = None,
    allow_templates: bool = False,
) -> Dict[str, Any]:
    registry = load_campaign_registry(registry_path, allow_templates=allow_templates)
    differences: List[Dict[str, Any]] = []
    current_public_manifest = None
    current_parameter_set = None
    if public_manifest_path is not None:
        current_public_manifest, manifest_differences = _registry_public_manifest_drift(
            registry,
            public_manifest_path,
            allow_templates=allow_templates,
        )
        differences.extend(manifest_differences)
    if parameter_set_path is not None:
        current_parameter_set, parameter_differences = _registry_parameter_set_drift(
            registry,
            parameter_set_path,
            allow_templates=allow_templates,
        )
        differences.extend(parameter_differences)
    drift_detected = bool(differences)
    return {
        "schema_version": CAMPAIGN_REGISTRY_COMPARISON_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "matching_registry": not drift_detected,
        "drift_detected": drift_detected,
        "registry": _campaign_registry_reference_record(registry_path, registry),
        "current_public_manifest": current_public_manifest,
        "current_parameter_set": current_parameter_set,
        "differences": differences,
        "withheld_labels_loaded": False,
    }


def verify_campaign_registry_for_inputs(
    registry_path: Path | str,
    manifest_path: Path | str,
    *,
    parameter_set_path: Optional[Path | str] = None,
    allow_templates: bool = False,
    require_approved: bool = False,
    require_locked: bool = False,
) -> Dict[str, Any]:
    registry = load_campaign_registry(
        registry_path,
        allow_templates=allow_templates,
        require_approved=require_approved,
        require_locked=require_locked,
    )
    parameter_requirement = registry.get("approved_parameter_requirement", {})
    if parameter_requirement.get("require_approved_for_holdout", True) and parameter_set_path is None:
        raise ManifestValidationError("Campaign registry requires an approved parameter set; pass --parameter-set")
    comparison = compare_campaign_registry(
        registry_path,
        public_manifest_path=manifest_path,
        parameter_set_path=parameter_set_path,
        allow_templates=allow_templates,
    )
    if comparison["drift_detected"]:
        raise ManifestValidationError("Campaign registry drift detected against current manifest or parameter set")
    record = _campaign_registry_reference_record(registry_path, registry)
    record["comparison"] = comparison
    return record


def _json_diff(reference: Any, candidate: Any, path: str = "$", *, max_differences: int = 100) -> List[Dict[str, Any]]:
    if max_differences <= 0:
        return []
    if type(reference) is not type(candidate):
        return [{"path": path, "reference": reference, "candidate": candidate, "difference": "type_or_value"}]
    if isinstance(reference, dict):
        differences: List[Dict[str, Any]] = []
        for key in sorted(set(reference) | set(candidate), key=str):
            if len(differences) >= max_differences:
                break
            child_path = f"{path}.{key}"
            if key not in reference:
                differences.append({"path": child_path, "reference": None, "candidate": candidate[key], "difference": "added"})
            elif key not in candidate:
                differences.append({"path": child_path, "reference": reference[key], "candidate": None, "difference": "removed"})
            else:
                differences.extend(
                    _json_diff(reference[key], candidate[key], child_path, max_differences=max_differences - len(differences))
                )
        return differences[:max_differences]
    if isinstance(reference, list):
        differences = []
        for idx in range(max(len(reference), len(candidate))):
            if len(differences) >= max_differences:
                break
            child_path = f"{path}[{idx}]"
            if idx >= len(reference):
                differences.append({"path": child_path, "reference": None, "candidate": candidate[idx], "difference": "added"})
            elif idx >= len(candidate):
                differences.append({"path": child_path, "reference": reference[idx], "candidate": None, "difference": "removed"})
            else:
                differences.extend(
                    _json_diff(reference[idx], candidate[idx], child_path, max_differences=max_differences - len(differences))
                )
        return differences[:max_differences]
    if reference != candidate:
        return [{"path": path, "reference": reference, "candidate": candidate, "difference": "value"}]
    return []


def compare_parameter_sets(reference_path: Path | str, candidate_path: Path | str) -> Dict[str, Any]:
    reference = load_parameter_set(reference_path, allow_templates=True)
    candidate = load_parameter_set(candidate_path, allow_templates=True)
    reference_payload = _parameter_set_hash_payload(reference)
    candidate_payload = _parameter_set_hash_payload(candidate)
    matching_hash = reference["parameter_set_hash"] == candidate["parameter_set_hash"]
    return {
        "schema_version": PARAMETER_COMPARISON_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "matching_hash": matching_hash,
        "changed": not matching_hash,
        "reference": _parameter_set_reference_record(reference_path, reference),
        "candidate": _parameter_set_reference_record(candidate_path, candidate),
        "differences": _json_diff(reference_payload, candidate_payload) if not matching_hash else [],
        "withheld_labels_loaded": False,
    }


ROBUSTNESS_PLAN_ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "template_only",
    "validation_id",
    "campaign_id",
    "robustness_plan_name",
    "created_at_utc",
    "computed_at_utc",
    "preregistration",
    "policy",
    "source_artifacts",
    "reviewer_attack_coverage",
    "variants",
    "notes",
    "robustness_plan_hash",
    "robustness_plan_id",
    "withheld_labels_loaded",
}

ROBUSTNESS_PLAN_NON_HASH_KEYS = {
    "robustness_plan_hash",
    "robustness_plan_id",
    "created_at_utc",
    "computed_at_utc",
    "template_only",
    "withheld_labels_loaded",
}

ROBUSTNESS_VARIANT_ALLOWED_KEYS = {
    "variant_id",
    "family",
    "scope",
    "arm",
    "description",
    "parameters",
    "parameter_overrides",
    "null_baseline",
    "stability_group",
    "provider",
    "comparison_arm",
    "target_selector",
    "apply_to_splits",
    "reviewer_attack",
    "expected_effect",
    "notes",
}

ROBUSTNESS_VARIANT_FAMILIES = {
    "baseline_reference",
    "void_threshold_sweep",
    "min_anomaly_voxel_threshold",
    "morphology_iterations",
    "connected_component_topology",
    "segmentation_topology",
    "connected_component_inflation",
    "depth_prior_pinn_regularization",
    "pinn_prior_regularization",
    "sar_preprocessing_variant",
    "sar_preprocessing_quality_filter",
    "top_k_candidate_cutoff",
    "candidate_volume_topk",
    "null_region_baseline",
    "random_spatial_baseline",
    "null_region_random_spatial_baseline",
    "repeat_product_date_stability",
}

ROBUSTNESS_REVIEWER_ATTACKS = {
    "threshold_sensitivity",
    "segmentation_topology",
    "connected_component_inflation",
    "pinn_prior_sensitivity",
    "sar_preprocessing_variants",
    "null_region_false_positives",
    "repeat_product_stability",
    "candidate_volume_topk",
    "physically_extreme_candidate_volume",
}


def _robustness_plan_hash_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in data.items()
        if str(key) not in ROBUSTNESS_PLAN_NON_HASH_KEYS
    }


def robustness_plan_hash(data: Dict[str, Any]) -> str:
    """Return the deterministic hash of a no-label robustness/ablation plan."""
    return _stable_json_sha256(_robustness_plan_hash_payload(data))


def robustness_plan_id_from_hash(hash_value: str) -> str:
    return f"{ROBUSTNESS_PLAN_ID_PREFIX}{str(hash_value)[:ROBUSTNESS_PLAN_ID_HASH_CHARS]}"


def _validate_json_metadata_payload(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ManifestValidationError(f"{field} must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_metadata_payload(item, f"{field}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ManifestValidationError(f"{field} object keys must be non-empty strings")
            normalized[key] = _validate_json_metadata_payload(child, f"{field}.{key}")
        return normalized
    raise ManifestValidationError(f"{field} must be JSON scalar, list, or object")


def _normalize_string_list(value: Any, field: str, *, allow_empty: bool = True) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestValidationError(f"{field} must be a list")
    normalized = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ManifestValidationError(f"{field}[{idx}] must be a non-empty string")
        normalized.append(item.strip())
    if not allow_empty and not normalized:
        raise ManifestValidationError(f"{field} must not be empty")
    return normalized


def _robustness_holdout_scope_allowed(plan: Dict[str, Any]) -> bool:
    preregistration = plan.get("preregistration", {}) if isinstance(plan.get("preregistration"), dict) else {}
    status = str(preregistration.get("status") or "").strip().lower()
    unblinding_status = str(preregistration.get("holdout_unblinding_status") or "").strip().lower()
    return status in {
        "preregistered_before_holdout_unblinding",
        "locked_before_holdout_unblinding",
    } and unblinding_status in {"not_unblinded", "withheld", "labels_withheld"}


def _normalize_robustness_variant(variant: Any, idx: int, *, holdout_scope_allowed: bool) -> Dict[str, Any]:
    field = f"variants[{idx}]"
    if not isinstance(variant, dict):
        raise ManifestValidationError(f"{field} must be an object")
    unknown = sorted(str(key) for key in variant if key not in ROBUSTNESS_VARIANT_ALLOWED_KEYS)
    if unknown:
        raise ManifestValidationError(f"{field} has unsupported key(s): {unknown}")
    normalized = dict(variant)
    variant_id = normalized.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id.strip():
        raise ManifestValidationError(f"{field}.variant_id must be a non-empty string")
    normalized["variant_id"] = _safe_name(variant_id)
    if normalized["variant_id"] != variant_id.strip():
        raise ManifestValidationError(f"{field}.variant_id must contain only letters, numbers, dash, or underscore")
    family = normalized.get("family")
    if not isinstance(family, str) or family not in ROBUSTNESS_VARIANT_FAMILIES:
        raise ManifestValidationError(f"{field}.family must be one of {sorted(ROBUSTNESS_VARIANT_FAMILIES)}")
    normalized["family"] = family
    scope = str(normalized.get("scope") or "calibration_only").strip().lower()
    if scope != "calibration_only" and not holdout_scope_allowed:
        raise ManifestValidationError(
            f"{field}.scope must be calibration_only unless the plan was preregistered before holdout unblinding"
        )
    normalized["scope"] = scope
    arm = normalized.get("arm") or family
    if not isinstance(arm, str) or not arm.strip():
        raise ManifestValidationError(f"{field}.arm must be a non-empty string")
    normalized["arm"] = _safe_name(arm)
    parameters = normalized.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ManifestValidationError(f"{field}.parameters must be an object")
    normalized["parameters"] = _validate_json_metadata_payload(parameters, f"{field}.parameters")
    if "parameter_overrides" in normalized:
        if not isinstance(normalized["parameter_overrides"], dict):
            raise ManifestValidationError(f"{field}.parameter_overrides must be an object")
        normalized["parameter_overrides"] = _validate_json_metadata_payload(
            normalized["parameter_overrides"], f"{field}.parameter_overrides"
        )
    normalized["null_baseline"] = bool(normalized.get("null_baseline", family in {"null_region_baseline", "random_spatial_baseline", "null_region_random_spatial_baseline"}))
    if "stability_group" in normalized and not isinstance(normalized["stability_group"], str):
        raise ManifestValidationError(f"{field}.stability_group must be a string")
    if "provider" in normalized:
        provider = normalized["provider"]
        if not isinstance(provider, str) or provider not in SUPPORTED_SAR_PROVIDERS:
            raise ManifestValidationError(f"{field}.provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}")
    for text_key in ("comparison_arm", "description", "reviewer_attack", "expected_effect", "notes"):
        if text_key in normalized and normalized[text_key] is not None and not isinstance(normalized[text_key], str):
            raise ManifestValidationError(f"{field}.{text_key} must be a string")
    if "apply_to_splits" in normalized:
        normalized["apply_to_splits"] = _normalize_string_list(normalized["apply_to_splits"], f"{field}.apply_to_splits")
    if "target_selector" in normalized:
        if not isinstance(normalized["target_selector"], dict):
            raise ManifestValidationError(f"{field}.target_selector must be an object")
        normalized["target_selector"] = _validate_json_metadata_payload(normalized["target_selector"], f"{field}.target_selector")
    return normalized


def load_robustness_plan(path: Path | str, *, allow_templates: bool = False) -> Dict[str, Any]:
    """Load and validate a public no-label robustness/ablation plan."""
    plan_path = Path(path)
    data = _load_json(plan_path)
    if data.get("schema_version") != ROBUSTNESS_PLAN_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Robustness plan schema_version must be {ROBUSTNESS_PLAN_SCHEMA_VERSION!r}"
        )
    unknown = sorted(str(key) for key in data if key not in ROBUSTNESS_PLAN_ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ManifestValidationError(f"Robustness plan has unsupported top-level key(s): {unknown}")
    if data.get("template_only") and not allow_templates:
        raise ManifestValidationError(
            "Robustness plan is marked template_only; copy it and remove template_only after filling real values"
        )
    _reject_campaign_registry_private_keys(data)
    validation_id = data.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestValidationError("Robustness plan validation_id must be a non-empty string")
    campaign_id = data.get("campaign_id")
    if campaign_id is not None and (not isinstance(campaign_id, str) or not campaign_id.strip()):
        raise ManifestValidationError("Robustness plan campaign_id must be a non-empty string when provided")
    preregistration = data.get("preregistration", {})
    if not isinstance(preregistration, dict):
        raise ManifestValidationError("Robustness plan preregistration must be an object")
    policy = data.get("policy", {})
    if not isinstance(policy, dict):
        raise ManifestValidationError("Robustness plan policy must be an object")
    policy = dict(policy)
    policy.setdefault("labels_withheld_from_runner", True)
    policy.setdefault("no_downloads_or_training", True)
    policy.setdefault("default_execution_mode", "dry_run_plan_only")
    policy.setdefault("ablation_use_rule", "calibration_only_unless_preregistered_before_holdout_unblinding")
    policy.setdefault("parameter_changes_after_holdout_unblinding", "forbidden")
    if policy.get("labels_withheld_from_runner") is not True:
        raise ManifestValidationError("Robustness plan policy.labels_withheld_from_runner must be true")
    if policy.get("no_downloads_or_training") is not True:
        raise ManifestValidationError("Robustness plan policy.no_downloads_or_training must be true")
    if policy.get("default_execution_mode") != "dry_run_plan_only":
        raise ManifestValidationError("Robustness plan policy.default_execution_mode must be dry_run_plan_only")
    if policy.get("parameter_changes_after_holdout_unblinding") != "forbidden":
        raise ManifestValidationError("Robustness plan policy.parameter_changes_after_holdout_unblinding must be forbidden")
    source_artifacts = data.get("source_artifacts", {})
    if not isinstance(source_artifacts, dict):
        raise ManifestValidationError("Robustness plan source_artifacts must be an object")
    coverage = _normalize_string_list(data.get("reviewer_attack_coverage", []), "reviewer_attack_coverage")
    unknown_coverage = sorted(item for item in coverage if item not in ROBUSTNESS_REVIEWER_ATTACKS)
    if unknown_coverage:
        raise ManifestValidationError(f"Robustness plan reviewer_attack_coverage has unsupported value(s): {unknown_coverage}")
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ManifestValidationError("Robustness plan variants must be a non-empty list")
    holdout_scope_allowed = _robustness_holdout_scope_allowed(data)
    normalized_variants = [
        _normalize_robustness_variant(variant, idx, holdout_scope_allowed=holdout_scope_allowed)
        for idx, variant in enumerate(variants)
    ]
    variant_ids = [variant["variant_id"] for variant in normalized_variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise ManifestValidationError("Robustness plan variant_id values must be unique")
    normalized = dict(data)
    normalized["validation_id"] = validation_id.strip()
    if campaign_id is not None:
        normalized["campaign_id"] = campaign_id.strip()
    normalized["preregistration"] = _validate_json_metadata_payload(preregistration, "preregistration")
    normalized["policy"] = _validate_json_metadata_payload(policy, "policy")
    normalized["source_artifacts"] = _validate_json_metadata_payload(source_artifacts, "source_artifacts")
    normalized["reviewer_attack_coverage"] = coverage
    normalized["variants"] = normalized_variants
    normalized["withheld_labels_loaded"] = False
    computed_hash = robustness_plan_hash(normalized)
    computed_id = robustness_plan_id_from_hash(computed_hash)
    recorded_hash = normalized.get("robustness_plan_hash")
    if recorded_hash is not None and str(recorded_hash) != computed_hash:
        raise ManifestValidationError("Robustness plan robustness_plan_hash does not match canonical content")
    recorded_id = normalized.get("robustness_plan_id")
    if recorded_id is not None and str(recorded_id) != computed_id:
        raise ManifestValidationError("Robustness plan robustness_plan_id does not match canonical content hash")
    normalized["robustness_plan_hash"] = computed_hash
    normalized["robustness_plan_id"] = computed_id
    return normalized


def _robustness_plan_reference_record(path: Path | str, data: Dict[str, Any]) -> Dict[str, Any]:
    plan_path = Path(path)
    return {
        "path": str(plan_path),
        "file_sha256": _sha256_file(plan_path) if plan_path.exists() else None,
        "schema_version": data.get("schema_version"),
        "validation_id": data.get("validation_id"),
        "campaign_id": data.get("campaign_id"),
        "robustness_plan_name": data.get("robustness_plan_name"),
        "robustness_plan_id": data.get("robustness_plan_id"),
        "robustness_plan_hash": data.get("robustness_plan_hash"),
        "variant_count": len(data.get("variants", [])) if isinstance(data.get("variants"), list) else 0,
        "hash_algorithm": "sha256_canonical_json_excluding_identity_time_and_template_metadata",
        "withheld_labels_loaded": False,
    }


def robustness_plan_validation_summary(plan_path: Path | str, *, allow_templates: bool = False) -> Dict[str, Any]:
    plan = load_robustness_plan(plan_path, allow_templates=allow_templates)
    variants = plan.get("variants", [])
    by_family: Dict[str, int] = {}
    by_scope: Dict[str, int] = {}
    reviewer_attacks: Dict[str, int] = {}
    for variant in variants:
        family = str(variant.get("family") or "unspecified")
        scope = str(variant.get("scope") or "unspecified")
        by_family[family] = by_family.get(family, 0) + 1
        by_scope[scope] = by_scope.get(scope, 0) + 1
        attack = variant.get("reviewer_attack") or variant.get("arm") or family
        reviewer_attacks[str(attack)] = reviewer_attacks.get(str(attack), 0) + 1
    return {
        "schema_version": ROBUSTNESS_PLAN_VALIDATION_SCHEMA_VERSION,
        "valid": True,
        "plan": _robustness_plan_reference_record(plan_path, plan),
        "summary": {
            "variant_count": len(variants),
            "calibration_only": all(str(variant.get("scope")) == "calibration_only" for variant in variants),
            "null_baseline_variant_count": sum(1 for variant in variants if variant.get("null_baseline")),
            "repeat_stability_variant_count": sum(1 for variant in variants if variant.get("stability_group")),
            "by_family": dict(sorted(by_family.items())),
            "by_scope": dict(sorted(by_scope.items())),
            "reviewer_attack_coverage": list(plan.get("reviewer_attack_coverage", [])),
            "variant_reviewer_attack_counts": dict(sorted(reviewer_attacks.items())),
        },
        "policy": plan.get("policy"),
        "withheld_labels_loaded": False,
    }


def _robustness_target_in_scope(target: Dict[str, Any], *, include_audit_only: bool = False, include_holdout: bool = False) -> bool:
    split = str(target.get("split") or target.get("split_designation") or "").strip().lower()
    if split == "calibration" or "calibration" in split:
        return True
    if include_audit_only and (target.get("audit_only") or split == "audit" or "audit" in split):
        return True
    if include_holdout and split == "holdout":
        return True
    return False


def _robustness_command_for_step(
    *,
    manifest_path: Path,
    output_dir: Path,
    target: Dict[str, Any],
    parameter_set_path: Path,
    registry_path: Path,
    robustness_plan_path: Path,
    variant: Dict[str, Any],
    include_audit_only: bool,
) -> List[str]:
    target_id = str(target["target_id"])
    variant_id = str(variant["variant_id"])
    target_output_dir = output_dir / "variants" / _safe_name(variant_id) / "targets" / _safe_name(target_id)
    command = [
        "python",
        "blind_validation.py",
        "run",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(target_output_dir),
        "--target-id",
        target_id,
        "--parameter-set",
        str(parameter_set_path),
        "--require-approved-parameters",
        "--campaign-registry",
        str(registry_path),
        "--require-locked-campaign-registry",
        "--robustness-plan",
        str(robustness_plan_path),
        "--variant-id",
        variant_id,
    ]
    if include_audit_only:
        command.append("--include-audit-only")
    return command


def plan_robustness_ablations(
    robustness_plan_path: Path | str,
    manifest_path: Path | str,
    registry_path: Path | str,
    parameter_set_path: Path | str,
    output_dir: Path | str,
    *,
    allow_templates: bool = False,
    include_audit_only: bool = False,
    include_preregistered_holdout: bool = False,
) -> Dict[str, Any]:
    """Build a deterministic dry-run-only robustness/ablation execution plan."""
    robustness_plan_path = Path(robustness_plan_path)
    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    parameter_set_path = Path(parameter_set_path)
    output_dir = Path(output_dir)
    robustness_plan = load_robustness_plan(robustness_plan_path, allow_templates=allow_templates)
    if include_preregistered_holdout and not _robustness_holdout_scope_allowed(robustness_plan):
        raise ManifestValidationError("Holdout ablation planning requires preregistration before holdout unblinding")
    public_manifest, registry, parameter_set, _product_lock, registry_record, _product_lock_record = _load_campaign_inputs_for_execution(
        manifest_path,
        registry_path,
        parameter_set_path,
        None,
        allow_templates=allow_templates,
        require_locked_registry=True,
        require_approved_parameters=True,
    )
    if robustness_plan.get("validation_id") != public_manifest.get("validation_id"):
        raise ManifestValidationError("Robustness plan validation_id does not match public manifest validation_id")
    if robustness_plan.get("campaign_id") and robustness_plan.get("campaign_id") != registry.get("campaign_id"):
        raise ManifestValidationError("Robustness plan campaign_id does not match campaign registry campaign_id")
    selected_targets = [
        target
        for target in public_manifest.get("targets", [])
        if isinstance(target, dict)
        and _robustness_target_in_scope(
            target,
            include_audit_only=include_audit_only,
            include_holdout=include_preregistered_holdout,
        )
    ]
    selected_targets = sorted(selected_targets, key=lambda item: str(item.get("target_id")))
    variants = sorted(robustness_plan.get("variants", []), key=lambda item: str(item.get("variant_id")))
    steps = []
    for target in selected_targets:
        target_id = str(target.get("target_id"))
        provider_summary = _target_provider_execution_summary(target)
        for variant in variants:
            variant_id = str(variant.get("variant_id"))
            command = _robustness_command_for_step(
                manifest_path=manifest_path,
                output_dir=output_dir,
                target=target,
                parameter_set_path=parameter_set_path,
                registry_path=registry_path,
                robustness_plan_path=robustness_plan_path,
                variant=variant,
                include_audit_only=include_audit_only,
            )
            target_output_dir = output_dir / "variants" / _safe_name(variant_id) / "targets" / _safe_name(target_id)
            step = {
                "step_id": f"{_safe_name(variant_id)}::{_safe_name(target_id)}",
                "variant_id": variant_id,
                "variant_family": variant.get("family"),
                "variant_arm": variant.get("arm"),
                "variant_scope": variant.get("scope"),
                "null_baseline": bool(variant.get("null_baseline")),
                "stability_group": variant.get("stability_group"),
                "variant_parameters": variant.get("parameters", {}),
                "target_id": target_id,
                "name": target.get("name"),
                "split": target.get("split"),
                "status": "planned_dry_run_only",
                "execution_mode": "dry_run_plan_only",
                "executable": True,
                "provider_execution": provider_summary,
                "target_output_dir": str(target_output_dir),
                "run_manifest_path": str(target_output_dir / "run_manifest.json"),
                "command": command,
                "command_display": " ".join(command),
                "downloads_attempted_by_plan": False,
                "training_attempted_by_plan": False,
                "withheld_labels_loaded": False,
            }
            step.update(_public_campaign_metadata_record(target))
            steps.append(step)
    by_family = {family: sum(1 for variant in variants if variant.get("family") == family) for family in {variant.get("family") for variant in variants}}
    by_target_split = {split: sum(1 for target in selected_targets if str(target.get("split") or "unspecified") == split) for split in {str(target.get("split") or "unspecified") for target in selected_targets}}
    plan = {
        "schema_version": ROBUSTNESS_EXECUTION_PLAN_SCHEMA_VERSION,
        "validation_id": public_manifest.get("validation_id"),
        "campaign_id": registry.get("campaign_id"),
        "campaign_registry": registry_record,
        "campaign_registry_id": registry_record.get("registry_id"),
        "campaign_registry_hash": registry_record.get("registry_hash"),
        "campaign_registry_status": registry_record.get("status"),
        "public_manifest": str(manifest_path),
        "public_manifest_sha256": _sha256_file(manifest_path),
        "parameter_set": _parameter_set_reference_record(parameter_set_path, parameter_set),
        "parameter_set_id": parameter_set.get("parameter_set_id"),
        "parameter_set_hash": parameter_set.get("parameter_set_hash"),
        "robustness_plan": _robustness_plan_reference_record(robustness_plan_path, robustness_plan),
        "robustness_plan_id": robustness_plan.get("robustness_plan_id"),
        "robustness_plan_hash": robustness_plan.get("robustness_plan_hash"),
        "output_dir": str(output_dir),
        "run_policy": {
            "default_no_download": True,
            "downloads_attempted_by_plan": False,
            "training_attempted_by_plan": False,
            "execute_real_requested": False,
            "explicit_real_confirmation_required": False,
            "synthetic_fallback_allowed": False,
            "ablation_use_rule": robustness_plan.get("policy", {}).get("ablation_use_rule"),
            "calibration_only_default": True,
            "audit_only_included": bool(include_audit_only),
            "preregistered_holdout_included": bool(include_preregistered_holdout),
        },
        "summary": {
            "target_count": len(selected_targets),
            "variant_count": len(variants),
            "planned_step_count": len(steps),
            "null_baseline_variant_count": sum(1 for variant in variants if variant.get("null_baseline")),
            "repeat_stability_variant_count": sum(1 for variant in variants if variant.get("stability_group")),
            "by_variant_family": dict(sorted((str(key), value) for key, value in by_family.items())),
            "by_target_split": dict(sorted(by_target_split.items())),
        },
        "steps": steps,
        "withheld_labels_loaded": False,
    }
    return _round_metric(plan)


def verify_parameter_set_for_scoring(
    run_manifest: Dict[str, Any],
    *,
    parameter_set_path: Optional[Path | str] = None,
    require_approved: bool = False,
) -> Dict[str, Any]:
    """Verify score-time parameter-set identity without loading or inspecting labels."""
    run_record = run_manifest.get("parameter_set") if isinstance(run_manifest.get("parameter_set"), dict) else None
    result: Dict[str, Any] = {
        "checked": bool(parameter_set_path is not None or require_approved or run_record),
        "required_approved_for_holdout": bool(require_approved),
        "run_parameter_set": run_record,
        "provided_parameter_set": None,
        "matching_hash": None,
        "approved_for_holdout": bool(run_record.get("approved_for_holdout")) if run_record else False,
        "status": "not_checked",
        "withheld_labels_loaded": False,
    }
    provided = None
    if parameter_set_path is not None:
        provided = load_parameter_set(parameter_set_path, require_approved=require_approved)
        result["provided_parameter_set"] = _parameter_set_reference_record(parameter_set_path, provided)
        if provided.get("validation_id") != run_manifest.get("validation_id"):
            raise ManifestValidationError("Parameter set validation_id does not match run manifest validation_id")
        if run_record is None:
            raise ManifestValidationError("Run manifest does not record a parameter set; cannot compare score-time set")
        matching_hash = provided["parameter_set_hash"] == run_record.get("parameter_set_hash")
        result["matching_hash"] = matching_hash
        if not matching_hash:
            raise ManifestValidationError("Score-time parameter set hash does not match run-manifest parameter set hash")
        result["approved_for_holdout"] = bool(provided.get("approved_for_holdout"))

    if require_approved:
        if run_record is None and provided is None:
            raise ManifestValidationError("Holdout scoring requires an approved parameter set, but run manifest has none")
        if not result["approved_for_holdout"]:
            raise ManifestValidationError("Parameter set is not approved for holdout scoring")
    result["status"] = "verified" if result["checked"] else "not_checked"
    return result


def _default_parameter_set_template(validation_id: str, parameter_set_name: str) -> Dict[str, Any]:
    template = {
        "schema_version": PARAMETER_SET_SCHEMA_VERSION,
        "template_only": True,
        "validation_id": validation_id,
        "parameter_set_name": parameter_set_name,
        "model_card": {
            "model_family": "Biondi SAR Doppler tomography validation harness",
            "intended_use": "Blind known-void validation with frozen no-label parameters",
            "training_policy": "No long training is started by parameter-set validation commands.",
        },
        "analysis_plan": {
            "primary_endpoint": "positive_site_hit_rate_with_negative_false_positive_rate",
            "secondary_endpoints": ["rank_of_first_hit", "localization_error_m", "candidates_per_km2"],
            "holdout_rule": "Do not alter this parameter set after withheld labels are exposed.",
        },
        "thresholds": {
            "void_probability_threshold": 0.35,
            "candidate_confidence_threshold": 0.0,
            "min_anomaly_voxels": 3,
        },
        "resolution_profile": {
            "name": "quick",
            "grid_nx": 64,
            "grid_ny": 64,
            "grid_nz": 32,
            "domain_width_m": 800.0,
            "max_depth_m": 500.0,
        },
        "pinn_settings": {
            "epochs": 500,
            "physics_weight": 1.0,
            "data_weight": 20.0,
            "sparsity_weight": 1.0,
            "regularization_weight": 0.1,
            "deep_prior_weight": 1.0,
            "surface_prior_weight": 0.0,
            "excitation_frequency_hz": 2.0,
        },
        "scoring_tolerances": {
            "default_horizontal_tolerance_m": DEFAULT_HORIZONTAL_TOLERANCE_M,
            "default_depth_tolerance_m": DEFAULT_DEPTH_TOLERANCE_M,
        },
        "acquisition_provider_preferences": {
            "primary_provider": "sentinel1_asf",
            "provider_preferences": ["sentinel1_asf", "umbra_open_data", "xband_spotlight_placeholder"],
            "comparison_arms": [
                {"provider": "sentinel1_asf", "comparison_arm": "sentinel1_cband_iw"},
                {"provider": "umbra_open_data", "comparison_arm": "umbra_xband_spotlight_placeholder"},
                {"provider": "xband_spotlight_placeholder", "comparison_arm": "commercial_xband_spotlight_placeholder"},
            ],
            "real_downloads_default": False,
        },
        "split_policy": {
            "calibration_split": "calibration",
            "holdout_split": "holdout",
            "labels_withheld_from_runner": True,
            "parameter_changes_after_holdout_unblinding": "forbidden",
        },
        "approval": {
            "status": "draft",
            "approved_by": None,
            "approved_at_utc": None,
            "approval_scope": "not_approved_for_holdout_until_status_is_approved",
        },
    }
    template["acquisition_provider_preferences"] = _validate_parameter_provider_preferences(
        template["acquisition_provider_preferences"],
        "acquisition_provider_preferences",
    )
    computed_hash = parameter_set_hash(template)
    template["parameter_set_hash"] = computed_hash
    template["parameter_set_id"] = parameter_set_id_from_hash(computed_hash)
    return template


def _provider_profile(provider: str) -> Dict[str, Any]:
    if provider not in SUPPORTED_SAR_PROVIDERS:
        raise ManifestValidationError(f"SAR provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}")
    return SAR_PROVIDER_PROFILES[provider]


def _normalize_provider_preferences(value: Any, field: str, primary_provider: str) -> List[str]:
    if value is None:
        return [primary_provider]
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(f"{field} must be a non-empty list")
    providers = []
    for idx, provider in enumerate(value):
        if not isinstance(provider, str) or provider not in SUPPORTED_SAR_PROVIDERS:
            raise ManifestValidationError(f"{field}[{idx}] must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}")
        providers.append(provider)
    if primary_provider not in providers:
        providers.insert(0, primary_provider)
    return providers


def _normalize_sar_search(search: Any, target_path: str) -> Dict[str, Any]:
    if search is None:
        return {}
    if not isinstance(search, dict):
        raise ManifestValidationError(f"{target_path}.sar_search must be an object")
    allowed_keys = {
        "provider",
        "provider_label",
        "provider_preferences",
        "comparison_arm",
        "comparison_group",
        "platform",
        "band",
        "processing_level",
        "beam_mode",
        "acquisition_mode",
        "polarization",
        "flight_direction",
        "start_date",
        "end_date",
        "max_results",
        "selection_count",
        "resolution_m",
        "placeholder_only",
        "notes",
    }
    unknown = sorted(str(key) for key in search if key not in allowed_keys)
    if unknown:
        raise ManifestValidationError(f"{target_path}.sar_search has unsupported key(s): {unknown}")
    normalized = dict(search)
    provider = normalized.get("provider", "sentinel1_asf")
    if not isinstance(provider, str) or provider not in SUPPORTED_SAR_PROVIDERS:
        raise ManifestValidationError(
            f"{target_path}.sar_search.provider must be one of {sorted(SUPPORTED_SAR_PROVIDERS)}"
        )
    profile = _provider_profile(provider)
    normalized["provider"] = provider
    normalized["provider_id"] = provider
    normalized["provider_label"] = normalized.get("provider_label", profile["provider_label"])
    normalized["provider_preferences"] = _normalize_provider_preferences(
        normalized.get("provider_preferences"), f"{target_path}.sar_search.provider_preferences", provider
    )
    normalized["comparison_arm"] = str(normalized.get("comparison_arm") or profile["default_comparison_arm"])
    if "comparison_group" in normalized and not isinstance(normalized["comparison_group"], str):
        raise ManifestValidationError(f"{target_path}.sar_search.comparison_group must be a string")
    normalized["platform"] = normalized.get("platform", profile["platform"])
    normalized["band"] = normalized.get("band", profile["band"])
    normalized["processing_level"] = normalized.get("processing_level", profile["processing_level"])
    normalized["beam_mode"] = normalized.get("beam_mode", profile["beam_mode"])
    normalized["acquisition_mode"] = normalized.get("acquisition_mode", profile["acquisition_mode"])
    normalized["polarization"] = normalized.get("polarization", "VV")
    normalized["search_supported"] = bool(profile["search_supported"])
    normalized["download_supported"] = bool(profile["download_supported"])
    normalized["real_lock_supported"] = bool(profile["real_lock_supported"])
    normalized["requires_credentials"] = bool(profile["requires_credentials"])
    normalized["limitations"] = list(profile.get("limitations", []))
    normalized["placeholder_only"] = bool(normalized.get("placeholder_only", not profile["search_supported"]))
    flight_direction = normalized.get("flight_direction")
    if flight_direction is not None:
        direction = str(flight_direction).upper()
        if direction not in {"ASCENDING", "DESCENDING"}:
            raise ManifestValidationError(
                f"{target_path}.sar_search.flight_direction must be ASCENDING or DESCENDING"
            )
        normalized["flight_direction"] = direction
    for key in (
        "start_date",
        "end_date",
        "polarization",
        "platform",
        "processing_level",
        "beam_mode",
        "band",
        "acquisition_mode",
        "provider_label",
        "comparison_arm",
    ):
        if key in normalized and not isinstance(normalized[key], str):
            raise ManifestValidationError(f"{target_path}.sar_search.{key} must be a string")
    if "max_results" in normalized:
        normalized["max_results"] = _safe_positive_int(
            normalized["max_results"], f"{target_path}.sar_search.max_results"
        )
    if "selection_count" in normalized:
        normalized["selection_count"] = _safe_positive_int(
            normalized["selection_count"], f"{target_path}.sar_search.selection_count"
        )
    if "resolution_m" in normalized:
        normalized["resolution_m"] = _as_float(
            normalized["resolution_m"], f"{target_path}.sar_search.resolution_m", minimum=0.0
        )
    return normalized


def _normalize_comparison_arms(value: Any, target_path: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(f"{target_path}.comparison_arms must be a non-empty list")
    normalized_arms = []
    seen = set()
    for idx, arm in enumerate(value):
        arm_path = f"{target_path}.comparison_arms[{idx}]"
        normalized = _normalize_sar_search(arm, arm_path)
        arm_name = normalized.get("comparison_arm")
        if arm_name in seen:
            raise ManifestValidationError(f"Duplicate comparison_arm {arm_name!r} in {target_path}.comparison_arms")
        seen.add(arm_name)
        normalized_arms.append(normalized)
    return normalized_arms


def _target_comparison_arms(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(target.get("comparison_arms"), list) and target["comparison_arms"]:
        return [dict(arm) for arm in target["comparison_arms"]]
    if isinstance(target.get("sar_search"), dict):
        return [dict(target["sar_search"])]
    return [_normalize_sar_search({}, "sar_search")]


def _provider_label_from_search(search: Dict[str, Any]) -> str:
    return str(search.get("provider_label") or search.get("provider") or "unspecified")


def _comparison_arm_from_search(search: Dict[str, Any]) -> str:
    return str(search.get("comparison_arm") or search.get("provider") or "unspecified")


def _load_env_file_secret_safe(env_path: Optional[Path] = None) -> Dict[str, Any]:
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    env_path = Path(env_path)
    result = {
        "loaded": False,
        "path": str(env_path),
        "entry_count": 0,
        "secret_values_recorded": False,
    }
    if not env_path.exists():
        return result
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
            result["entry_count"] += 1
    result["loaded"] = True
    return result


def _detect_auth_mode(environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = os.environ if environ is None else environ
    token_source = None
    for key in ("EARTHDATA_TOKEN", "EARTHDATA_BEARER_TOKEN"):
        if str(env.get(key, "")).strip():
            token_source = key
            break
    username_present = bool(str(env.get("EARTHDATA_USERNAME", "")).strip())
    password_present = bool(str(env.get("EARTHDATA_PASSWORD", "")).strip())
    if token_source:
        mode = "token"
    elif username_present and password_present:
        mode = "credentials"
    elif username_present or password_present:
        mode = "incomplete_credentials"
    else:
        mode = "none"
    return {
        "mode": mode,
        "token_source": token_source,
        "username_present": username_present,
        "password_present": password_present,
        "secret_values_recorded": False,
    }


def _redact_env_secret_values(text: str) -> str:
    redacted = str(text)
    secret_values = []
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in SENSITIVE_KEY_MARKERS):
            cleaned = str(value).strip()
            if cleaned:
                secret_values.append(cleaned)
                if cleaned.lower().startswith("bearer "):
                    secret_values.append(cleaned[7:].strip())
    for secret in sorted(set(secret_values), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _safe_error_summary(exc: BaseException, max_chars: int = 800) -> str:
    summary = _redact_env_secret_values(f"{exc.__class__.__name__}: {exc}")
    summary = summary.replace("\n", " ").replace("\r", " ")
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def _target_bbox(target: Dict[str, Any]) -> Tuple[float, float, float, float]:
    center = target["center"]
    buffer_deg = float(target.get("buffer_deg", 0.02))
    lat = float(center["lat"])
    lon = float(center["lon"])
    return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)


def _resolve_target_search_params(
    target: Dict[str, Any],
    *,
    search: Optional[Dict[str, Any]] = None,
    start_date: str = DEFAULT_SAR_SEARCH_START_DATE,
    end_date: str = DEFAULT_SAR_SEARCH_END_DATE,
    max_results: int = DEFAULT_SAR_MAX_RESULTS,
    selection_count: int = DEFAULT_PRODUCT_SELECTION_COUNT,
) -> Dict[str, Any]:
    search = dict(search if search is not None else target.get("sar_search", {}))
    provider = search.get("provider", "sentinel1_asf")
    profile = SAR_PROVIDER_PROFILES.get(str(provider), SAR_PROVIDER_PROFILES["sentinel1_asf"])
    resolved = {
        "provider": provider,
        "provider_label": search.get("provider_label", profile["provider_label"]),
        "provider_preferences": list(search.get("provider_preferences", [provider])),
        "comparison_arm": search.get("comparison_arm", profile["default_comparison_arm"]),
        "comparison_group": search.get("comparison_group"),
        "platform": search.get("platform", profile["platform"]),
        "band": search.get("band", profile["band"]),
        "processing_level": search.get("processing_level", profile["processing_level"]),
        "beam_mode": search.get("beam_mode", profile["beam_mode"]),
        "acquisition_mode": search.get("acquisition_mode", profile["acquisition_mode"]),
        "polarization": search.get("polarization", "VV"),
        "flight_direction": search.get("flight_direction"),
        "start_date": search.get("start_date", start_date),
        "end_date": search.get("end_date", end_date),
        "max_results": int(search.get("max_results", max_results)),
        "selection_count": int(search.get("selection_count", selection_count)),
        "search_supported": bool(search.get("search_supported", profile["search_supported"])),
        "download_supported": bool(search.get("download_supported", profile["download_supported"])),
        "real_lock_supported": bool(search.get("real_lock_supported", profile["real_lock_supported"])),
        "placeholder_only": bool(search.get("placeholder_only", not profile["search_supported"])),
        "requires_credentials": bool(search.get("requires_credentials", profile["requires_credentials"])),
        "limitations": list(search.get("limitations", profile.get("limitations", []))),
    }
    if "resolution_m" in search:
        resolved["resolution_m"] = search.get("resolution_m")
    resolved["bbox"] = list(_target_bbox(target))
    return resolved


def _normalize_product_metadata(product: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    product_id = (
        product.get("product_id")
        or product.get("file_id")
        or product.get("granule_name")
        or product.get("sceneName")
        or product.get("id")
        or f"result_{source_index:04d}"
    )
    product_name = product.get("granule_name") or product.get("product_name") or product_id
    size_mb = _finite_or_none(product.get("size_mb"))
    if size_mb is None and _finite_or_none(product.get("bytes")) is not None:
        size_mb = float(product["bytes"]) / (1024.0 * 1024.0)
    normalized = {
        "product_id": str(product_id),
        "product_name": str(product_name),
        "provider": product.get("provider") or product.get("provider_label") or "ASF Sentinel-1",
        "provider_id": product.get("provider_id"),
        "comparison_arm": product.get("comparison_arm"),
        "comparison_group": product.get("comparison_group"),
        "band": product.get("band"),
        "acquisition_mode": product.get("acquisition_mode"),
        "platform": product.get("platform") or "Sentinel-1",
        "processing_level": product.get("processing_level") or product.get("processingLevel") or "SLC",
        "beam_mode": product.get("beam_mode") or product.get("beamMode") or product.get("beamModeType"),
        "polarization": product.get("polarization"),
        "flight_direction": product.get("flight_direction") or product.get("flightDirection"),
        "start_time": product.get("start_time") or product.get("startTime") or product.get("datetime"),
        "stop_time": product.get("stop_time") or product.get("stopTime"),
        "relative_orbit": product.get("relative_orbit") or product.get("pathNumber"),
        "orbit_number": product.get("orbit_number") or product.get("orbit"),
        "frame_number": product.get("frame_number") or product.get("frameNumber"),
        "size_mb": size_mb,
        "source_result_index": int(source_index),
        "download_url_recorded": False,
        "secret_values_recorded": False,
    }
    return {key: value for key, value in normalized.items() if value is not None}


def _parse_product_time(value: Any) -> float:
    if not value:
        return float("-inf")
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")


def _sort_products_for_lock(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        products,
        key=lambda product: (
            -_parse_product_time(product.get("start_time")),
            str(product.get("product_id") or product.get("product_name") or ""),
        ),
    )


def _estimated_products_size_mb(products: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Return summed known product sizes in MB, or None when no sizes are known."""
    known_sizes = []
    for product in products:
        if not isinstance(product, dict):
            continue
        size_mb = _finite_or_none(product.get("size_mb"))
        if size_mb is not None and size_mb >= 0:
            known_sizes.append(size_mb)
    if not known_sizes:
        return None
    return round(sum(known_sizes), 3)


def _mb_to_gb(size_mb: Optional[float]) -> Optional[float]:
    return round(float(size_mb) / 1024.0, 3) if size_mb is not None else None


def _load_optional_previous_lock(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    data = _load_json(path)
    if data.get("schema_version") != PRODUCT_LOCK_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Product lock schema_version must be {PRODUCT_LOCK_SCHEMA_VERSION!r}"
        )
    return data


def _lock_target_key(target: Dict[str, Any]) -> str:
    return f"{target.get('target_id')}::{target.get('comparison_arm') or 'default'}"


def load_product_lock(path: Path | str) -> Dict[str, Any]:
    lock_path = Path(path)
    data = _load_json(lock_path)
    if data.get("schema_version") != PRODUCT_LOCK_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Product lock schema_version must be {PRODUCT_LOCK_SCHEMA_VERSION!r}"
        )
    if not isinstance(data.get("targets"), list):
        raise ManifestValidationError("Product lock targets must be a list")
    return data


def _selected_ids_by_target(lock: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        _lock_target_key(target): [str(item) for item in target.get("selected_product_ids", [])]
        for target in lock.get("targets", [])
        if target.get("target_id") is not None
    }


def build_product_lock(
    inventory: Dict[str, Any],
    *,
    source_inventory_path: Optional[Path | str] = None,
    previous_lock_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build a deterministic selected-product lock from a SAR inventory."""
    if inventory.get("schema_version") != SAR_INVENTORY_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Inventory schema_version must be {SAR_INVENTORY_SCHEMA_VERSION!r}"
        )
    previous_path = Path(previous_lock_path) if previous_lock_path is not None else None
    previous_lock = _load_optional_previous_lock(previous_path)
    previous_ids = _selected_ids_by_target(previous_lock) if previous_lock else {}
    target_locks = []
    selection_changed = False
    for target in inventory.get("targets", []):
        selected_products = target.get("selected_products", [])
        selected_ids = [str(product.get("product_id")) for product in selected_products]
        previous_target_ids = previous_ids.get(_lock_target_key(target))
        target_changed = previous_target_ids is not None and previous_target_ids != selected_ids
        selection_changed = selection_changed or target_changed
        estimated_download_size_mb = _estimated_products_size_mb(selected_products)
        target_lock = {
            "target_id": target.get("target_id"),
            "name": target.get("name"),
            "status": target.get("status"),
            "provider": target.get("provider"),
            "provider_id": target.get("provider_id"),
            "provider_label": target.get("provider_label"),
            "comparison_arm": target.get("comparison_arm"),
            "comparison_group": target.get("comparison_group"),
            "placeholder_only": target.get("placeholder_only"),
            "selection_policy": inventory.get("selection_policy", PRODUCT_SELECTION_POLICY),
            "selection_count": len(selected_products),
            "selected_product_ids": selected_ids,
            "selected_products": selected_products,
            "estimated_download_size_mb": estimated_download_size_mb,
            "estimated_download_size_gb": _mb_to_gb(estimated_download_size_mb),
            "search_parameters": target.get("search_parameters"),
            "selection_changed_from_previous_lock": target_changed,
        }
        target_lock.update(_public_campaign_metadata_record(target))
        if isinstance(target.get("public_campaign_metadata"), dict):
            target_lock["public_campaign_metadata"] = dict(target["public_campaign_metadata"])
        target_locks.append(target_lock)
    locked_products = [
        product
        for target in target_locks
        for product in target.get("selected_products", [])
        if isinstance(product, dict)
    ]
    estimated_download_size_mb = _estimated_products_size_mb(locked_products)
    lock = {
        "schema_version": PRODUCT_LOCK_SCHEMA_VERSION,
        "validation_id": inventory.get("validation_id"),
        "created_at_utc": _utc_now_iso(),
        "source_inventory": str(source_inventory_path) if source_inventory_path is not None else None,
        "source_inventory_sha256": _sha256_file(Path(source_inventory_path))
        if source_inventory_path is not None and Path(source_inventory_path).exists()
        else _stable_json_sha256(inventory),
        "public_manifest": inventory.get("public_manifest"),
        "public_manifest_sha256": inventory.get("public_manifest_sha256"),
        "campaign_registry": inventory.get("campaign_registry"),
        "campaign_registry_id": (inventory.get("campaign_registry") or {}).get("registry_id")
        if isinstance(inventory.get("campaign_registry"), dict) else None,
        "campaign_registry_hash": (inventory.get("campaign_registry") or {}).get("registry_hash")
        if isinstance(inventory.get("campaign_registry"), dict) else None,
        "parameter_set": inventory.get("parameter_set"),
        "selection_policy": inventory.get("selection_policy", PRODUCT_SELECTION_POLICY),
        "search_only": True,
        "no_download": True,
        "estimated_download_size_mb": estimated_download_size_mb,
        "estimated_download_size_gb": _mb_to_gb(estimated_download_size_mb),
        "withheld_labels_loaded": False,
        "previous_lock_compared": previous_lock is not None,
        "previous_lock": str(previous_path) if previous_path is not None and previous_path.exists() else None,
        "previous_lock_sha256": _sha256_file(previous_path)
        if previous_path is not None and previous_path.exists()
        else None,
        "selection_changed_from_previous_lock": selection_changed,
        "targets": target_locks,
    }
    return lock


def _lock_targets_by_id(lock: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    targets: Dict[str, Dict[str, Any]] = {}
    for target in lock.get("targets", []):
        target_id = target.get("target_id") if isinstance(target, dict) else None
        if target_id is not None:
            key = _lock_target_key(target)
            targets[key] = target
            if str(target_id) not in targets and target.get("comparison_arm") in {None, "sentinel1_cband_iw"}:
                targets[str(target_id)] = target
    return targets


def _locked_products_by_target(lock: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    products_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for target in lock.get("targets", []):
        if not isinstance(target, dict) or target.get("target_id") is None:
            continue
        products = target.get("selected_products", [])
        key = _lock_target_key(target)
        product_records = [
            dict(item) for item in products if isinstance(item, dict)
        ] if isinstance(products, list) else []
        products_by_target[key] = product_records
        if str(target["target_id"]) not in products_by_target and target.get("comparison_arm") in {None, "sentinel1_cband_iw"}:
            products_by_target[str(target["target_id"])] = product_records
    return products_by_target


def verify_product_lock_for_manifest(
    lock_path: Path | str,
    public_manifest: Dict[str, Any],
    manifest_path: Path | str,
    *,
    require_single_product_per_real_target: bool = False,
    target_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    lock_path = Path(lock_path)
    manifest_path = Path(manifest_path)
    lock = load_product_lock(lock_path)
    if lock.get("validation_id") != public_manifest.get("validation_id"):
        raise ManifestValidationError(
            "Product lock validation_id does not match public manifest validation_id"
        )
    public_hash = _sha256_file(manifest_path)
    lock_public_hash = lock.get("public_manifest_sha256")
    if not lock_public_hash:
        raise ManifestValidationError(
            "Product lock lacks public_manifest_sha256; cannot verify it was built from the current manifest"
        )
    if lock_public_hash != public_hash:
        raise ManifestValidationError(
            "Product lock public_manifest_sha256 does not match the current public manifest"
        )

    lock_targets = _lock_targets_by_id(lock)
    target_id_filter = {str(target_id) for target_id in target_ids} if target_ids is not None else None
    verified_targets = []
    for public_target in public_manifest.get("targets", []):
        target_id = str(public_target.get("target_id"))
        if target_id_filter is not None and target_id not in target_id_filter:
            continue
        if public_target.get("data_mode", "real_slc") != "real_slc":
            continue
        sentinel_arms = [
            arm for arm in _target_comparison_arms(public_target)
            if arm.get("provider") == "sentinel1_asf" and arm.get("real_lock_supported", True)
        ]
        if not sentinel_arms:
            sentinel_arms = [_normalize_sar_search({}, "sar_search")]
        target_lock = None
        for arm in sentinel_arms:
            arm_key = f"{target_id}::{arm.get('comparison_arm') or 'default'}"
            target_lock = lock_targets.get(arm_key) or lock_targets.get(target_id)
            if target_lock is not None:
                break
        if target_lock is None:
            raise ManifestValidationError(f"Product lock is missing Sentinel-1 real_slc target {target_id!r}")
        if target_lock.get("provider") not in {None, "sentinel1_asf"}:
            raise ManifestValidationError(
                "Current locked real execution only supports Sentinel-1 product locks; "
                f"target {target_id!r} lock provider is {target_lock.get('provider')!r}"
            )
        selected_ids = [str(item) for item in target_lock.get("selected_product_ids", [])]
        selected_products = [
            dict(item) for item in target_lock.get("selected_products", []) if isinstance(item, dict)
        ]
        if not selected_ids:
            raise ManifestValidationError(f"Product lock target {target_id!r} has no selected_product_ids")
        if not selected_products:
            raise ManifestValidationError(
                f"Product lock target {target_id!r} has no selected_products metadata for acquisition"
            )
        if len(selected_products) != len(selected_ids):
            raise ManifestValidationError(
                f"Product lock target {target_id!r} selected_products count does not match selected_product_ids"
            )
        metadata_ids = [str(product.get("product_id")) for product in selected_products]
        if metadata_ids != selected_ids:
            raise ManifestValidationError(
                f"Product lock target {target_id!r} selected product metadata does not match selected_product_ids"
            )
        if require_single_product_per_real_target and len(selected_ids) != 1:
            raise ManifestValidationError(
                "Current locked real execution can enforce exactly one Sentinel-1 product per target; "
                f"target {target_id!r} lock selected {len(selected_ids)} product(s). Regenerate the lock "
                "with sar_search.selection_count=1 or refactor acquisition for multi-product locked processing."
            )
        verified_targets.append(
            {
                "target_id": target_id,
                "provider": target_lock.get("provider"),
                "provider_label": target_lock.get("provider_label"),
                "comparison_arm": target_lock.get("comparison_arm"),
                "selected_product_ids": selected_ids,
                "selected_product_count": len(selected_ids),
                "estimated_download_size_mb": target_lock.get("estimated_download_size_mb"),
                "estimated_download_size_gb": target_lock.get("estimated_download_size_gb"),
                "status": target_lock.get("status"),
            }
        )

    locked_products = [
        product
        for products in _locked_products_by_target(lock).values()
        for product in products
        if isinstance(product, dict)
    ]
    estimated_download_size_mb = _estimated_products_size_mb(locked_products)

    return {
        "path": str(lock_path),
        "sha256": _sha256_file(lock_path),
        "validation_id": lock.get("validation_id"),
        "public_manifest_sha256": lock_public_hash,
        "public_manifest_sha256_matches": True,
        "selection_policy": lock.get("selection_policy"),
        "verified_real_slc_target_count": len(verified_targets),
        "estimated_download_size_mb": estimated_download_size_mb,
        "estimated_download_size_gb": _mb_to_gb(estimated_download_size_mb),
        "targets": verified_targets,
        "enforcement": "single_locked_sentinel1_product_pre_download_gate"
        if require_single_product_per_real_target else "lock_manifest_verified",
        "remaining_limitations": [
            "The current real pipeline can enforce one locked Sentinel-1 product per target; "
            "multi-product locked execution is blocked until acquisition and processing are refactored."
        ],
    }


SearchExecutor = Callable[..., List[Dict[str, Any]]]


def build_sar_inventory(
    manifest_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    lock_output_path: Optional[Path | str] = None,
    parameter_set_path: Optional[Path | str] = None,
    campaign_registry_path: Optional[Path | str] = None,
    allow_templates: bool = False,
    searcher: Optional[SearchExecutor] = None,
    load_dotenv: bool = True,
    env_path: Optional[Path | str] = None,
    start_date: str = DEFAULT_SAR_SEARCH_START_DATE,
    end_date: str = DEFAULT_SAR_SEARCH_END_DATE,
    max_results: int = DEFAULT_SAR_MAX_RESULTS,
    selection_count: int = DEFAULT_PRODUCT_SELECTION_COUNT,
    search_timeout_seconds: Optional[int] = None,
    search_max_retries: Optional[int] = None,
    search_retry_backoff_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Search-only SAR inventory for all public manifest targets.

    This function deliberately accepts no label path and performs no downloads.
    """
    manifest_path = Path(manifest_path)
    public_manifest = load_public_manifest(manifest_path, allow_templates=allow_templates)
    parameter_set_record = None
    if parameter_set_path is not None:
        parameter_set = load_parameter_set(parameter_set_path, require_approved=False)
        if parameter_set.get("validation_id") != public_manifest.get("validation_id"):
            raise ManifestValidationError("Parameter set validation_id does not match public manifest validation_id")
        parameter_set_record = _parameter_set_reference_record(parameter_set_path, parameter_set)
    campaign_registry_record = None
    if campaign_registry_path is not None:
        campaign_registry_record = verify_campaign_registry_for_inputs(
            campaign_registry_path,
            manifest_path,
            parameter_set_path=parameter_set_path,
            allow_templates=allow_templates,
            require_approved=False,
            require_locked=False,
        )
    dotenv_result = _load_env_file_secret_safe(Path(env_path) if env_path is not None else None) if load_dotenv else {
        "loaded": False,
        "path": str(env_path) if env_path is not None else None,
        "entry_count": 0,
        "secret_values_recorded": False,
    }
    auth_record = _detect_auth_mode()
    if searcher is None:
        from slc_data_fetcher import search_sentinel1_slc

        searcher = search_sentinel1_slc
    inventory_targets = []
    for target in public_manifest["targets"]:
        for arm in _target_comparison_arms(target):
            params = _resolve_target_search_params(
                target,
                search=arm,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
                selection_count=selection_count,
            )
            target_record: Dict[str, Any] = {
                "target_id": target["target_id"],
                "name": target["name"],
                "data_mode": target.get("data_mode", "real_slc"),
                "center": target["center"],
                "buffer_deg": target.get("buffer_deg", 0.02),
                "provider": params["provider"],
                "provider_id": params["provider"],
                "provider_label": params["provider_label"],
                "comparison_arm": params["comparison_arm"],
                "comparison_group": params.get("comparison_group"),
                "band": params.get("band"),
                "acquisition_mode": params.get("acquisition_mode"),
                "placeholder_only": params.get("placeholder_only"),
                "search_supported": params.get("search_supported"),
                "download_supported": params.get("download_supported"),
                "real_lock_supported": params.get("real_lock_supported"),
                "requires_credentials": params.get("requires_credentials"),
                "limitations": params.get("limitations", []),
                "search_parameters": params,
                "status": "failed",
                "products_found": 0,
                "products": [],
                "selected_products": [],
                "selected_product_ids": [],
                "estimated_available_size_mb": None,
                "estimated_download_size_mb": None,
                "estimated_download_size_gb": None,
                "withheld_labels_loaded": False,
                "public_campaign_metadata": _public_campaign_metadata_record(target),
            }
            target_record.update(_public_campaign_metadata_record(target))
            if not params.get("search_supported", True):
                target_record.update(
                    {
                        "status": "placeholder_not_searched",
                        "search_skipped_reason": "provider_search_not_implemented_no_credentials_or_downloads_used",
                    }
                )
                inventory_targets.append(target_record)
                continue
            try:
                raw_products = searcher(
                    bbox=tuple(params["bbox"]),
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    max_results=params["max_results"],
                    flight_direction=params.get("flight_direction"),
                    polarization=params.get("polarization", "VV"),
                    search_timeout_seconds=search_timeout_seconds,
                    search_max_retries=search_max_retries,
                    search_retry_backoff_seconds=search_retry_backoff_seconds,
                )
                products = []
                for idx, product in enumerate(raw_products or []):
                    if not isinstance(product, dict):
                        continue
                    product_with_arm = dict(product)
                    product_with_arm.setdefault("provider_id", params["provider"])
                    product_with_arm.setdefault("provider", params["provider_label"])
                    product_with_arm.setdefault("platform", params["platform"])
                    product_with_arm.setdefault("band", params.get("band"))
                    product_with_arm.setdefault("beam_mode", params.get("beam_mode"))
                    product_with_arm.setdefault("processing_level", params.get("processing_level"))
                    product_with_arm.setdefault("acquisition_mode", params.get("acquisition_mode"))
                    product_with_arm.setdefault("comparison_arm", params["comparison_arm"])
                    product_with_arm.setdefault("comparison_group", params.get("comparison_group"))
                    products.append(_normalize_product_metadata(product_with_arm, idx))
                sorted_products = _sort_products_for_lock(products)
                selected_products = sorted_products[: max(0, int(params["selection_count"]))]
                estimated_available_size_mb = _estimated_products_size_mb(sorted_products)
                estimated_download_size_mb = _estimated_products_size_mb(selected_products)
                target_record.update(
                    {
                        "status": "success",
                        "products_found": len(products),
                        "products": sorted_products,
                        "selected_products": selected_products,
                        "selected_product_ids": [product["product_id"] for product in selected_products],
                        "estimated_available_size_mb": estimated_available_size_mb,
                        "estimated_download_size_mb": estimated_download_size_mb,
                        "estimated_download_size_gb": _mb_to_gb(estimated_download_size_mb),
                    }
                )
            except Exception as exc:
                target_record["error"] = _safe_error_summary(exc)
            inventory_targets.append(target_record)
    selected_products_all = [
        product
        for target in inventory_targets
        for product in target.get("selected_products", [])
        if isinstance(product, dict)
    ]
    estimated_download_size_mb = _estimated_products_size_mb(selected_products_all)
    inventory = {
        "schema_version": SAR_INVENTORY_SCHEMA_VERSION,
        "validation_id": public_manifest["validation_id"],
        "created_at_utc": _utc_now_iso(),
        "public_manifest": str(manifest_path),
        "public_manifest_sha256": _sha256_file(manifest_path),
        "campaign_registry": campaign_registry_record,
        "campaign_registry_id": campaign_registry_record.get("registry_id") if campaign_registry_record else None,
        "campaign_registry_hash": campaign_registry_record.get("registry_hash") if campaign_registry_record else None,
        "parameter_set": parameter_set_record,
        "search_only": True,
        "no_download": True,
        "downloads_attempted": False,
        "estimated_download_size_mb": estimated_download_size_mb,
        "estimated_download_size_gb": _mb_to_gb(estimated_download_size_mb),
        "selected_product_count": len(selected_products_all),
        "download_size_estimate_note": "Summed from known selected product size_mb metadata; compare against geoanomaly health disk.free_gb before explicit real execution.",
        "withheld_labels_loaded": False,
        "selection_policy": PRODUCT_SELECTION_POLICY,
        "default_search_parameters": {
            "provider": "sentinel1_asf",
            "provider_label": "ASF Sentinel-1",
            "comparison_arm": "sentinel1_cband_iw",
            "platform": "Sentinel-1",
            "band": "C",
            "processing_level": "SLC",
            "beam_mode": "IW",
            "polarization": "VV",
            "start_date": start_date,
            "end_date": end_date,
            "max_results": int(max_results),
            "selection_count": int(selection_count),
        },
        "auth": auth_record,
        "dotenv": dotenv_result,
        "targets": inventory_targets,
    }
    if lock_output_path is not None:
        lock_path = Path(lock_output_path)
        previous_path = lock_path if lock_path.exists() else None
        lock = build_product_lock(inventory, previous_lock_path=previous_path)
        _write_json(lock_path, lock)
        inventory["product_lock"] = str(lock_path)
        inventory["product_lock_sha256"] = _sha256_file(lock_path)
        inventory["product_lock_selection_changed_from_previous_lock"] = lock[
            "selection_changed_from_previous_lock"
        ]
    if output_path is not None:
        _write_json(Path(output_path), inventory)
    return inventory


def _public_target_to_pipeline_target(public_target: Dict[str, Any]) -> Dict[str, Any]:
    max_depth = float(public_target.get("max_depth_m", 1000.0) or 1000.0)
    return {
        "name": public_target["name"],
        "lat": public_target["center"]["lat"],
        "lon": public_target["center"]["lon"],
        "buffer_deg": public_target.get("buffer_deg", 0.02),
        "description": public_target.get("description", "Blind validation target"),
        # This is a neutral processing-depth hint, not withheld truth.
        "expected_depth_m": max_depth / 2.0,
    }


PipelineExecutor = Callable[..., Dict[str, Any]]


def _target_campaign_execution_excluded(target: Dict[str, Any]) -> Tuple[bool, str]:
    if bool(target.get("audit_only") or target.get("prior_run_audit_only")):
        return True, "audit_only_target_excluded_from_primary_execution"
    split = str(target.get("split") or target.get("split_designation") or "").strip().lower()
    if split == "audit" or "audit_only" in split:
        return True, "audit_split_excluded_from_primary_execution"
    scoring_status = str(target.get("public_scoring_status") or "").strip().lower()
    if scoring_status.startswith("not_") or "not_primary" in scoring_status:
        return True, "public_scoring_status_marks_not_primary_holdout"
    return False, "primary_execution_eligible"


def _target_provider_execution_summary(target: Dict[str, Any]) -> Dict[str, Any]:
    arms = _target_comparison_arms(target)
    arm_records = []
    primary_supported = False
    unsupported = False
    selected_primary_arm: Optional[Dict[str, Any]] = None
    for arm in arms:
        provider = str(arm.get("provider") or "sentinel1_asf")
        profile = SAR_PROVIDER_PROFILES.get(provider, SAR_PROVIDER_PROFILES["sentinel1_asf"])
        comparison_arm = str(arm.get("comparison_arm") or profile["default_comparison_arm"])
        real_lock_supported = bool(arm.get("real_lock_supported", profile["real_lock_supported"]))
        download_supported = bool(arm.get("download_supported", profile["download_supported"]))
        placeholder_only = bool(arm.get("placeholder_only", not profile["search_supported"]))
        supported_for_strict_real_execution = bool(
            provider == "sentinel1_asf" and real_lock_supported and download_supported and not placeholder_only
        )
        arm_record = {
            "provider": provider,
            "provider_label": str(arm.get("provider_label") or profile["provider_label"]),
            "comparison_arm": comparison_arm,
            "comparison_group": arm.get("comparison_group"),
            "band": str(arm.get("band") or profile["band"]),
            "placeholder_only": placeholder_only,
            "search_supported": bool(arm.get("search_supported", profile["search_supported"])),
            "download_supported": download_supported,
            "real_lock_supported": real_lock_supported,
            "supported_for_strict_real_execution": supported_for_strict_real_execution,
            "limitations": list(arm.get("limitations", profile.get("limitations", []))),
        }
        arm_records.append(arm_record)
        unsupported = unsupported or not supported_for_strict_real_execution
        if supported_for_strict_real_execution and selected_primary_arm is None:
            selected_primary_arm = arm_record
            primary_supported = True
    if selected_primary_arm is None and arm_records:
        selected_primary_arm = arm_records[0]
    return {
        "primary_supported": primary_supported,
        "contains_unsupported_or_placeholder_arm": unsupported,
        "selected_primary_arm": selected_primary_arm,
        "arms": arm_records,
    }


def _load_campaign_inputs_for_execution(
    manifest_path: Path | str,
    registry_path: Path | str,
    parameter_set_path: Path | str,
    product_lock_path: Optional[Path | str],
    *,
    allow_templates: bool = False,
    require_locked_registry: bool = True,
    require_approved_parameters: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    parameter_set_path = Path(parameter_set_path)
    registry_path = Path(registry_path)
    public_manifest = load_public_manifest(manifest_path, allow_templates=allow_templates)
    parameter_set = load_parameter_set(
        parameter_set_path,
        allow_templates=allow_templates,
        require_approved=require_approved_parameters,
    )
    if parameter_set.get("validation_id") != public_manifest.get("validation_id"):
        raise ManifestValidationError("Parameter set validation_id does not match public manifest validation_id")
    registry_record = verify_campaign_registry_for_inputs(
        registry_path,
        manifest_path,
        parameter_set_path=parameter_set_path,
        allow_templates=allow_templates,
        require_approved=require_approved_parameters,
        require_locked=require_locked_registry,
    )
    if isinstance(registry_record.get("comparison"), dict):
        stable_comparison = dict(registry_record["comparison"])
        stable_comparison.pop("created_at_utc", None)
        registry_record = dict(registry_record)
        registry_record["comparison"] = stable_comparison
    product_lock = load_product_lock(product_lock_path) if product_lock_path is not None else None
    product_lock_record: Dict[str, Any] = {"provided": product_lock_path is not None, "path": str(product_lock_path) if product_lock_path is not None else None}
    if product_lock is not None:
        if product_lock.get("validation_id") != public_manifest.get("validation_id"):
            raise ManifestValidationError("Product lock validation_id does not match public manifest validation_id")
        if product_lock.get("public_manifest_sha256") != _sha256_file(manifest_path):
            raise ManifestValidationError("Product lock public_manifest_sha256 does not match the current public manifest")
        if product_lock.get("parameter_set"):
            lock_parameter_hash = product_lock.get("parameter_set", {}).get("parameter_set_hash")
            if lock_parameter_hash and lock_parameter_hash != parameter_set.get("parameter_set_hash"):
                raise ManifestValidationError("Product lock parameter set hash does not match current parameter set")
        if product_lock.get("campaign_registry"):
            lock_registry_hash = product_lock.get("campaign_registry", {}).get("registry_hash")
            if lock_registry_hash and lock_registry_hash != registry_record.get("registry_hash"):
                raise ManifestValidationError("Product lock campaign registry hash does not match current campaign registry")
        product_lock_record.update(
            {
                "sha256": _sha256_file(Path(product_lock_path)),
                "schema_version": product_lock.get("schema_version"),
                "validation_id": product_lock.get("validation_id"),
                "public_manifest_sha256": product_lock.get("public_manifest_sha256"),
                "campaign_registry_id": product_lock.get("campaign_registry_id"),
                "campaign_registry_hash": product_lock.get("campaign_registry_hash"),
                "parameter_set_id": (product_lock.get("parameter_set") or {}).get("parameter_set_id")
                if isinstance(product_lock.get("parameter_set"), dict) else None,
                "parameter_set_hash": (product_lock.get("parameter_set") or {}).get("parameter_set_hash")
                if isinstance(product_lock.get("parameter_set"), dict) else None,
                "target_count": len(product_lock.get("targets", [])) if isinstance(product_lock.get("targets"), list) else None,
                "estimated_download_size_mb": product_lock.get("estimated_download_size_mb"),
                "estimated_download_size_gb": product_lock.get("estimated_download_size_gb"),
                "no_download": bool(product_lock.get("no_download", False)),
                "search_only": bool(product_lock.get("search_only", False)),
            }
        )
    registry = load_campaign_registry(registry_path, allow_templates=allow_templates, require_approved=require_approved_parameters, require_locked=require_locked_registry)
    return public_manifest, registry, parameter_set, product_lock, registry_record, product_lock_record


def _campaign_command_for_target(
    *,
    manifest_path: Path,
    output_dir: Path,
    target: Dict[str, Any],
    parameter_set_path: Path,
    registry_path: Path,
    product_lock_path: Optional[Path],
    execute_real: bool,
    include_audit_only: bool,
    allow_synthetic_fallback: bool,
    confirm_real_downloads_and_training: bool,
) -> List[str]:
    target_output_dir = output_dir / "targets" / _safe_name(str(target["target_id"]))
    command = [
        "python",
        "blind_validation.py",
        "run",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(target_output_dir),
        "--target-id",
        str(target["target_id"]),
        "--parameter-set",
        str(parameter_set_path),
        "--require-approved-parameters",
        "--campaign-registry",
        str(registry_path),
        "--require-locked-campaign-registry",
    ]
    if product_lock_path is not None:
        command.extend(["--product-lock", str(product_lock_path), "--require-product-lock"])
    if execute_real:
        command.append("--execute-real")
        if confirm_real_downloads_and_training:
            command.append("--confirm-real-downloads-and-training")
    if include_audit_only:
        command.append("--include-audit-only")
    if allow_synthetic_fallback:
        command.append("--allow-synthetic-fallback")
    return command


def plan_campaign_execution(
    manifest_path: Path | str,
    registry_path: Path | str,
    parameter_set_path: Path | str,
    output_dir: Path | str,
    *,
    product_lock_path: Optional[Path | str] = None,
    execute_real: bool = False,
    confirm_real_downloads_and_training: bool = False,
    allow_synthetic_fallback: bool = False,
    include_audit_only: bool = False,
    allow_unsupported_provider_arms: bool = False,
    allow_templates: bool = False,
) -> Dict[str, Any]:
    """Build a deterministic no-label multi-site campaign execution plan."""
    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    parameter_set_path = Path(parameter_set_path)
    output_dir = Path(output_dir)
    product_lock_path_obj = Path(product_lock_path) if product_lock_path is not None else None
    if allow_synthetic_fallback:
        raise ManifestValidationError("Campaign execution refuses synthetic fallback; remove --allow-synthetic-fallback for locked validation")
    if execute_real and not confirm_real_downloads_and_training:
        raise ManifestValidationError(
            "Campaign real execution may download SAR products and start training; pass "
            "--confirm-real-downloads-and-training only after product locks, disk estimates, "
            "and non-synthetic settings have been reviewed."
        )
    if execute_real and product_lock_path_obj is None:
        raise ManifestValidationError("Campaign real execution requires --product-lock")

    public_manifest, registry, parameter_set, product_lock, registry_record, product_lock_record = _load_campaign_inputs_for_execution(
        manifest_path,
        registry_path,
        parameter_set_path,
        product_lock_path_obj,
        allow_templates=allow_templates,
        require_locked_registry=True,
        require_approved_parameters=True,
    )
    if public_manifest.get("template_only") or registry.get("template_only") or parameter_set.get("template_only"):
        raise ManifestValidationError("Campaign execution requires non-template public manifest, parameter set, and locked registry")
    locked_targets = _lock_targets_by_id(product_lock) if product_lock is not None else {}
    steps = []
    executable_count = 0
    skipped_count = 0
    flagged_count = 0
    total_estimated_download_size_mb = 0.0
    total_estimated_download_size_known = False
    for target in sorted(public_manifest.get("targets", []), key=lambda item: str(item.get("target_id"))):
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id"))
        excluded, exclusion_reason = _target_campaign_execution_excluded(target)
        provider_summary = _target_provider_execution_summary(target)
        primary_arm = provider_summary.get("selected_primary_arm") or {}
        comparison_arm = str(primary_arm.get("comparison_arm") or "default")
        locked_key = f"{target_id}::{comparison_arm}"
        locked_target = locked_targets.get(locked_key) or locked_targets.get(target_id)
        locked_ids = [str(item) for item in locked_target.get("selected_product_ids", [])] if isinstance(locked_target, dict) else []
        unsupported = not bool(provider_summary.get("primary_supported"))
        reasons = []
        status = "planned"
        if excluded and not include_audit_only:
            status = "skipped_audit_only"
            reasons.append(exclusion_reason)
        if unsupported and not allow_unsupported_provider_arms:
            status = "skipped_unsupported_provider"
            reasons.append("no_supported_sentinel1_primary_execution_arm")
        if execute_real and not locked_ids:
            status = "blocked_missing_product_lock"
            reasons.append("missing_locked_sentinel1_product_selection")
        if target.get("data_mode", "real_slc") == "template":
            status = "blocked_template_target"
            reasons.append("template_target_cannot_execute")
        executable = status == "planned"
        if executable:
            executable_count += 1
        else:
            skipped_count += 1 if status.startswith("skipped") else 0
            flagged_count += 1 if status.startswith("blocked") else 0
        estimated_mb = locked_target.get("estimated_download_size_mb") if isinstance(locked_target, dict) else None
        if executable and _finite_or_none(estimated_mb) is not None:
            total_estimated_download_size_mb += float(estimated_mb)
            total_estimated_download_size_known = True
        command = _campaign_command_for_target(
            manifest_path=manifest_path,
            output_dir=output_dir,
            target=target,
            parameter_set_path=parameter_set_path,
            registry_path=registry_path,
            product_lock_path=product_lock_path_obj,
            execute_real=execute_real,
            include_audit_only=include_audit_only,
            allow_synthetic_fallback=False,
            confirm_real_downloads_and_training=confirm_real_downloads_and_training,
        )
        step = {
            "target_id": target_id,
            "name": target.get("name"),
            "status": status,
            "execution_mode": "real_locked" if execute_real else "dry_run_plan",
            "executable": executable,
            "skip_or_block_reasons": sorted(set(reasons)),
            "data_mode": target.get("data_mode", "real_slc"),
            "split": target.get("split"),
            "split_designation": target.get("split_designation"),
            "audit_only": bool(target.get("audit_only") or target.get("prior_run_audit_only")),
            "public_scoring_status": target.get("public_scoring_status"),
            "provider_execution": provider_summary,
            "selected_execution_arm": primary_arm,
            "product_lock": {
                "required_for_real_execution": bool(execute_real),
                "provided": product_lock is not None,
                "matched": bool(locked_ids),
                "target_lock_key": locked_key if locked_ids else None,
                "selected_product_ids": locked_ids,
                "selected_product_count": len(locked_ids),
                "estimated_download_size_mb": estimated_mb,
                "estimated_download_size_gb": _mb_to_gb(_finite_or_none(estimated_mb)),
            },
            "target_output_dir": str(output_dir / "targets" / _safe_name(target_id)),
            "run_manifest_path": str(output_dir / "targets" / _safe_name(target_id) / "run_manifest.json"),
            "command": command,
            "command_display": " ".join(command),
            "withheld_labels_loaded": False,
        }
        step.update(_public_campaign_metadata_record(target))
        steps.append(step)
    if execute_real and flagged_count:
        raise ManifestValidationError("Campaign real execution blocked by missing product locks or template targets")
    if not allow_unsupported_provider_arms and any(step["status"] == "skipped_unsupported_provider" for step in steps) and not any(step["executable"] for step in steps):
        raise ManifestValidationError("No campaign targets have supported provider arms for execution")
    run_policy = {
        "default_no_download": not bool(execute_real),
        "downloads_attempted_by_plan": False,
        "training_attempted_by_plan": False,
        "execute_real_requested": bool(execute_real),
        "explicit_real_confirmation_required": bool(execute_real),
        "explicit_real_confirmation_provided": bool(confirm_real_downloads_and_training),
        "synthetic_fallback_allowed": False,
        "synthetic_fallback_refused_by_default": True,
        "audit_only_included": bool(include_audit_only),
        "unsupported_provider_arms_allowed": bool(allow_unsupported_provider_arms),
    }
    plan = {
        "schema_version": CAMPAIGN_EXECUTION_PLAN_SCHEMA_VERSION,
        "validation_id": public_manifest.get("validation_id"),
        "campaign_id": registry.get("campaign_id"),
        "campaign_registry": registry_record,
        "campaign_registry_id": registry_record.get("registry_id"),
        "campaign_registry_hash": registry_record.get("registry_hash"),
        "campaign_registry_status": registry_record.get("status"),
        "public_manifest": str(manifest_path),
        "public_manifest_sha256": _sha256_file(manifest_path),
        "parameter_set": _parameter_set_reference_record(parameter_set_path, parameter_set),
        "parameter_set_id": parameter_set.get("parameter_set_id"),
        "parameter_set_hash": parameter_set.get("parameter_set_hash"),
        "product_lock": product_lock_record,
        "output_dir": str(output_dir),
        "run_policy": run_policy,
        "summary": {
            "target_count": len(steps),
            "planned_executable_targets": executable_count,
            "skipped_targets": skipped_count,
            "blocked_or_flagged_targets": flagged_count,
            "audit_only_targets": sum(1 for step in steps if step.get("audit_only")),
            "unsupported_or_placeholder_arm_targets": sum(
                1 for step in steps if step.get("provider_execution", {}).get("contains_unsupported_or_placeholder_arm")
            ),
            "estimated_download_size_mb_for_executable_targets": round(total_estimated_download_size_mb, 3) if total_estimated_download_size_known else None,
            "estimated_download_size_gb_for_executable_targets": _mb_to_gb(total_estimated_download_size_mb) if total_estimated_download_size_known else None,
            "by_status": dict(sorted({status: sum(1 for step in steps if step["status"] == status) for status in {step["status"] for step in steps}}.items())),
        },
        "steps": steps,
        "withheld_labels_loaded": False,
    }
    return _round_metric(plan)


def _load_campaign_plan(plan_path: Path | str) -> Dict[str, Any]:
    plan = _load_json(Path(plan_path))
    if plan.get("schema_version") != CAMPAIGN_EXECUTION_PLAN_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Campaign execution plan schema_version must be {CAMPAIGN_EXECUTION_PLAN_SCHEMA_VERSION!r}"
        )
    return plan


def _run_manifest_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "schema_version": None,
            "target_count": 0,
            "candidate_count": 0,
            "downloads_attempted": False,
            "dry_run": None,
            "sha256": None,
        }
    data = _load_json(path)
    if data.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(f"Run manifest {path} schema_version must be {RUN_MANIFEST_SCHEMA_VERSION!r}")
    targets = data.get("targets", []) if isinstance(data.get("targets"), list) else []
    return {
        "exists": True,
        "path": str(path),
        "schema_version": data.get("schema_version"),
        "target_count": len(targets),
        "candidate_count": sum(int(target.get("candidate_count", 0) or 0) for target in targets if isinstance(target, dict)),
        "downloads_attempted": bool(data.get("downloads_attempted")),
        "dry_run": data.get("dry_run"),
        "sha256": _sha256_file(path),
    }


def campaign_execution_status(plan_path: Path | str) -> Dict[str, Any]:
    """Summarize existing per-target campaign run manifests without loading labels."""
    plan_path = Path(plan_path)
    plan = _load_campaign_plan(plan_path)
    target_records = []
    completed = 0
    planned = 0
    skipped = 0
    blocked = 0
    candidate_count = 0
    downloads_attempted = False
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        run_manifest_path = Path(step.get("run_manifest_path") or "")
        run_status = _run_manifest_status(run_manifest_path)
        if step.get("executable"):
            planned += 1
            completed += 1 if run_status["exists"] else 0
        elif str(step.get("status", "")).startswith("skipped"):
            skipped += 1
        else:
            blocked += 1
        candidate_count += int(run_status.get("candidate_count", 0) or 0)
        downloads_attempted = downloads_attempted or bool(run_status.get("downloads_attempted"))
        target_records.append(
            {
                "target_id": step.get("target_id"),
                "name": step.get("name"),
                "planned_status": step.get("status"),
                "planned_executable": bool(step.get("executable")),
                "run_manifest": run_status,
                "complete": bool(step.get("executable") and run_status["exists"]),
                "audit_only": bool(step.get("audit_only")),
                "withheld_labels_loaded": False,
            }
        )
    return {
        "schema_version": CAMPAIGN_EXECUTION_STATUS_SCHEMA_VERSION,
        "validation_id": plan.get("validation_id"),
        "campaign_id": plan.get("campaign_id"),
        "campaign_registry_id": plan.get("campaign_registry_id"),
        "campaign_registry_hash": plan.get("campaign_registry_hash"),
        "plan": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "summary": {
            "planned_executable_targets": planned,
            "completed_targets": completed,
            "pending_targets": max(0, planned - completed),
            "skipped_targets": skipped,
            "blocked_or_flagged_targets": blocked,
            "total_candidate_count": candidate_count,
            "downloads_attempted": downloads_attempted,
        },
        "targets": target_records,
        "withheld_labels_loaded": False,
    }


def run_campaign_execution_plan(
    plan_path: Path | str,
    *,
    resume: bool = True,
    execute_real: bool = False,
    confirm_real_downloads_and_training: bool = False,
    pipeline_executor: Optional[PipelineExecutor] = None,
) -> Dict[str, Any]:
    """Run executable campaign plan steps; default is dry-run-safe."""
    plan_path = Path(plan_path)
    plan = _load_campaign_plan(plan_path)
    plan_execute_real = bool(plan.get("run_policy", {}).get("execute_real_requested"))
    if execute_real != plan_execute_real:
        if execute_real:
            raise ManifestValidationError("Plan was not created for real execution; rebuild it with --execute-real")
        raise ManifestValidationError("Plan was created for real execution; pass --execute-real to acknowledge real-run intent")
    if execute_real and not confirm_real_downloads_and_training:
        raise ManifestValidationError("Campaign real execution requires --confirm-real-downloads-and-training")
    if plan.get("run_policy", {}).get("synthetic_fallback_allowed"):
        raise ManifestValidationError("Campaign execution refuses plans with synthetic fallback enabled")
    results = []
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        target_id = str(step.get("target_id"))
        run_manifest_path = Path(step.get("run_manifest_path") or "")
        if not step.get("executable"):
            results.append(
                {
                    "target_id": target_id,
                    "status": "skipped",
                    "reason": step.get("skip_or_block_reasons", []),
                    "run_manifest_path": str(run_manifest_path),
                }
            )
            continue
        if resume and run_manifest_path.exists():
            results.append(
                {
                    "target_id": target_id,
                    "status": "resumed_existing",
                    "run_manifest_path": str(run_manifest_path),
                    "run_manifest_sha256": _sha256_file(run_manifest_path),
                }
            )
            continue
        run_manifest = run_blind_validation(
            plan["public_manifest"],
            Path(step["target_output_dir"]),
            dry_run=not execute_real,
            allow_real_downloads=execute_real,
            allow_synthetic_fallback=False,
            product_lock_path=plan.get("product_lock", {}).get("path"),
            require_product_lock=bool(execute_real or plan.get("product_lock", {}).get("provided")),
            confirm_real_downloads_and_training=confirm_real_downloads_and_training,
            parameter_set_path=plan.get("parameter_set", {}).get("path"),
            require_approved_parameters=True,
            campaign_registry_path=plan.get("campaign_registry", {}).get("path"),
            require_locked_campaign_registry=True,
            target_ids=[target_id],
            include_audit_only=bool(plan.get("run_policy", {}).get("audit_only_included")),
            command_args={
                "command": "campaign-run",
                "plan": str(plan_path),
                "target_id": target_id,
                "execute_real": execute_real,
                "resume": resume,
            },
            pipeline_executor=pipeline_executor,
        )
        results.append(
            {
                "target_id": target_id,
                "status": "executed",
                "run_manifest_path": str(run_manifest_path),
                "run_manifest_sha256": _sha256_file(run_manifest_path),
                "candidate_count": sum(int(target.get("candidate_count", 0) or 0) for target in run_manifest.get("targets", [])),
            }
        )
    status = campaign_execution_status(plan_path)
    status["run_results"] = results
    return status


def package_campaign_no_label_evidence(
    plan_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    status_path: Optional[Path | str] = None,
    inventory_path: Optional[Path | str] = None,
    product_lock_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build a deterministic no-label campaign evidence summary."""
    plan_path = Path(plan_path)
    plan = _load_campaign_plan(plan_path)
    if status_path is not None:
        status = _load_json(Path(status_path))
        if status.get("schema_version") != CAMPAIGN_EXECUTION_STATUS_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Campaign status schema_version must be {CAMPAIGN_EXECUTION_STATUS_SCHEMA_VERSION!r}"
            )
    else:
        status = campaign_execution_status(plan_path)
    artifact_inputs = [
        ("campaign_plan", plan_path),
        ("campaign_status", Path(status_path) if status_path is not None else None),
        ("public_manifest", Path(plan["public_manifest"])),
        ("parameter_set", Path(plan.get("parameter_set", {}).get("path")))
        if plan.get("parameter_set", {}).get("path") else ("parameter_set", None),
        ("campaign_registry", Path(plan.get("campaign_registry", {}).get("path")))
        if plan.get("campaign_registry", {}).get("path") else ("campaign_registry", None),
        ("sar_inventory", Path(inventory_path) if inventory_path is not None else None),
        ("product_lock", Path(product_lock_path) if product_lock_path is not None else None),
    ]
    artifacts = []
    for role, path in artifact_inputs:
        if path is None:
            continue
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "exists": path.exists(),
                "sha256": _sha256_file(path) if path.exists() and path.is_file() else None,
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
            }
        )
    target_evidence = []
    for target in status.get("targets", []):
        if not isinstance(target, dict):
            continue
        run_info = target.get("run_manifest", {}) if isinstance(target.get("run_manifest"), dict) else {}
        target_evidence.append(
            {
                "target_id": target.get("target_id"),
                "planned_status": target.get("planned_status"),
                "complete": bool(target.get("complete")),
                "audit_only": bool(target.get("audit_only")),
                "run_manifest_path": run_info.get("path"),
                "run_manifest_sha256": run_info.get("sha256"),
                "candidate_count": run_info.get("candidate_count"),
                "downloads_attempted": bool(run_info.get("downloads_attempted")),
                "withheld_labels_loaded": False,
            }
        )
    evidence = {
        "schema_version": CAMPAIGN_EVIDENCE_PACKAGE_SCHEMA_VERSION,
        "validation_id": plan.get("validation_id"),
        "campaign_id": plan.get("campaign_id"),
        "campaign_registry_id": plan.get("campaign_registry_id"),
        "campaign_registry_hash": plan.get("campaign_registry_hash"),
        "claim_boundary": REPORT_CANDIDATE_CLAIM,
        "labels_included": False,
        "withheld_labels_loaded": False,
        "summary": {
            "planned_executable_targets": status.get("summary", {}).get("planned_executable_targets"),
            "completed_targets": status.get("summary", {}).get("completed_targets"),
            "pending_targets": status.get("summary", {}).get("pending_targets"),
            "skipped_targets": status.get("summary", {}).get("skipped_targets"),
            "blocked_or_flagged_targets": status.get("summary", {}).get("blocked_or_flagged_targets"),
            "total_candidate_count": status.get("summary", {}).get("total_candidate_count"),
            "downloads_attempted": status.get("summary", {}).get("downloads_attempted"),
        },
        "artifacts": artifacts,
        "targets": target_evidence,
        "limitations": [
            REPORT_CANDIDATE_CLAIM,
            "This campaign evidence summary intentionally excludes withheld labels and score outputs.",
            "No-label status/package outputs support audit and execution readiness only; they do not establish field accuracy.",
        ],
    }
    if output_path is not None:
        _write_json(Path(output_path), evidence)
    return evidence


def run_blind_validation(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    dry_run: bool = True,
    allow_real_downloads: bool = False,
    allow_synthetic_fallback: bool = False,
    product_lock_path: Optional[Path | str] = None,
    require_product_lock: bool = False,
    confirm_real_downloads_and_training: bool = False,
    parameter_set_path: Optional[Path | str] = None,
    require_approved_parameters: bool = False,
    campaign_registry_path: Optional[Path | str] = None,
    require_locked_campaign_registry: bool = False,
    robustness_plan_path: Optional[Path | str] = None,
    robustness_variant_id: Optional[str] = None,
    target_ids: Optional[Sequence[str]] = None,
    include_audit_only: bool = True,
    command_args: Optional[Dict[str, Any]] = None,
    pipeline_executor: Optional[PipelineExecutor] = None,
) -> Dict[str, Any]:
    """Run or dry-run public targets and freeze candidate outputs.

    Withheld labels are deliberately not accepted by this API.
    """
    manifest_path = Path(manifest_path)
    public_manifest = load_public_manifest(manifest_path)
    target_id_filter = {str(target_id) for target_id in target_ids} if target_ids is not None else None
    if target_id_filter is not None:
        available_target_ids = {str(target.get("target_id")) for target in public_manifest.get("targets", []) if isinstance(target, dict)}
        missing_target_ids = sorted(target_id_filter - available_target_ids)
        if missing_target_ids:
            raise ManifestValidationError(f"Requested target_id(s) not present in public manifest: {missing_target_ids}")
    parameter_set_record = None
    if parameter_set_path is not None:
        parameter_set = load_parameter_set(parameter_set_path, require_approved=require_approved_parameters)
        if parameter_set.get("validation_id") != public_manifest.get("validation_id"):
            raise ManifestValidationError("Parameter set validation_id does not match public manifest validation_id")
        parameter_set_record = _parameter_set_reference_record(parameter_set_path, parameter_set)
    elif require_approved_parameters:
        raise ManifestValidationError("--require-approved-parameters requires --parameter-set")
    campaign_registry_record = None
    if campaign_registry_path is not None:
        campaign_registry_record = verify_campaign_registry_for_inputs(
            campaign_registry_path,
            manifest_path,
            parameter_set_path=parameter_set_path,
            require_approved=False,
            require_locked=require_locked_campaign_registry,
        )
    elif require_locked_campaign_registry:
        raise ManifestValidationError("--require-locked-campaign-registry requires --campaign-registry")
    robustness_plan_record = None
    robustness_variant_record = None
    if robustness_plan_path is not None:
        robustness_plan = load_robustness_plan(robustness_plan_path)
        if robustness_plan.get("validation_id") != public_manifest.get("validation_id"):
            raise ManifestValidationError("Robustness plan validation_id does not match public manifest validation_id")
        if campaign_registry_record and robustness_plan.get("campaign_id") and robustness_plan.get("campaign_id") != campaign_registry_record.get("campaign_id"):
            raise ManifestValidationError("Robustness plan campaign_id does not match campaign registry campaign_id")
        robustness_plan_record = _robustness_plan_reference_record(robustness_plan_path, robustness_plan)
        if robustness_variant_id is not None:
            variants_by_id = {str(variant.get("variant_id")): variant for variant in robustness_plan.get("variants", []) if isinstance(variant, dict)}
            if str(robustness_variant_id) not in variants_by_id:
                raise ManifestValidationError(f"Robustness variant_id {robustness_variant_id!r} is not present in robustness plan")
            robustness_variant_record = dict(variants_by_id[str(robustness_variant_id)])
    elif robustness_variant_id is not None:
        raise ManifestValidationError("--variant-id requires --robustness-plan")
    output_dir = Path(output_dir)
    run_manifest_path = output_dir / "run_manifest.json"
    target_root = output_dir / "targets"

    if allow_synthetic_fallback and not allow_real_downloads:
        raise ManifestValidationError(
            "Synthetic fallback can only be enabled together with explicit real execution"
        )
    if require_product_lock and allow_synthetic_fallback:
        raise ManifestValidationError(
            "Synthetic fallback cannot be combined with required locked-product real execution"
        )
    if not dry_run and not allow_real_downloads:
        raise ManifestValidationError(
            "Real processing is opt-in only; use dry_run or set allow_real_downloads=True"
        )
    if require_product_lock and not product_lock_path:
        raise ManifestValidationError("--require-product-lock requires --product-lock")
    if not dry_run and allow_real_downloads and not confirm_real_downloads_and_training:
        raise ManifestValidationError(
            "Real execution may download SAR products and start training; pass "
            "--confirm-real-downloads-and-training only after product locks, disk estimates, "
            "and non-synthetic settings have been reviewed."
        )

    strict_lock_execution = bool(product_lock_path and not dry_run and allow_real_downloads)
    product_lock_verification: Optional[Dict[str, Any]] = None
    locked_products_by_target: Dict[str, List[Dict[str, Any]]] = {}
    loaded_product_lock: Optional[Dict[str, Any]] = None
    if product_lock_path is not None:
        product_lock_verification = verify_product_lock_for_manifest(
            product_lock_path,
            public_manifest,
            manifest_path,
            require_single_product_per_real_target=strict_lock_execution,
            target_ids=sorted(target_id_filter) if target_id_filter is not None else None,
        )
        loaded_product_lock = load_product_lock(product_lock_path)
        if parameter_set_record and loaded_product_lock.get("parameter_set"):
            lock_parameter_hash = loaded_product_lock.get("parameter_set", {}).get("parameter_set_hash")
            if lock_parameter_hash and lock_parameter_hash != parameter_set_record.get("parameter_set_hash"):
                raise ManifestValidationError("Product lock parameter set hash does not match current parameter set")
        if campaign_registry_record and loaded_product_lock.get("campaign_registry"):
            lock_registry_hash = loaded_product_lock.get("campaign_registry", {}).get("registry_hash")
            if lock_registry_hash and lock_registry_hash != campaign_registry_record.get("registry_hash"):
                raise ManifestValidationError("Product lock campaign registry hash does not match current campaign registry")
        locked_products_by_target = _locked_products_by_target(loaded_product_lock)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_targets = []
    for public_target in public_manifest["targets"]:
        target_id = public_target["target_id"]
        if target_id_filter is not None and str(target_id) not in target_id_filter:
            continue
        excluded_from_primary, exclusion_reason = _target_campaign_execution_excluded(public_target)
        if excluded_from_primary and not include_audit_only:
            raise ManifestValidationError(
                f"Target {target_id!r} is excluded from primary campaign execution: {exclusion_reason}; "
                "pass --include-audit-only to run audit-only targets intentionally"
            )
        target_dir = target_root / _safe_name(target_id)
        frozen_csv = target_dir / "frozen_candidates.csv"
        source_csv_value = public_target.get("existing_candidates_csv")
        source_csv = _manifest_relative_path(source_csv_value, manifest_path) if source_csv_value else None
        status = "dry_run"
        mode = public_target.get("data_mode", "real_slc")
        pipeline_result: Optional[Dict[str, Any]] = None
        freeze_reason = "empty_no_candidates_available"

        if mode == "template":
            raise ManifestValidationError(f"Target {target_id!r} is marked template and cannot be run")

        if dry_run or mode == "existing_candidates":
            candidate_count, freeze_reason = _copy_or_empty_candidate_csv(source_csv, frozen_csv)
        else:
            if mode != "real_slc":
                candidate_count, freeze_reason = _copy_or_empty_candidate_csv(source_csv, frozen_csv)
            else:
                if pipeline_executor is None:
                    # The legacy SAR-vibrometry executor was removed after it
                    # failed ground-truth validation (Carlsbad vs barren
                    # control, 2026-07). Real execution now requires an
                    # explicit pipeline_executor (e.g. a deformation_intel
                    # based runner).
                    raise ManifestValidationError(
                        "Real execution requires a pipeline_executor: the legacy "
                        "vibrometry pipeline was removed after failing blind "
                        "ground-truth validation. Provide an executor built on "
                        "deformation_intel, or run with dry_run/existing_candidates."
                    )
                pipeline_target = _public_target_to_pipeline_target(public_target)
                pipeline_kwargs: Dict[str, Any] = {
                    "target": pipeline_target,
                    "credentials": _redacted_env_credentials(),
                    "use_synthetic_fallback": allow_synthetic_fallback,
                    "resolution": public_target.get("resolution", "quick"),
                }
                locked_products = locked_products_by_target.get(target_id)
                if locked_products:
                    pipeline_kwargs["locked_sentinel1_products"] = locked_products
                    pipeline_kwargs["require_locked_sentinel1"] = True
                elif require_product_lock:
                    raise ManifestValidationError(
                        f"Product lock verification did not provide locked products for target {target_id!r}"
                    )
                pipeline_result = pipeline_executor(**pipeline_kwargs)
                result_outputs = pipeline_result.get("outputs", {}) if isinstance(pipeline_result, dict) else {}
                result_csv_value = (
                    result_outputs.get("anomaly_catalog")
                    or result_outputs.get("detected_anomalies_csv")
                    or result_outputs.get("anomalies_csv")
                )
                result_csv = Path(result_csv_value) if result_csv_value else None
                candidate_count, freeze_reason = _copy_or_empty_candidate_csv(result_csv, frozen_csv)
                status = str(pipeline_result.get("status", "unknown")) if isinstance(pipeline_result, dict) else "unknown"

        target_record = {
            "target_id": target_id,
            "name": public_target["name"],
            "status": status,
                "data_mode": mode,
                "provider": public_target.get("sar_search", {}).get("provider"),
                "provider_label": public_target.get("sar_search", {}).get("provider_label"),
                "comparison_arm": public_target.get("sar_search", {}).get("comparison_arm"),
                "comparison_group": public_target.get("sar_search", {}).get("comparison_group"),
                "candidate_count": candidate_count,
            "freeze_reason": freeze_reason,
            "frozen_candidates_csv": _relative_to_base(frozen_csv, output_dir),
                "public_target": public_target,
                "public_campaign_metadata": _public_campaign_metadata_record(public_target),
                "product_lock_enforced": bool(
                    locked_products_by_target.get(target_id)
                    or locked_products_by_target.get(f"{target_id}::{public_target.get('sar_search', {}).get('comparison_arm') or 'default'}")
                ) and not dry_run,
                "locked_product_ids": [
                str(product.get("product_id")) for product in (
                    locked_products_by_target.get(target_id)
                    or locked_products_by_target.get(f"{target_id}::{public_target.get('sar_search', {}).get('comparison_arm') or 'default'}")
                    or []
                )
            ],
        }
        target_record.update(_public_campaign_metadata_record(public_target))
        if pipeline_result is not None:
            target_record["pipeline_result_status"] = pipeline_result.get("status")
            target_record["pipeline_result_error"] = pipeline_result.get("error")
        run_targets.append(target_record)

    product_lock_hash = _sha256_file(Path(product_lock_path)) if product_lock_path is not None else None
    effective_command_args = command_args if command_args is not None else {
        "api": "run_blind_validation",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "dry_run": bool(dry_run),
        "allow_real_downloads": bool(allow_real_downloads),
        "allow_synthetic_fallback": bool(allow_synthetic_fallback),
        "product_lock_path": str(product_lock_path) if product_lock_path is not None else None,
        "require_product_lock": bool(require_product_lock),
        "confirm_real_downloads_and_training": bool(confirm_real_downloads_and_training),
        "parameter_set_path": str(parameter_set_path) if parameter_set_path is not None else None,
        "require_approved_parameters": bool(require_approved_parameters),
        "campaign_registry_path": str(campaign_registry_path) if campaign_registry_path is not None else None,
        "require_locked_campaign_registry": bool(require_locked_campaign_registry),
        "robustness_plan_path": str(robustness_plan_path) if robustness_plan_path is not None else None,
        "robustness_variant_id": robustness_variant_id,
        "target_ids": sorted(target_id_filter) if target_id_filter is not None else None,
        "include_audit_only": bool(include_audit_only),
    }
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "validation_id": public_manifest["validation_id"],
        "public_manifest": str(manifest_path),
        "public_manifest_sha256": _sha256_file(manifest_path),
        "campaign_registry": campaign_registry_record,
        "campaign_registry_id": campaign_registry_record.get("registry_id") if campaign_registry_record else None,
        "campaign_registry_hash": campaign_registry_record.get("registry_hash") if campaign_registry_record else None,
        "campaign_registry_status": campaign_registry_record.get("status") if campaign_registry_record else None,
        "parameter_set": parameter_set_record,
        "parameter_set_id": parameter_set_record.get("parameter_set_id") if parameter_set_record else None,
        "parameter_set_hash": parameter_set_record.get("parameter_set_hash") if parameter_set_record else None,
        "parameter_set_approved_for_holdout": bool(parameter_set_record.get("approved_for_holdout")) if parameter_set_record else False,
        "robustness_plan": robustness_plan_record,
        "robustness_plan_id": robustness_plan_record.get("robustness_plan_id") if robustness_plan_record else None,
        "robustness_plan_hash": robustness_plan_record.get("robustness_plan_hash") if robustness_plan_record else None,
        "robustness_variant": robustness_variant_record,
        "robustness_variant_id": robustness_variant_record.get("variant_id") if robustness_variant_record else None,
        "product_lock": str(product_lock_path) if product_lock_path is not None else None,
        "product_lock_sha256": product_lock_hash,
        "product_lock_verification": product_lock_verification,
        "product_lock_enforcement": {
            "required": bool(require_product_lock),
            "lock_provided": product_lock_path is not None,
            "verified": product_lock_verification is not None,
            "mode": product_lock_verification.get("enforcement") if product_lock_verification else "not_requested",
            "strict_real_execution": strict_lock_execution,
            "remaining_limitations": product_lock_verification.get("remaining_limitations", [])
            if product_lock_verification else [],
        },
        "created_at_unix": int(time.time()),
        "created_at_utc": _utc_now_iso(),
        "dry_run": bool(dry_run),
        "real_downloads_allowed": bool(allow_real_downloads),
        "downloads_attempted": bool((not dry_run) and allow_real_downloads),
        "real_execution_confirmation": {
            "required": bool((not dry_run) and allow_real_downloads),
            "confirmed": bool(confirm_real_downloads_and_training),
            "flag": "--confirm-real-downloads-and-training",
            "policy": "Real execution is refused unless this explicit confirmation is supplied after product locks and disk estimates are reviewed.",
        },
        "synthetic_fallback_allowed": bool(allow_synthetic_fallback),
        "target_filter": sorted(target_id_filter) if target_id_filter is not None else None,
        "include_audit_only": bool(include_audit_only),
        "no_synthetic_fallback_status": {
            "synthetic_fallback_allowed": bool(allow_synthetic_fallback),
            "synthetic_fallback_disabled": not bool(allow_synthetic_fallback),
            "policy": "Synthetic fallback is disabled unless explicitly requested; locked-product execution forbids it.",
        },
        "reproducibility": {
            "public_manifest": str(manifest_path),
            "public_manifest_sha256": _sha256_file(manifest_path),
            "campaign_registry": campaign_registry_record,
            "campaign_registry_id": campaign_registry_record.get("registry_id") if campaign_registry_record else None,
            "campaign_registry_hash": campaign_registry_record.get("registry_hash") if campaign_registry_record else None,
            "campaign_registry_status": campaign_registry_record.get("status") if campaign_registry_record else None,
            "campaign_registry_required_locked": bool(require_locked_campaign_registry),
            "parameter_set": parameter_set_record,
            "parameter_set_id": parameter_set_record.get("parameter_set_id") if parameter_set_record else None,
            "parameter_set_hash": parameter_set_record.get("parameter_set_hash") if parameter_set_record else None,
            "parameter_set_required_approved_for_holdout": bool(require_approved_parameters),
            "robustness_plan": robustness_plan_record,
            "robustness_plan_id": robustness_plan_record.get("robustness_plan_id") if robustness_plan_record else None,
            "robustness_plan_hash": robustness_plan_record.get("robustness_plan_hash") if robustness_plan_record else None,
            "robustness_variant": robustness_variant_record,
            "product_lock": str(product_lock_path) if product_lock_path is not None else None,
            "product_lock_sha256": product_lock_hash,
            "real_execution_confirmation": bool(confirm_real_downloads_and_training),
            "code_fingerprints": _key_module_fingerprints(),
            "command_args": _json_safe_secret_redacted(effective_command_args),
            "runtime": _runtime_metadata(),
            "random_seed_policy": {
                "deterministic_seed_set_by_runner": False,
                "seed_value": None,
                "policy": "The blind runner does not set random seeds; dry-run candidate freezing is deterministic, while downstream real processing may depend on module-level randomness unless configured separately.",
            },
            "no_synthetic_fallback_status": {
                "synthetic_fallback_allowed": bool(allow_synthetic_fallback),
                "synthetic_fallback_disabled": not bool(allow_synthetic_fallback),
            },
        },
        "withheld_labels_loaded": False,
        "targets": run_targets,
    }
    _write_json(run_manifest_path, run_manifest)
    return run_manifest


def _candidate_confidence(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    for key in (
        "fused_confidence_score",
        "deep_target_score",
        "void_evidence_score",
        "mean_void_probability",
        "confidence",
        "score",
    ):
        value = _finite_or_none(row.get(key))
        if value is not None:
            return value, key
    return None, None


def _candidate_rank(row: Dict[str, Any], row_index: int) -> Tuple[int, str]:
    for key in ("deep_target_rank", "rank", "candidate_rank"):
        value = _int_or_none(row.get(key))
        if value is not None and value > 0:
            return value, key
    return row_index + 1, "csv_row_order"


def _candidate_position(row: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    assumptions: List[str] = []
    domain_width = _finite_or_none(target.get("domain_width_m"))
    center = target.get("center", {})
    center_lat = _finite_or_none(center.get("lat"))
    center_lon = _finite_or_none(center.get("lon"))

    centroid_m = _parse_numeric_sequence(row.get("centroid_m"), min_len=2)
    if centroid_m is not None:
        depth = centroid_m[2] if len(centroid_m) >= 3 else _finite_or_none(row.get("depth_m"))
        assumptions.append("candidate centroid_m interpreted as meters relative to public target center")
        return {
            "x_m": centroid_m[0],
            "y_m": centroid_m[1],
            "depth_m": depth,
            "location_source": "centroid_m",
            "location_confidence": "high",
            "assumptions": assumptions,
        }

    x = _finite_or_none(row.get("centroid_x_m"))
    y = _finite_or_none(row.get("centroid_y_m"))
    z = _finite_or_none(row.get("centroid_z_m"))
    if x is None:
        x = _finite_or_none(row.get("x_m"))
    if y is None:
        y = _finite_or_none(row.get("y_m"))
    if z is None:
        z = _finite_or_none(row.get("z_m"))
    if x is not None and y is not None:
        assumptions.append("candidate x/y meter fields interpreted relative to public target center")
        return {
            "x_m": x,
            "y_m": y,
            "depth_m": z if z is not None else _finite_or_none(row.get("depth_m")),
            "location_source": "xy_meter_fields",
            "location_confidence": "high",
            "assumptions": assumptions,
        }

    lat = _finite_or_none(row.get("centroid_lat"))
    lon = _finite_or_none(row.get("centroid_lon"))
    if lat is None:
        lat = _finite_or_none(row.get("lat"))
    if lon is None:
        lon = _finite_or_none(row.get("lon"))
    if lat is not None and lon is not None and center_lat is not None and center_lon is not None:
        x, y = _latlon_to_offset_m(lat, lon, center_lat, center_lon)
        assumptions.append("candidate lat/lon converted to local meters around public target center")
        return {
            "x_m": x,
            "y_m": y,
            "depth_m": _finite_or_none(row.get("depth_m")),
            "lat": lat,
            "lon": lon,
            "location_source": "lat_lon",
            "location_confidence": "medium",
            "assumptions": assumptions,
        }

    centroid_px = _parse_numeric_sequence(row.get("centroid_px"), min_len=2)
    if centroid_px is None:
        centroid_px = _parse_numeric_sequence(row.get("centroid_pixel"), min_len=2)
    if centroid_px is not None and domain_width is not None:
        nx = _finite_or_none(row.get("grid_nx")) or _finite_or_none(row.get("nx"))
        ny = _finite_or_none(row.get("grid_ny")) or _finite_or_none(row.get("ny"))
        if nx and ny and nx > 1 and ny > 1:
            x = (centroid_px[0] / (nx - 1.0) - 0.5) * domain_width
            y = (centroid_px[1] / (ny - 1.0) - 0.5) * domain_width
            assumptions.append("candidate pixel centroid converted using public domain_width_m and grid dimensions")
        else:
            x = 0.0
            y = 0.0
            assumptions.append("candidate pixel centroid present but grid dimensions absent; scored at target center")
        return {
            "x_m": x,
            "y_m": y,
            "depth_m": _finite_or_none(row.get("depth_m")),
            "location_source": "pixel_domain_fallback",
            "location_confidence": "low",
            "assumptions": assumptions,
        }

    assumptions.append("candidate location absent; scored at public target center")
    return {
        "x_m": 0.0,
        "y_m": 0.0,
        "depth_m": _finite_or_none(row.get("depth_m")),
        "location_source": "target_center_fallback",
        "location_confidence": "very_low",
        "assumptions": assumptions,
    }


def load_candidate_csv(path: Path | str, target: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    candidates = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        for row_index, row in enumerate(reader):
            rank, rank_source = _candidate_rank(row, row_index)
            confidence, confidence_source = _candidate_confidence(row)
            position = _candidate_position(row, target)
            candidates.append(
                {
                    "row_index": row_index,
                    "candidate_id": row.get("id") or row.get("candidate_id") or str(row_index + 1),
                    "rank": rank,
                    "rank_source": rank_source,
                    "confidence": confidence,
                    "confidence_source": confidence_source,
                    "x_m": position["x_m"],
                    "y_m": position["y_m"],
                    "depth_m": position.get("depth_m"),
                    "location_source": position["location_source"],
                    "location_confidence": position["location_confidence"],
                    "assumptions": position["assumptions"],
                    "raw": row,
                }
            )
    candidates.sort(key=lambda item: (item["rank"], item["row_index"]))
    return candidates


def _label_position(label: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    assumptions: List[str] = []
    offset = _parse_numeric_sequence(label.get("offset_m"), min_len=2)
    if offset is None:
        offset = _parse_numeric_sequence(label.get("centroid_m"), min_len=2)
    if offset is not None:
        depth = label.get("depth_m")
        if depth is None and len(offset) >= 3:
            depth = offset[2]
        return {
            "x_m": offset[0],
            "y_m": offset[1],
            "depth_m": _finite_or_none(depth),
            "source": "withheld_offset_m",
            "assumptions": assumptions,
        }
    lat = _finite_or_none(label.get("lat"))
    lon = _finite_or_none(label.get("lon"))
    center = target.get("center", {})
    center_lat = _finite_or_none(center.get("lat"))
    center_lon = _finite_or_none(center.get("lon"))
    if lat is not None and lon is not None and center_lat is not None and center_lon is not None:
        x, y = _latlon_to_offset_m(lat, lon, center_lat, center_lon)
        assumptions.append("withheld label lat/lon converted to local meters around public target center")
        return {
            "x_m": x,
            "y_m": y,
            "depth_m": _finite_or_none(label.get("depth_m")),
            "source": "withheld_lat_lon",
            "assumptions": assumptions,
        }
    raise ManifestValidationError(f"Withheld label {label.get('void_id', '?')} lacks a usable location")


def _match_label_to_candidates(
    label: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    label_pos = _label_position(label, target)
    horizontal_tolerance = _finite_or_none(label.get("horizontal_tolerance_m")) or DEFAULT_HORIZONTAL_TOLERANCE_M
    depth_tolerance = _finite_or_none(label.get("depth_tolerance_m")) or DEFAULT_DEPTH_TOLERANCE_M
    label_depth = label_pos.get("depth_m")
    comparisons = []
    best_pass = None
    best_any = None

    for candidate in candidates:
        dx = float(candidate["x_m"]) - float(label_pos["x_m"])
        dy = float(candidate["y_m"]) - float(label_pos["y_m"])
        horizontal_error = math.hypot(dx, dy)
        candidate_depth = candidate.get("depth_m")
        depth_error = None
        depth_pass = True
        if label_depth is not None:
            if candidate_depth is None:
                depth_pass = True
            else:
                depth_error = abs(float(candidate_depth) - float(label_depth))
                depth_pass = depth_error <= depth_tolerance
        horizontal_pass = horizontal_error <= horizontal_tolerance
        passed = bool(horizontal_pass and depth_pass)
        comparison = {
            "candidate": candidate,
            "horizontal_error_m": horizontal_error,
            "depth_error_m": depth_error,
            "horizontal_pass": horizontal_pass,
            "depth_pass": depth_pass,
            "passed": passed,
        }
        comparisons.append(comparison)
        if best_any is None or (
            horizontal_error,
            depth_error if depth_error is not None else -1.0,
            candidate["rank"],
        ) < (
            best_any["horizontal_error_m"],
            best_any["depth_error_m"] if best_any["depth_error_m"] is not None else -1.0,
            best_any["candidate"]["rank"],
        ):
            best_any = comparison
        if passed and (
            best_pass is None
            or (candidate["rank"], horizontal_error) < (best_pass["candidate"]["rank"], best_pass["horizontal_error_m"])
        ):
            best_pass = comparison

    best = best_pass or best_any
    matched = best_pass is not None
    assumptions = list(label_pos.get("assumptions", []))
    if matched and best and label_depth is not None and best["candidate"].get("depth_m") is None:
        assumptions.append("candidate depth absent; hit determined by horizontal distance only")

    return {
        "void_id": label["void_id"],
        "matched": matched,
        "horizontal_tolerance_m": horizontal_tolerance,
        "depth_tolerance_m": depth_tolerance if label_depth is not None else None,
        "best_candidate_id": best["candidate"]["candidate_id"] if best else None,
        "best_candidate_rank": best["candidate"]["rank"] if best else None,
        "best_candidate_confidence": best["candidate"].get("confidence") if best else None,
        "horizontal_error_m": best["horizontal_error_m"] if best else None,
        "depth_error_m": best["depth_error_m"] if best else None,
        "label_depth_m": label_depth,
        "candidate_depth_m": best["candidate"].get("depth_m") if best else None,
        "location_source": label_pos["source"],
        "assumptions": assumptions,
        "matched_candidate_row_index": best["candidate"]["row_index"] if matched and best else None,
    }


def _summarize_confidences(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [c["confidence"] for c in candidates if c.get("confidence") is not None]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "top_rank_confidence": None,
        }
    top = min(candidates, key=lambda c: (c["rank"], c["row_index"]))
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "top_rank_confidence": top.get("confidence"),
    }


def _score_target(
    run_target: Dict[str, Any],
    label_entry: Dict[str, Any],
    run_manifest_dir: Path,
) -> Dict[str, Any]:
    target = run_target.get("public_target", {})
    if not target:
        target = {
            "target_id": run_target["target_id"],
            "name": run_target.get("name", run_target["target_id"]),
            "center": run_target.get("center", {"lat": 0.0, "lon": 0.0}),
        }
    csv_value = run_target.get("frozen_candidates_csv")
    csv_path = Path(csv_value) if csv_value else Path("missing.csv")
    if not csv_path.is_absolute():
        csv_path = run_manifest_dir / csv_path
    candidates = load_candidate_csv(csv_path, target)
    site_class = label_entry["site_class"]
    known_voids = label_entry.get("known_voids", [])
    label_results = [
        _match_label_to_candidates(label, candidates, target)
        for label in known_voids
    ]

    matched_rows = {
        result["matched_candidate_row_index"]
        for result in label_results
        if result.get("matched_candidate_row_index") is not None
    }
    positive_site_hit = bool(site_class == "positive" and any(r["matched"] for r in label_results))
    positive_site_miss = bool(site_class == "positive" and not positive_site_hit)
    negative_site_false_positive = bool(site_class == "negative" and len(candidates) > 0)
    first_hit_results = [r for r in label_results if r["matched"]]
    first_hit = min(first_hit_results, key=lambda r: (r["best_candidate_rank"], r["horizontal_error_m"])) if first_hit_results else None
    area_km2 = _finite_or_none(target.get("area_km2"))
    candidates_per_km2 = (len(candidates) / area_km2) if area_km2 and area_km2 > 0 else None
    assumptions = []
    for candidate in candidates:
        assumptions.extend(candidate.get("assumptions", []))
    for result in label_results:
        assumptions.extend(result.get("assumptions", []))
    assumptions = sorted(set(assumptions))
    public_metadata = _public_campaign_metadata_record(target)
    split = (
        target.get("split")
        or target.get("split_designation")
        or run_target.get("split")
        or run_target.get("split_designation")
        or label_entry.get("split")
        or "unspecified"
    )
    site_type = target.get("site_type") or label_entry.get("site_type") or label_entry.get("site_subclass")
    public_site_category = target.get("public_site_category") or run_target.get("public_site_category")
    sar_search = target.get("sar_search", {}) if isinstance(target.get("sar_search"), dict) else {}
    provider = run_target.get("provider") or sar_search.get("provider") or "unspecified"
    provider_label = run_target.get("provider_label") or sar_search.get("provider_label") or str(provider)
    comparison_arm = run_target.get("comparison_arm") or sar_search.get("comparison_arm") or str(provider)
    comparison_group = run_target.get("comparison_group") or sar_search.get("comparison_group")

    return _round_metric(
        {
            "target_id": run_target["target_id"],
            "name": run_target.get("name"),
            "split": str(split),
            "provider": str(provider),
            "provider_label": str(provider_label),
            "comparison_arm": str(comparison_arm),
            "comparison_group": str(comparison_group) if comparison_group is not None else None,
            "split_designation": target.get("split_designation") or run_target.get("split_designation"),
            "pair_id": target.get("pair_id") or run_target.get("pair_id"),
            "group_id": target.get("group_id") or run_target.get("group_id"),
            "public_site_category": public_site_category,
            "public_strata": target.get("public_strata") or target.get("public_stratum") or run_target.get("public_strata"),
            "acquisition_stratum": target.get("acquisition_stratum") or run_target.get("acquisition_stratum"),
            "terrain_descriptor": target.get("terrain_descriptor") or run_target.get("terrain_descriptor"),
            "land_cover_descriptor": target.get("land_cover_descriptor") or run_target.get("land_cover_descriptor"),
            "lithology_descriptor": target.get("lithology_descriptor") or run_target.get("lithology_descriptor"),
            "campaign_tier": target.get("campaign_tier") or run_target.get("campaign_tier"),
            "prior_run_status": target.get("prior_run_status") or run_target.get("prior_run_status"),
            "audit_only": bool(target.get("audit_only") or run_target.get("audit_only")),
            "prior_run_audit_only": bool(target.get("prior_run_audit_only") or run_target.get("prior_run_audit_only")),
            "public_scoring_status": target.get("public_scoring_status") or run_target.get("public_scoring_status"),
            "public_campaign_metadata": public_metadata,
            "site_class": site_class,
            "site_type": site_type,
            "candidate_count": len(candidates),
            "known_void_count": len(known_voids),
            "matched_known_void_count": sum(1 for r in label_results if r["matched"]),
            "matched_candidate_count": len(matched_rows),
            "positive_site_hit": positive_site_hit,
            "positive_site_miss": positive_site_miss,
            "negative_site_false_positive": negative_site_false_positive,
            "area_km2": area_km2,
            "candidates_per_km2": candidates_per_km2,
            "rank_of_first_hit": first_hit["best_candidate_rank"] if first_hit else None,
            "first_hit_distance_error_m": first_hit["horizontal_error_m"] if first_hit else None,
            "first_hit_depth_error_m": first_hit["depth_error_m"] if first_hit else None,
            "first_hit_confidence": first_hit["best_candidate_confidence"] if first_hit else None,
            "confidence_summary": _summarize_confidences(candidates),
            "candidate_summaries": [
                {
                    "candidate_id": c["candidate_id"],
                    "rank": c["rank"],
                    "confidence": c["confidence"],
                    "x_m": c["x_m"],
                    "y_m": c["y_m"],
                    "depth_m": c.get("depth_m"),
                    "location_source": c["location_source"],
                    "location_confidence": c["location_confidence"],
                    "matched": c["row_index"] in matched_rows,
                }
                for c in candidates
            ],
            "label_results": label_results,
            "scoring_assumptions": assumptions,
        }
    )


def _build_summary(target_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive_scores = [s for s in target_scores if s["site_class"] == "positive"]
    negative_scores = [s for s in target_scores if s["site_class"] == "negative"]
    positive_hits = sum(1 for s in positive_scores if s["positive_site_hit"])
    negative_fps = sum(1 for s in negative_scores if s["negative_site_false_positive"])
    total_candidates = sum(int(s["candidate_count"]) for s in target_scores)
    matched_candidates = sum(int(s["matched_candidate_count"]) for s in target_scores)
    total_labels = sum(int(s["known_void_count"]) for s in target_scores)
    matched_labels = sum(int(s["matched_known_void_count"]) for s in target_scores)
    precision_denom = positive_hits + negative_fps
    candidate_precision_denom = total_candidates
    ranks = [s["rank_of_first_hit"] for s in target_scores if s["rank_of_first_hit"] is not None]
    distances = [s["first_hit_distance_error_m"] for s in target_scores if s["first_hit_distance_error_m"] is not None]
    depths = [s["first_hit_depth_error_m"] for s in target_scores if s["first_hit_depth_error_m"] is not None]
    all_conf = []
    top_conf = []
    for score in target_scores:
        conf_summary = score.get("confidence_summary", {})
        if conf_summary.get("top_rank_confidence") is not None:
            top_conf.append(conf_summary["top_rank_confidence"])
        for candidate in score.get("candidate_summaries", []):
            if candidate.get("confidence") is not None:
                all_conf.append(candidate["confidence"])

    summary = {
        "target_count": len(target_scores),
        "positive_sites": len(positive_scores),
        "positive_site_hits": positive_hits,
        "positive_site_misses": sum(1 for s in positive_scores if s["positive_site_miss"]),
        "negative_sites": len(negative_scores),
        "negative_site_false_positive_sites": negative_fps,
        "negative_site_true_negative_sites": len(negative_scores) - negative_fps,
        "site_precision_like": positive_hits / precision_denom if precision_denom else None,
        "site_recall_like": positive_hits / len(positive_scores) if positive_scores else None,
        "candidate_precision_like": matched_candidates / candidate_precision_denom if candidate_precision_denom else None,
        "known_void_label_recall_like": matched_labels / total_labels if total_labels else None,
        "total_candidates": total_candidates,
        "matched_candidate_count": matched_candidates,
        "total_known_void_labels": total_labels,
        "matched_known_void_labels": matched_labels,
        "rank_summary": {
            "first_hit_ranks": ranks,
            "mean_first_hit_rank": mean(ranks) if ranks else None,
            "median_first_hit_rank": median(ranks) if ranks else None,
        },
        "distance_error_m_summary": {
            "mean_first_hit_distance_error_m": mean(distances) if distances else None,
            "median_first_hit_distance_error_m": median(distances) if distances else None,
            "max_first_hit_distance_error_m": max(distances) if distances else None,
        },
        "depth_error_m_summary": {
            "mean_first_hit_depth_error_m": mean(depths) if depths else None,
            "median_first_hit_depth_error_m": median(depths) if depths else None,
            "max_first_hit_depth_error_m": max(depths) if depths else None,
        },
        "confidence_summary": {
            "all_candidate_count_with_confidence": len(all_conf),
            "all_candidate_confidence_mean": mean(all_conf) if all_conf else None,
            "all_candidate_confidence_median": median(all_conf) if all_conf else None,
            "all_candidate_confidence_min": min(all_conf) if all_conf else None,
            "all_candidate_confidence_max": max(all_conf) if all_conf else None,
            "top_rank_confidence_mean": mean(top_conf) if top_conf else None,
            "top_rank_confidence_median": median(top_conf) if top_conf else None,
        },
    }
    return _round_metric(summary)


def _metadata_value_counts(targets: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for target in targets:
        value = str(target.get(key) or "unspecified")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def score_blind_validation(
    run_manifest_path: Path | str,
    labels_path: Path | str,
    output_path: Optional[Path | str] = None,
    *,
    parameter_set_path: Optional[Path | str] = None,
    require_approved_parameters: bool = False,
) -> Dict[str, Any]:
    """Score frozen blind candidates against withheld known-void labels."""
    run_manifest_path = Path(run_manifest_path)
    labels_path = Path(labels_path)
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Run manifest schema_version must be {RUN_MANIFEST_SCHEMA_VERSION!r}"
        )
    parameter_verification = verify_parameter_set_for_scoring(
        run_manifest,
        parameter_set_path=parameter_set_path,
        require_approved=require_approved_parameters,
    )
    labels = load_withheld_labels(labels_path)
    if labels.get("validation_id") != run_manifest.get("validation_id"):
        raise ManifestValidationError(
            "Withheld labels validation_id does not match run manifest validation_id"
        )
    run_targets = run_manifest.get("targets")
    if not isinstance(run_targets, list) or not run_targets:
        raise ManifestValidationError("Run manifest targets must be a non-empty list")
    label_by_target = {entry["target_id"]: entry for entry in labels["labels"]}
    run_target_ids = {target.get("target_id") for target in run_targets}
    missing = sorted(t for t in run_target_ids if t not in label_by_target)
    extra = sorted(t for t in label_by_target if t not in run_target_ids)
    if missing:
        raise ManifestValidationError(f"Withheld labels missing target_id(s): {missing}")
    if extra:
        raise ManifestValidationError(f"Withheld labels include target_id(s) not present in run: {extra}")

    run_manifest_dir = run_manifest_path.parent
    target_scores = [
        _score_target(target, label_by_target[target["target_id"]], run_manifest_dir)
        for target in run_targets
    ]
    report = {
        "schema_version": SCORE_REPORT_SCHEMA_VERSION,
        "validation_id": run_manifest["validation_id"],
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": _sha256_file(run_manifest_path),
        "campaign_registry": run_manifest.get("campaign_registry"),
        "campaign_registry_id": run_manifest.get("campaign_registry_id"),
        "campaign_registry_hash": run_manifest.get("campaign_registry_hash"),
        "campaign_registry_status": run_manifest.get("campaign_registry_status"),
        "parameter_set": run_manifest.get("parameter_set"),
        "parameter_set_id": run_manifest.get("parameter_set_id"),
        "parameter_set_hash": run_manifest.get("parameter_set_hash"),
        "parameter_set_verification": parameter_verification,
        "withheld_labels": str(labels_path),
        "withheld_labels_sha256": _sha256_file(labels_path),
        "summary": _build_summary(target_scores),
        "targets": target_scores,
    }
    report["summary"]["by_provider"] = _grouped_target_summaries(target_scores, "provider")
    report["summary"]["by_provider_label"] = _grouped_target_summaries(target_scores, "provider_label")
    report["summary"]["by_comparison_arm"] = _grouped_target_summaries(target_scores, "comparison_arm")
    report["summary"]["by_split"] = _grouped_target_summaries(target_scores, "split")
    report["summary"]["metadata_counts"] = {
        "provider": _metadata_value_counts(target_scores, "provider"),
        "provider_label": _metadata_value_counts(target_scores, "provider_label"),
        "comparison_arm": _metadata_value_counts(target_scores, "comparison_arm"),
        "split": _metadata_value_counts(target_scores, "split"),
        "public_site_category": _metadata_value_counts(target_scores, "public_site_category"),
        "campaign_tier": _metadata_value_counts(target_scores, "campaign_tier"),
        "acquisition_stratum": _metadata_value_counts(target_scores, "acquisition_stratum"),
        "prior_run_status": _metadata_value_counts(target_scores, "prior_run_status"),
    }
    report = _round_metric(report)
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def _percentile(sorted_values: List[float], fraction: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _numeric_distribution(values: Iterable[Any]) -> Dict[str, Any]:
    nums = [float(value) for value in values if _finite_or_none(value) is not None]
    nums.sort()
    return _round_metric(
        {
            "count": len(nums),
            "mean": mean(nums) if nums else None,
            "median": median(nums) if nums else None,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
            "p25": _percentile(nums, 0.25),
            "p75": _percentile(nums, 0.75),
        }
    )


def _target_collection_summary(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive = [target for target in targets if target.get("site_class") == "positive"]
    negative = [target for target in targets if target.get("site_class") == "negative"]
    positive_hits = sum(1 for target in positive if target.get("positive_site_hit"))
    negative_fp_sites = sum(1 for target in negative if target.get("negative_site_false_positive"))
    total_candidates = sum(int(target.get("candidate_count", 0) or 0) for target in targets)
    matched_candidates = sum(int(target.get("matched_candidate_count", 0) or 0) for target in targets)
    total_labels = sum(int(target.get("known_void_count", 0) or 0) for target in targets)
    matched_labels = sum(int(target.get("matched_known_void_count", 0) or 0) for target in targets)
    precision_denom = positive_hits + negative_fp_sites
    return _round_metric(
        {
            "target_count": len(targets),
            "positive_sites": len(positive),
            "positive_site_hits": positive_hits,
            "positive_site_misses": sum(1 for target in positive if target.get("positive_site_miss")),
            "positive_hit_rate": positive_hits / len(positive) if positive else None,
            "negative_sites": len(negative),
            "negative_false_positive_sites": negative_fp_sites,
            "negative_true_negative_sites": len(negative) - negative_fp_sites,
            "negative_false_positive_site_rate": negative_fp_sites / len(negative) if negative else None,
            "site_precision_like": positive_hits / precision_denom if precision_denom else None,
            "site_recall_like": positive_hits / len(positive) if positive else None,
            "candidate_precision_like": matched_candidates / total_candidates if total_candidates else None,
            "known_void_label_recall_like": matched_labels / total_labels if total_labels else None,
            "total_candidates": total_candidates,
            "matched_candidate_count": matched_candidates,
            "total_known_void_labels": total_labels,
            "matched_known_void_labels": matched_labels,
        }
    )


def _grouped_target_summaries(targets: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for target in targets:
        groups.setdefault(str(target.get(key) or "unspecified"), []).append(target)
    return {group_key: _target_collection_summary(items) for group_key, items in sorted(groups.items())}


def _false_positive_area_summary(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    negative_targets = [target for target in targets if target.get("site_class") == "negative"]
    negative_candidates = sum(int(target.get("candidate_count", 0) or 0) for target in negative_targets)
    negative_area_values = [
        float(target.get("area_km2"))
        for target in negative_targets
        if _finite_or_none(target.get("area_km2")) is not None and float(target.get("area_km2")) > 0
    ]
    all_area_values = [
        float(target.get("area_km2"))
        for target in targets
        if _finite_or_none(target.get("area_km2")) is not None and float(target.get("area_km2")) > 0
    ]
    all_candidates = sum(int(target.get("candidate_count", 0) or 0) for target in targets)
    negative_area = sum(negative_area_values)
    all_area = sum(all_area_values)
    return _round_metric(
        {
            "negative_candidate_false_positives": negative_candidates,
            "negative_area_km2_with_area": negative_area,
            "negative_false_positive_candidates_per_km2": negative_candidates / negative_area
            if negative_area > 0 else None,
            "all_candidates": all_candidates,
            "all_area_km2_with_area": all_area,
            "all_candidates_per_km2": all_candidates / all_area if all_area > 0 else None,
            "targets_missing_area_km2": sum(1 for target in targets if _finite_or_none(target.get("area_km2")) is None),
        }
    )


def _confidence_calibration_bins(targets: List[Dict[str, Any]], bin_count: int = 5) -> List[Dict[str, Any]]:
    bin_count = max(1, int(bin_count))
    bins = []
    for idx in range(bin_count):
        lower = idx / bin_count
        upper = (idx + 1) / bin_count
        bins.append(
            {
                "bin_index": idx,
                "lower_inclusive": lower,
                "upper_exclusive": upper if idx < bin_count - 1 else None,
                "upper_inclusive": upper if idx == bin_count - 1 else None,
                "candidate_count": 0,
                "matched_count": 0,
                "confidence_sum": 0.0,
            }
        )
    out_of_range = {
        "bin_index": "out_of_range",
        "lower_inclusive": None,
        "upper_exclusive": None,
        "upper_inclusive": None,
        "candidate_count": 0,
        "matched_count": 0,
        "confidence_sum": 0.0,
    }
    for target in targets:
        for candidate in target.get("candidate_summaries", []):
            confidence = _finite_or_none(candidate.get("confidence"))
            if confidence is None:
                continue
            bucket = bins[min(int(confidence * bin_count), bin_count - 1)] if 0.0 <= confidence <= 1.0 else out_of_range
            bucket["candidate_count"] += 1
            bucket["matched_count"] += 1 if candidate.get("matched") else 0
            bucket["confidence_sum"] += float(confidence)
    output_bins = bins + ([out_of_range] if out_of_range["candidate_count"] else [])
    for bucket in output_bins:
        confidence_sum = float(bucket.pop("confidence_sum"))
        candidate_count = int(bucket["candidate_count"])
        bucket["mean_confidence"] = confidence_sum / candidate_count if candidate_count else None
        bucket["observed_match_rate"] = bucket["matched_count"] / candidate_count if candidate_count else None
    return _round_metric(output_bins)


def build_baseline_report(score_paths: Sequence[Path | str], *, confidence_bins: int = 5) -> Dict[str, Any]:
    paths = [Path(path) for path in score_paths]
    if not paths:
        raise ManifestValidationError("At least one score JSON is required")
    score_records = []
    targets: List[Dict[str, Any]] = []
    for path in paths:
        score = _load_json(path)
        if score.get("schema_version") != SCORE_REPORT_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Score report {path} schema_version must be {SCORE_REPORT_SCHEMA_VERSION!r}"
            )
        score_targets = score.get("targets")
        if not isinstance(score_targets, list):
            raise ManifestValidationError(f"Score report {path} targets must be a list")
        score_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "validation_id": score.get("validation_id"),
                "run_manifest": score.get("run_manifest"),
                "run_manifest_sha256": score.get("run_manifest_sha256"),
                "parameter_set_id": score.get("parameter_set_id"),
                "parameter_set_hash": score.get("parameter_set_hash"),
                "parameter_set_approved_for_holdout": bool(
                    (score.get("parameter_set") or {}).get("approved_for_holdout")
                ) if isinstance(score.get("parameter_set"), dict) else False,
                "withheld_labels_sha256": score.get("withheld_labels_sha256"),
                "target_count": len(score_targets),
            }
        )
        for target in score_targets:
            if isinstance(target, dict):
                copied = dict(target)
                copied["source_score_report"] = str(path)
                copied["validation_id"] = score.get("validation_id")
                targets.append(copied)

    rank_values = [target.get("rank_of_first_hit") for target in targets if target.get("rank_of_first_hit") is not None]
    rank_histogram: Dict[str, int] = {}
    for rank in rank_values:
        key = str(int(float(rank)))
        rank_histogram[key] = rank_histogram.get(key, 0) + 1
    localization_errors = []
    depth_errors = []
    for target in targets:
        for result in target.get("label_results", []):
            if result.get("matched"):
                if _finite_or_none(result.get("horizontal_error_m")) is not None:
                    localization_errors.append(result.get("horizontal_error_m"))
                if _finite_or_none(result.get("depth_error_m")) is not None:
                    depth_errors.append(result.get("depth_error_m"))

    split_groups: Dict[str, List[Dict[str, Any]]] = {}
    for target in targets:
        split_groups.setdefault(str(target.get("split") or "unspecified"), []).append(target)
    by_split = {split: _target_collection_summary(items) for split, items in sorted(split_groups.items())}
    by_split_and_site_class = {
        split: _grouped_target_summaries(items, "site_class") for split, items in sorted(split_groups.items())
    }
    summary = _target_collection_summary(targets)
    summary.update(
        {
            "score_report_count": len(score_records),
            "by_site_class": _grouped_target_summaries(targets, "site_class"),
            "by_provider": _grouped_target_summaries(targets, "provider"),
            "by_provider_label": _grouped_target_summaries(targets, "provider_label"),
            "by_comparison_arm": _grouped_target_summaries(targets, "comparison_arm"),
            "by_split": by_split,
            "by_split_and_site_class": by_split_and_site_class,
            "metadata_counts": {
                "provider": _metadata_value_counts(targets, "provider"),
                "provider_label": _metadata_value_counts(targets, "provider_label"),
                "comparison_arm": _metadata_value_counts(targets, "comparison_arm"),
            },
            "false_positives_per_area": _false_positive_area_summary(targets),
            "rank_of_first_hit_distribution": {
                "values": [int(float(value)) for value in rank_values],
                "histogram": dict(sorted(rank_histogram.items(), key=lambda item: int(item[0]))),
                "summary": _numeric_distribution(rank_values),
            },
            "localization_error_m_summary": _numeric_distribution(localization_errors),
            "depth_error_m_summary": _numeric_distribution(depth_errors),
            "confidence_calibration_bins": _confidence_calibration_bins(targets, confidence_bins),
        }
    )
    compact_targets = [
        {
            "source_score_report": target.get("source_score_report"),
            "validation_id": target.get("validation_id"),
            "target_id": target.get("target_id"),
            "name": target.get("name"),
            "split": target.get("split", "unspecified"),
            "provider": target.get("provider"),
            "provider_label": target.get("provider_label"),
            "comparison_arm": target.get("comparison_arm"),
            "comparison_group": target.get("comparison_group"),
            "site_class": target.get("site_class"),
            "candidate_count": target.get("candidate_count"),
            "known_void_count": target.get("known_void_count"),
            "matched_known_void_count": target.get("matched_known_void_count"),
            "positive_site_hit": target.get("positive_site_hit"),
            "negative_site_false_positive": target.get("negative_site_false_positive"),
            "area_km2": target.get("area_km2"),
            "rank_of_first_hit": target.get("rank_of_first_hit"),
            "first_hit_distance_error_m": target.get("first_hit_distance_error_m"),
            "first_hit_depth_error_m": target.get("first_hit_depth_error_m"),
            "first_hit_confidence": target.get("first_hit_confidence"),
        }
        for target in targets
    ]
    return {
        "schema_version": BASELINE_REPORT_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "score_reports": score_records,
        "summary": _round_metric(summary),
        "targets": _round_metric(compact_targets),
    }


def _score_variant_record(score: Dict[str, Any]) -> Dict[str, Any]:
    raw = score.get("robustness_variant")
    if not isinstance(raw, dict):
        raw = score.get("ablation_variant")
    if not isinstance(raw, dict):
        raw = {}
    variant_id = raw.get("variant_id") or score.get("robustness_variant_id") or score.get("ablation_variant_id") or "baseline"
    family = raw.get("family") or score.get("robustness_family") or score.get("ablation_family") or "unspecified"
    arm = raw.get("arm") or score.get("robustness_arm") or score.get("ablation_arm") or family
    return {
        "variant_id": str(variant_id),
        "family": str(family),
        "arm": str(arm),
        "null_baseline": bool(raw.get("null_baseline", score.get("null_baseline", False))),
        "stability_group": raw.get("stability_group") or score.get("stability_group"),
    }


def _variant_grouped_summary(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(key) or "unspecified"), []).append(record)
    return {group_key: _target_collection_summary(items) for group_key, items in sorted(groups.items())}


def build_robustness_summary(score_paths: Sequence[Path | str]) -> Dict[str, Any]:
    """Aggregate existing score JSONs by robustness/ablation metadata."""
    paths = [Path(path) for path in score_paths]
    if not paths:
        raise ManifestValidationError("At least one score JSON is required")
    score_records = []
    targets: List[Dict[str, Any]] = []
    for path in paths:
        score = _load_json(path)
        if score.get("schema_version") != SCORE_REPORT_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Score report {path} schema_version must be {SCORE_REPORT_SCHEMA_VERSION!r}"
            )
        score_targets = score.get("targets")
        if not isinstance(score_targets, list):
            raise ManifestValidationError(f"Score report {path} targets must be a list")
        variant = _score_variant_record(score)
        score_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "validation_id": score.get("validation_id"),
                "run_manifest": score.get("run_manifest"),
                "run_manifest_sha256": score.get("run_manifest_sha256"),
                "parameter_set_id": score.get("parameter_set_id"),
                "parameter_set_hash": score.get("parameter_set_hash"),
                "variant": variant,
                "target_count": len(score_targets),
            }
        )
        for target in score_targets:
            if isinstance(target, dict):
                copied = dict(target)
                copied["source_score_report"] = str(path)
                copied["validation_id"] = score.get("validation_id")
                copied["variant_id"] = variant["variant_id"]
                copied["variant_family"] = variant["family"]
                copied["variant_arm"] = variant["arm"]
                copied["null_baseline"] = bool(variant["null_baseline"])
                copied["null_baseline_group"] = "true" if variant["null_baseline"] else "false"
                copied["stability_group"] = variant.get("stability_group")
                targets.append(copied)
    summary = _target_collection_summary(targets)
    summary.update(
        {
            "score_report_count": len(score_records),
            "by_variant_id": _variant_grouped_summary(targets, "variant_id"),
            "by_family": _variant_grouped_summary(targets, "variant_family"),
            "by_arm": _variant_grouped_summary(targets, "variant_arm"),
            "by_provider": _variant_grouped_summary(targets, "provider"),
            "by_comparison_arm": _variant_grouped_summary(targets, "comparison_arm"),
            "by_null_baseline": _variant_grouped_summary(targets, "null_baseline_group"),
            "by_stability_group": _variant_grouped_summary(targets, "stability_group"),
            "metadata_counts": {
                "variant_id": _metadata_value_counts(targets, "variant_id"),
                "variant_family": _metadata_value_counts(targets, "variant_family"),
                "variant_arm": _metadata_value_counts(targets, "variant_arm"),
                "provider": _metadata_value_counts(targets, "provider"),
                "comparison_arm": _metadata_value_counts(targets, "comparison_arm"),
                "null_baseline": _metadata_value_counts(targets, "null_baseline_group"),
            },
            "false_positives_per_area": _false_positive_area_summary(targets),
        }
    )
    compact_targets = [
        {
            "source_score_report": target.get("source_score_report"),
            "validation_id": target.get("validation_id"),
            "variant_id": target.get("variant_id"),
            "variant_family": target.get("variant_family"),
            "variant_arm": target.get("variant_arm"),
            "null_baseline": target.get("null_baseline"),
            "stability_group": target.get("stability_group"),
            "target_id": target.get("target_id"),
            "split": target.get("split", "unspecified"),
            "provider": target.get("provider"),
            "comparison_arm": target.get("comparison_arm"),
            "site_class": target.get("site_class"),
            "candidate_count": target.get("candidate_count"),
            "matched_candidate_count": target.get("matched_candidate_count"),
            "known_void_count": target.get("known_void_count"),
            "matched_known_void_count": target.get("matched_known_void_count"),
            "positive_site_hit": target.get("positive_site_hit"),
            "negative_site_false_positive": target.get("negative_site_false_positive"),
        }
        for target in targets
    ]
    return {
        "schema_version": ROBUSTNESS_SUMMARY_SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "score_reports": score_records,
        "summary": _round_metric(summary),
        "targets": _round_metric(compact_targets),
        "withheld_labels_loaded": False,
    }


def format_baseline_report_text(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = [
        "Blind validation baseline report",
        f"Generated UTC: {report.get('created_at_utc')}",
        f"Score reports: {summary.get('score_report_count', len(report.get('score_reports', [])))}",
        f"Targets: {summary.get('target_count')}",
        "",
        "Precision/recall-like metrics:",
        f"  Site precision-like: {fmt(summary.get('site_precision_like'))}",
        f"  Site recall-like: {fmt(summary.get('site_recall_like'))}",
        f"  Candidate precision-like: {fmt(summary.get('candidate_precision_like'))}",
        f"  Known-void label recall-like: {fmt(summary.get('known_void_label_recall_like'))}",
        "",
        "Site outcomes:",
        f"  Positive hits: {summary.get('positive_site_hits')}/{summary.get('positive_sites')}",
        f"  Negative false-positive sites: {summary.get('negative_false_positive_sites')}/{summary.get('negative_sites')}",
    ]
    area_summary = summary.get("false_positives_per_area", {})
    lines.extend(
        [
            f"  Negative false-positive candidates per km2: {fmt(area_summary.get('negative_false_positive_candidates_per_km2'))}",
            "",
            "Rank/localization:",
            f"  First-hit rank median: {fmt(summary.get('rank_of_first_hit_distribution', {}).get('summary', {}).get('median'))}",
            f"  Localization error median m: {fmt(summary.get('localization_error_m_summary', {}).get('median'))}",
            f"  Depth error median m: {fmt(summary.get('depth_error_m_summary', {}).get('median'))}",
            "",
            "Hit rate by split:",
        ]
    )
    for split, split_summary in summary.get("by_split", {}).items():
        lines.append(
            f"  {split}: positive_hit_rate={fmt(split_summary.get('positive_hit_rate'))}, "
            f"negative_fp_rate={fmt(split_summary.get('negative_false_positive_site_rate'))}, "
            f"targets={split_summary.get('target_count')}"
        )
    lines.append("")
    lines.append("Hit rate by provider:")
    for provider, provider_summary in summary.get("by_provider", {}).items():
        lines.append(
            f"  {provider}: positive_hit_rate={fmt(provider_summary.get('positive_hit_rate'))}, "
            f"negative_fp_rate={fmt(provider_summary.get('negative_false_positive_site_rate'))}, "
            f"targets={provider_summary.get('target_count')}"
        )
    lines.append("")
    lines.append("Hit rate by comparison arm:")
    for arm, arm_summary in summary.get("by_comparison_arm", {}).items():
        lines.append(
            f"  {arm}: positive_hit_rate={fmt(arm_summary.get('positive_hit_rate'))}, "
            f"negative_fp_rate={fmt(arm_summary.get('negative_false_positive_site_rate'))}, "
            f"targets={arm_summary.get('target_count')}"
        )
    lines.append("")
    lines.append("Confidence calibration bins:")
    for bucket in summary.get("confidence_calibration_bins", []):
        if bucket.get("bin_index") == "out_of_range":
            label = "out_of_range"
        elif bucket.get("upper_exclusive") is not None:
            label = f"[{bucket.get('lower_inclusive'):.2f}, {bucket.get('upper_exclusive'):.2f})"
        else:
            label = f"[{bucket.get('lower_inclusive'):.2f}, {bucket.get('upper_inclusive'):.2f}]"
        lines.append(
            f"  {label}: n={bucket.get('candidate_count')}, matched={bucket.get('matched_count')}, "
            f"observed_match_rate={fmt(bucket.get('observed_match_rate'))}, "
            f"mean_confidence={fmt(bucket.get('mean_confidence'))}"
        )
    return "\n".join(lines) + "\n"


REPORT_ARTIFACT_ROLE_ORDER = [
    "public_manifest",
    "parameter_set",
    "sar_inventory",
    "product_lock",
    "run_manifest",
    "score_json",
    "baseline_json",
    "baseline_text",
    "command_log",
    "notes_file",
    "notes",
]

REPORT_REQUIRED_ARTIFACT_ROLES = ("public_manifest", "run_manifest", "score_json")

REPORT_EVIDENCE_LABELS = {
    "public_manifest": "EVIDENCE_PUBLIC_NO_LABEL_INPUT",
    "parameter_set": "EVIDENCE_FROZEN_NO_LABEL_PARAMETERS",
    "sar_inventory": "EVIDENCE_SEARCH_ONLY_NO_DOWNLOAD_INVENTORY",
    "product_lock": "EVIDENCE_DETERMINISTIC_PRODUCT_SELECTION_LOCK",
    "run_manifest": "EVIDENCE_FROZEN_CANDIDATE_RUN_MANIFEST",
    "score_json": "EVIDENCE_WITHHELD_LABEL_SCORE_OUTPUT",
    "baseline_json": "EVIDENCE_AGGREGATED_BASELINE_METRICS_JSON",
    "baseline_text": "EVIDENCE_AGGREGATED_BASELINE_METRICS_TEXT",
    "command_log": "EVIDENCE_COMMAND_LOG_REDACTED",
    "notes_file": "EVIDENCE_OPERATOR_NOTES_REDACTED",
    "notes": "EVIDENCE_OPERATOR_NOTES_REDACTED",
}

REPORT_EVIDENCE_DESCRIPTIONS = {
    "public_manifest": "Public no-label manifest used by the runner.",
    "parameter_set": "No-label parameter set intended to freeze thresholds, resolution, tolerances, and approval state.",
    "sar_inventory": "Search-only SAR product inventory; no product downloads are represented by this artifact.",
    "product_lock": "Deterministic selected-product lock built from an inventory.",
    "run_manifest": "Frozen run manifest with candidate CSV references and reproducibility metadata.",
    "score_json": "Score output produced only after frozen candidates are compared with withheld labels.",
    "baseline_json": "Aggregate baseline metrics derived from one or more score JSON files.",
    "baseline_text": "Human-readable baseline metrics summary.",
    "command_log": "Command log copied into the package after secret redaction.",
    "notes_file": "Operator notes copied into the package after secret redaction.",
    "notes": "Inline operator notes written after secret redaction.",
}


def _write_json_any(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        dump_strict_json(data, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def _ascii_safe_text(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("ascii", errors="backslashreplace").decode("ascii")


def _write_ascii_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ascii_safe_text(text), encoding="ascii", newline="\n")


def _is_secret_like_key(key: Any) -> bool:
    upper = str(key).upper()
    compact = upper.replace("-", "_")
    return any(marker in upper for marker in SENSITIVE_KEY_MARKERS) or any(
        marker in compact
        for marker in (
            "AUTHORIZATION",
            "BEARER",
            "API_KEY",
            "ACCESS_KEY",
            "PRIVATE_KEY",
            "CLIENT_SECRET",
            "REFRESH_TOKEN",
        )
    )


def _redact_report_text(text: str) -> Tuple[str, bool]:
    before = str(text)
    redacted = _redact_env_secret_values(before)
    redacted = _AUTHORIZATION_BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = _BARE_BEARER_RE.sub(r"\1<redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", redacted)
    return redacted, redacted != before


def _redact_report_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            key_text = str(key)
            if _is_secret_like_key(key_text):
                output[key_text] = "<redacted>"
            else:
                output[key_text] = _redact_report_json_value(child)
        return output
    if isinstance(value, list):
        return [_redact_report_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_report_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        redacted, _ = _redact_report_text(value)
        return redacted
    return value


def _report_role_sort_key(role: str) -> Tuple[int, str]:
    try:
        index = REPORT_ARTIFACT_ROLE_ORDER.index(role)
    except ValueError:
        index = len(REPORT_ARTIFACT_ROLE_ORDER)
    return (index, role)


def _package_relpath(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _artifact_destination(output_dir: Path, role: str, source_path: Path, role_index: int) -> Path:
    suffix = source_path.suffix.lower()
    if suffix not in {".json", ".txt", ".csv", ".log", ".md"}:
        suffix = ".txt"
    prefix = role if role_index <= 1 else f"{role}_{role_index:03d}"
    safe_base = _safe_name(source_path.stem or role)
    return output_dir / "artifacts" / f"{prefix}__{safe_base}{suffix}"


def _copy_redacted_artifact(source_path: Path, destination_path: Path) -> Dict[str, Any]:
    if not source_path.exists() or not source_path.is_file():
        raise ManifestValidationError(f"Report package artifact does not exist or is not a file: {source_path}")
    source_sha = _sha256_file(source_path)
    source_size = int(source_path.stat().st_size)
    raw = source_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    redaction_applied = "\ufffd" in text
    content_kind = "text"
    if source_path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            redacted_text, text_redacted = _redact_report_text(text)
            _write_ascii_text(destination_path, redacted_text)
            redaction_applied = bool(redaction_applied or text_redacted)
            content_kind = "text_invalid_json"
        else:
            redacted_json = _redact_report_json_value(parsed)
            before = dumps_strict_json(parsed, sort_keys=True, ensure_ascii=True)
            after = dumps_strict_json(redacted_json, sort_keys=True, ensure_ascii=True)
            redaction_applied = before != after
            _write_json_any(destination_path, redacted_json)
            content_kind = "json"
    else:
        redacted_text, text_redacted = _redact_report_text(text)
        redaction_applied = bool(redaction_applied or text_redacted)
        _write_ascii_text(destination_path, redacted_text)
    return {
        "content_kind": content_kind,
        "redaction_applied": bool(redaction_applied),
        "source_sha256": source_sha,
        "source_size_bytes": source_size,
        "packaged_sha256": _sha256_file(destination_path),
        "packaged_size_bytes": int(destination_path.stat().st_size),
    }


def _write_inline_notes_artifact(notes: str, destination_path: Path) -> Dict[str, Any]:
    redacted_notes, redaction_applied = _redact_report_text(notes)
    _write_ascii_text(destination_path, redacted_notes)
    return {
        "content_kind": "text",
        "redaction_applied": bool(redaction_applied),
        "source_sha256": None,
        "source_size_bytes": None,
        "packaged_sha256": _sha256_file(destination_path),
        "packaged_size_bytes": int(destination_path.stat().st_size),
    }


def _source_json_summary(role: str, path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists() or path.suffix.lower() != ".json":
        return {}
    try:
        data = _load_json(path)
    except ManifestValidationError:
        return {"parse_status": "invalid_json"}
    summary = {
        "parse_status": "ok",
        "schema_version": data.get("schema_version"),
        "validation_id": data.get("validation_id"),
    }
    if role == "parameter_set":
        try:
            params = load_parameter_set(path, allow_templates=True)
        except ManifestValidationError as exc:
            summary["validation_status"] = f"invalid: {_safe_error_summary(exc)}"
        else:
            summary.update(
                {
                    "validation_status": "ok",
                    "parameter_set_id": params.get("parameter_set_id"),
                    "parameter_set_hash": params.get("parameter_set_hash"),
                    "approval_status": params.get("approval_status"),
                    "approved_for_holdout": bool(params.get("approved_for_holdout")),
                }
            )
    elif role == "public_manifest":
        try:
            manifest = load_public_manifest(path, allow_templates=True)
        except ManifestValidationError as exc:
            summary["validation_status"] = f"invalid: {_safe_error_summary(exc)}"
        else:
            summary.update(
                {
                    "validation_status": "ok",
                    "target_count": len(manifest.get("targets", [])),
                    "withheld_labels_present": False,
                }
            )
    elif role == "run_manifest":
        summary.update(
            {
                "target_count": len(data.get("targets", [])) if isinstance(data.get("targets"), list) else None,
                "dry_run": data.get("dry_run"),
                "downloads_attempted": data.get("downloads_attempted"),
                "parameter_set_id": data.get("parameter_set_id"),
                "parameter_set_hash": data.get("parameter_set_hash"),
                "product_lock_sha256": data.get("product_lock_sha256"),
            }
        )
    elif role == "score_json":
        score_summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        summary.update(
            {
                "target_count": score_summary.get("target_count"),
                "positive_site_hits": score_summary.get("positive_site_hits"),
                "positive_sites": score_summary.get("positive_sites"),
                "negative_false_positive_sites": score_summary.get("negative_site_false_positive_sites"),
                "negative_sites": score_summary.get("negative_sites"),
                "site_recall_like": score_summary.get("site_recall_like"),
                "site_precision_like": score_summary.get("site_precision_like"),
                "parameter_set_id": data.get("parameter_set_id"),
                "parameter_set_hash": data.get("parameter_set_hash"),
            }
        )
    elif role == "baseline_json":
        baseline_summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        summary.update(
            {
                "score_report_count": baseline_summary.get("score_report_count"),
                "target_count": baseline_summary.get("target_count"),
                "positive_site_hits": baseline_summary.get("positive_site_hits"),
                "positive_sites": baseline_summary.get("positive_sites"),
                "negative_false_positive_sites": baseline_summary.get("negative_false_positive_sites"),
                "negative_sites": baseline_summary.get("negative_sites"),
                "site_recall_like": baseline_summary.get("site_recall_like"),
                "site_precision_like": baseline_summary.get("site_precision_like"),
            }
        )
    elif role == "sar_inventory":
        summary.update(
            {
                "target_count": len(data.get("targets", [])) if isinstance(data.get("targets"), list) else None,
                "search_only": data.get("search_only"),
                "no_download": data.get("no_download"),
                "downloads_attempted": data.get("downloads_attempted"),
            }
        )
    elif role == "product_lock":
        summary.update(
            {
                "target_count": len(data.get("targets", [])) if isinstance(data.get("targets"), list) else None,
                "selection_policy": data.get("selection_policy"),
                "no_download": data.get("no_download"),
            }
        )
    return _redact_report_json_value(summary)


def _score_findings(score_path: Optional[Path], baseline_path: Optional[Path]) -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "score_summary_available": False,
        "baseline_summary_available": False,
    }
    if score_path is not None and score_path.exists():
        try:
            score = _load_json(score_path)
        except ManifestValidationError:
            score = {}
        summary = score.get("summary", {}) if isinstance(score.get("summary"), dict) else {}
        if summary:
            findings["score_summary_available"] = True
            findings["score_summary"] = _round_metric(
                {
                    "target_count": summary.get("target_count"),
                    "positive_sites": summary.get("positive_sites"),
                    "positive_site_hits": summary.get("positive_site_hits"),
                    "positive_site_misses": summary.get("positive_site_misses"),
                    "negative_sites": summary.get("negative_sites"),
                    "negative_false_positive_sites": summary.get("negative_site_false_positive_sites"),
                    "negative_true_negative_sites": summary.get("negative_site_true_negative_sites"),
                    "site_recall_like": summary.get("site_recall_like"),
                    "site_precision_like": summary.get("site_precision_like"),
                    "candidate_precision_like": summary.get("candidate_precision_like"),
                    "known_void_label_recall_like": summary.get("known_void_label_recall_like"),
                    "total_candidates": summary.get("total_candidates"),
                    "matched_known_void_labels": summary.get("matched_known_void_labels"),
                    "total_known_void_labels": summary.get("total_known_void_labels"),
                }
            )
    if baseline_path is not None and baseline_path.exists():
        try:
            baseline = _load_json(baseline_path)
        except ManifestValidationError:
            baseline = {}
        summary = baseline.get("summary", {}) if isinstance(baseline.get("summary"), dict) else {}
        if summary:
            findings["baseline_summary_available"] = True
            findings["baseline_summary"] = _round_metric(
                {
                    "score_report_count": summary.get("score_report_count"),
                    "target_count": summary.get("target_count"),
                    "positive_sites": summary.get("positive_sites"),
                    "positive_site_hits": summary.get("positive_site_hits"),
                    "negative_sites": summary.get("negative_sites"),
                    "negative_false_positive_sites": summary.get("negative_false_positive_sites"),
                    "site_recall_like": summary.get("site_recall_like"),
                    "site_precision_like": summary.get("site_precision_like"),
                    "candidate_precision_like": summary.get("candidate_precision_like"),
                    "known_void_label_recall_like": summary.get("known_void_label_recall_like"),
                }
            )
    findings["claim_boundary"] = REPORT_CANDIDATE_CLAIM
    return findings


def _package_hash_manifest(output_dir: Path) -> Dict[str, Any]:
    records = []
    for path in sorted(output_dir.rglob("*"), key=lambda item: item.relative_to(output_dir).as_posix()):
        if not path.is_file() or path.name == "file_hash_manifest.json":
            continue
        stat = path.stat()
        records.append(
            {
                "path": _package_relpath(path, output_dir),
                "sha256": _sha256_file(path),
                "size_bytes": int(stat.st_size),
            }
        )
    return {
        "schema_version": f"{REPORT_PACKAGE_SCHEMA_VERSION}-hash-manifest",
        "hash_algorithm": "sha256",
        "reproducibility_note": "Wall-clock time and file mtimes are intentionally omitted.",
        "files": records,
    }


def _format_report_package_text(summary: Dict[str, Any]) -> str:
    findings = summary.get("findings", {})

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = [
        "# Blind Known-Void Validation Report Package",
        "",
        "## Claim boundary",
        REPORT_CANDIDATE_CLAIM,
        "",
        "This package supports audit and review of a validation workflow. It is not field verification,",
        "does not establish operational accuracy by itself, and must not be used to claim confirmed",
        "subsurface structures without independent evidence.",
        "",
        "## Evidence labels",
    ]
    for artifact in summary.get("artifacts", []):
        lines.append(
            f"- {artifact.get('role')}: {artifact.get('evidence_label')} - "
            f"{artifact.get('description')} Packaged as {artifact.get('packaged_path')} "
            f"with sha256 {artifact.get('packaged_sha256')}."
        )
    lines.extend(
        [
            "",
            "## Methods summary",
            "1. The public manifest is the only runner input and is checked for withheld-label leakage.",
            "2. Parameter-set artifacts, when present, document the no-label thresholds, resolution, PINN settings, and approval state.",
            "3. SAR inventory artifacts, when present, are search-only and record product metadata without download evidence.",
            "4. Product locks, when present, freeze deterministic product selections before any real execution.",
            "5. Run manifests freeze candidate CSV outputs before withheld-label scoring.",
            "6. Score JSON is produced only after frozen candidates are compared with withheld labels by the scoring stage.",
            "7. Baseline reports aggregate score outputs into precision/recall-like and false-positive summaries.",
            "8. All copied text/JSON artifacts in this package are redacted for secret-like keys, tokens, passwords, and bearer strings.",
            "",
            "## Results summary",
        ]
    )
    score_summary = findings.get("score_summary", {}) if isinstance(findings.get("score_summary"), dict) else {}
    baseline_summary = findings.get("baseline_summary", {}) if isinstance(findings.get("baseline_summary"), dict) else {}
    if score_summary:
        lines.extend(
            [
                "Score JSON summary:",
                f"- Targets: {fmt(score_summary.get('target_count'))}",
                f"- Positive-site hits: {fmt(score_summary.get('positive_site_hits'))}/{fmt(score_summary.get('positive_sites'))}",
                f"- Negative false-positive sites: {fmt(score_summary.get('negative_false_positive_sites'))}/{fmt(score_summary.get('negative_sites'))}",
                f"- Site recall-like: {fmt(score_summary.get('site_recall_like'))}",
                f"- Site precision-like: {fmt(score_summary.get('site_precision_like'))}",
            ]
        )
    else:
        lines.append("Score JSON summary: n/a")
    if baseline_summary:
        lines.extend(
            [
                "Baseline JSON summary:",
                f"- Score reports: {fmt(baseline_summary.get('score_report_count'))}",
                f"- Targets: {fmt(baseline_summary.get('target_count'))}",
                f"- Site recall-like: {fmt(baseline_summary.get('site_recall_like'))}",
                f"- Site precision-like: {fmt(baseline_summary.get('site_precision_like'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
        ]
    )
    for limitation in summary.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "- JSON files are written with sorted keys and ASCII-safe output.",
            "- The package omits wall-clock generation time and file mtimes.",
            "- file_hash_manifest.json records sha256 hashes for packaged files except itself.",
            "- Source artifact sha256 values are recorded before redaction; packaged sha256 values are recorded after redaction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _prepare_report_output_dir(output_dir: Path, *, overwrite: bool = False) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            resolved = output_dir.resolve()
            if resolved == Path.cwd().resolve() or resolved == Path(resolved.anchor):
                raise ManifestValidationError("Refusing to overwrite the workspace root or filesystem root")
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise ManifestValidationError(
                f"Report package output directory is not empty: {output_dir}. Use --overwrite to replace it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def package_validation_report(
    output_dir: Path | str,
    *,
    public_manifest: Optional[Path | str] = None,
    parameter_set: Optional[Path | str] = None,
    sar_inventory: Optional[Path | str] = None,
    product_lock: Optional[Path | str] = None,
    run_manifest: Optional[Path | str] = None,
    score_json: Optional[Path | str] = None,
    baseline_json: Optional[Path | str] = None,
    baseline_text: Optional[Path | str] = None,
    command_logs: Optional[Sequence[Path | str]] = None,
    notes: Optional[str] = None,
    notes_file: Optional[Path | str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write a deterministic, redacted validation report package directory."""
    output_dir = Path(output_dir)
    _prepare_report_output_dir(output_dir, overwrite=overwrite)

    role_paths: List[Tuple[str, Optional[Path]]] = [
        ("public_manifest", Path(public_manifest) if public_manifest is not None else None),
        ("parameter_set", Path(parameter_set) if parameter_set is not None else None),
        ("sar_inventory", Path(sar_inventory) if sar_inventory is not None else None),
        ("product_lock", Path(product_lock) if product_lock is not None else None),
        ("run_manifest", Path(run_manifest) if run_manifest is not None else None),
        ("score_json", Path(score_json) if score_json is not None else None),
        ("baseline_json", Path(baseline_json) if baseline_json is not None else None),
        ("baseline_text", Path(baseline_text) if baseline_text is not None else None),
    ]
    for log_path in command_logs or []:
        role_paths.append(("command_log", Path(log_path)))
    if notes_file is not None:
        role_paths.append(("notes_file", Path(notes_file)))

    provided_roles = {role for role, path in role_paths if path is not None}
    missing_required = [role for role in REPORT_REQUIRED_ARTIFACT_ROLES if role not in provided_roles]
    if missing_required:
        raise ManifestValidationError(f"Report package missing required artifact role(s): {missing_required}")

    ordered_role_paths = sorted(
        [(role, path) for role, path in role_paths if path is not None],
        key=lambda item: (_report_role_sort_key(item[0]), str(item[1])),
    )
    role_counts: Dict[str, int] = {}
    artifacts = []
    source_summaries: Dict[str, Any] = {}
    score_path: Optional[Path] = None
    baseline_json_path: Optional[Path] = None
    for role, source_path in ordered_role_paths:
        assert source_path is not None
        role_counts[role] = role_counts.get(role, 0) + 1
        destination = _artifact_destination(output_dir, role, source_path, role_counts[role])
        copy_record = _copy_redacted_artifact(source_path, destination)
        artifact_record = {
            "role": role,
            "evidence_label": REPORT_EVIDENCE_LABELS.get(role, "EVIDENCE_UNSPECIFIED"),
            "description": REPORT_EVIDENCE_DESCRIPTIONS.get(role, "User-supplied report package artifact."),
            "source_path": str(source_path),
            "packaged_path": _package_relpath(destination, output_dir),
            **copy_record,
        }
        artifacts.append(artifact_record)
        if role not in source_summaries:
            source_summaries[role] = _source_json_summary(role, source_path)
        else:
            source_summaries.setdefault(f"{role}_{role_counts[role]:03d}", _source_json_summary(role, source_path))
        if role == "score_json" and score_path is None:
            score_path = source_path
        if role == "baseline_json" and baseline_json_path is None:
            baseline_json_path = source_path

    if notes is not None:
        role = "notes"
        role_counts[role] = role_counts.get(role, 0) + 1
        destination = output_dir / "artifacts" / "notes__inline.txt"
        notes_record = _write_inline_notes_artifact(notes, destination)
        artifacts.append(
            {
                "role": role,
                "evidence_label": REPORT_EVIDENCE_LABELS[role],
                "description": REPORT_EVIDENCE_DESCRIPTIONS[role],
                "source_path": None,
                "packaged_path": _package_relpath(destination, output_dir),
                **notes_record,
            }
        )

    validation_ids = sorted(
        {
            str(summary.get("validation_id"))
            for summary in source_summaries.values()
            if isinstance(summary, dict) and summary.get("validation_id") is not None
        }
    )
    limitations = [
        REPORT_CANDIDATE_CLAIM,
        "Packaged artifacts are redacted copies; source hashes identify original inputs, while packaged hashes identify redacted package files.",
        "This package does not include private withheld-label source files unless a user explicitly supplies one as a generic log or note, which is not recommended.",
        "Fixture or dry-run outputs verify workflow separation and determinism only; they do not establish real-world field accuracy.",
        "SAR provider, product-lock, and real-execution limitations remain governed by the source run and inventory manifests.",
    ]
    summary = {
        "schema_version": REPORT_PACKAGE_SCHEMA_VERSION,
        "validation_ids": validation_ids,
        "validation_id_consistent": len(validation_ids) <= 1,
        "required_artifact_roles": list(REPORT_REQUIRED_ARTIFACT_ROLES),
        "provided_artifact_roles": sorted(provided_roles | ({"notes"} if notes is not None else set()), key=_report_role_sort_key),
        "claim_boundary": REPORT_CANDIDATE_CLAIM,
        "evidence_label_definitions": {
            role: {
                "label": REPORT_EVIDENCE_LABELS[role],
                "description": REPORT_EVIDENCE_DESCRIPTIONS[role],
            }
            for role in REPORT_ARTIFACT_ROLE_ORDER
        },
        "artifacts": artifacts,
        "source_summaries": source_summaries,
        "findings": _score_findings(score_path, baseline_json_path),
        "limitations": limitations,
        "reproducibility": {
            "json_sort_keys": True,
            "ascii_safe_outputs": True,
            "wall_clock_time_recorded": False,
            "file_mtimes_recorded": False,
            "hash_algorithm": "sha256",
            "redaction_policy": "secret_like_keys_tokens_passwords_bearer_strings_and_current_environment_secret_values",
        },
    }
    summary_path = output_dir / "validation_summary.json"
    _write_json_any(summary_path, summary)
    report_path = output_dir / "methods_limitations_claim_boundary.md"
    _write_ascii_text(report_path, _format_report_package_text(summary))
    hash_manifest = _package_hash_manifest(output_dir)
    hash_manifest_path = output_dir / "file_hash_manifest.json"
    _write_json_any(hash_manifest_path, hash_manifest)
    package_record = {
        **summary,
        "package_files": {
            "validation_summary": _package_relpath(summary_path, output_dir),
            "methods_limitations_claim_boundary": _package_relpath(report_path, output_dir),
            "file_hash_manifest": _package_relpath(hash_manifest_path, output_dir),
        },
    }
    return package_record


def _cmd_validate_public(args: argparse.Namespace) -> int:
    load_public_manifest(args.manifest, allow_templates=args.allow_templates)
    print(f"OK public manifest: {args.manifest}")
    return 0


def _cmd_validate_labels(args: argparse.Namespace) -> int:
    load_withheld_labels(args.labels, allow_templates=args.allow_templates)
    print(f"OK withheld labels: {args.labels}")
    return 0


def _cmd_init_parameters(args: argparse.Namespace) -> int:
    template = _default_parameter_set_template(args.validation_id, args.name)
    if args.approved:
        template["template_only"] = False
        template["approval"] = {
            "status": "approved_for_holdout",
            "approved_by": args.approved_by or "validation_custodian",
            "approved_at_utc": args.approved_at_utc or _utc_now_iso(),
            "approval_scope": "holdout_scoring_allowed_only_for_this_canonical_parameter_hash",
        }
        template["approval_status"] = "approved_for_holdout"
        template["approved_for_holdout"] = True
    _write_json(args.output, template)
    print(f"Wrote parameter set template: {args.output}")
    print(f"Parameter set id: {template['parameter_set_id']}")
    print(f"Parameter set hash: {template['parameter_set_hash']}")
    print(f"Approved for holdout: {bool(template.get('approved_for_holdout', False))}")
    return 0


def _cmd_validate_parameters(args: argparse.Namespace) -> int:
    data = load_parameter_set(
        args.parameter_set,
        allow_templates=args.allow_templates,
        require_approved=args.require_approved,
    )
    print(f"OK parameter set: {args.parameter_set}")
    print(f"Parameter set id: {data['parameter_set_id']}")
    print(f"Parameter set hash: {data['parameter_set_hash']}")
    print(f"Approval status: {data['approval_status']}")
    print(f"Approved for holdout: {data['approved_for_holdout']}")
    return 0


def _cmd_compare_parameters(args: argparse.Namespace) -> int:
    comparison = compare_parameter_sets(args.reference, args.candidate)
    if args.output is not None:
        _write_json(args.output, comparison)
        print(f"Wrote parameter comparison: {args.output}")
    else:
        print(dumps_strict_json(comparison, indent=2, sort_keys=True))
    print(f"Matching hash: {comparison['matching_hash']}")
    print(f"Changed: {comparison['changed']}")
    return 2 if comparison["changed"] and args.fail_on_changed else 0


def _cmd_init_campaign_registry(args: argparse.Namespace) -> int:
    registry = _default_campaign_registry_template(
        args.validation_id,
        args.campaign_id,
        args.manifest,
        args.parameter_set,
        campaign_name=args.name,
        status=args.status,
        allow_templates=args.allow_templates,
        approved_by=args.approved_by,
        approved_at_utc=args.approved_at_utc,
        locked_by=args.locked_by,
        locked_at_utc=args.locked_at_utc,
    )
    _write_json(args.output, registry)
    print(f"Wrote campaign registry: {args.output}")
    print(f"Campaign registry id: {registry['registry_id']}")
    print(f"Campaign registry hash: {registry['registry_hash']}")
    print(f"Status: {registry['status']}")
    print(f"Public manifest sha256: {registry['public_manifest']['file_sha256']}")
    print(f"Parameter set hash: {registry['parameter_set']['parameter_set_hash']}")
    return 0


def _cmd_validate_campaign_registry(args: argparse.Namespace) -> int:
    registry = load_campaign_registry(
        args.registry,
        allow_templates=args.allow_templates,
        require_approved=args.require_approved,
        require_locked=args.require_locked,
    )
    print(f"OK campaign registry: {args.registry}")
    print(f"Campaign registry id: {registry['registry_id']}")
    print(f"Campaign registry hash: {registry['registry_hash']}")
    print(f"Status: {registry['status']}")
    print(f"Public manifest sha256: {registry['public_manifest']['file_sha256']}")
    print(f"Parameter set hash: {registry['parameter_set']['parameter_set_hash']}")
    return 0


def _cmd_compare_campaign_registry(args: argparse.Namespace) -> int:
    comparison = compare_campaign_registry(
        args.registry,
        public_manifest_path=args.manifest,
        parameter_set_path=args.parameter_set,
        allow_templates=args.allow_templates,
    )
    if args.output is not None:
        _write_json(args.output, comparison)
        print(f"Wrote campaign registry comparison: {args.output}")
    else:
        print(dumps_strict_json(comparison, indent=2, sort_keys=True))
    print(f"Matching registry: {comparison['matching_registry']}")
    print(f"Drift detected: {comparison['drift_detected']}")
    return 2 if comparison["drift_detected"] and args.fail_on_drift else 0


def _cmd_set_campaign_registry_status(args: argparse.Namespace) -> int:
    registry = load_campaign_registry(args.registry, allow_templates=args.allow_templates)
    status_value = args.status
    if status_value is None:
        status_value = "locked" if args.command == "lock-campaign-registry" else "approved"
    status = _campaign_status(status_value)
    parameter_record = registry.get("parameter_set", {}) if isinstance(registry.get("parameter_set"), dict) else {}
    if status in {"approved", "locked"} and not parameter_record.get("approved_for_holdout"):
        raise ManifestValidationError("Approved or locked campaign registry requires an approved parameter set")
    registry["status"] = status
    registry["template_only"] = bool(args.template_only) if args.template_only else False
    approval = dict(registry.get("approval", {})) if isinstance(registry.get("approval"), dict) else {}
    if status in {"approved", "locked"}:
        approval.update(
            {
                "status": status,
                "approved_by": args.approved_by or approval.get("approved_by") or "validation_custodian",
                "approved_at_utc": args.approved_at_utc or approval.get("approved_at_utc") or _utc_now_iso(),
                "approval_scope": approval.get("approval_scope") or "campaign_registry_identity_and_parameter_set_lock",
            }
        )
    else:
        approval.update(
            {
                "status": "draft",
                "approved_by": None,
                "approved_at_utc": None,
                "approval_scope": "not_approved_until_status_is_approved_or_locked",
            }
        )
    lock = dict(registry.get("lock", {})) if isinstance(registry.get("lock"), dict) else {}
    if status == "locked":
        lock.update(
            {
                "locked": True,
                "locked_by": args.locked_by or lock.get("locked_by") or args.approved_by or "validation_custodian",
                "locked_at_utc": args.locked_at_utc or lock.get("locked_at_utc") or _utc_now_iso(),
                "lock_scope": lock.get("lock_scope") or "public_manifest_parameter_set_scoring_metrics_provider_arms_and_immutable_artifact_hashes",
            }
        )
    else:
        lock.update(
            {
                "locked": False,
                "locked_by": None,
                "locked_at_utc": None,
                "lock_scope": lock.get("lock_scope") or "public_manifest_parameter_set_scoring_metrics_provider_arms_and_immutable_artifact_hashes",
            }
        )
    registry["approval"] = approval
    registry["lock"] = lock
    registry["registry_hash"] = campaign_registry_hash(registry)
    registry["registry_id"] = campaign_registry_id_from_hash(registry["registry_hash"])
    output = args.output or args.registry
    _write_json(output, registry)
    print(f"Wrote campaign registry: {output}")
    print(f"Campaign registry id: {registry['registry_id']}")
    print(f"Campaign registry hash: {registry['registry_hash']}")
    print(f"Status: {registry['status']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    dry_run = not args.execute_real
    manifest = run_blind_validation(
        args.manifest,
        args.output_dir,
        dry_run=dry_run,
        allow_real_downloads=args.execute_real,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        product_lock_path=args.product_lock,
        require_product_lock=args.require_product_lock,
        confirm_real_downloads_and_training=args.confirm_real_downloads_and_training,
        parameter_set_path=args.parameter_set,
        require_approved_parameters=args.require_approved_parameters,
        campaign_registry_path=args.campaign_registry,
        require_locked_campaign_registry=args.require_locked_campaign_registry,
        robustness_plan_path=args.robustness_plan,
        robustness_variant_id=args.variant_id,
        target_ids=args.target_id,
        include_audit_only=args.include_audit_only,
        command_args=_namespace_to_safe_dict(args),
    )
    print(f"Wrote run manifest: {Path(args.output_dir) / 'run_manifest.json'}")
    print(f"Targets frozen: {len(manifest['targets'])}")
    print(f"Dry run: {manifest['dry_run']}")
    return 0


def _cmd_campaign_plan(args: argparse.Namespace) -> int:
    plan = plan_campaign_execution(
        args.manifest,
        args.campaign_registry,
        args.parameter_set,
        args.output_dir,
        product_lock_path=args.product_lock,
        execute_real=args.execute_real,
        confirm_real_downloads_and_training=args.confirm_real_downloads_and_training,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        include_audit_only=args.include_audit_only,
        allow_unsupported_provider_arms=args.allow_unsupported_provider_arms,
        allow_templates=args.allow_templates,
    )
    if args.output is not None:
        _write_json(args.output, plan)
        print(f"Wrote campaign execution plan: {args.output}")
    else:
        print(dumps_strict_json(plan, indent=2, sort_keys=True))
    return 0


def _cmd_campaign_status(args: argparse.Namespace) -> int:
    status = campaign_execution_status(args.plan)
    if args.output is not None:
        _write_json(args.output, status)
        print(f"Wrote campaign execution status: {args.output}")
    else:
        print(dumps_strict_json(status, indent=2, sort_keys=True))
    return 0


def _cmd_campaign_run(args: argparse.Namespace) -> int:
    status = run_campaign_execution_plan(
        args.plan,
        resume=not args.no_resume,
        execute_real=args.execute_real,
        confirm_real_downloads_and_training=args.confirm_real_downloads_and_training,
    )
    if args.output is not None:
        _write_json(args.output, status)
        print(f"Wrote campaign execution status: {args.output}")
    else:
        print(dumps_strict_json(status, indent=2, sort_keys=True))
    return 0


def _cmd_campaign_package(args: argparse.Namespace) -> int:
    evidence = package_campaign_no_label_evidence(
        args.plan,
        args.output,
        status_path=args.status,
        inventory_path=args.sar_inventory,
        product_lock_path=args.product_lock,
    )
    if args.output is None:
        print(dumps_strict_json(evidence, indent=2, sort_keys=True))
    else:
        print(f"Wrote campaign no-label evidence summary: {args.output}")
    return 0


def _cmd_validate_robustness_plan(args: argparse.Namespace) -> int:
    summary = robustness_plan_validation_summary(args.plan, allow_templates=args.allow_templates)
    if args.output is not None:
        _write_json(args.output, summary)
        print(f"Wrote robustness plan validation: {args.output}")
    else:
        print(dumps_strict_json(summary, indent=2, sort_keys=True))
    return 0


def _cmd_robustness_plan(args: argparse.Namespace) -> int:
    plan = plan_robustness_ablations(
        args.plan,
        args.manifest,
        args.campaign_registry,
        args.parameter_set,
        args.output_dir,
        allow_templates=args.allow_templates,
        include_audit_only=args.include_audit_only,
        include_preregistered_holdout=args.include_preregistered_holdout,
    )
    if args.output is not None:
        _write_json(args.output, plan)
        print(f"Wrote robustness ablation execution plan: {args.output}")
    else:
        print(dumps_strict_json(plan, indent=2, sort_keys=True))
    return 0


def _cmd_robustness_summary(args: argparse.Namespace) -> int:
    scores = list(args.scores or []) + list(args.score or [])
    if not scores:
        raise ManifestValidationError("At least one score report path is required")
    summary = build_robustness_summary(scores)
    if args.output is not None:
        _write_json(args.output, summary)
        print(f"Wrote robustness summary: {args.output}")
    else:
        print(dumps_strict_json(summary, indent=2, sort_keys=True))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    report = score_blind_validation(
        args.run_manifest,
        args.labels,
        args.output,
        parameter_set_path=args.parameter_set,
        require_approved_parameters=args.require_approved_parameters,
    )
    print(f"Wrote score report: {args.output}")
    summary = report["summary"]
    print(
        "Summary: "
        f"positive_hits={summary['positive_site_hits']}/{summary['positive_sites']}, "
        f"negative_false_positive_sites={summary['negative_site_false_positive_sites']}/{summary['negative_sites']}, "
        f"site_recall_like={summary['site_recall_like']}, "
        f"site_precision_like={summary['site_precision_like']}"
    )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    output = args.output or (Path(args.output_dir) / "sar_inventory.json" if args.output_dir else None)
    lock_output = args.lock_output
    if lock_output is None and args.output_dir and args.write_lock:
        lock_output = Path(args.output_dir) / "product_lock.json"
    inventory = build_sar_inventory(
        args.manifest,
        output,
        lock_output_path=lock_output,
        parameter_set_path=args.parameter_set,
        allow_templates=args.allow_templates,
        load_dotenv=not args.no_dotenv,
        env_path=args.env_path,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
        selection_count=args.selection_count,
        search_timeout_seconds=args.timeout,
        search_max_retries=args.retries,
        search_retry_backoff_seconds=args.retry_backoff,
        campaign_registry_path=args.campaign_registry,
    )
    if output is None:
        print(dumps_strict_json(inventory, indent=2, sort_keys=True))
    else:
        print(f"Wrote SAR inventory: {output}")
    if lock_output is not None:
        print(f"Wrote product lock: {lock_output}")
    print("No downloads attempted: True")
    failures = [target for target in inventory["targets"] if target.get("status") != "success"]
    return 2 if failures and args.fail_on_search_error else 0


def _cmd_lock_products(args: argparse.Namespace) -> int:
    inventory = _load_json(args.inventory)
    lock = build_product_lock(
        inventory,
        source_inventory_path=args.inventory,
        previous_lock_path=args.previous_lock,
    )
    _write_json(args.output, lock)
    print(f"Wrote product lock: {args.output}")
    print(f"Selection changed from previous lock: {lock['selection_changed_from_previous_lock']}")
    return 2 if lock["selection_changed_from_previous_lock"] and args.fail_on_changed_selection else 0


def _cmd_baseline_report(args: argparse.Namespace) -> int:
    scores = list(args.scores or []) + list(args.score or [])
    if not scores:
        raise ManifestValidationError("At least one score report path is required")
    report = build_baseline_report(scores, confidence_bins=args.confidence_bins)
    _write_json(args.output_json, report)
    text = format_baseline_report_text(report)
    text_path = args.output_text
    if text_path is None:
        text_path = args.output_json.with_suffix(".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    print(f"Wrote baseline report JSON: {args.output_json}")
    print(f"Wrote baseline report text: {text_path}")
    print(
        "Summary: "
        f"targets={report['summary']['target_count']}, "
        f"site_recall_like={report['summary']['site_recall_like']}, "
        f"site_precision_like={report['summary']['site_precision_like']}"
    )
    return 0


def _cmd_package_report(args: argparse.Namespace) -> int:
    package = package_validation_report(
        args.output_dir,
        public_manifest=args.public_manifest,
        parameter_set=args.parameter_set,
        sar_inventory=args.sar_inventory,
        product_lock=args.product_lock,
        run_manifest=args.run_manifest,
        score_json=args.score_json,
        baseline_json=args.baseline_json,
        baseline_text=args.baseline_text,
        command_logs=args.command_log,
        notes=args.notes,
        notes_file=args.notes_file,
        overwrite=args.overwrite,
    )
    files = package["package_files"]
    print(f"Wrote validation report package: {args.output_dir}")
    print(f"Wrote validation summary: {Path(args.output_dir) / files['validation_summary']}")
    print(f"Wrote methods/limitations report: {Path(args.output_dir) / files['methods_limitations_claim_boundary']}")
    print(f"Wrote file hash manifest: {Path(args.output_dir) / files['file_hash_manifest']}")
    print(f"Claim boundary: {package['claim_boundary']}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind known-void validation runner and scorer"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate_public = sub.add_parser("validate-public", help="Validate a public no-label manifest")
    validate_public.add_argument("--manifest", required=True, type=Path)
    validate_public.add_argument("--allow-templates", action="store_true")
    validate_public.set_defaults(func=_cmd_validate_public)

    validate_labels = sub.add_parser("validate-labels", help="Validate withheld labels for scorer use")
    validate_labels.add_argument("--labels", required=True, type=Path)
    validate_labels.add_argument("--allow-templates", action="store_true")
    validate_labels.set_defaults(func=_cmd_validate_labels)

    init_parameters = sub.add_parser(
        "init-parameters",
        help="Write a deterministic no-label validation parameter-set template",
    )
    init_parameters.add_argument("--validation-id", required=True)
    init_parameters.add_argument("--name", default="draft_validation_parameters")
    init_parameters.add_argument("--output", required=True, type=Path)
    init_parameters.add_argument(
        "--approved",
        action="store_true",
        help="Mark the generated set approved_for_holdout. Use only after calibration review.",
    )
    init_parameters.add_argument("--approved-by", default=None)
    init_parameters.add_argument("--approved-at-utc", default=None)
    init_parameters.set_defaults(func=_cmd_init_parameters)

    validate_parameters = sub.add_parser(
        "validate-parameters",
        aliases=["validate-parameter-set"],
        help="Validate a no-label parameter set and print its canonical hash/id",
    )
    validate_parameters.add_argument("--parameter-set", required=True, type=Path)
    validate_parameters.add_argument("--allow-templates", action="store_true")
    validate_parameters.add_argument("--require-approved", action="store_true")
    validate_parameters.set_defaults(func=_cmd_validate_parameters)

    compare_parameters = sub.add_parser(
        "compare-parameters",
        aliases=["compare-parameter-sets"],
        help="Compare two parameter sets by canonical hash and content diff",
    )
    compare_parameters.add_argument("--reference", required=True, type=Path)
    compare_parameters.add_argument("--candidate", required=True, type=Path)
    compare_parameters.add_argument("--output", type=Path, default=None)
    compare_parameters.add_argument("--fail-on-changed", action="store_true")
    compare_parameters.set_defaults(func=_cmd_compare_parameters)

    init_registry = sub.add_parser(
        "init-campaign-registry",
        help="Write a deterministic no-label campaign registry tied to a public manifest and parameter set",
    )
    init_registry.add_argument("--validation-id", required=True)
    init_registry.add_argument("--campaign-id", required=True)
    init_registry.add_argument("--name", default="blind_multi_site_campaign")
    init_registry.add_argument("--manifest", required=True, type=Path)
    init_registry.add_argument("--parameter-set", required=True, type=Path)
    init_registry.add_argument("--output", required=True, type=Path)
    init_registry.add_argument("--status", choices=sorted(CAMPAIGN_REGISTRY_STATUSES), default="draft")
    init_registry.add_argument("--allow-templates", action="store_true")
    init_registry.add_argument("--approved-by", default=None)
    init_registry.add_argument("--approved-at-utc", default=None)
    init_registry.add_argument("--locked-by", default=None)
    init_registry.add_argument("--locked-at-utc", default=None)
    init_registry.set_defaults(func=_cmd_init_campaign_registry)

    validate_registry = sub.add_parser(
        "validate-campaign-registry",
        aliases=["validate-registry"],
        help="Validate a no-label campaign registry and print its canonical hash/id",
    )
    validate_registry.add_argument("--registry", required=True, type=Path)
    validate_registry.add_argument("--allow-templates", action="store_true")
    validate_registry.add_argument("--require-approved", action="store_true")
    validate_registry.add_argument("--require-locked", action="store_true")
    validate_registry.set_defaults(func=_cmd_validate_campaign_registry)

    compare_registry = sub.add_parser(
        "compare-campaign-registry",
        aliases=["compare-registry"],
        help="Compare a campaign registry against the current public manifest and parameter set",
    )
    compare_registry.add_argument("--registry", required=True, type=Path)
    compare_registry.add_argument("--manifest", type=Path, default=None)
    compare_registry.add_argument("--parameter-set", type=Path, default=None)
    compare_registry.add_argument("--output", type=Path, default=None)
    compare_registry.add_argument("--allow-templates", action="store_true")
    compare_registry.add_argument("--fail-on-drift", action="store_true")
    compare_registry.set_defaults(func=_cmd_compare_campaign_registry)

    registry_status = sub.add_parser(
        "set-campaign-registry-status",
        aliases=["approve-campaign-registry", "lock-campaign-registry"],
        help="Rewrite a campaign registry with draft, approved, or locked status metadata",
    )
    registry_status.add_argument("--registry", required=True, type=Path)
    registry_status.add_argument("--output", type=Path, default=None)
    registry_status.add_argument("--status", choices=sorted(CAMPAIGN_REGISTRY_STATUSES), default=None)
    registry_status.add_argument("--allow-templates", action="store_true")
    registry_status.add_argument("--template-only", action="store_true")
    registry_status.add_argument("--approved-by", default=None)
    registry_status.add_argument("--approved-at-utc", default=None)
    registry_status.add_argument("--locked-by", default=None)
    registry_status.add_argument("--locked-at-utc", default=None)
    registry_status.set_defaults(func=_cmd_set_campaign_registry_status)

    run = sub.add_parser("run", help="Freeze blind candidate outputs from a public manifest")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument(
        "--execute-real",
        action="store_true",
        help="Opt in to real pipeline execution/downloads. Default is dry-run freeze only.",
    )
    run.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Opt in to synthetic fallback during real execution. Disabled by default.",
    )
    run.add_argument(
        "--product-lock",
        type=Path,
        default=None,
        help="Product lock generated by preflight/lock-products. When used with --execute-real, the runner verifies and passes locked product metadata into acquisition.",
    )
    run.add_argument(
        "--require-product-lock",
        action="store_true",
        help="Fail unless --product-lock is supplied and can be enforced before real downloads.",
    )
    run.add_argument(
        "--confirm-real-downloads-and-training",
        action="store_true",
        help="Required with --execute-real. Confirms the operator reviewed product locks, estimated download sizes, disk space, and non-synthetic settings.",
    )
    run.add_argument(
        "--parameter-set",
        type=Path,
        default=None,
        help="No-label validation parameter set to record in the run manifest.",
    )
    run.add_argument(
        "--require-approved-parameters",
        action="store_true",
        help="Fail unless --parameter-set is approved for holdout scoring.",
    )
    run.add_argument(
        "--campaign-registry",
        type=Path,
        default=None,
        help="No-label campaign registry to verify against the current public manifest and parameter set.",
    )
    run.add_argument(
        "--require-locked-campaign-registry",
        action="store_true",
        help="Fail unless --campaign-registry is supplied and has locked status.",
    )
    run.add_argument(
        "--robustness-plan",
        type=Path,
        default=None,
        help="No-label robustness/ablation plan to record in the run manifest for fixture or calibration-only variant runs.",
    )
    run.add_argument(
        "--variant-id",
        default=None,
        help="Variant ID from --robustness-plan to record in the run manifest. Does not alter processing by itself.",
    )
    run.add_argument(
        "--target-id",
        action="append",
        default=None,
        help="Run only this public target_id. May be repeated; used by campaign-level drivers.",
    )
    run.add_argument(
        "--include-audit-only",
        action="store_true",
        help="Allow explicitly requested audit-only targets to run. Campaign primary execution excludes them by default.",
    )
    run.set_defaults(func=_cmd_run)

    campaign_plan = sub.add_parser(
        "campaign-plan",
        aliases=["plan-campaign"],
        help="Create a strict JSON no-label campaign execution plan with per-target commands",
    )
    campaign_plan.add_argument("--manifest", required=True, type=Path)
    campaign_plan.add_argument("--campaign-registry", required=True, type=Path)
    campaign_plan.add_argument("--parameter-set", required=True, type=Path)
    campaign_plan.add_argument("--product-lock", type=Path, default=None)
    campaign_plan.add_argument("--output-dir", required=True, type=Path)
    campaign_plan.add_argument("--output", type=Path, default=None)
    campaign_plan.add_argument("--execute-real", action="store_true")
    campaign_plan.add_argument("--confirm-real-downloads-and-training", action="store_true")
    campaign_plan.add_argument("--allow-synthetic-fallback", action="store_true")
    campaign_plan.add_argument("--include-audit-only", action="store_true")
    campaign_plan.add_argument("--allow-unsupported-provider-arms", action="store_true")
    campaign_plan.add_argument("--allow-templates", action="store_true")
    campaign_plan.set_defaults(func=_cmd_campaign_plan)

    campaign_status = sub.add_parser(
        "campaign-status",
        aliases=["status-campaign"],
        help="Read a campaign execution plan and summarize existing per-target outputs as strict JSON",
    )
    campaign_status.add_argument("--plan", required=True, type=Path)
    campaign_status.add_argument("--output", type=Path, default=None)
    campaign_status.set_defaults(func=_cmd_campaign_status)

    campaign_run = sub.add_parser(
        "campaign-run",
        aliases=["run-campaign"],
        help="Execute or dry-run campaign plan steps, resuming existing per-target run manifests by default",
    )
    campaign_run.add_argument("--plan", required=True, type=Path)
    campaign_run.add_argument("--output", type=Path, default=None)
    campaign_run.add_argument("--execute-real", action="store_true")
    campaign_run.add_argument("--confirm-real-downloads-and-training", action="store_true")
    campaign_run.add_argument("--no-resume", action="store_true")
    campaign_run.set_defaults(func=_cmd_campaign_run)

    campaign_package = sub.add_parser(
        "campaign-package",
        aliases=["package-campaign-evidence"],
        help="Write a deterministic no-label campaign evidence/status summary",
    )
    campaign_package.add_argument("--plan", required=True, type=Path)
    campaign_package.add_argument("--output", type=Path, default=None)
    campaign_package.add_argument("--status", type=Path, default=None)
    campaign_package.add_argument("--sar-inventory", type=Path, default=None)
    campaign_package.add_argument("--product-lock", type=Path, default=None)
    campaign_package.set_defaults(func=_cmd_campaign_package)

    validate_robustness = sub.add_parser(
        "validate-robustness-plan",
        aliases=["validate-ablation-plan"],
        help="Validate a public no-label robustness/ablation plan and emit strict JSON",
    )
    validate_robustness.add_argument("--plan", required=True, type=Path)
    validate_robustness.add_argument("--output", type=Path, default=None)
    validate_robustness.add_argument("--allow-templates", action="store_true")
    validate_robustness.set_defaults(func=_cmd_validate_robustness_plan)

    robustness_plan = sub.add_parser(
        "robustness-plan",
        aliases=["ablation-plan", "plan-robustness"],
        help="Create a deterministic dry-run-only robustness/ablation execution plan",
    )
    robustness_plan.add_argument("--plan", required=True, type=Path, help="Robustness/ablation plan JSON")
    robustness_plan.add_argument("--manifest", required=True, type=Path)
    robustness_plan.add_argument("--campaign-registry", required=True, type=Path)
    robustness_plan.add_argument("--parameter-set", required=True, type=Path)
    robustness_plan.add_argument("--output-dir", required=True, type=Path)
    robustness_plan.add_argument("--output", type=Path, default=None)
    robustness_plan.add_argument("--allow-templates", action="store_true")
    robustness_plan.add_argument("--include-audit-only", action="store_true")
    robustness_plan.add_argument("--include-preregistered-holdout", action="store_true")
    robustness_plan.set_defaults(func=_cmd_robustness_plan)

    robustness_summary = sub.add_parser(
        "robustness-summary",
        aliases=["ablation-summary", "summarize-robustness"],
        help="Aggregate score JSON files by robustness/ablation variant metadata",
    )
    robustness_summary.add_argument("scores", nargs="*", type=Path, help="Score JSON path(s)")
    robustness_summary.add_argument(
        "--score",
        action="append",
        type=Path,
        default=[],
        help="Score JSON path. May be repeated and may be combined with positional score paths.",
    )
    robustness_summary.add_argument("--output", type=Path, default=None)
    robustness_summary.set_defaults(func=_cmd_robustness_summary)

    score = sub.add_parser("score", help="Score frozen candidates against withheld labels")
    score.add_argument("--run-manifest", required=True, type=Path)
    score.add_argument("--labels", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument(
        "--parameter-set",
        type=Path,
        default=None,
        help="Score-time parameter set. Hash must match the run manifest when supplied.",
    )
    score.add_argument(
        "--require-approved-parameters",
        action="store_true",
        help="Fail holdout scoring unless the run/provided parameter set is approved.",
    )
    score.set_defaults(func=_cmd_score)

    preflight = sub.add_parser(
        "preflight",
        aliases=["inventory"],
        help="Build a search-only SAR product inventory for every public manifest target",
    )
    preflight.add_argument("--manifest", required=True, type=Path)
    preflight.add_argument("--output", type=Path, default=None)
    preflight.add_argument("--output-dir", type=Path, default=None)
    preflight.add_argument("--lock-output", type=Path, default=None)
    preflight.add_argument(
        "--parameter-set",
        type=Path,
        default=None,
        help="No-label validation parameter set to record in the SAR inventory and product lock.",
    )
    preflight.add_argument(
        "--campaign-registry",
        type=Path,
        default=None,
        help="No-label campaign registry to verify and record in the SAR inventory and product lock.",
    )
    preflight.add_argument("--write-lock", action="store_true", help="Also write product_lock.json under --output-dir")
    preflight.add_argument("--allow-templates", action="store_true")
    preflight.add_argument("--start-date", default=DEFAULT_SAR_SEARCH_START_DATE)
    preflight.add_argument("--end-date", default=DEFAULT_SAR_SEARCH_END_DATE)
    preflight.add_argument("--max-results", type=int, default=DEFAULT_SAR_MAX_RESULTS)
    preflight.add_argument("--selection-count", type=int, default=DEFAULT_PRODUCT_SELECTION_COUNT)
    preflight.add_argument("--timeout", type=int, default=None, help="ASF/CMR timeout seconds")
    preflight.add_argument("--retries", type=int, default=None, help="ASF/CMR search attempts")
    preflight.add_argument("--retry-backoff", type=float, default=None, help="Search retry backoff seconds")
    preflight.add_argument("--env-path", type=Path, default=None)
    preflight.add_argument("--no-dotenv", action="store_true", help="Do not load local .env")
    preflight.add_argument("--fail-on-search-error", action="store_true")
    preflight.set_defaults(func=_cmd_preflight)

    lock_products = sub.add_parser(
        "lock-products",
        help="Write a deterministic product lock from a previously written SAR inventory",
    )
    lock_products.add_argument("--inventory", required=True, type=Path)
    lock_products.add_argument("--output", required=True, type=Path)
    lock_products.add_argument("--previous-lock", type=Path, default=None)
    lock_products.add_argument("--fail-on-changed-selection", action="store_true")
    lock_products.set_defaults(func=_cmd_lock_products)

    baseline_report = sub.add_parser(
        "baseline-report",
        aliases=["summarize-scores"],
        help="Aggregate one or more score JSON files into a baseline metrics report",
    )
    baseline_report.add_argument("scores", nargs="*", type=Path, help="Score JSON path(s)")
    baseline_report.add_argument(
        "--score",
        action="append",
        type=Path,
        default=[],
        help="Score JSON path. May be repeated and may be combined with positional score paths.",
    )
    baseline_report.add_argument("--output-json", required=True, type=Path)
    baseline_report.add_argument("--output-text", type=Path, default=None)
    baseline_report.add_argument("--confidence-bins", type=int, default=5)
    baseline_report.set_defaults(func=_cmd_baseline_report)

    package_report = sub.add_parser(
        "package-report",
        aliases=["report-package"],
        help="Write a deterministic redacted validation report package directory",
    )
    package_report.add_argument("--output-dir", required=True, type=Path)
    package_report.add_argument("--public-manifest", required=True, type=Path)
    package_report.add_argument("--run-manifest", required=True, type=Path)
    package_report.add_argument("--score-json", required=True, type=Path)
    package_report.add_argument("--parameter-set", type=Path, default=None)
    package_report.add_argument("--sar-inventory", type=Path, default=None)
    package_report.add_argument("--product-lock", type=Path, default=None)
    package_report.add_argument("--baseline-json", type=Path, default=None)
    package_report.add_argument("--baseline-text", type=Path, default=None)
    package_report.add_argument(
        "--command-log",
        action="append",
        type=Path,
        default=[],
        help="Command log path to copy with redaction. May be repeated.",
    )
    package_report.add_argument("--notes-file", type=Path, default=None)
    package_report.add_argument("--notes", default=None)
    package_report.add_argument("--overwrite", action="store_true")
    package_report.set_defaults(func=_cmd_package_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
