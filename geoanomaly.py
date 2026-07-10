#!/usr/bin/env python3
"""Stable validation-first CLI surface for GeoAnomalyMapper.

The top-level product CLI intentionally exposes the research validation workflow
before any UI or product-polish commands. It delegates validation subcommands to
``blind_validation.py`` so the canonical runner/scorer/report implementation
stays in one place.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import blind_validation
from json_utils import dumps_strict_json


VERSION = "0.1.0"
MIN_PYTHON = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parent

VALIDATION_COMMANDS = [
    "validate-public",
    "validate-labels",
    "init-parameters",
    "validate-parameters",
    "compare-parameters",
    "init-campaign-registry",
    "validate-campaign-registry",
    "compare-campaign-registry",
    "set-campaign-registry-status",
    "approve-campaign-registry",
    "lock-campaign-registry",
    "campaign-plan",
    "plan-campaign",
    "campaign-status",
    "status-campaign",
    "campaign-run",
    "run-campaign",
    "campaign-package",
    "package-campaign-evidence",
    "validate-robustness-plan",
    "validate-ablation-plan",
    "robustness-plan",
    "ablation-plan",
    "plan-robustness",
    "robustness-summary",
    "ablation-summary",
    "summarize-robustness",
    "preflight",
    "inventory",
    "lock-products",
    "run",
    "score",
    "baseline-report",
    "summarize-scores",
    "package-report",
    "report-package",
]

FIXTURE_WORKFLOW_COMMANDS = [
    "python geoanomaly.py validation validate-public --manifest validation_examples/public_manifest_fixture.json",
    "python geoanomaly.py validation run --manifest validation_examples/public_manifest_fixture.json --output-dir data/blind_validation/fixture_run",
    "python geoanomaly.py validation score --run-manifest data/blind_validation/fixture_run/run_manifest.json --labels validation_examples/withheld_labels_fixture.json --output data/blind_validation/fixture_score.json",
    "python geoanomaly.py validation baseline-report data/blind_validation/fixture_score.json --output-json data/blind_validation/fixture_baseline_report.json --output-text data/blind_validation/fixture_baseline_report.txt",
    "python geoanomaly.py validation validate-robustness-plan --plan validation_examples/robustness_ablation_plan_template.json --allow-templates",
    "python geoanomaly.py validation package-report --public-manifest validation_examples/public_manifest_fixture.json --run-manifest data/blind_validation/fixture_run/run_manifest.json --score-json data/blind_validation/fixture_score.json --baseline-json data/blind_validation/fixture_baseline_report.json --baseline-text data/blind_validation/fixture_baseline_report.txt --output-dir data/blind_validation/fixture_report_package",
]

CLEAN_ENVIRONMENT_COMMANDS = [
    "python -m venv .venv-clean",
    r".venv-clean\Scripts\python -m pip install --upgrade pip",
    r".venv-clean\Scripts\python -m pip install -e .",
    r".venv-clean\Scripts\python -m compileall -q blind_validation.py geoanomaly.py json_utils.py tests",
    r".venv-clean\Scripts\python -m unittest discover -s tests -p test_release_hardening.py",
    r".venv-clean\Scripts\python -m unittest discover -s tests -p test_blind_validation.py",
]

CORE_MODULES = [
    "blind_validation",
    "geoanomaly",
    "json_utils",
    "slc_data_fetcher",
    "project_paths",
    "deformation_intel.opera",
    "deformation_intel.timeseries",
    "deformation_intel.sources",
    "deformation_intel.detect",
]

REQUIREMENT_IMPORTS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("rasterio", "rasterio"),
    ("pyproj", "pyproj"),
    ("xarray", "xarray"),
    ("h5netcdf", "h5netcdf"),
    ("earthaccess", "earthaccess"),
    ("asf_search", "asf_search"),
    ("hyp3_sdk", "hyp3_sdk"),
    ("matplotlib", "matplotlib"),
    ("requests", "requests"),
]

HEALTH_DIRECTORIES = [
    "data",
    "data/blind_validation",
    "outputs",
    "results",
]

EXPECTED_ENV_KEYS = [
    "EARTHDATA_TOKEN",
    "EARTHDATA_BEARER_TOKEN",
    "EARTHDATA_USERNAME",
    "EARTHDATA_PASSWORD",
]

FIXTURE_WORKFLOW_INPUTS = [
    "geoanomaly.py",
    "blind_validation.py",
    "validation_examples/public_manifest_fixture.json",
    "validation_examples/withheld_labels_fixture.json",
    "validation_examples/fixture_candidates/positive_detected_anomalies.csv",
    "validation_examples/fixture_candidates/negative_detected_anomalies.csv",
]


def _module_status(module_name: str) -> Dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    return {
        "module": module_name,
        "available": spec is not None,
        "checked_by": "importlib.util.find_spec",
    }


def _dependency_status(package_name: str, module_name: str) -> Dict[str, Any]:
    status = _module_status(module_name)
    status["package"] = package_name
    return status


def _env_status(root: Path) -> Dict[str, Any]:
    env_path = root / ".env"
    example_path = root / ".env.example"
    discovered_keys = set()
    parse_error = None
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key:
                    discovered_keys.add(key)
        except OSError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "dotenv": {
            "path": ".env",
            "present": env_path.exists(),
            "expected_keys_present": {key: key in discovered_keys for key in EXPECTED_ENV_KEYS},
            "values_printed": False,
            "parse_error": parse_error,
        },
        "example": {
            "path": ".env.example",
            "present": example_path.exists(),
        },
    }


def _directory_status(root: Path, create_dirs: bool) -> List[Dict[str, Any]]:
    records = []
    for rel_path in HEALTH_DIRECTORIES:
        path = root / rel_path
        created = False
        error = None
        if create_dirs and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created = True
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        if exists:
            writable = os.access(path, os.W_OK)
        else:
            parent = path.parent
            writable = parent.exists() and os.access(parent, os.W_OK)
        records.append(
            {
                "path": rel_path,
                "exists": exists,
                "is_dir": is_dir,
                "writable": bool(writable),
                "created": created,
                "error": error,
            }
        )
    return records


def _disk_status(root: Path) -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    free_gb = usage.free / (1024 ** 3)
    return {
        "available": True,
        "path": ".",
        "free_bytes": usage.free,
        "free_gb": round(free_gb, 3),
        "total_bytes": usage.total,
    }


def _gpu_status(probe_gpu: bool) -> Dict[str, Any]:
    if not probe_gpu:
        return {
            "probed": False,
            "available": None,
            "status": "skipped",
        }
    if importlib.util.find_spec("torch") is None:
        return {
            "probed": True,
            "available": False,
            "status": "torch_not_installed",
        }
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as exc:  # pragma: no cover - defensive around optional GPU runtimes
        return {
            "probed": True,
            "available": False,
            "status": "probe_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "probed": True,
        "available": cuda_available,
        "status": "cuda_available" if cuda_available else "cpu_only",
        "device_count": device_count,
    }


def _command_status(root: Path) -> List[Dict[str, Any]]:
    fixture_inputs_available = all((root / rel_path).exists() for rel_path in FIXTURE_WORKFLOW_INPUTS)
    tests_available = (root / "tests").is_dir()
    commands = [
        {
            "name": "release_health",
            "command": "python geoanomaly.py health --json --skip-gpu",
            "available": (root / "geoanomaly.py").exists(),
            "no_downloads_or_training": True,
        },
        {
            "name": "py_compile",
            "command": "python -m compileall -q blind_validation.py geoanomaly.py tests",
            "available": (root / "blind_validation.py").exists() and (root / "geoanomaly.py").exists() and tests_available,
            "no_downloads_or_training": True,
        },
        {
            "name": "unit_tests",
            "command": "python -m unittest discover -s tests",
            "available": tests_available,
            "no_downloads_or_training": True,
        },
        {
            "name": "clean_env_create",
            "command": CLEAN_ENVIRONMENT_COMMANDS[0],
            "available": True,
            "no_downloads_or_training": True,
        },
        {
            "name": "clean_env_install_editable",
            "command": CLEAN_ENVIRONMENT_COMMANDS[2],
            "available": (root / "pyproject.toml").exists(),
            "no_downloads_or_training": True,
        },
        {
            "name": "clean_env_smoke_tests",
            "command": CLEAN_ENVIRONMENT_COMMANDS[4],
            "available": tests_available,
            "no_downloads_or_training": True,
        },
    ]
    for index, command in enumerate(FIXTURE_WORKFLOW_COMMANDS, start=1):
        commands.append(
            {
                "name": f"fixture_workflow_step_{index}",
                "command": command,
                "available": fixture_inputs_available,
                "no_downloads_or_training": "--execute-real" not in command,
            }
        )
    return commands


def build_health_report(
    root: Path | str = PROJECT_ROOT,
    *,
    create_dirs: bool = False,
    probe_gpu: bool = True,
) -> Dict[str, Any]:
    root_path = Path(root)
    python_ok = sys.version_info >= MIN_PYTHON
    core_modules = [_module_status(module_name) for module_name in CORE_MODULES]
    dependencies = [_dependency_status(package_name, module_name) for package_name, module_name in REQUIREMENT_IMPORTS]
    directories = _directory_status(root_path, create_dirs)
    commands = _command_status(root_path)
    warnings = []
    failures = []
    if not python_ok:
        failures.append(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required")
    missing_core = [record["module"] for record in core_modules if not record["available"]]
    if missing_core:
        failures.append("Missing core modules: " + ", ".join(missing_core))
    missing_deps = [record["package"] for record in dependencies if not record["available"]]
    if missing_deps:
        warnings.append("Missing optional/installed dependency modules: " + ", ".join(missing_deps))
    missing_dirs = [record["path"] for record in directories if not record["exists"]]
    if missing_dirs:
        warnings.append("Local data/output directories are missing: " + ", ".join(missing_dirs))
    unavailable_commands = [record["name"] for record in commands if not record["available"]]
    if unavailable_commands:
        warnings.append("Some test/example commands are not available: " + ", ".join(unavailable_commands))
    env = _env_status(root_path)
    if not env["example"]["present"]:
        warnings.append(".env.example is missing")
    return {
        "schema_version": "geoanomaly-health-v1",
        "status": "fail" if failures else "ok",
        "version": VERSION,
        "root": ".",
        "no_downloads_attempted": True,
        "no_training_attempted": True,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
            "minimum_required": f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            "ok": python_ok,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "core_modules": core_modules,
        "dependencies": dependencies,
        "directories": directories,
        "secrets": env,
        "disk": _disk_status(root_path),
        "gpu": _gpu_status(probe_gpu),
        "commands": commands,
        "warnings": warnings,
        "failures": failures,
    }


def format_health_report(report: Dict[str, Any]) -> str:
    lines = [
        "GeoAnomalyMapper release health report",
        f"Status: {report['status'].upper()}",
        f"Version: {report['version']}",
        f"No downloads attempted: {report['no_downloads_attempted']}",
        f"No training attempted: {report['no_training_attempted']}",
        (
            "Python: "
            f"{report['python']['version']} "
            f"(requires >= {report['python']['minimum_required']}; ok={report['python']['ok']})"
        ),
    ]
    disk = report["disk"]
    if disk.get("available"):
        lines.append(f"Disk free: {disk['free_gb']} GB")
    else:
        lines.append(f"Disk free: unavailable ({disk.get('error')})")
    dotenv = report["secrets"]["dotenv"]
    present_keys = ",".join(key for key, present in dotenv["expected_keys_present"].items() if present)
    lines.append(
        ".env: "
        f"present={dotenv['present']}; "
        f"values_printed={dotenv['values_printed']}; "
        f"expected_keys_present={present_keys}"
    )
    gpu = report["gpu"]
    lines.append(f"GPU: status={gpu['status']}; available={gpu['available']}")
    available_core = sum(1 for item in report["core_modules"] if item["available"])
    lines.append(f"Core modules available: {available_core}/{len(report['core_modules'])}")
    available_deps = sum(1 for item in report["dependencies"] if item["available"])
    lines.append(f"Dependency modules available: {available_deps}/{len(report['dependencies'])}")
    lines.append("Directories:")
    for item in report["directories"]:
        status = "OK" if item["exists"] and item["is_dir"] else "WARN"
        lines.append(f"  {status} {item['path']} exists={item['exists']} writable={item['writable']}")
    lines.append("Commands:")
    for item in report["commands"]:
        status = "OK" if item["available"] else "WARN"
        lines.append(f"  {status} {item['name']}: {item['command']}")
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report["warnings"])
    if report["failures"]:
        lines.append("Failures:")
        lines.extend(f"  - {failure}" for failure in report["failures"])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoanomaly.py",
        description="GeoAnomalyMapper research product CLI; validation-first, non-UI surface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"GeoAnomalyMapper validation-first CLI v{VERSION}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validation = sub.add_parser(
        "validation",
        add_help=False,
        help="Delegate to blind known-void validation workflow commands",
        description=(
            "Delegate to blind_validation.py. Use `python geoanomaly.py validation --help` "
            "for canonical validation commands."
        ),
    )
    validation.add_argument(
        "validation_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to blind_validation.py",
    )

    commands = sub.add_parser(
        "commands",
        help="Print canonical validation-first command examples",
    )
    commands.set_defaults(func=_cmd_commands)

    health = sub.add_parser(
        "health",
        aliases=["preflight"],
        help="Run lightweight release health checks without downloads or training",
    )
    health.add_argument("--json", action="store_true", dest="json_output", help="Print machine-readable JSON")
    health.add_argument("--strict", action="store_true", help="Return non-zero on warnings as well as failures")
    health.add_argument("--create-dirs", action="store_true", help="Create local data/output directories if missing")
    health.add_argument("--skip-gpu", action="store_true", help="Skip optional torch CUDA probing")
    health.set_defaults(func=_cmd_health)

    validation.set_defaults(func=_cmd_validation)
    return parser


def _cmd_commands(args: argparse.Namespace) -> int:
    lines = [
        "GeoAnomalyMapper validation-first commands:",
        "  python geoanomaly.py health --json --skip-gpu",
        *[f"  {command}" for command in FIXTURE_WORKFLOW_COMMANDS],
        "",
        "Clean environment smoke commands:",
        *[f"  {command}" for command in CLEAN_ENVIRONMENT_COMMANDS],
        "",
        "Canonical delegated validation subcommands:",
        "  " + ", ".join(VALIDATION_COMMANDS),
    ]
    print("\n".join(lines))
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    report = build_health_report(
        PROJECT_ROOT,
        create_dirs=args.create_dirs,
        probe_gpu=not args.skip_gpu,
    )
    if args.json_output:
        print(dumps_strict_json(report, indent=2, sort_keys=True))
    else:
        print(format_health_report(report))
    if report["failures"] or (args.strict and report["warnings"]):
        return 2
    return 0


def _cmd_validation(args: argparse.Namespace) -> int:
    forwarded = list(args.validation_args or [])
    if not forwarded:
        forwarded = ["--help"]
    return int(blind_validation.main(forwarded))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
