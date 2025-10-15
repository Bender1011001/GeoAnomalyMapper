#!/usr/bin/env python3
"""
GeoAnomalyMapper Environment Setup & Diagnostics Utility

Safe-by-default diagnostics and optional guided setup for GeoAnomalyMapper (GAM).

Key principles:
- Non-invasive by default: 'check' performs read-only diagnostics without imports that
  trigger directory creation.
- Explicit confirmation required before any setup actions or "deep" imports that may
  cause side effects (e.g., shims creating directories).
- Works without any existing configuration; integrates with v1/v2 config/paths when enabled.
- Does not alter installation requirements or existing user workflows.

Commands:
  - check        Run environment diagnostics (safe by default)
  - setup        Optional guided environment setup (requires confirmation)
  - requirements Show scenario-based requirements and suggestions

Examples:
  - Basic diagnostics (safe):
      python setup_environment.py check

  - Diagnostics with JSON output (for CI/troubleshooting):
      python setup_environment.py check --json report.json

  - Deep diagnostics (imports v2 shims and Stage 1–4 components; may create dirs):
      python setup_environment.py check --deep --yes

  - Include network DNS preflight for Copernicus/NASA (non-default):
      python setup_environment.py check --network --yes

  - Guided setup (create directory structure, optional .env copy):
      python setup_environment.py setup --yes

  - Requirements summary:
      python setup_environment.py requirements

Exit codes:
  0 = success
  1 = warnings/issues found (non-fatal)
  2 = errors encountered

Author: GeoAnomalyMapper Stage 5 Integration
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, asdict
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------
# Utilities (no side effects)
# ---------------------------

def _bool_icon(ok: bool) -> str:
    return "✓" if ok else "✗"


def _safe_import_version(pkg: str, import_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Attempt to detect a Python package and its version without raising.

    Args:
        pkg: Distribution or module name to check spec for (e.g., 'rasterio')
        import_name: Optional import name if different from package (e.g., 'osgeo.gdal')

    Returns:
        (installed: bool, version: Optional[str])
    """
    name = import_name or pkg
    try:
        if find_spec(name) is None:
            return False, None
        mod = import_module(name)
        ver = getattr(mod, "__version__", None)
        # osgeo.gdal version location
        if ver is None and name.startswith("osgeo"):
            try:
                from osgeo import gdal  # type: ignore
                ver = gdal.VersionInfo()  # Returns a string like '3040200'
                if isinstance(ver, str) and ver.isdigit():
                    # Convert to dotted string (e.g., 3040200 -> 3.4.2)
                    ver = f"{int(ver[0])}.{int(ver[1:3])}.{int(ver[3:5])}"
            except Exception:
                pass
        return True, ver
    except Exception:
        return False, None


def _which_all(cmds: List[str]) -> Dict[str, bool]:
    return {c: shutil.which(c) is not None for c in cmds}


def _human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= step and i < len(units) - 1:
        f /= step
        i += 1
    return f"{f:.1f} {units[i]}"


def _disk_usage_for(path: Path) -> Dict[str, str]:
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "total": _human_bytes(usage.total),
            "used": _human_bytes(usage.used),
            "free": _human_bytes(usage.free),
        }
    except Exception:
        # Fallback to project root
        usage = shutil.disk_usage(".")
        return {
            "total": _human_bytes(usage.total),
            "used": _human_bytes(usage.used),
            "free": _human_bytes(usage.free),
        }


def _try_psutil_memory() -> Optional[Dict[str, str]]:
    try:
        if find_spec("psutil") is None:
            return None
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {
            "total": _human_bytes(vm.total),
            "available": _human_bytes(vm.available),
            "percent": f"{vm.percent:.1f}%",
        }
    except Exception:
        return None


def _ask_confirmation(prompt: str) -> bool:
    """
    Ask user to confirm on STDIN if --yes not provided.
    """
    print(prompt + " [y/N]: ", end="", flush=True)
    try:
        ans = input().strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# ---------------------------
# Diagnostics data models
# ---------------------------

@dataclass
class SystemInfo:
    os: str
    os_release: str
    os_version: str
    machine: str
    processor: str
    python_version: str
    python_executable: str
    cpu_count: int
    memory: Optional[Dict[str, str]]
    disk_project: Dict[str, str]
    disk_data_dir: Dict[str, str]


@dataclass
class PackageInfo:
    installed: bool
    version: Optional[str]


@dataclass
class PythonPackages:
    core: Dict[str, PackageInfo]
    optional: Dict[str, PackageInfo]


@dataclass
class CLITools:
    tools: Dict[str, bool]


@dataclass
class DataDirsStatus:
    base: str
    exists: bool
    subdirs: Dict[str, bool]


@dataclass
class ConfigModeStatus:
    env_flag: str
    v2_enabled: bool
    safe_shim_imported: bool
    shims: Optional[Dict[str, str]]  # Only filled in deep mode
    warning: Optional[str]


@dataclass
class StageComponentsStatus:
    deep_checked: bool
    imports_ok: Dict[str, bool]
    robust_downloader_ok: bool
    network_dns_ok: Optional[bool]


@dataclass
class DiagnosticsReport:
    system: SystemInfo
    python: PythonPackages
    cli_tools: CLITools
    data_dirs: DataDirsStatus
    config_mode: ConfigModeStatus
    stage_components: StageComponentsStatus
    issues: List[str]
    warnings: List[str]


# ---------------------------
# Core diagnostic routines
# ---------------------------

def gather_system_info(project_root: Path, data_dir_guess: Path) -> SystemInfo:
    mem = _try_psutil_memory()
    return SystemInfo(
        os=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=sys.version.replace("\n", " "),
        python_executable=sys.executable,
        cpu_count=os.cpu_count() or 1,
        memory=mem,
        disk_project=_disk_usage_for(project_root),
        disk_data_dir=_disk_usage_for(data_dir_guess if data_dir_guess.exists() else project_root),
    )


def gather_python_packages() -> PythonPackages:
    core_names = {
        "numpy": "numpy",
        "rasterio": "rasterio",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "requests": "requests",
    }
    optional_names = {
        "fiona": "fiona",
        "geopandas": "geopandas",
        "pyproj": "pyproj",
        "shapely": "shapely",
        "osgeo.gdal": "osgeo.gdal",  # GDAL Python bindings
        "rio_cogeo": "rio_cogeo",    # as module 'rio_cogeo'
        "isce2": "isce2",
        "mintpy": "mintpy",
        "psutil": "psutil",
    }

    def info_map(names: Dict[str, str]) -> Dict[str, PackageInfo]:
        out: Dict[str, PackageInfo] = {}
        for key, import_name in names.items():
            installed, version = _safe_import_version(import_name)
            out[key] = PackageInfo(installed=installed, version=version)
        return out

    return PythonPackages(
        core=info_map(core_names),
        optional=info_map(optional_names),
    )


def gather_cli_tools() -> CLITools:
    tools_to_check = ["gdalinfo", "gdal_translate", "gdalbuildvrt"]
    return CLITools(tools=_which_all(tools_to_check))


def expected_data_subdirs(base: Path) -> Dict[str, bool]:
    """
    Expected directory layout from repo structure and docs.
    (Read-only check; do not create.)
    """
    subdirs = [
        "raw/gravity",
        "raw/magnetic",
        "raw/dem",
        "raw/insar/sentinel1",
        "processed/gravity",
        "processed/magnetic",
        "processed/dem",
        "processed/insar",
        "outputs/final",
        "outputs/multi_resolution",
        "outputs/void_detection",
        "outputs/visualizations",
        "outputs/reports",
        "cache",
    ]
    return {sd: (base / sd).exists() for sd in subdirs}


def gather_data_dirs_status(project_root: Path) -> DataDirsStatus:
    # Safe default based on repo structure and shims' fallback defaults
    base = project_root / "data"
    return DataDirsStatus(
        base=str(base),
        exists=base.exists(),
        subdirs=expected_data_subdirs(base),
    )


def check_config_mode(deep: bool, confirm_side_effects: bool) -> ConfigModeStatus:
    """
    Safe config mode check first; optional deep import of shims to verify v2/v1 behavior.

    WARNING: Importing utils.config/paths creates directories by design.
    Only do this if confirm_side_effects is True.
    """
    env_flag = os.getenv("GAM_USE_V2_CONFIG", "false")
    v2_enabled = env_flag.lower() == "true"

    if not deep:
        return ConfigModeStatus(
            env_flag=env_flag,
            v2_enabled=v2_enabled,
            safe_shim_imported=False,
            shims=None,
            warning=(
                "Deep shims not imported; pass --deep --yes to verify v2/v1 behavior. "
                "Note: deep import may create directories."
            ),
        )

    if deep and not confirm_side_effects:
        return ConfigModeStatus(
            env_flag=env_flag,
            v2_enabled=v2_enabled,
            safe_shim_imported=False,
            shims=None,
            warning="Deep import requested without confirmation; skipping to stay non-invasive.",
        )

    shim_info: Dict[str, str] = {}
    try:
        # Import guarded, acknowledging side effects
        from utils import config_shim as cshim  # type: ignore
        from utils import paths_shim as pshim   # type: ignore

        shim_info["config_v2_enabled"] = str(cshim.is_v2_config_enabled())
        # Accessors may use env/get, not directly create; but import already did side-effect if any.
        # Fallback getters will just read env or defaults.
        shim_info["DATA_DIR"] = str(os.getenv("DATA_DIR", "data"))
        shim_info["OUTPUT_DIR"] = str(os.getenv("OUTPUT_DIR", "data/outputs"))

        # Path shim getters (import creates dirs in v2 paths manager; acknowledged in deep mode)
        try:
            shim_info["paths_v2_enabled"] = str(pshim.is_v2_paths_enabled())
            shim_info["data_dir_path"] = str(pshim.get_data_dir())
            shim_info["output_dir_path"] = str(pshim.get_output_dir())
        except Exception as e:
            shim_info["paths_error"] = f"{e}"
        return ConfigModeStatus(
            env_flag=env_flag,
            v2_enabled=v2_enabled,
            safe_shim_imported=True,
            shims=shim_info,
            warning=None,
        )
    except Exception as e:
        return ConfigModeStatus(
            env_flag=env_flag,
            v2_enabled=v2_enabled,
            safe_shim_imported=False,
            shims={"error": str(e)},
            warning="Failed to import shims in deep mode.",
        )


def check_stage_components(deep: bool, confirm_side_effects: bool, do_network: bool) -> StageComponentsStatus:
    """
    Validate Stage 1–4 components. Safe by default (no imports).
    Deep mode will import modules (may cause side effects via shims) only with confirmation.
    Optional network DNS preflight is off by default.
    """
    imports_ok: Dict[str, bool] = {
        "gam_data_agent": False,
        "multi_resolution_fusion": False,
        "detect_voids": False,
        "validate_against_known_features": False,
    }
    robust_downloader_ok = False
    network_dns_ok: Optional[bool] = None

    if not deep or not confirm_side_effects:
        # Shallow result; recommend deep mode if needed
        return StageComponentsStatus(
            deep_checked=False,
            imports_ok=imports_ok,
            robust_downloader_ok=robust_downloader_ok,
            network_dns_ok=network_dns_ok,
        )

    # Deep imports (acknowledged side effects)
    try:
        import_module("gam_data_agent")
        imports_ok["gam_data_agent"] = True
    except Exception:
        imports_ok["gam_data_agent"] = False

    try:
        import_module("multi_resolution_fusion")
        imports_ok["multi_resolution_fusion"] = True
    except Exception:
        imports_ok["multi_resolution_fusion"] = False

    try:
        import_module("detect_voids")
        imports_ok["detect_voids"] = True
    except Exception:
        imports_ok["detect_voids"] = False

    try:
        import_module("validate_against_known_features")
        imports_ok["validate_against_known_features"] = True
    except Exception:
        imports_ok["validate_against_known_features"] = False

    # RobustDownloader availability (no network call)
    try:
        from utils.error_handling import RobustDownloader  # type: ignore
        _ = RobustDownloader(max_retries=1, base_delay=0.1, circuit_breaker=False)
        robust_downloader_ok = True
    except Exception:
        robust_downloader_ok = False

    # Optional network DNS check (reduced timeouts to avoid long waits)
    if do_network:
        try:
            from utils.error_handling import ensure_dns, DEFAULT_SERVICES  # type: ignore
            hosts: List[str] = []
            for svc in DEFAULT_SERVICES.values():
                hosts.extend(svc.get("hosts", []))
            unique_hosts = sorted(set(hosts))
            # Use shorter timeouts/retries than defaults to keep this fast
            ensure_dns(unique_hosts, timeout=2, max_retries=1)
            network_dns_ok = True
        except Exception:
            network_dns_ok = False

    return StageComponentsStatus(
        deep_checked=True,
        imports_ok=imports_ok,
        robust_downloader_ok=robust_downloader_ok,
        network_dns_ok=network_dns_ok,
    )


def build_diagnostics_report(
    project_root: Path,
    deep: bool,
    yes: bool,
    do_network: bool,
) -> DiagnosticsReport:
    data_dir_guess = project_root / "data"
    system = gather_system_info(project_root, data_dir_guess)
    python = gather_python_packages()
    cli_tools = gather_cli_tools()
    data_dirs = gather_data_dirs_status(project_root)
    config_mode = check_config_mode(deep=deep, confirm_side_effects=yes)
    stage_components = check_stage_components(deep=deep, confirm_side_effects=yes, do_network=do_network)

    issues: List[str] = []
    warnings: List[str] = []

    # Core packages issues
    for name, info in python.core.items():
        if not info.installed:
            issues.append(f"Core Python package missing: {name}")

    # Raster stack warning
    if not python.core.get("rasterio", PackageInfo(False, None)).installed:
        warnings.append("rasterio not detected; raster processing will be unavailable.")
    if not any(cli_tools.tools.values()):
        warnings.append("GDAL CLI tools not found in PATH (gdalinfo/gdal_translate/gdalbuildvrt). Recommended for some workflows.")

    # Data directories warnings
    if not data_dirs.exists:
        warnings.append("Data directory 'data/' does not exist yet (this is normal for new setups).")
    else:
        # Require at least one raw source subdir for meaningful processing
        raw_any = any(
            data_dirs.subdirs.get(k, False)
            for k in ("raw/gravity", "raw/magnetic", "raw/dem", "raw/insar/sentinel1")
        )
        if not raw_any:
            warnings.append("No raw data subdirectories detected under data/raw/. Populate raw datasets to proceed.")

    # Shims notes
    if config_mode.warning:
        warnings.append(config_mode.warning)

    # Stage components warnings
    if deep and not stage_components.deep_checked:
        warnings.append("Deep stage component checks were skipped due to lack of confirmation.")
    elif deep:
        for mod, ok in stage_components.imports_ok.items():
            if not ok:
                warnings.append(f"Module import failed (deep): {mod}")
        if stage_components.network_dns_ok is False:
            warnings.append("Network DNS preflight failed. Check connectivity to provider endpoints.")

    return DiagnosticsReport(
        system=system,
        python=python,
        cli_tools=cli_tools,
        data_dirs=data_dirs,
        config_mode=config_mode,
        stage_components=stage_components,
        issues=issues,
        warnings=warnings,
    )


def print_diagnostics(report: DiagnosticsReport) -> None:
    # System
    s = report.system
    print("\n=== System Information ===")
    print(f"OS:           {s.os} {s.os_release} ({s.os_version})")
    print(f"Machine:      {s.machine} | CPU: {s.cpu_count} | Python: {s.python_version}")
    print(f"Executable:   {s.python_executable}")
    if s.memory:
        print(f"Memory:       total={s.memory['total']}, available={s.memory['available']}, used={s.memory['percent']}")
    print(f"Disk (proj):  total={s.disk_project['total']}, used={s.disk_project['used']}, free={s.disk_project['free']}")
    print(f"Disk (data):  total={s.disk_data_dir['total']}, used={s.disk_data_dir['used']}, free={s.disk_data_dir['free']}")

    # Python packages
    print("\n=== Python Packages ===")
    print("Core:")
    for name, info in report.python.core.items():
        print(f"  {name:12} {_bool_icon(info.installed)}  {info.version or '-'}")
    print("Optional:")
    for name, info in report.python.optional.items():
        print(f"  {name:12} {_bool_icon(info.installed)}  {info.version or '-'}")

    # CLI tools
    print("\n=== CLI Tools (PATH) ===")
    for tool, ok in report.cli_tools.tools.items():
        print(f"  {tool:14} {_bool_icon(ok)}")

    # Data directories
    dd = report.data_dirs
    print("\n=== Data Directories ===")
    print(f"Base: {dd.base}  exists={_bool_icon(dd.exists)}")
    for sub, ok in dd.subdirs.items():
        print(f"  {sub:32} {_bool_icon(ok)}")

    # Config mode
    cm = report.config_mode
    print("\n=== Config/Paths Mode ===")
    print(f"GAM_USE_V2_CONFIG: {cm.env_flag} (v2_enabled={_bool_icon(cm.v2_enabled)})")
    if cm.safe_shim_imported:
        print("Shims imported (deep mode):")
        if cm.shims:
            for k, v in cm.shims.items():
                print(f"  {k}: {v}")
    else:
        if cm.warning:
            print(f"Note: {cm.warning}")

    # Stage components
    sc = report.stage_components
    print("\n=== Stage 1–4 Components ===")
    if not sc.deep_checked:
        print("Deep checks were not performed (safe mode). Use --deep --yes to verify imports.")
    else:
        for mod, ok in sc.imports_ok.items():
            print(f"  import {mod:28} {_bool_icon(ok)}")
        print(f"  RobustDownloader available: {_bool_icon(sc.robust_downloader_ok)}")
        if sc.network_dns_ok is not None:
            print(f"  DNS preflight (providers):  {_bool_icon(sc.network_dns_ok)}")

    # Summary
    print("\n=== Summary ===")
    if report.issues:
        print(f"Issues ({len(report.issues)}):")
        for it in report.issues:
            print(f"  - {it}")
    else:
        print("Issues: none")

    if report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("Warnings: none")
    print("")


# ---------------------------
# Guided setup (explicit)
# ---------------------------

def perform_setup(yes: bool, use_v2_shims: bool, project_root: Path) -> int:
    """
    Optional guided setup:
      - Creates directory structure under data/
      - Optionally copies .env.example -> .env (without adding secrets)
      - If use_v2_shims is true (and confirmed), imports shims to create v2-managed dirs
    """
    # Confirm up-front if not --yes
    if not yes:
        if not _ask_confirmation("Proceed with guided setup (create directories, optional .env copy)?"):
            print("Aborted. No changes made.")
            return 0

    # Path structure based on repo and docs; safe to create
    data = project_root / "data"
    dirs_to_create = [
        data / "raw" / "gravity",
        data / "raw" / "magnetic",
        data / "raw" / "dem",
        data / "raw" / "insar" / "sentinel1",
        data / "processed" / "gravity",
        data / "processed" / "magnetic",
        data / "processed" / "dem",
        data / "processed" / "insar",
        data / "outputs" / "final",
        data / "outputs" / "multi_resolution",
        data / "outputs" / "void_detection",
        data / "outputs" / "visualizations",
        data / "outputs" / "reports",
        data / "cache",
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory structure under: {data}")

    # .env example copy (only if .env missing)
    env_example = project_root / ".env.example"
    env_target = project_root / ".env"
    if env_example.exists() and not env_target.exists():
        do_copy = yes or _ask_confirmation(f"Copy {env_example} -> {env_target}? (you can edit credentials locally)")
        if do_copy:
            shutil.copy2(str(env_example), str(env_target))
            print(f"Copied .env template to: {env_target}")
        else:
            print("Skipped copying .env")

    # Optional v2 shims directory creation
    if use_v2_shims:
        if yes or _ask_confirmation(
            "Import v2 shims to create any managed directories (may create/ensure paths)?"
        ):
            try:
                from utils import config_shim as cshim  # type: ignore
                from utils import paths_shim as pshim   # type: ignore
                # Touch a few getters (import already ensures dirs in v2 modes)
                _ = cshim.is_v2_config_enabled()
                _ = pshim.get_data_dir()
                _ = pshim.get_output_dir()
                print("v2 shims imported and paths ensured where applicable.")
            except Exception as e:
                print(f"Warning: failed to import v2 shims: {e}")
        else:
            print("Skipped importing v2 shims.")
    else:
        print("Skipped v2 shims path creation (default).")

    print("Setup complete.")
    return 0


# ---------------------------
# Requirements summary
# ---------------------------

def print_requirements_summary() -> None:
    """
    Scenario-based requirements summary. Does not modify system or install anything.
    """
    print("\n=== Requirements Summary ===")
    print("\nCore processing (gravity/magnetic/DEM, fusion):")
    print("  Python >= 3.9")
    print("  pip install -e .")
    print("  Includes: numpy, rasterio, scipy, matplotlib, requests, affine")
    print("  Recommended system tools: GDAL CLI (gdalinfo, gdal_translate, gdalbuildvrt)")

    print("\nOptional geospatial/vector stack:")
    print("  fiona, geopandas, pyproj, shapely")

    print("\nOptional enhancements:")
    print("  rio-cogeo          # Better COG generation")
    print("  osgeo.gdal         # GDAL Python bindings (if needed)")
    print("  psutil             # System diagnostics (memory), optional")

    print("\nInSAR processing (advanced):")
    print("  isce2, mintpy      # Install only if you plan to process InSAR locally")
    print("  SNAP / ISCE2 tools # External apps, not installed by this repo")

    print("\nDevelopment & troubleshooting:")
    print("  JSON diagnostics:  python setup_environment.py check --json diag.json")
    print("  Deep mode checks:  python setup_environment.py check --deep --yes")
    print("")

# ---------------------------
# CLI
# ---------------------------

def run_check(args: argparse.Namespace) -> int:
    project_root = Path(__file__).parent.resolve()
    deep = bool(args.deep)
    yes = bool(args.yes)
    do_network = bool(args.network)

    if deep and not yes:
        print("Deep mode requested. This may create directories due to v2 shims. Use --yes to confirm.")
        # Continue with safe-only check instead of aborting
        deep = False

    report = build_diagnostics_report(
        project_root=project_root,
        deep=deep,
        yes=yes,
        do_network=do_network,
    )

    print_diagnostics(report)

    if args.json:
        try:
            out = Path(args.json)
            payload = asdict(report)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"JSON report written to: {out}")
        except Exception as e:
            print(f"Failed to write JSON report: {e}")
            return 2

    # Exit code logic: errors > warnings > ok
    if report.issues:
        return 2
    if report.warnings:
        return 1
    return 0


def run_setup(args: argparse.Namespace) -> int:
    project_root = Path(__file__).parent.resolve()
    yes = bool(args.yes)
    use_v2 = bool(args.v2)
    return perform_setup(yes=yes, use_v2_shims=use_v2, project_root=project_root)


def run_requirements(_: argparse.Namespace) -> int:
    print_requirements_summary()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GeoAnomalyMapper Environment Setup & Diagnostics Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe diagnostics (no side effects)
  python setup_environment.py check

  # Diagnostics with JSON output
  python setup_environment.py check --json diag.json

  # Deep diagnostics (imports shims and Stage 1–4 modules; may create dirs)
  python setup_environment.py check --deep --yes

  # Include DNS preflight for provider endpoints (non-default)
  python setup_environment.py check --network --yes

  # Guided setup (create directory skeleton; optional .env copy)
  python setup_environment.py setup --yes

  # Use v2 shims to ensure v2-managed directories during setup
  python setup_environment.py setup --v2 --yes

  # Requirements summary
  python setup_environment.py requirements
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Run environment diagnostics (safe by default)")
    p_check.add_argument("--json", help="Write diagnostics as JSON to specified file")
    p_check.add_argument(
        "--deep",
        action="store_true",
        help="Deep diagnostics: import shims and Stage 1–4 modules (may create directories)",
    )
    p_check.add_argument(
        "--yes",
        action="store_true",
        help="Confirm to allow deep diagnostics that may cause side effects",
    )
    p_check.add_argument(
        "--network",
        action="store_true",
        help="Include provider DNS preflight (non-default; read-only but remote)",
    )
    p_check.set_defaults(func=run_check)

    p_setup = sub.add_parser("setup", help="Guided setup (explicit confirmation required)")
    p_setup.add_argument(
        "--yes", action="store_true", help="Run non-interactively with confirmation"
    )
    p_setup.add_argument(
        "--v2",
        action="store_true",
        help="Use v2 shims to ensure directories (may create dirs under v2 paths)",
    )
    p_setup.set_defaults(func=run_setup)

    p_req = sub.add_parser("requirements", help="Show scenario-based requirements and suggestions")
    p_req.set_defaults(func=run_requirements)

    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 2


if __name__ == "__main__":
    sys.exit(main())