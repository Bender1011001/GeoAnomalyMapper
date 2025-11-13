#!/usr/bin/env python3
"""
GeoAnomalyMapper v2 Final Verification Harness

- Runs comprehensive checks for v1 (default) and v2 (opt-in) modes
- Validates imports, CLI help, shims/config/paths behavior
- Exercises dynamic weighting in-memory (no dataset required)
- Verifies data agent dry-run behavior and status/report outputs
- Runs environment diagnostics (safe and deep)
- Confirms GitHub Pages structure presence
- Executes a minimal end-to-end processing path (non-fatal if data missing)
- Emits a structured JSON report and console summary

Outputs:
  - integration_verification_report.json (repository root)
Exit code:
  - 0 if all mandatory checks passed
  - 1 otherwise (with detailed reasons in JSON)
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).parent.resolve()

# ------------- Utilities ------------- #

def _env_for(mode: str) -> Dict[str, str]:
    env = dict(os.environ)
    if mode == "v1":
        env["GAM_USE_V2_CONFIG"] = "false"
        env.pop("GAM_DYNAMIC_WEIGHTING", None)
        env.pop("GAM_DATA_AGENT_ENABLED", None)
        env.pop("GAM_VALIDATION_ENABLED", None)
    elif mode == "v2":
        env["GAM_USE_V2_CONFIG"] = "true"
        env["GAM_DYNAMIC_WEIGHTING"] = "true"
        env["GAM_DATA_AGENT_ENABLED"] = "true"
        env["GAM_VALIDATION_ENABLED"] = "true"
    else:
        raise ValueError("mode must be 'v1' or 'v2'")
    return env

def run_py(args: List[str], env: Dict[str, str]) -> Tuple[int, str]:
    """Run a python script with controlled environment and capture output."""
    cmd = [sys.executable] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out
    except Exception as e:
        return 1, f"Exception running {' '.join(args)}: {e}"

def check_imports(mods: List[str], env: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
    """Attempt to import modules in-process (uses current interpreter env)."""
    # Ensure this process has the same env flags as requested
    for k, v in env.items():
        if k.startswith("GAM_"):
            os.environ[k] = v

    results: Dict[str, Any] = {}
    ok = True
    for m in mods:
        try:
            importlib.invalidate_caches()
            importlib.import_module(m)
            results[m] = "OK"
        except Exception as e:
            ok = False
            results[m] = f"FAIL: {e}"
    return ok, results

def docs_structure_ok() -> Tuple[bool, List[str]]:
    required = [
        "docs/index.html",
        "docs/assets/main.js",
        "docs/css/styles.css",
        "docs/js/app.js",
        "docs/data/datasets.json",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    return len(missing) == 0, missing

def read_shims_snapshot(env: Dict[str, str]) -> Dict[str, Any]:
    """Interrogate config/paths shims in a subprocess to avoid side effects bleed."""
    code = r"""
from utils import config_shim, paths_shim
from pathlib import Path
snap = {
  "config_v2_enabled": bool(config_shim.is_v2_config_enabled()),
  "data_dir": str(paths_shim.get_data_dir()),
  "output_dir": str(paths_shim.get_output_dir()),
  "processed_dir": str(paths_shim.get_processed_dir()),
  "cache_dir": str(paths_shim.get_cache_dir())
}
print(__import__("json").dumps(snap))
"""
    rc, out = run_py(["-c", code], env)
    if rc != 0:
        return {"error": out.strip()}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"Parse error: {e}", "raw": out}

def exercise_dynamic_weighting(env: Dict[str, str]) -> Dict[str, Any]:
    """Run in-memory fusion with synthetic data to validate GAM_DYNAMIC_WEIGHTING gating."""
    code = r"""
import numpy as np, json, logging
logging.getLogger('multi_resolution_fusion').setLevel(logging.ERROR)
from multi_resolution_fusion import DataLayer, fuse_weighted
# Two simple layers with different resolutions and synthetic uncertainties
a = np.ones((50, 60), dtype=np.float32)
b = np.ones((50, 60), dtype=np.float32) * 2.0
la = DataLayer(name="A", data=a, resolution=0.001, bounds=(-105,32,-104,33), uncertainty=np.full(a.shape, 0.1, dtype=np.float32), weight=1.0)
lb = DataLayer(name="B", data=b, resolution=0.01,  bounds=(-105,32,-104,33), uncertainty=np.full(b.shape, 0.2, dtype=np.float32), weight=1.0)

# Dynamic = False
f_std = fuse_weighted([la, lb], use_uncertainty=True, dynamic=False)
# Dynamic = True
f_dyn = fuse_weighted([la, lb], use_uncertainty=True, dynamic=True, target_res=0.001)

# Basic assertions
ok_shape = (f_std.data.shape == a.shape) and (f_dyn.data.shape == a.shape)
# Verify output arrays are numeric
ok_values = (np.isfinite(f_std.data).any()) and (np.isfinite(f_dyn.data).any())

print(json.dumps({"ok_shape": bool(ok_shape), "ok_values": bool(ok_values), "std_mean": float(np.nanmean(f_std.data)), "dyn_mean": float(np.nanmean(f_dyn.data))}))
"""
    rc, out = run_py(["-c", code], env)
    res: Dict[str, Any] = {"rc": rc, "raw": out}
    if rc == 0:
        try:
            # Find first JSON line
            for line in (out or "").splitlines():
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    res.update(json.loads(s))
                    break
        except Exception as e:
            res["parse_error"] = str(e)
    return res

def cli_helps(env: Dict[str, str]) -> Dict[str, Any]:
    targets = [
        "gam_data_agent.py",
        "multi_resolution_fusion.py",
        "detect_voids.py",
        "validate_against_known_features.py",
        "setup_environment.py",
        "process_data.py",
        "process_insar_data.py",
        "create_visualization.py",
        "create_enhanced_visualization.py",
    ]
    results: Dict[str, Any] = {}
    for t in targets:
        rc, out = run_py([str(ROOT / t), "-h"], env)
        results[t] = {"rc": rc, "ok": rc == 0, "snippet": "\n".join((out or "").splitlines()[:10])}
    return results

def env_diagnostics(env: Dict[str, str], deep: bool = False) -> Dict[str, Any]:
    args = ["setup_environment.py", "check"]
    if deep:
        args += ["--deep", "--yes"]
    rc, out = run_py(args, env)
    return {"rc": rc, "ok": rc in (0,1), "exit_code_meaning": "0=ok,1=warnings,2=errors", "snippet": "\n".join((out or "").splitlines()[-40:])}

def data_agent_status_and_report(env: Dict[str, str]) -> Dict[str, Any]:
    # Status + report file existence
    rc, out = run_py(["gam_data_agent.py", "status", "--report"], env)
    report_path = ROOT / "data" / "outputs" / "data_status_report.md"
    return {"rc": rc, "ok": rc == 0, "report_exists": report_path.exists(), "report_path": str(report_path), "snippet": "\n".join((out or "").splitlines()[:20])}

def data_agent_dry_run_free(env: Dict[str, str]) -> Dict[str, Any]:
    rc, out = run_py(["gam_data_agent.py", "download", "free", "--bbox=-105,32,-104,33", "--dry-run"], env)
    status_path = ROOT / "data" / "data_status.json"
    return {"rc": rc, "ok": rc == 0, "status_exists": status_path.exists(), "status_path": str(status_path), "snippet": "\n".join((out or "").splitlines()[-20:])}

def process_all_minimal(env: Dict[str, str]) -> Dict[str, Any]:
    # This may warn if data missing; success is non-fatal; we capture rc/snippet and artifact presence
    rc, out = run_py(["process_data.py", "--region=-105.0,32.0,-104.0,33.0"], env)
    log_path = ROOT / "data" / "processed" / "processing_log.json"
    return {"rc": rc, "ok": rc in (0, ), "log_exists": log_path.exists(), "log_path": str(log_path), "snippet": "\n".join((out or "").splitlines()[-30:])}

# ------------- Test Suites ------------- #

def run_v1_suite() -> Dict[str, Any]:
    env = _env_for("v1")
    results: Dict[str, Any] = {"mode": "v1"}

    # Imports
    import_ok, import_results = check_imports(
        [
            "utils.config_shim",
            "utils.paths_shim",
            "multi_resolution_fusion",
            "detect_voids",
            "validate_against_known_features",
            "setup_environment",
            "process_data",
            "process_insar_data",
            "create_visualization",
            "create_enhanced_visualization",
        ],
        env,
    )
    results["imports"] = {"ok": import_ok, "details": import_results}

    # CLI -h
    results["cli_help"] = cli_helps(env)

    # Environment diagnostics (safe)
    results["env_check_safe"] = env_diagnostics(env, deep=False)

    # Shims snapshot (expect v2 disabled, default paths)
    shims = read_shims_snapshot(env)
    results["shims"] = shims

    # Data agent status/report (safe)
    results["data_agent_status"] = data_agent_status_and_report(env)

    # Docs structure
    docs_ok, missing = docs_structure_ok()
    results["docs"] = {"ok": docs_ok, "missing": missing}

    # Determine v1 overall ok
    mandatory_ok = (
        results["imports"]["ok"]
        and all(v.get("ok") for v in results["cli_help"].values())
        and results["env_check_safe"]["ok"]
        and isinstance(shims, dict)
        and (shims.get("config_v2_enabled") is False)
        and results["docs"]["ok"]
    )
    results["ok"] = bool(mandatory_ok)
    return results

def run_v2_suite() -> Dict[str, Any]:
    env = _env_for("v2")
    results: Dict[str, Any] = {"mode": "v2"}

    # Imports
    import_ok, import_results = check_imports(
        [
            "utils.config_shim",
            "utils.paths_shim",
            "multi_resolution_fusion",
            "detect_voids",
            "validate_against_known_features",
            "setup_environment",
            "gam_data_agent",
        ],
        env,
    )
    results["imports"] = {"ok": import_ok, "details": import_results}

    # CLI -h
    results["cli_help"] = cli_helps(env)

    # Environment diagnostics (deep; may create dirs)
    results["env_check_deep"] = env_diagnostics(env, deep=True)

    # Shims snapshot (expect v2 enabled)
    shims = read_shims_snapshot(env)
    results["shims"] = shims

    # Dynamic weighting (in-memory exercise)
    results["dynamic_weighting"] = exercise_dynamic_weighting(env)

    # Data agent dry-run (non-invasive) and status file
    results["data_agent_dry_run"] = data_agent_dry_run_free(env)
    results["data_agent_status"] = data_agent_status_and_report(env)

    # Minimal end-to-end (non-fatal if data missing)
    results["process_all"] = process_all_minimal(env)

    # Determine v2 overall ok (mandatory subset)
    mandatory_ok = (
        results["imports"]["ok"]
        and all(v.get("ok") for v in results["cli_help"].values())
        and results["env_check_deep"]["ok"]
        and isinstance(shims, dict)
        and (shims.get("config_v2_enabled") is True)
        and (results["dynamic_weighting"].get("rc") == 0)
        and (results["dynamic_weighting"].get("ok_shape") is True)
        and (results["dynamic_weighting"].get("ok_values") is True)
        and results["data_agent_dry_run"]["ok"]
    )
    results["ok"] = bool(mandatory_ok)
    return results

# ------------- Main ------------- #

def main() -> int:
    report: Dict[str, Any] = {
        "project_root": str(ROOT),
        "py_executable": sys.executable,
        "v1": None,
        "v2": None,
        "docs": None,
        "summary": {},
    }

    v1 = run_v1_suite()
    v2 = run_v2_suite()
    report["v1"] = v1
    report["v2"] = v2

    # Docs consistency already checked in v1; replicate top-level for convenience
    docs_ok, missing = docs_structure_ok()
    report["docs"] = {"ok": docs_ok, "missing": missing}

    # Summary
    report["summary"] = {
        "v1_ok": v1.get("ok", False),
        "v2_ok": v2.get("ok", False),
        "docs_ok": bool(docs_ok),
        "overall_ok": bool(v1.get("ok", False) and v2.get("ok", False) and docs_ok),
    }

    # Write JSON report
    out_path = ROOT / "integration_verification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Human-readable summary
    print("\n=== FINAL VERIFICATION SUMMARY ===")
    print(f"V1 (default)         : {'OK' if report['summary']['v1_ok'] else 'FAIL'}")
    print(f"V2 (opt-in)          : {'OK' if report['summary']['v2_ok'] else 'FAIL'}")
    print(f"GitHub Pages (docs/)  : {'OK' if report['summary']['docs_ok'] else 'FAIL'}")
    print(f"OVERALL               : {'OK' if report['summary']['overall_ok'] else 'FAIL'}")
    if not report["summary"]["overall_ok"]:
        print("\nIssues detected. See integration_verification_report.json for details.")

    return 0 if report["summary"]["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())