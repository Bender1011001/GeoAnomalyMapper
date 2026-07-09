#!/usr/bin/env python3
"""Controlled positive regression for known injected subsurface voids.

This is a lightweight local control: it writes a deterministic wave-speed
volume with one low-velocity spherical void, then runs the production
visualization/extraction/reporting pipeline without remote SAR acquisition or
PINN training. The goal is to prove that a valid low-speed void volume is not
inverted, clipped, or lost by the model-output-to-visualizer path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from json_utils import dump_strict_json, dumps_strict_json, to_strict_jsonable
from project_paths import DATA_DIR
import visualize_3d_subsurface


DEFAULT_OUTPUT_DIR = DATA_DIR / "controlled_void_regression" / "latest"


def _json_safe(value: Any) -> Any:
    return to_strict_jsonable(value)


def build_controlled_void_volume(
    shape_zyx: Tuple[int, int, int] = (24, 24, 24),
    center_zyx: Tuple[int, int, int] = (12, 11, 13),
    radius_voxels: float = 3.25,
    background_wave_speed: float = 3500.0,
    void_wave_speed: float = 450.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic wave-speed volume and the injected void mask."""
    z, y, x = np.indices(shape_zyx, dtype=np.float32)
    cz, cy, cx = center_zyx
    distance_voxels = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    mask = distance_voxels <= float(radius_voxels)

    wave_speed = np.full(shape_zyx, float(background_wave_speed), dtype=np.float32)
    wave_speed[mask] = float(void_wave_speed)
    return wave_speed, mask


def _expected_physical_centroid(
    mask: np.ndarray,
    max_depth_m: float,
    domain_width_m: float,
) -> Sequence[float]:
    coords = np.argwhere(mask)
    nz, ny, nx = mask.shape
    dz = float(max_depth_m) / nz
    dx = float(domain_width_m) / nx
    dy = float(domain_width_m) / ny
    centroid_voxel = coords.mean(axis=0)
    return [
        float((centroid_voxel[2] + 0.5) * dx - domain_width_m / 2.0),
        float((centroid_voxel[1] + 0.5) * dy - domain_width_m / 2.0),
        float((centroid_voxel[0] + 0.5) * dz),
    ]


def _centroid_distance_m(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def run_controlled_void_regression(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    clean: bool = False,
    enable_pyvista: bool = False,
) -> Dict[str, Any]:
    """Run the known-void positive control and return a JSON-safe summary."""
    output_path = Path(output_dir)
    if clean and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    background_speed = 3500.0
    void_speed = 450.0
    max_depth_m = 240.0
    domain_width_m = 240.0
    void_threshold = 0.35
    min_anomaly_voxels = 10

    wave_speed, injected_mask = build_controlled_void_volume(
        background_wave_speed=background_speed,
        void_wave_speed=void_speed,
    )
    wave_speed_path = output_path / "controlled_wave_speed_volume.npy"
    mask_path = output_path / "controlled_injected_void_mask.npy"
    np.save(wave_speed_path, wave_speed)
    np.save(mask_path, injected_mask.astype(np.uint8))

    config = {
        "void_threshold": void_threshold,
        "min_anomaly_voxels": min_anomaly_voxels,
        "max_depth_m": max_depth_m,
        "domain_width_m": domain_width_m,
        "background_wave_speed": background_speed,
        "min_wave_speed": 300.0,
        "depth_slices": [40.0, 80.0, 120.0, 130.0, 160.0, 200.0],
        "enable_pyvista": bool(enable_pyvista),
    }

    outputs = visualize_3d_subsurface.run_visualization_pipeline(
        wave_speed_path=str(wave_speed_path),
        output_dir=str(output_path),
        void_probability_path=None,
        config=config,
        interactive=False,
    )

    anomalies = outputs.get("anomaly_list", [])
    expected_centroid_m = _expected_physical_centroid(
        injected_mask,
        max_depth_m=max_depth_m,
        domain_width_m=domain_width_m,
    )
    voxel_size_m = max_depth_m / wave_speed.shape[0]
    best_anomaly = anomalies[0] if anomalies else None

    centroid_error_m = None
    if best_anomaly:
        centroid_error_m = _centroid_distance_m(best_anomaly["centroid_m"], expected_centroid_m)

    detected = bool(
        best_anomaly
        and outputs.get("anomaly_count", 0) >= 1
        and best_anomaly.get("voxel_count", 0) >= min_anomaly_voxels
        and float(best_anomaly.get("mean_void_probability", 0.0)) > void_threshold
        and float(best_anomaly.get("mean_wave_speed_ms", background_speed)) < background_speed * 0.7
        and centroid_error_m is not None
        and centroid_error_m <= 2.0 * voxel_size_m
    )

    manifest = {}
    audit_manifest_path = outputs.get("audit_manifest")
    if audit_manifest_path and Path(audit_manifest_path).exists():
        with open(audit_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    threshold_diagnostics = manifest.get("diagnostics", {}).get("threshold_crossing", {})
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "controlled_void_detected": detected,
        "anomaly_count": int(outputs.get("anomaly_count", 0)),
        "anomalies_detected": int(outputs.get("anomalies_detected", outputs.get("anomaly_count", 0))),
        "expected": {
            "background_wave_speed_ms": background_speed,
            "void_wave_speed_ms": void_speed,
            "void_threshold": void_threshold,
            "min_anomaly_voxels": min_anomaly_voxels,
            "injected_voxels": int(injected_mask.sum()),
            "expected_centroid_m": expected_centroid_m,
            "max_allowed_centroid_error_m": float(2.0 * voxel_size_m),
        },
        "observed": {
            "best_anomaly": best_anomaly,
            "centroid_error_m": centroid_error_m,
            "threshold_diagnostics": threshold_diagnostics,
        },
        "artifacts": {
            "output_dir": str(output_path),
            "wave_speed_volume": str(wave_speed_path),
            "injected_void_mask": str(mask_path),
            "computed_void_probability_volume": outputs.get("void_probability_volume"),
            "anomaly_report": outputs.get("anomaly_report"),
            "anomaly_catalog": outputs.get("anomalies_csv") or outputs.get("detected_anomalies_csv"),
            "audit_manifest": audit_manifest_path,
            "cross_sections": outputs.get("cross_sections"),
            "vtk_volume": outputs.get("vtk_volume"),
            "void_mesh_stl": outputs.get("void_mesh_stl"),
        },
    }

    summary_path = output_path / "controlled_void_regression_summary.json"
    summary["artifacts"]["summary"] = str(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        dump_strict_json(summary, f, indent=2)

    return _json_safe(summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the controlled injected-void regression.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Delete the output directory before running.")
    parser.add_argument("--enable-pyvista", action="store_true", help="Also run optional PyVista 3-D rendering/mesh export.")
    parser.add_argument("--no-fail", action="store_true", help="Return exit code 0 even if the control is not detected.")
    args = parser.parse_args(argv)

    summary = run_controlled_void_regression(
        output_dir=args.output_dir,
        clean=args.clean,
        enable_pyvista=args.enable_pyvista,
    )
    print(dumps_strict_json(summary, indent=2))
    if not summary.get("controlled_void_detected") and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
