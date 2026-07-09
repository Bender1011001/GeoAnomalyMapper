#!/usr/bin/env python3
"""
3D Subsurface Visualization and Target Extraction
===================================================

This module provides 3D volumetric visualization and anomaly extraction
for the Biondi SAR Doppler Tomography pipeline. It takes the 3D density
and wave-speed volumes produced by pinn_vibro_inversion.py and:

1. Extracts 3D isosurfaces of voids, dense bodies, and anomalous structures
   using the Marching Cubes algorithm
2. Classifies anomalies by shape (spherical, cylindrical, planar -> natural vs artificial)
3. Generates interactive 3D visualizations with PyVista
4. Exports results to standard 3D formats (VTK, STL, OBJ)
5. Produces 2D depth-slice cross-sections for publication

Usage:
    python visualize_3d_subsurface.py --volume data/inversion_3d/outputs/wave_speed_volume.npy
    python visualize_3d_subsurface.py --volume ws.npy --density dc.npy --threshold 0.6
"""

import os
import logging
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, gaussian_filter

from json_utils import dump_strict_json, to_strict_jsonable
from project_paths import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# Configuration
# ============================================================
VIZ_DIR = DATA_DIR / "visualization_3d"

DEFAULT_VIZ_CONFIG = {
    "void_threshold": 0.35,         # Lowered for sigmoid mapping (was 0.5)
    "dense_threshold": 0.7,         # Dense body threshold (wave speed > background * factor)
    "min_anomaly_voxels": 10,       # Minimum voxels for a detected anomaly
    "smoothing_sigma": 0.5,         # Gaussian smoothing before isosurface extraction
    "enable_pyvista": True,          # Can be disabled for lightweight CPU-only audit runs/tests
    "max_depth_m": 2000.0,          # From inversion config
    "domain_width_m": 5000.0,       # From inversion config
    "background_wave_speed": 3500.0,
    "min_wave_speed": 300.0,
    "morphology_iterations": 0,      # Keep thin tunnels/shafts unless cleanup is explicitly requested
    "deep_score_boost": 0.75,        # Boost evidence score for deeper bodies without ignoring evidence
    "depth_slices": [50, 100, 200, 500, 1000, 1500],  # Depth cross-sections (meters)
}

POSITIVE_ANOMALY_CATALOG_FILENAMES = (
    "detected_anomalies.csv",
)
POSITIVE_VOID_MESH_FILENAMES = (
    "void_surface.stl",
    "void_surface.obj",
    "void_surface.ply",
    "void_surface.vtp",
    "void_surface.vtk",
)
AUDIT_HASH_LIMIT_BYTES = 64 * 1024 * 1024


def _json_safe(value: Any) -> Any:
    """Convert common numpy/container values to JSON-native objects."""
    return to_strict_jsonable(value)


def compute_void_probability_from_wave_speed(
    wave_speed: np.ndarray,
    background_wave_speed: float = 3500.0,
    void_speed_threshold_ratio: float = 0.7,
    temperature_ratio: float = 0.05,
) -> np.ndarray:
    """Map wave speed to void probability with the pipeline's low-speed convention.

    Lower wave speed means higher void probability. The mapping is intentionally
    conservative: the 50% point is at ``background_wave_speed *
    void_speed_threshold_ratio`` and the transition width is controlled as a
    fraction of background speed. This helper is used by the visualizer when a
    PINN-provided void-probability volume is absent and by controlled regression
    tests to verify the sign convention without changing production thresholds.
    """
    wave = np.asarray(wave_speed, dtype=np.float64)
    bg_speed = float(background_wave_speed)
    threshold_ratio = float(void_speed_threshold_ratio)
    temperature = max(abs(bg_speed) * float(temperature_ratio), 1e-9)
    anomaly_threshold = bg_speed * threshold_ratio

    z_scores = -(wave - anomaly_threshold) / temperature
    z_scores = np.clip(z_scores, -20.0, 20.0)
    void_probability = 1.0 / (1.0 + np.exp(-z_scores))
    return void_probability.astype(np.float32)


def _array_stats(array: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Return lightweight finite-value statistics for an in-memory array."""
    arr = np.asarray(array)
    finite_mask = np.isfinite(arr)
    finite_values = arr[finite_mask]

    stats: Dict[str, Any] = {
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "total_count": int(arr.size),
        "finite_count": int(finite_mask.sum()),
        "nonfinite_count": int(arr.size - finite_mask.sum()),
    }

    if finite_values.size:
        finite_float = finite_values.astype(np.float64, copy=False)
        quantile_levels = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        quantile_values = np.quantile(finite_float, quantile_levels)

        def _quantile_key(level: float) -> str:
            pct = int(round(level * 100))
            return "p100" if pct >= 100 else f"p{pct:02d}"

        stats.update({
            "min": float(finite_float.min()),
            "max": float(finite_float.max()),
            "mean": float(finite_float.mean()),
            "std": float(finite_float.std()),
            "quantiles": {
                _quantile_key(level): float(value)
                for level, value in zip(quantile_levels, quantile_values)
            },
        })
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "quantiles": {}})

    if threshold is not None:
        threshold = float(threshold)
        above_mask = finite_mask & (arr > threshold)
        at_or_above_mask = finite_mask & (arr >= threshold)
        finite_count = max(int(finite_mask.sum()), 1)
        stats["threshold_counts"] = {
            "threshold": threshold,
            "comparison": ">",
            "voxels_above_threshold": int(above_mask.sum()),
            "voxels_at_or_above_threshold": int(at_or_above_mask.sum()),
            "fraction_above_threshold": float(above_mask.sum() / finite_count),
        }

    return stats


def _threshold_component_diagnostics(
    void_probability: np.ndarray,
    threshold: float,
    min_component_voxels: int,
) -> Dict[str, Any]:
    """Summarize threshold crossings and connected components for auditability."""
    arr = np.asarray(void_probability)
    finite_mask = np.isfinite(arr)
    threshold = float(threshold)
    min_component_voxels = int(min_component_voxels)
    crossing_mask = finite_mask & (arr > threshold)
    crossing_voxels = int(crossing_mask.sum())

    labeled_array, num_features = label(crossing_mask)
    if num_features > 0:
        component_counts = np.bincount(labeled_array.ravel())[1:].astype(np.int64)
    else:
        component_counts = np.array([], dtype=np.int64)

    significant_counts = component_counts[component_counts >= min_component_voxels]
    top_counts = sorted((int(v) for v in component_counts), reverse=True)[:10]

    return {
        "void_threshold": threshold,
        "comparison": ">",
        "min_component_voxels": min_component_voxels,
        "finite_voxel_count": int(finite_mask.sum()),
        "voxels_crossing_threshold": crossing_voxels,
        "any_voxel_crosses_threshold": bool(crossing_voxels > 0),
        "connected_components_crossing_threshold": int(num_features),
        "largest_component_voxels": int(component_counts.max()) if component_counts.size else 0,
        "components_meeting_min_voxels": int(significant_counts.size),
        "any_component_meets_min_voxels": bool(significant_counts.size > 0),
        "component_voxel_counts_top10": top_counts,
    }


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 hash for a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Optional[str], hash_limit_bytes: int = AUDIT_HASH_LIMIT_BYTES) -> Dict[str, Any]:
    """Return path, existence, size, timestamp, and bounded hash metadata."""
    if not path:
        return {"path": None, "exists": False}

    p = Path(path)
    metadata: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return metadata

    try:
        stat = p.stat()
        metadata.update({
            "type": "directory" if p.is_dir() else "file",
            "size_bytes": int(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
        if p.is_file():
            if stat.st_size <= hash_limit_bytes:
                metadata["sha256"] = _sha256_file(p)
            else:
                metadata["sha256"] = None
                metadata["hash_skipped"] = f"file_size_exceeds_{hash_limit_bytes}_bytes"
    except OSError as exc:
        metadata["metadata_error"] = str(exc)
    return metadata


def _cleanup_stale_positive_artifacts(
    output_dir: str,
    remove_catalogs: bool,
    remove_meshes: bool,
    reason: str,
) -> List[str]:
    """Remove stale positive-looking artifacts while preserving reports/VTK volumes."""
    output_path = Path(output_dir)
    candidate_names: List[str] = []
    if remove_catalogs:
        candidate_names.extend(POSITIVE_ANOMALY_CATALOG_FILENAMES)
    if remove_meshes:
        candidate_names.extend(POSITIVE_VOID_MESH_FILENAMES)

    removed: List[str] = []
    for filename in sorted(set(candidate_names)):
        artifact_path = output_path / filename
        try:
            if artifact_path.exists() and (artifact_path.is_file() or artifact_path.is_symlink()):
                artifact_path.unlink()
                removed.append(str(artifact_path))
                logger.info(f"Removed stale positive artifact ({reason}): {artifact_path}")
        except OSError as exc:
            logger.warning(f"Could not remove stale artifact {artifact_path}: {exc}")
    return removed


def write_audit_manifest(
    output_dir: str,
    wave_speed_path: str,
    wave_speed: np.ndarray,
    void_probability_path: Optional[str],
    void_probability: np.ndarray,
    density_contrast_path: Optional[str] = None,
    density_contrast: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    outputs: Optional[Dict] = None,
    anomalies: Optional[List[Dict]] = None,
    cleanup_removed: Optional[List[str]] = None,
    cleanup_reason: Optional[str] = None,
    void_probability_source: Optional[str] = None,
) -> str:
    """Write a lightweight local manifest for per-run no-void auditing."""
    cfg = DEFAULT_VIZ_CONFIG.copy()
    if config:
        cfg.update(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = outputs or {}
    anomalies = anomalies or []
    cleanup_removed = cleanup_removed or []
    void_threshold = float(cfg["void_threshold"])
    min_anomaly_voxels = int(cfg.get("min_anomaly_voxels", DEFAULT_VIZ_CONFIG["min_anomaly_voxels"]))
    threshold_diagnostics = _threshold_component_diagnostics(
        void_probability,
        threshold=void_threshold,
        min_component_voxels=min_anomaly_voxels,
    )

    expected_artifacts: Dict[str, str] = {
        "anomaly_report": str(output_path / "anomaly_report.txt"),
        "detected_anomalies_csv": str(output_path / "detected_anomalies.csv"),
        "void_mesh_stl": str(output_path / "void_surface.stl"),
        "void_probability_volume": str(output_path / "void_probability_volume.npy"),
        "vtk_volume": str(output_path / "subsurface_volume.vtk"),
        "cross_sections": str(output_path / "depth_cross_sections.png"),
        "isosurface_render": str(output_path / "isosurface_3d.png"),
    }
    for key, value in outputs.items():
        if isinstance(value, str):
            expected_artifacts.setdefault(key, value)

    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_path),
        "config": _json_safe({
            "void_threshold": cfg.get("void_threshold"),
            "min_anomaly_voxels": cfg.get("min_anomaly_voxels"),
            "max_depth_m": cfg.get("max_depth_m"),
            "domain_width_m": cfg.get("domain_width_m"),
            "background_wave_speed": cfg.get("background_wave_speed"),
            "depth_slices": cfg.get("depth_slices"),
        }),
        "inputs": {
            "wave_speed_volume": _file_metadata(wave_speed_path),
            "void_probability_volume": _file_metadata(void_probability_path),
            "density_contrast_volume": _file_metadata(density_contrast_path),
        },
        "input_sources": {
            "void_probability": void_probability_source or ("file" if void_probability_path and Path(void_probability_path).exists() else "computed_from_wave_speed"),
            "density_contrast": "file" if density_contrast_path and Path(density_contrast_path).exists() else "not_available",
        },
        "arrays": {
            "wave_speed": _array_stats(wave_speed),
            "void_probability": _array_stats(void_probability, threshold=void_threshold),
        },
        "counts": {
            "anomaly_count": int(len(anomalies)),
            "anomalies_detected": int(len(anomalies)),
            "void_contour_points": int(outputs.get("void_contour_points", 0) or 0),
            "void_contour_cells": int(outputs.get("void_contour_cells", 0) or 0),
            "voxels_crossing_void_threshold": int(threshold_diagnostics["voxels_crossing_threshold"]),
            "connected_components_crossing_void_threshold": int(threshold_diagnostics["connected_components_crossing_threshold"]),
            "components_meeting_min_anomaly_voxels": int(threshold_diagnostics["components_meeting_min_voxels"]),
        },
        "diagnostics": {
            "threshold_crossing": threshold_diagnostics,
            "calibration_summary": {
                "any_voxel_crosses_configured_threshold": bool(threshold_diagnostics["any_voxel_crosses_threshold"]),
                "any_connected_component_meets_min_voxels": bool(threshold_diagnostics["any_component_meets_min_voxels"]),
                "anomalies_require_both_threshold_and_min_voxels": True,
                "note": "No positive anomaly is claimed unless a connected component crosses the configured threshold and survives the min_anomaly_voxels filter.",
            },
        },
        "outputs": {
            key: _file_metadata(path)
            for key, path in sorted(expected_artifacts.items())
        },
        "stale_artifact_cleanup": {
            "reason": cleanup_reason,
            "removed_count": int(len(cleanup_removed)),
            "removed": list(cleanup_removed),
        },
    }

    if density_contrast is not None:
        manifest["arrays"]["density_contrast"] = _array_stats(density_contrast)

    manifest_path = output_path / "audit_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        dump_strict_json(manifest, f, indent=2)
    logger.info(f"Saved audit manifest: {manifest_path}")
    return str(manifest_path)


def _get_matplotlib_pyplot():
    """Import matplotlib lazily so CLI help/imports do not require plotting dependencies."""
    import matplotlib.pyplot as plt
    return plt


# ============================================================
# 1. Anomaly Extraction via Marching Cubes
# ============================================================
def extract_isosurfaces(
    volume: np.ndarray,
    threshold: float,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smoothing: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D isosurface meshes from a volumetric field using
    the Marching Cubes algorithm.

    This is the core function that converts the continuous 3D density
    field from the PINN into discrete geometric surfaces representing
    void boundaries, ore body shells, and other subsurface interfaces.

    Parameters
    ----------
    volume : np.ndarray
        3D array (nz, ny, nx) of scalar values.
    threshold : float
        Isosurface level. Voxels above this value are "inside".
    spacing : tuple
        Physical spacing between voxels in (z, y, x) meters.
    smoothing : float
        Gaussian smoothing sigma applied before extraction.

    Returns
    -------
    vertices : np.ndarray
        (N, 3) mesh vertex coordinates in physical units.
    faces : np.ndarray
        (M, 3) triangle face indices.
    normals : np.ndarray
        (N, 3) vertex normals.
    """
    from skimage.measure import marching_cubes

    # Smooth volume to reduce mesh noise
    if smoothing > 0:
        volume_smooth = gaussian_filter(volume.astype(np.float64), sigma=smoothing)
    else:
        volume_smooth = volume.astype(np.float64)

    # Check if threshold is within data range
    vmin, vmax = volume_smooth.min(), volume_smooth.max()
    if threshold < vmin or threshold > vmax:
        logger.warning(
            f"Threshold {threshold:.3f} outside data range [{vmin:.3f}, {vmax:.3f}]. "
            f"No isosurface will be generated."
        )
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(int), np.array([]).reshape(0, 3)

    # Marching Cubes
    vertices, faces, normals, values = marching_cubes(
        volume_smooth,
        level=threshold,
        spacing=spacing,
    )

    logger.info(
        f"Extracted isosurface: {len(vertices)} vertices, {len(faces)} triangles, "
        f"threshold={threshold:.3f}"
    )

    return vertices, faces, normals


def extract_anomaly_bodies(
    void_probability: np.ndarray,
    wave_speed: np.ndarray,
    config: Optional[Dict] = None,
    embedding_anomaly_map: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Identify and characterize individual anomaly bodies from the 3D volumes.

    Uses connected-component labeling on the thresholded void probability
    to identify separate anomalies, then classifies each by shape and properties.

    Parameters
    ----------
    void_probability : np.ndarray
        3D array (nz, ny, nx) of void probabilities (0-1).
    wave_speed : np.ndarray
        3D array (nz, ny, nx) of wave speeds (m/s).
    config : dict, optional
        Visualization configuration.
    embedding_anomaly_map : np.ndarray, optional
        Cap 6: 2-D (ny, nx) float32 surface embedding anomaly score map in [0,1].
        When provided, the deep_target_score is scaled by a surface corroboration
        factor: anomaly bodies whose centroid projects onto an anomalous surface
        pixel are boosted; bodies over normal surface are mildly reduced.
        This does NOT create detections - it only re-weights existing PINN results.

    Returns
    -------
    list of dict
        Each detected anomaly with properties:
        - id, centroid, volume_m3, depth_range_m
        - mean_wave_speed, shape_classification
        - bounding_box, aspect_ratios
        - deep_target_score, fused_confidence_score (Cap 6)
    """
    cfg = DEFAULT_VIZ_CONFIG.copy()
    if config:
        cfg.update(config)

    nz, ny, nx = void_probability.shape
    max_depth = cfg["max_depth_m"]
    domain_width = cfg["domain_width_m"]

    # Pixel sizes
    dz = max_depth / nz  # meters per voxel (depth)
    dx = domain_width / nx  # meters per voxel (horizontal)
    dy = domain_width / ny

    # Binary thresholding
    binary_voids = void_probability > cfg["void_threshold"]

    # Optional cleanup. Keep disabled by default because erosion can erase thin
    # tunnels, shafts, or narrow deep bodies before connected-component labeling.
    cleanup_iterations = int(cfg.get("morphology_iterations", 0))
    if cleanup_iterations > 0:
        binary_clean = binary_erosion(binary_voids, iterations=cleanup_iterations)
        binary_clean = binary_dilation(binary_clean, iterations=cleanup_iterations)
    else:
        binary_clean = binary_voids

    # Connected component labeling
    labeled_array, num_features = label(binary_clean)
    logger.info(f"Found {num_features} connected anomaly bodies")

    anomalies = []

    for label_id in range(1, num_features + 1):
        component_mask = labeled_array == label_id
        voxel_count = component_mask.sum()

        if voxel_count < cfg["min_anomaly_voxels"]:
            continue

        # Get coordinates of this component
        coords = np.argwhere(component_mask)  # (N, 3) as (iz, iy, ix)

        # Centroid
        centroid_voxel = coords.mean(axis=0)
        centroid_physical = np.array([
            (centroid_voxel[2] + 0.5) * dx - domain_width / 2,  # x (meters from center)
            (centroid_voxel[1] + 0.5) * dy - domain_width / 2,  # y
            (centroid_voxel[0] + 0.5) * dz,                     # z (depth in meters)
        ])

        # Bounding box
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        extent_z = (bbox_max[0] - bbox_min[0] + 1) * dz
        extent_y = (bbox_max[1] - bbox_min[1] + 1) * dy
        extent_x = (bbox_max[2] - bbox_min[2] + 1) * dx

        # Volume
        volume_m3 = voxel_count * dx * dy * dz

        # Depth range
        depth_min = bbox_min[0] * dz
        depth_max = (bbox_max[0] + 1) * dz

        # Mean wave speed within anomaly
        mean_ws = wave_speed[component_mask].mean()
        min_ws = wave_speed[component_mask].min()

        # Mean void probability
        mean_vp = void_probability[component_mask].mean()

        # Shape classification based on aspect ratios
        extents = sorted([extent_x, extent_y, extent_z])
        aspect_1 = extents[2] / max(extents[0], 1e-6)  # longest / shortest
        aspect_2 = extents[1] / max(extents[0], 1e-6)  # middle / shortest

        if aspect_1 < 2.0 and aspect_2 < 2.0:
            shape = "SPHERICAL"
            shape_note = "Natural void or ore body"
        elif aspect_1 > 5.0 and aspect_2 < 2.0:
            shape = "CYLINDRICAL"
            shape_note = "Possible tunnel, shaft, or pipe"
        elif aspect_1 > 3.0 and aspect_2 > 3.0:
            shape = "PLANAR"
            shape_note = "Possible room, layer, or fault"
        elif aspect_1 > 3.0:
            shape = "ELONGATED"
            shape_note = "Linear feature (vein, channel, or conduit)"
        else:
            shape = "IRREGULAR"
            shape_note = "Complex or natural formation"

        # Artificial vs Natural scoring
        # Sharp boundaries + geometric shape -> higher artificiality score
        component_crop = void_probability[
            bbox_min[0]:bbox_max[0]+1,
            bbox_min[1]:bbox_max[1]+1,
            bbox_min[2]:bbox_max[2]+1
        ]
        if min(component_crop.shape) < 2:
            edge_sharpness = 0.0
        else:
            gradient_magnitude = np.gradient(component_crop)
            grad_mag = np.sqrt(sum(g**2 for g in gradient_magnitude))
            # AUDIT FIX (3.2): Only compute mean over boundary voxels where
            # gradient is significant. The old mean(abs(gradient)) over the
            # entire bounding box dilutes as 1/R for larger voids because
            # the interior gradient is ~0.
            boundary_mask = grad_mag > 0.05
            if boundary_mask.any():
                edge_sharpness = float(grad_mag[boundary_mask].mean())
            else:
                edge_sharpness = 0.0

        artificiality_score = 0.0
        if shape in ["CYLINDRICAL", "PLANAR"]:
            artificiality_score += 0.4
        if edge_sharpness > 0.3:
            artificiality_score += 0.3
        if min_ws < 500:  # Very low wave speed = likely air/void
            artificiality_score += 0.3
        artificiality_score = min(artificiality_score, 1.0)

        depth_fraction = float(np.clip(centroid_physical[2] / max(max_depth, 1e-9), 0.0, 1.0))
        bottom_depth_fraction = float(np.clip(depth_max / max(max_depth, 1e-9), 0.0, 1.0))
        depth_priority_score = 0.65 * depth_fraction + 0.35 * bottom_depth_fraction

        speed_drop_score = float(
            np.clip(
                (cfg["background_wave_speed"] - mean_ws)
                / max(cfg["background_wave_speed"] - cfg.get("min_wave_speed", 300.0), 1e-9),
                0.0,
                1.0,
            )
        )
        edge_score = float(np.clip(edge_sharpness / 0.3, 0.0, 1.0))
        void_evidence_score = float(
            np.clip(
                0.45 * mean_vp
                + 0.25 * speed_drop_score
                + 0.15 * edge_score
                + 0.15 * artificiality_score,
                0.0,
                1.0,
            )
        )
        deep_score_boost = max(float(cfg.get("deep_score_boost", 0.75)), 0.0)
        deep_target_score = float(
            void_evidence_score
            * (1.0 + deep_score_boost * depth_priority_score)
            / (1.0 + deep_score_boost)
        )

        anomaly = {
            "id": label_id,
            "centroid_m": centroid_physical.tolist(),
            "depth_m": float(centroid_physical[2]),
            "depth_range_m": [float(depth_min), float(depth_max)],
            "extent_m": [float(extent_x), float(extent_y), float(extent_z)],
            "volume_m3": float(volume_m3),
            "voxel_count": int(voxel_count),
            "mean_wave_speed_ms": float(mean_ws),
            "min_wave_speed_ms": float(min_ws),
            "mean_void_probability": float(mean_vp),
            "shape_classification": shape,
            "shape_note": shape_note,
            "aspect_ratios": [float(aspect_1), float(aspect_2)],
            "edge_sharpness": float(edge_sharpness),
            "artificiality_score": float(artificiality_score),
            "depth_priority_score": float(depth_priority_score),
            "void_evidence_score": float(void_evidence_score),
            "deep_target_score": float(deep_target_score),
        }

        # --- Cap 6: Surface embedding corroboration ---
        # Sample the 2-D anomaly map at the body's centroid (ix, iy) position.
        # The centroid is already in physical meters from domain center, so we
        # convert back to [0,1] fractional pixel coords.
        surface_anomaly_at_centroid: Optional[float] = None
        fused_confidence_score: float = deep_target_score  # default: no change
        if embedding_anomaly_map is not None:
            em_h, em_w = embedding_anomaly_map.shape
            # centroid_voxel: (iz, iy, ix) in voxel space
            frac_x = float(centroid_voxel[2]) / max(nx - 1, 1)  # [0,1]
            frac_y = float(centroid_voxel[1]) / max(ny - 1, 1)
            px_col = int(np.clip(round(frac_x * (em_w - 1)), 0, em_w - 1))
            px_row = int(np.clip(round(frac_y * (em_h - 1)), 0, em_h - 1))
            val = float(embedding_anomaly_map[px_row, px_col])
            if np.isfinite(val):
                surface_anomaly_at_centroid = val
                # Factor: 0.7 at surface_anomaly=0, 1.0 at surface_anomaly=1
                surface_factor = 0.70 + 0.30 * np.clip(val, 0.0, 1.0)
                fused_confidence_score = float(deep_target_score * surface_factor)

        anomaly["surface_anomaly_at_centroid"] = surface_anomaly_at_centroid
        anomaly["fused_confidence_score"] = fused_confidence_score
        anomalies.append(anomaly)

    # Rank by fused_confidence_score if embedding map was provided,
    # otherwise fall back to deep_target_score (identical when no map).
    rank_key = "fused_confidence_score" if embedding_anomaly_map is not None else "deep_target_score"
    anomalies.sort(
        key=lambda a: (
            a[rank_key],
            a["depth_m"],
            a["mean_void_probability"],
            a["volume_m3"],
        ),
        reverse=True,
    )
    for rank, anomaly in enumerate(anomalies, start=1):
        anomaly["deep_target_rank"] = rank

    logger.info(f"Characterized {len(anomalies)} significant anomalies")
    for a in anomalies[:10]:
        fused = a.get("fused_confidence_score", a["deep_target_score"])
        surf = a.get("surface_anomaly_at_centroid")
        surf_str = f"{float(surf):.3f}" if surf is not None else "n/a"
        logger.info(
            f"  Rank {a['deep_target_rank']} anomaly {a['id']}: depth={a['depth_m']:.0f}m, "
            f"shape={a['shape_classification']}, "
            f"vol={a['volume_m3']:.0f}m^3, "
            f"deep_score={a['deep_target_score']:.3f}, "
            f"fused={fused:.3f}, surface_anomaly={surf_str}"
        )

    return anomalies


# ============================================================
# 2. 3D Visualization with PyVista
# ============================================================
def render_3d_subsurface(
    wave_speed: np.ndarray,
    void_probability: np.ndarray,
    anomalies: List[Dict],
    output_dir: str,
    config: Optional[Dict] = None,
    interactive: bool = False,
) -> Dict[str, str]:
    """
    Generate 3D visualization of subsurface structure using PyVista.

    Produces:
    - 3D isosurface rendering of detected voids/structures
    - Volumetric rendering with opacity mapping
    - Cross-sectional slices at specified depths
    - Exported mesh files (VTK, STL)

    Parameters
    ----------
    wave_speed : np.ndarray
        3D wave speed volume (nz, ny, nx).
    void_probability : np.ndarray
        3D void probability volume (nz, ny, nx).
    anomalies : list of dict
        Detected anomaly bodies.
    output_dir : str
        Output directory.
    config : dict, optional
        Visualization config.
    interactive : bool
        If True, opens interactive 3D viewer. If False, saves screenshots.

    Returns
    -------
    dict
        Paths to generated visualizations and meshes.
    """
    cfg = DEFAULT_VIZ_CONFIG.copy()
    if config:
        cfg.update(config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = {"void_contour_points": 0, "void_contour_cells": 0}
    threshold_diagnostics = _threshold_component_diagnostics(
        void_probability,
        threshold=float(cfg["void_threshold"]),
        min_component_voxels=int(cfg.get("min_anomaly_voxels", DEFAULT_VIZ_CONFIG["min_anomaly_voxels"])),
    )
    wave_speed_stats = _array_stats(wave_speed)
    void_probability_stats = _array_stats(void_probability, threshold=float(cfg["void_threshold"]))

    nz, ny, nx = wave_speed.shape
    max_depth = cfg["max_depth_m"]
    domain_width = cfg["domain_width_m"]

    if bool(cfg.get("enable_pyvista", True)):
        try:
            import pyvista as pv
            pv.set_plot_theme("dark")
            HAS_PYVISTA = True
        except ImportError:
            logger.warning("PyVista not installed. Falling back to matplotlib cross-sections only.")
            HAS_PYVISTA = False
    else:
        logger.info("PyVista rendering disabled by config; generating lightweight audit plots/report only.")
        HAS_PYVISTA = False

    if HAS_PYVISTA:
        # Create structured grid
        x = np.linspace(-domain_width / 2, domain_width / 2, nx)
        y = np.linspace(-domain_width / 2, domain_width / 2, ny)
        z = np.linspace(0, -max_depth, nz)  # Negative = depth below surface

        grid = pv.RectilinearGrid(x, y, z)
        grid["wave_speed"] = wave_speed.flatten(order="C")
        grid["void_probability"] = void_probability.flatten(order="C")

        # Wave speed anomaly (deviation from background)
        ws_anomaly = wave_speed - cfg["background_wave_speed"]
        grid["wave_speed_anomaly"] = ws_anomaly.flatten(order="C")

        # === Isosurface Render ===
        logger.info("Rendering isosurfaces...")
        plotter = pv.Plotter(off_screen=not interactive, window_size=[1920, 1080])
        plotter.set_background("black")

        # Extract void isosurfaces at multiple thresholds
        for thresh, color, opacity in [
            (0.8, "red", 0.9),
            (0.6, "orange", 0.6),
            (0.4, "yellow", 0.3),
        ]:
            try:
                contour = grid.contour(
                    isosurfaces=[thresh],
                    scalars="void_probability",
                )
                if contour.n_points > 0:
                    plotter.add_mesh(
                        contour, color=color, opacity=opacity,
                        label=f"Void prob > {thresh}"
                    )
            except Exception as e:
                logger.debug(f"No isosurface at threshold {thresh}: {e}")

        # Add surface plane (semi-transparent)
        surface_plane = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=domain_width,
            j_size=domain_width,
        )
        plotter.add_mesh(surface_plane, color="green", opacity=0.1, label="Surface")

        # Add anomaly markers
        for anom in anomalies[:10]:
            cx, cy, cz = anom["centroid_m"]
            sphere = pv.Sphere(
                radius=min(anom["extent_m"]) / 4,
                center=(cx, cy, -cz),  # Negative depth for visualization
            )
            color = "cyan" if anom["artificiality_score"] > 0.5 else "magenta"
            plotter.add_mesh(
                sphere, color=color, opacity=0.8,
                label=f"Anomaly {anom['id']}: {anom['shape_classification']}"
            )

        plotter.add_legend()
        plotter.add_axes()
        plotter.add_title(
            "SAR Doppler Tomography - 3D Subsurface Structure",
            font_size=14,
        )

        # Camera position: bird's eye at angle
        plotter.camera_position = [
            (domain_width * 1.5, domain_width * 1.5, max_depth * 0.5),
            (0, 0, -max_depth / 2),
            (0, 0, 1),
        ]

        if interactive:
            plotter.show()
        else:
            iso_path = Path(output_dir) / "isosurface_3d.png"
            plotter.screenshot(str(iso_path))
            outputs["isosurface_render"] = str(iso_path)
            logger.info(f"Saved isosurface render: {iso_path}")

        plotter.close()

        # === Save VTK grid for external viewers ===
        vtk_path = Path(output_dir) / "subsurface_volume.vtk"
        grid.save(str(vtk_path))
        outputs["vtk_volume"] = str(vtk_path)
        logger.info(f"Saved VTK volume: {vtk_path}")

        # === Extract and save STL mesh of primary void ===
        try:
            primary_contour = grid.contour(
                isosurfaces=[cfg["void_threshold"]],
                scalars="void_probability",
            )
            contour_points = int(getattr(primary_contour, "n_points", 0) or 0)
            contour_cells = int(getattr(primary_contour, "n_cells", 0) or 0)
            outputs["void_contour_points"] = contour_points
            outputs["void_contour_cells"] = contour_cells
            if contour_points > 0 and anomalies:
                stl_path = Path(output_dir) / "void_surface.stl"
                primary_contour.save(str(stl_path))
                outputs["void_mesh_stl"] = str(stl_path)
                logger.info(f"Saved void mesh STL: {stl_path}")
            elif contour_points > 0:
                logger.info(
                    "Primary void contour exists but no significant anomalies were detected; "
                    "not exporting a positive void mesh."
                )
        except Exception as e:
            outputs["void_contour_error"] = str(e)
            logger.warning(f"Could not extract STL: {e}")

    # ============================================================
    # 3. Depth Cross-Sections (matplotlib - always available)
    # ============================================================
    logger.info("Generating depth cross-sections...")
    plt = _get_matplotlib_pyplot()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "SAR Doppler Tomography - Depth Cross-Sections",
        fontsize=16, fontweight="bold"
    )

    depth_values = np.linspace(0, max_depth, nz)

    for idx, depth_m in enumerate(cfg["depth_slices"][:6]):
        ax = axes[idx // 3, idx % 3]

        # Find nearest depth slice
        iz = np.argmin(np.abs(depth_values - depth_m))

        # Wave speed at this depth
        slice_data = wave_speed[iz]
        anomaly_slice = void_probability[iz]

        # Plot wave speed as background. Slices are (ny, nx): rows map to the
        # y axis and columns to the x axis, matching the centroid convention,
        # so no transpose — transposing would swap axes under the markers.
        im = ax.imshow(
            slice_data,
            extent=[-domain_width/2, domain_width/2, -domain_width/2, domain_width/2],
            cmap="RdYlBu",
            origin="lower",
            vmin=cfg["min_wave_speed"] if "min_wave_speed" in cfg else 300,
            vmax=cfg["max_wave_speed"] if "max_wave_speed" in cfg else 6000,
        )

        # Overlay void probability contours
        if anomaly_slice.max() > 0.3:
            x_extent = np.linspace(-domain_width/2, domain_width/2, nx)
            y_extent = np.linspace(-domain_width/2, domain_width/2, ny)
            ax.contour(
                x_extent, y_extent, anomaly_slice,
                levels=[0.3, 0.5, 0.7, 0.9],
                colors=["yellow", "orange", "red", "white"],
                linewidths=[0.5, 1.0, 1.5, 2.0],
            )

        # Mark anomaly centroids at this depth
        for anom in anomalies:
            anom_depth = anom["depth_m"]
            if abs(anom_depth - depth_m) < max_depth / nz * 2:
                cx, cy, _ = anom["centroid_m"]
                marker = "^" if anom["artificiality_score"] > 0.5 else "o"
                color = "cyan" if anom["artificiality_score"] > 0.5 else "magenta"
                ax.plot(cx, cy, marker, color=color, markersize=8, markeredgecolor="white")

        ax.set_title(f"Depth: {depth_m}m", fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(im, ax=ax, label="Wave Speed (m/s)", shrink=0.8)

    plt.tight_layout()
    cross_path = Path(output_dir) / "depth_cross_sections.png"
    plt.savefig(str(cross_path), dpi=200, bbox_inches="tight")
    plt.close()
    outputs["cross_sections"] = str(cross_path)
    logger.info(f"Saved cross-sections: {cross_path}")

    # ============================================================
    # 4. Anomaly Report
    # ============================================================
    report_path = Path(output_dir) / "anomaly_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SAR DOPPLER TOMOGRAPHY - SUBSURFACE ANOMALY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Domain: {domain_width:.0f} x {domain_width:.0f} x {max_depth:.0f} meters\n")
        f.write(f"Volume shape: {wave_speed.shape}\n")
        f.write(f"Background wave speed: {cfg['background_wave_speed']:.0f} m/s\n")
        f.write(f"Void threshold: {cfg['void_threshold']:.2f}\n")
        f.write("Ranking: deep_target_score prioritizes deeper connected void/object bodies while retaining evidence strength\n")
        f.write(f"Total anomalies detected: {len(anomalies)}\n\n")

        vp_quantiles = void_probability_stats.get("quantiles", {})
        ws_quantiles = wave_speed_stats.get("quantiles", {})
        f.write("Calibration diagnostics (configured thresholds; not post-hoc tuned)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Void probability range: {void_probability_stats['min']:.6f} to {void_probability_stats['max']:.6f}\n")
        f.write(
            "  Void probability quantiles: "
            f"p50={vp_quantiles.get('p50', float('nan')):.6f}, "
            f"p90={vp_quantiles.get('p90', float('nan')):.6f}, "
            f"p95={vp_quantiles.get('p95', float('nan')):.6f}, "
            f"p99={vp_quantiles.get('p99', float('nan')):.6f}\n"
        )
        f.write(f"  Wave speed range: {wave_speed_stats['min']:.3f} to {wave_speed_stats['max']:.3f} m/s\n")
        f.write(
            "  Wave speed quantiles (m/s): "
            f"p01={ws_quantiles.get('p01', float('nan')):.3f}, "
            f"p05={ws_quantiles.get('p05', float('nan')):.3f}, "
            f"p50={ws_quantiles.get('p50', float('nan')):.3f}, "
            f"p95={ws_quantiles.get('p95', float('nan')):.3f}\n"
        )
        f.write(f"  Voxels > threshold: {threshold_diagnostics['voxels_crossing_threshold']}\n")
        f.write(f"  Connected components > threshold: {threshold_diagnostics['connected_components_crossing_threshold']}\n")
        f.write(f"  Largest component voxels: {threshold_diagnostics['largest_component_voxels']}\n")
        f.write(f"  Components meeting min_anomaly_voxels: {threshold_diagnostics['components_meeting_min_voxels']}\n")
        f.write(f"  Any voxel crosses threshold: {threshold_diagnostics['any_voxel_crosses_threshold']}\n")
        f.write(f"  Any component survives size filter: {threshold_diagnostics['any_component_meets_min_voxels']}\n\n")

        for anom in anomalies:
            f.write("-" * 50 + "\n")
            f.write(f"RANK #{anom['deep_target_rank']} - ANOMALY #{anom['id']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Shape: {anom['shape_classification']} ({anom['shape_note']})\n")
            f.write(f"  Depth: {anom['depth_m']:.0f}m (range: {anom['depth_range_m'][0]:.0f}-{anom['depth_range_m'][1]:.0f}m)\n")
            f.write(f"  Centroid: ({anom['centroid_m'][0]:.0f}, {anom['centroid_m'][1]:.0f}, {anom['centroid_m'][2]:.0f}) m\n")
            f.write(f"  Extent: {anom['extent_m'][0]:.0f} x {anom['extent_m'][1]:.0f} x {anom['extent_m'][2]:.0f} m\n")
            f.write(f"  Volume: {anom['volume_m3']:.0f} m^3\n")
            f.write(f"  Mean wave speed: {anom['mean_wave_speed_ms']:.0f} m/s\n")
            f.write(f"  Min wave speed: {anom['min_wave_speed_ms']:.0f} m/s\n")
            f.write(f"  Void probability: {anom['mean_void_probability']:.3f}\n")
            f.write(f"  Deep target score: {anom['deep_target_score']:.3f}\n")
            fused = anom.get('fused_confidence_score', anom['deep_target_score'])
            surf = anom.get('surface_anomaly_at_centroid')
            if surf is not None:
                f.write(f"  Fused confidence score: {fused:.3f} (surface anomaly: {float(surf):.3f})\n")
            f.write(f"  Void evidence score: {anom['void_evidence_score']:.3f}\n")
            f.write(f"  Depth priority score: {anom['depth_priority_score']:.3f}\n")
            f.write(f"  Artificiality score: {anom['artificiality_score']:.2f}\n")
            f.write(f"  Aspect ratios: {anom['aspect_ratios']}\n")
            f.write(f"  Edge sharpness: {anom['edge_sharpness']:.4f}\n\n")

    outputs["anomaly_report"] = str(report_path)
    logger.info(f"Saved anomaly report: {report_path}")

    return outputs


# ============================================================
# 5. Full Visualization Pipeline
# ============================================================
def run_visualization_pipeline(
    wave_speed_path: str,
    output_dir: Optional[str] = None,
    void_probability_path: Optional[str] = None,
    density_contrast_path: Optional[str] = None,
    config: Optional[Dict] = None,
    interactive: bool = False,
    embedding_anomaly_map: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Run the complete 3D visualization pipeline.

    Parameters
    ----------
    wave_speed_path : str
        Path to wave speed volume (.npy).
    output_dir : str, optional
        Output directory.
    void_probability_path : str, optional
        Path to void probability volume. Auto-computed if not provided.
    density_contrast_path : str, optional
        Path to density contrast volume.
    config : dict, optional
        Configuration overrides.
    interactive : bool
        Open interactive 3D viewer.
    embedding_anomaly_map : np.ndarray, optional
        Cap 6: 2-D surface embedding anomaly score map to pass to
        extract_anomaly_bodies for fused confidence scoring.

    Returns
    -------
    dict
        All output file paths.
    """
    cfg = DEFAULT_VIZ_CONFIG.copy()
    if config:
        cfg.update(config)

    # FAILSAFE DOMAIN BUG FIX: Auto-parse metadata from PINN output
    # If domain_width_m/max_depth_m weren't passed in config (old codepath),
    # read them from the inversion_metadata.txt produced by the PINN.
    wave_speed_file = Path(wave_speed_path)
    metadata_file = wave_speed_file.parent / "inversion_metadata.txt"
    if metadata_file.exists():
        logger.info(f"Loading domain config from {metadata_file}")
        with open(metadata_file, 'r') as mf:
            for line in mf:
                line = line.strip()
                if line.startswith("max_depth_m:") and "max_depth_m" not in (config or {}):
                    cfg["max_depth_m"] = float(line.split(":")[1].strip())
                elif line.startswith("domain_width_m:") and "domain_width_m" not in (config or {}):
                    cfg["domain_width_m"] = float(line.split(":")[1].strip())
                elif line.startswith("background_wave_speed_ms:") and "background_wave_speed" not in (config or {}):
                    cfg["background_wave_speed"] = float(line.split(":")[1].strip())
        logger.info(f"  Domain: {cfg.get('domain_width_m')}m wide x {cfg.get('max_depth_m')}m deep")

    if output_dir is None:
        output_dir = str(VIZ_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load volumes
    logger.info(f"Loading wave speed volume: {wave_speed_path}")
    wave_speed = np.load(wave_speed_path)
    logger.info(f"  Shape: {wave_speed.shape}, range: [{wave_speed.min():.0f}, {wave_speed.max():.0f}] m/s")

    void_probability_source = "file"
    void_probability_manifest_path = void_probability_path
    if void_probability_path and os.path.exists(void_probability_path):
        void_prob = np.load(void_probability_path)
    else:
        # Compute from wave speed using Sigmoid mapping
        # (Matches pinn_vibro_inversion.py - see Wyllie's Time-Average Equation)
        logger.info("Computing void probability from wave speed (Sigmoid mapping)...")
        void_prob = compute_void_probability_from_wave_speed(
            wave_speed,
            background_wave_speed=cfg.get("background_wave_speed", 3500.0),
            void_speed_threshold_ratio=cfg.get("void_speed_threshold_ratio", 0.7),
            temperature_ratio=cfg.get("void_probability_temperature_ratio", 0.05),
        )
        void_probability_source = "computed_from_wave_speed"
        computed_void_probability_path = Path(output_dir) / "void_probability_volume.npy"
        np.save(computed_void_probability_path, void_prob)
        void_probability_manifest_path = str(computed_void_probability_path)
        logger.info(f"Saved computed void probability volume: {computed_void_probability_path}")

    if density_contrast_path and os.path.exists(density_contrast_path):
        density_contrast = np.load(density_contrast_path)
    else:
        density_contrast = None

    # Extract anomalies
    logger.info("Extracting anomaly bodies...")
    anomalies = extract_anomaly_bodies(
        void_prob, wave_speed, config=cfg, embedding_anomaly_map=embedding_anomaly_map
    )

    # Render
    logger.info("Generating 3D visualizations...")
    viz_outputs = render_3d_subsurface(
        wave_speed, void_prob, anomalies,
        output_dir, config=cfg, interactive=interactive,
    )

    viz_outputs["anomaly_list"] = anomalies
    viz_outputs["anomaly_count"] = len(anomalies)
    viz_outputs["anomalies_detected"] = len(anomalies)
    if void_probability_manifest_path:
        viz_outputs["void_probability_volume"] = str(void_probability_manifest_path)

    # Save anomaly list as CSV
    if anomalies:
        import csv
        csv_path = Path(output_dir) / "detected_anomalies.csv"
        fieldnames = list(anomalies[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for a in anomalies:
                # Convert lists to strings for CSV
                row = {k: str(v) if isinstance(v, list) else v for k, v in a.items()}
                writer.writerow(row)
        viz_outputs["anomalies_csv"] = str(csv_path)
        viz_outputs["detected_anomalies_csv"] = str(csv_path)

    # Remove positive-looking artifacts from previous runs when the current run
    # has no significant detections and/or no current mesh contour at threshold.
    contour_points = int(viz_outputs.get("void_contour_points", 0) or 0)
    cleanup_reasons: List[str] = []
    if not anomalies:
        cleanup_reasons.append("no_anomalies")
    if contour_points <= 0:
        cleanup_reasons.append("no_contour_points")

    cleanup_reason = ",".join(cleanup_reasons) if cleanup_reasons else None
    cleanup_removed: List[str] = []
    if cleanup_reasons:
        cleanup_removed = _cleanup_stale_positive_artifacts(
            output_dir=output_dir,
            remove_catalogs=not anomalies,
            remove_meshes=(not anomalies or contour_points <= 0),
            reason=cleanup_reason,
        )
    viz_outputs["stale_artifacts_removed"] = cleanup_removed
    if cleanup_reason:
        viz_outputs["stale_artifact_cleanup_reason"] = cleanup_reason

    audit_manifest_path = write_audit_manifest(
        output_dir=output_dir,
        wave_speed_path=wave_speed_path,
        wave_speed=wave_speed,
        void_probability_path=void_probability_manifest_path if void_probability_manifest_path and os.path.exists(void_probability_manifest_path) else None,
        void_probability=void_prob,
        density_contrast_path=density_contrast_path if density_contrast_path and os.path.exists(density_contrast_path) else None,
        density_contrast=density_contrast,
        config=cfg,
        outputs=viz_outputs,
        anomalies=anomalies,
        cleanup_removed=cleanup_removed,
        cleanup_reason=cleanup_reason,
        void_probability_source=void_probability_source,
    )
    viz_outputs["audit_manifest"] = audit_manifest_path

    return viz_outputs


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Subsurface Visualization and Anomaly Extraction"
    )
    parser.add_argument("--volume", required=True, help="Wave speed volume (.npy)")
    parser.add_argument("--void-prob", default=None, help="Void probability volume (.npy)")
    parser.add_argument("--density", default=None, help="Density contrast volume (.npy)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_VIZ_CONFIG["void_threshold"], help="Void threshold")
    parser.add_argument("--depth", type=float, default=DEFAULT_VIZ_CONFIG["max_depth_m"], help="Max depth (m)")
    parser.add_argument("--width", type=float, default=DEFAULT_VIZ_CONFIG["domain_width_m"], help="Domain width (m)")
    parser.add_argument("--bg-speed", type=float, default=DEFAULT_VIZ_CONFIG["background_wave_speed"], help="Background wave speed (m/s)")
    parser.add_argument("--interactive", action="store_true", help="Open interactive 3D viewer")

    args = parser.parse_args()

    config = {
        "void_threshold": args.threshold,
        "max_depth_m": args.depth,
        "domain_width_m": args.width,
        "background_wave_speed": args.bg_speed,
    }

    outputs = run_visualization_pipeline(
        args.volume,
        output_dir=args.output_dir,
        void_probability_path=args.void_prob,
        density_contrast_path=args.density,
        config=config,
        interactive=args.interactive,
    )

    print("\nOutputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
