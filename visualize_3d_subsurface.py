#!/usr/bin/env python3
"""
3D Subsurface Visualization and Target Extraction
===================================================

This module provides 3D volumetric visualization and anomaly extraction
for the Biondi SAR Doppler Tomography pipeline. It takes the 3D density
and wave-speed volumes produced by pinn_vibro_inversion.py and:

1. Extracts 3D isosurfaces of voids, dense bodies, and anomalous structures
   using the Marching Cubes algorithm
2. Classifies anomalies by shape (spherical, cylindrical, planar → natural vs artificial)
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
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from project_paths import DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# Configuration
# ============================================================
VIZ_DIR = DATA_DIR / "visualization_3d"

DEFAULT_VIZ_CONFIG = {
    "void_threshold": 0.5,          # Void probability threshold for isosurface
    "dense_threshold": 0.7,         # Dense body threshold (wave speed > background * factor)
    "min_anomaly_voxels": 10,       # Minimum voxels for a detected anomaly
    "smoothing_sigma": 0.5,         # Gaussian smoothing before isosurface extraction
    "max_depth_m": 2000.0,          # From inversion config
    "domain_width_m": 5000.0,       # From inversion config
    "background_wave_speed": 3500.0,
    "depth_slices": [50, 100, 200, 500, 1000, 1500],  # Depth cross-sections (meters)
}


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

    Returns
    -------
    list of dict
        Each detected anomaly with properties:
        - id, centroid, volume_m3, depth_range_m
        - mean_wave_speed, shape_classification
        - bounding_box, aspect_ratios
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

    # Morphological cleanup (remove single-voxel noise)
    binary_clean = binary_erosion(binary_voids, iterations=1)
    binary_clean = binary_dilation(binary_clean, iterations=1)

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
            centroid_voxel[2] * dx - domain_width / 2,  # x (meters from center)
            centroid_voxel[1] * dy - domain_width / 2,  # y
            centroid_voxel[0] * dz,                      # z (depth in meters)
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
        aspect_1 = extents[2] / max(extents[0], 1)  # longest / shortest
        aspect_2 = extents[1] / max(extents[0], 1)  # middle / shortest

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
        # Sharp boundaries + geometric shape → higher artificiality score
        gradient_magnitude = np.gradient(void_probability[
            bbox_min[0]:bbox_max[0]+1,
            bbox_min[1]:bbox_max[1]+1,
            bbox_min[2]:bbox_max[2]+1
        ])
        edge_sharpness = np.mean([np.abs(g).mean() for g in gradient_magnitude])

        artificiality_score = 0.0
        if shape in ["CYLINDRICAL", "PLANAR"]:
            artificiality_score += 0.4
        if edge_sharpness > 0.3:
            artificiality_score += 0.3
        if min_ws < 500:  # Very low wave speed = likely air/void
            artificiality_score += 0.3
        artificiality_score = min(artificiality_score, 1.0)

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
        }
        anomalies.append(anomaly)

    # Sort by void probability (highest first)
    anomalies.sort(key=lambda a: a["mean_void_probability"], reverse=True)

    logger.info(f"Characterized {len(anomalies)} significant anomalies")
    for a in anomalies[:10]:
        logger.info(
            f"  Anomaly {a['id']}: depth={a['depth_m']:.0f}m, "
            f"shape={a['shape_classification']}, "
            f"vol={a['volume_m3']:.0f}m³, "
            f"artificiality={a['artificiality_score']:.2f}"
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
    outputs = {}

    nz, ny, nx = wave_speed.shape
    max_depth = cfg["max_depth_m"]
    domain_width = cfg["domain_width_m"]

    try:
        import pyvista as pv
        pv.set_plot_theme("dark")
        HAS_PYVISTA = True
    except ImportError:
        logger.warning("PyVista not installed. Falling back to matplotlib cross-sections only.")
        HAS_PYVISTA = False

    if HAS_PYVISTA:
        # Create structured grid
        x = np.linspace(-domain_width / 2, domain_width / 2, nx)
        y = np.linspace(-domain_width / 2, domain_width / 2, ny)
        z = np.linspace(0, -max_depth, nz)  # Negative = depth below surface

        grid = pv.RectilinearGrid(x, y, z)
        grid["wave_speed"] = wave_speed.flatten(order="F")
        grid["void_probability"] = void_probability.flatten(order="F")

        # Wave speed anomaly (deviation from background)
        ws_anomaly = wave_speed - cfg["background_wave_speed"]
        grid["wave_speed_anomaly"] = ws_anomaly.flatten(order="F")

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
            "SAR Doppler Tomography — 3D Subsurface Structure",
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
            if primary_contour.n_points > 0:
                stl_path = Path(output_dir) / "void_surface.stl"
                primary_contour.save(str(stl_path))
                outputs["void_mesh_stl"] = str(stl_path)
                logger.info(f"Saved void mesh STL: {stl_path}")
        except Exception as e:
            logger.warning(f"Could not extract STL: {e}")

    # ============================================================
    # 3. Depth Cross-Sections (matplotlib — always available)
    # ============================================================
    logger.info("Generating depth cross-sections...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "SAR Doppler Tomography — Depth Cross-Sections",
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

        # Plot wave speed as background
        im = ax.imshow(
            slice_data.T,
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
                x_extent, y_extent, anomaly_slice.T,
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
        f.write("SAR DOPPLER TOMOGRAPHY — SUBSURFACE ANOMALY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Domain: {domain_width:.0f} x {domain_width:.0f} x {max_depth:.0f} meters\n")
        f.write(f"Volume shape: {wave_speed.shape}\n")
        f.write(f"Background wave speed: {cfg['background_wave_speed']:.0f} m/s\n")
        f.write(f"Void threshold: {cfg['void_threshold']:.2f}\n")
        f.write(f"Total anomalies detected: {len(anomalies)}\n\n")

        for anom in anomalies:
            f.write("-" * 50 + "\n")
            f.write(f"ANOMALY #{anom['id']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Shape: {anom['shape_classification']} ({anom['shape_note']})\n")
            f.write(f"  Depth: {anom['depth_m']:.0f}m (range: {anom['depth_range_m'][0]:.0f}-{anom['depth_range_m'][1]:.0f}m)\n")
            f.write(f"  Centroid: ({anom['centroid_m'][0]:.0f}, {anom['centroid_m'][1]:.0f}, {anom['centroid_m'][2]:.0f}) m\n")
            f.write(f"  Extent: {anom['extent_m'][0]:.0f} x {anom['extent_m'][1]:.0f} x {anom['extent_m'][2]:.0f} m\n")
            f.write(f"  Volume: {anom['volume_m3']:.0f} m³\n")
            f.write(f"  Mean wave speed: {anom['mean_wave_speed_ms']:.0f} m/s\n")
            f.write(f"  Min wave speed: {anom['min_wave_speed_ms']:.0f} m/s\n")
            f.write(f"  Void probability: {anom['mean_void_probability']:.3f}\n")
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

    if void_probability_path and os.path.exists(void_probability_path):
        void_prob = np.load(void_probability_path)
    else:
        # Compute from wave speed
        logger.info("Computing void probability from wave speed...")
        bg_speed = cfg["background_wave_speed"]
        min_speed = cfg.get("min_wave_speed", 300.0)
        void_prob = np.clip(
            1.0 - (wave_speed - min_speed) / (bg_speed - min_speed),
            0, 1
        )

    if density_contrast_path and os.path.exists(density_contrast_path):
        density_contrast = np.load(density_contrast_path)
    else:
        density_contrast = None

    # Extract anomalies
    logger.info("Extracting anomaly bodies...")
    anomalies = extract_anomaly_bodies(void_prob, wave_speed, config=cfg)

    # Render
    logger.info("Generating 3D visualizations...")
    viz_outputs = render_3d_subsurface(
        wave_speed, void_prob, anomalies,
        output_dir, config=cfg, interactive=interactive,
    )

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
    parser.add_argument("--threshold", type=float, default=0.5, help="Void threshold")
    parser.add_argument("--depth", type=float, default=2000.0, help="Max depth (m)")
    parser.add_argument("--width", type=float, default=5000.0, help="Domain width (m)")
    parser.add_argument("--bg-speed", type=float, default=3500.0, help="Background wave speed (m/s)")
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
