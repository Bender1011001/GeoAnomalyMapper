"""
GeoAnomalyMapper V2 - Scientific Core Implementation
High-level functional code for the master implementation guide.

This file provides the core functions to be integrated into the existing
GeoAnomalyMapper codebase by the development team.

- uncertainty_fusion: Implements bootstrap ensemble for uncertainty quantification.
- validation_metrics: Implements PR/ROC curve generation and calibration plots.
- resolution_governance: Implements resolution capping and metadata sidecars.
- insar_tools: Implements automated InSAR quality control.
- advanced_denoising: Implements tiled TV denoising.

"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Scikit-learn for validation and calibration
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

# Scikit-image for denoising
from skimage.restoration import denoise_tv_chambolle

# Assume existing functions are available for import
# from multi_resolution_fusion import fuse_weighted
# from utils.paths import get_raster_metadata, save_raster

# --- Placeholder for demonstration ---
def fuse_weighted(layers: List[np.ndarray]) -> np.ndarray:
    """Placeholder for the existing fuse_weighted function."""
    if not layers:
        return np.array([])
    # Simple average for demonstration
    return np.nanmean(np.stack(layers), axis=0)

def save_raster(path: Path, data: np.ndarray, profile: Dict):
    """Placeholder for saving a raster."""
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
# --- End Placeholder ---

# -----------------------------------------------------------------------------
# 1. Uncertainty-Aware Fusion (to be integrated into multi_resolution_fusion.py)
# -----------------------------------------------------------------------------

def uncertainty_fusion(
    layers: List[np.ndarray],
    profile: Dict,
    output_path_mean: Path,
    output_path_stddev: Path,
    n_iterations: int = 32,
    tile_size: int = 1024
):
    """
    Performs bootstrap ensemble fusion on tiled data to quantify uncertainty.

    Args:
        layers (List[np.ndarray]): List of input data layers as numpy arrays.
        profile (Dict): Rasterio profile for output rasters.
        output_path_mean (Path): Path to save the mean fused raster.
        output_path_stddev (Path): Path to save the std. dev. (uncertainty) raster.
        n_iterations (int): Number of bootstrap iterations.
        tile_size (int): The size of tiles to process in memory.
    """
    height, width = layers[0].shape
    
    # Create empty output rasters
    with rasterio.open(output_path_mean, 'w', **profile) as dst_mean, \
         rasterio.open(output_path_stddev, 'w', **profile) as dst_stddev:

        for i in tqdm(range(0, height, tile_size), desc="Tiled Uncertainty Fusion"):
            for j in range(0, width, tile_size):
                window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                
                tile_layers = [layer[window.row_off:window.row_off + window.height,
                                     window.col_off:window.col_off + window.width] for layer in layers]
                
                bootstrap_results = []
                for _ in range(n_iterations):
                    # Sample layers with replacement
                    indices = np.random.choice(len(tile_layers), len(tile_layers), replace=True)
                    sample_layers = [tile_layers[k] for k in indices]
                    
                    fused_tile = fuse_weighted(sample_layers)
                    bootstrap_results.append(fused_tile)
                
                stacked_results = np.stack(bootstrap_results, axis=0)
                
                # Calculate mean and std dev for the tile
                mean_tile = np.nanmean(stacked_results, axis=0).astype(profile['dtype'])
                stddev_tile = np.nanstd(stacked_results, axis=0).astype(profile['dtype'])
                
                dst_mean.write(mean_tile, window=window, indexes=1)
                dst_stddev.write(stddev_tile, window=window, indexes=1)

    print(f"Uncertainty fusion complete. Mean saved to {output_path_mean}, StdDev to {output_path_stddev}")


# -----------------------------------------------------------------------------
# 2. Enhanced Validation & Calibration (to be integrated into validate_against_known_features.py)
# -----------------------------------------------------------------------------

def generate_validation_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_dir: Path
):
    """
    Generates and saves PR/ROC curves and a calibration plot.

    Args:
        y_true (np.ndarray): Array of true binary labels (0 or 1).
        y_scores (np.ndarray): Array of predicted scores or probabilities.
        output_dir (Path): Directory to save the output plots.
    """
    output_dir.mkdir(exist_ok=True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_dir / "precision_recall_curve.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    # Calibration Plot
    plt.figure(figsize=(10, 10))
    ax_cal = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_hist = plt.subplot2grid((3, 1), (2, 0))
    disp = CalibrationDisplay.from_predictions(y_true, y_scores, n_bins=10, ax=ax_cal)
    ax_hist.hist(disp.y_prob, range=(0, 1), bins=10, histtype="step", lw=2)
    plt.suptitle("Calibration Plot")
    plt.savefig(output_dir / "calibration_plot.png")
    plt.close()
    
    print(f"Validation metrics saved to {output_dir}")


# -----------------------------------------------------------------------------
# 3. Resolution & Provenance Governance (to be integrated into fusion and processing scripts)
# -----------------------------------------------------------------------------

class ResolutionWarning(Warning):
    pass

def check_and_cap_resolution(
    input_resolutions: List[float],
    requested_resolution: float,
    force: bool = False
) -> float:
    """
    Checks if requested resolution is valid and caps it if necessary.
    """
    coarsest_res = max(input_resolutions)
    # Allow oversampling by a factor of 2, but warn.
    if requested_resolution < coarsest_res / 2:
        if force:
            print(f"Warning: Requested resolution ({requested_resolution}) is much finer than the coarsest input ({coarsest_res}). Forcing as requested.")
            return requested_resolution
        else:
            capped_res = coarsest_res / 2
            raise ResolutionWarning(
                f"Requested resolution ({requested_resolution}) is too fine. "
                f"Capping at {capped_res} based on coarsest input. Use --force-resolution to override."
            )
    return requested_resolution

def write_raster_with_metadata(
    raster_path: Path,
    data: np.ndarray,
    profile: Dict,
    metadata: Dict[str, Any]
):
    """
    Saves a raster and an accompanying JSON metadata file.
    """
    save_raster(raster_path, data, profile)
    metadata_path = raster_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Raster saved to {raster_path} with metadata at {metadata_path}")


# -----------------------------------------------------------------------------
# 4. InSAR Tools (to be its own module in utils/insar_tools.py)
# -----------------------------------------------------------------------------

def apply_coherence_mask(
    displacement_map: np.ndarray,
    coherence_map: np.ndarray,
    threshold: float = 0.4
) -> np.ndarray:
    """
    Masks displacement values where coherence is below a threshold.
    """
    masked_displacement = displacement_map.copy()
    masked_displacement[coherence_map < threshold] = np.nan
    return masked_displacement

def project_los_to_vertical(
    los_displacement: np.ndarray,
    incidence_angle_deg: np.ndarray
) -> np.ndarray:
    """
    Projects Line-of-Sight (LOS) displacement to vertical displacement.
    Assumes incidence angle is in degrees.
    """
    incidence_angle_rad = np.deg2rad(incidence_angle_deg)
    vertical_displacement = los_displacement / np.cos(incidence_angle_rad)
    return vertical_displacement


# -----------------------------------------------------------------------------
# 5. Advanced Denoising (to be integrated into multi_resolution_fusion.py)
# -----------------------------------------------------------------------------

def denoise_raster_tiled(
    input_path: Path,
    output_path: Path,
    weight: float = 0.1,
    tile_size: int = 512
):
    """
    Applies Total Variation (TV) denoising to a raster in a tiled manner.
    
    Args:
        input_path (Path): Path to the input raster.
        output_path (Path): Path to save the denoised raster.
        weight (float): Denoising weight. Higher values mean more smoothing.
        tile_size (int): Tile size for processing to manage memory.
    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in tqdm(range(0, src.height, tile_size), desc="Tiled Denoising"):
                for j in range(0, src.width, tile_size):
                    window = Window(j, i, min(tile_size, src.width - j), min(tile_size, src.height - i))
                    tile = src.read(1, window=window)
                    
                    # Handle NaN values by replacing with mean for denoising, then putting back
                    nan_mask = np.isnan(tile)
                    tile_mean = np.nanmean(tile)
                    tile_filled = np.nan_to_num(tile, nan=tile_mean)
                    
                    denoised_tile = denoise_tv_chambolle(tile_filled, weight=weight)
                    
                    denoised_tile[nan_mask] = np.nan
                    
                    dst.write(denoised_tile.astype(profile['dtype']), window=window, indexes=1)
    
    print(f"Denoised raster saved to {output_path}")
