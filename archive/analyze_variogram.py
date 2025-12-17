#!/usr/bin/env python3
"""
Variogram Analysis Script for GeoAnomalyMapper

This script calculates the spatial autocorrelation range of geological data
(gravity and magnetic) using semi-variogram analysis. The calculated range
will be used as the buffer_radius in subsequent spatial analysis steps.

The script:
1. Loads gravity residual and magnetic raster data
2. Samples random points to avoid memory issues
3. Calculates empirical semi-variograms
4. Fits theoretical models (Spherical and Exponential)
5. Extracts the range parameter
6. Visualizes and saves results

Dependencies:
    - scikit-gstat (or falls back to numpy-based implementation)
    - rasterio, numpy, scipy, matplotlib
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    import rasterio
    from project_paths import DATA_DIR, OUTPUTS_DIR, PROCESSED_DIR, ensure_directories
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Try to import scikit-gstat, fallback to numpy implementation if unavailable
try:
    import skgstat as skg
    HAS_SKGSTAT = True
    logger.info("Using scikit-gstat for variogram analysis")
except ImportError:
    HAS_SKGSTAT = False
    logger.warning("scikit-gstat not available, using numpy-based implementation")


# ============================================================================
# Theoretical Variogram Models
# ============================================================================

def spherical_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """
    Spherical variogram model.
    
    γ(h) = nugget + (sill - nugget) * [1.5 * (h / range) - 0.5 * (h / range)^3]  for h <= range
    γ(h) = sill  for h > range
    
    Args:
        h: Lag distances (array).
        nugget: Nugget effect (variance at distance 0).
        sill: Sill (total variance at plateau).
        range_param: Range (distance where variance plateaus).
    
    Returns:
        Semi-variance values for each lag distance.
    """
    gamma = np.zeros_like(h, dtype=float)
    mask = h <= range_param
    
    if range_param > 0:
        gamma[mask] = nugget + (sill - nugget) * (
            1.5 * (h[mask] / range_param) - 0.5 * (h[mask] / range_param) ** 3
        )
    gamma[~mask] = sill
    
    return gamma


def exponential_model(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
    """
    Exponential variogram model.
    
    γ(h) = nugget + (sill - nugget) * [1 - exp(-3 * h / range)]
    
    Note: Effective range is typically defined at 95% of sill, which occurs at ~3*range_param
    
    Args:
        h: Lag distances (array).
        nugget: Nugget effect.
        sill: Sill (total variance).
        range_param: Practical range parameter.
    
    Returns:
        Semi-variance values for each lag distance.
    """
    if range_param <= 0:
        return np.full_like(h, sill, dtype=float)
    
    return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / range_param))


# ============================================================================
# Variogram Calculation (NumPy Implementation)
# ============================================================================

def calculate_empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 20,
    max_lag: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate empirical semi-variogram using numpy.
    
    γ(h) = (1 / (2 * N(h))) * Σ [z(x_i) - z(x_j)]^2
    
    where N(h) is the number of pairs separated by distance h.
    
    Args:
        coords: Nx2 array of coordinates (x, y).
        values: N-length array of values at each coordinate.
        n_lags: Number of lag bins.
        max_lag: Maximum lag distance to consider (auto-calculated if None).
    
    Returns:
        Tuple of (lag_distances, semi_variances, pair_counts).
    """
    logger.info("Calculating empirical variogram...")
    
    # Calculate pairwise distances
    distances = pdist(coords)
    
    if max_lag is None:
        max_lag = np.percentile(distances, 50)  # Use median distance as max lag
    
    # Calculate pairwise squared differences
    value_diffs = pdist(values.reshape(-1, 1))
    semi_variances_all = 0.5 * value_diffs ** 2
    
    # Create lag bins
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    
    # Bin the data
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)
    
    for i in range(n_lags):
        mask = (distances >= lag_bins[i]) & (distances < lag_bins[i + 1])
        if mask.any():
            gamma[i] = np.mean(semi_variances_all[mask])
            counts[i] = np.sum(mask)
    
    # Filter out empty bins
    valid = counts > 0
    
    return lag_centers[valid], gamma[valid], counts[valid]


def fit_variogram_model(
    lags: np.ndarray,
    gamma: np.ndarray,
    model_type: str = 'spherical'
) -> Dict[str, Any]:
    """
    Fit a theoretical variogram model to empirical data.
    
    Args:
        lags: Lag distances.
        gamma: Empirical semi-variances.
        model_type: Model type ('spherical' or 'exponential').
    
    Returns:
        Dictionary with fitted parameters and metrics.
    """
    logger.info(f"Fitting {model_type} model...")
    
    # Initial parameter estimates
    nugget_init = gamma[0] if len(gamma) > 0 else 0.0
    sill_init = np.percentile(gamma, 95)
    range_init = lags[np.argmax(gamma >= 0.95 * sill_init)] if len(lags) > 0 else lags[-1] / 2
    
    # Select model function
    if model_type.lower() == 'spherical':
        model_func = spherical_model
    elif model_type.lower() == 'exponential':
        model_func = exponential_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit model
    try:
        params, covariance = curve_fit(
            model_func,
            lags,
            gamma,
            p0=[nugget_init, sill_init, range_init],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        
        nugget, sill, range_param = params
        
        # Calculate R² goodness of fit
        predicted = model_func(lags, nugget, sill, range_param)
        ss_res = np.sum((gamma - predicted) ** 2)
        ss_tot = np.sum((gamma - np.mean(gamma)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'nugget': nugget,
            'sill': sill,
            'range': range_param,
            'r_squared': r_squared,
            'model_type': model_type,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Failed to fit {model_type} model: {e}")
        return {
            'nugget': nugget_init,
            'sill': sill_init,
            'range': range_init,
            'r_squared': 0.0,
            'model_type': model_type,
            'success': False
        }


# ============================================================================
# Data Loading and Sampling
# ============================================================================

def load_raster_data(raster_path: Path, sample_size: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raster data and extract random sample of valid points.
    
    Args:
        raster_path: Path to raster file.
        sample_size: Number of random points to sample.
    
    Returns:
        Tuple of (coordinates, values) where coordinates is Nx2 and values is N-length.
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    logger.info(f"Loading raster: {raster_path.name}")
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        
        # Get valid (non-NaN, non-NoData) pixels
        nodata = src.nodata
        if nodata is not None:
            valid_mask = (~np.isnan(data)) & (data != nodata)
        else:
            valid_mask = ~np.isnan(data)
        
        # Get row, col indices of valid pixels
        rows, cols = np.where(valid_mask)
        values = data[rows, cols]
        
        # Convert pixel coordinates to geographic coordinates
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        coords = np.column_stack([xs, ys])
        
        logger.info(f"Found {len(values):,} valid pixels")
        
        # Random sampling to avoid memory issues
        if len(values) > sample_size:
            logger.info(f"Sampling {sample_size:,} random points")
            indices = np.random.choice(len(values), size=sample_size, replace=False)
            coords = coords[indices]
            values = values[indices]
        
        return coords, values


# ============================================================================
# Visualization
# ============================================================================

def plot_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    fitted_models: Dict[str, Dict[str, Any]],
    data_name: str,
    output_path: Path
) -> None:
    """
    Plot empirical variogram with fitted models.
    
    Args:
        lags: Lag distances.
        gamma: Empirical semi-variances.
        fitted_models: Dictionary of fitted model results.
        data_name: Name of dataset (for title).
        output_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot empirical variogram
    ax.plot(lags, gamma, 'ko', markersize=6, label='Empirical', zorder=3)
    
    # Plot fitted models
    h_fine = np.linspace(0, lags.max(), 200)
    colors = {'spherical': 'blue', 'exponential': 'red'}
    
    for model_name, model_result in fitted_models.items():
        if not model_result['success']:
            continue
        
        if model_name == 'spherical':
            model_func = spherical_model
        else:
            model_func = exponential_model
        
        gamma_fitted = model_func(
            h_fine,
            model_result['nugget'],
            model_result['sill'],
            model_result['range']
        )
        
        label = (f"{model_name.capitalize()} "
                f"(Range={model_result['range']:.3f}°, "
                f"R²={model_result['r_squared']:.3f})")
        
        ax.plot(h_fine, gamma_fitted, color=colors.get(model_name, 'gray'),
                linewidth=2, label=label, zorder=2)
        
        # Mark the range
        ax.axvline(model_result['range'], color=colors.get(model_name, 'gray'),
                   linestyle='--', alpha=0.5, zorder=1)
    
    ax.set_xlabel('Lag Distance (degrees)', fontsize=12)
    ax.set_ylabel('Semi-variance', fontsize=12)
    ax.set_title(f'Variogram Analysis - {data_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved variogram plot: {output_path}")
    plt.close()


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_dataset(
    raster_path: Path,
    data_name: str,
    sample_size: int = 5000,
    n_lags: int = 20
) -> Dict[str, Any]:
    """
    Complete variogram analysis for a single dataset.
    
    Args:
        raster_path: Path to raster file.
        data_name: Dataset name.
        sample_size: Number of points to sample.
        n_lags: Number of lag bins.
    
    Returns:
        Dictionary with analysis results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {data_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    coords, values = load_raster_data(raster_path, sample_size)
    
    # Calculate empirical variogram
    lags, gamma, counts = calculate_empirical_variogram(coords, values, n_lags=n_lags)
    
    logger.info(f"Empirical variogram calculated with {len(lags)} lag bins")
    
    # Fit models
    models = {}
    for model_type in ['spherical', 'exponential']:
        models[model_type] = fit_variogram_model(lags, gamma, model_type)
    
    # Create visualization
    output_dir = OUTPUTS_DIR / "variogram_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / f"variogram_{data_name.lower().replace(' ', '_')}.png"
    plot_variogram(lags, gamma, models, data_name, plot_path)
    
    return {
        'data_name': data_name,
        'n_samples': len(values),
        'lags': lags,
        'gamma': gamma,
        'counts': counts,
        'models': models,
        'plot_path': plot_path
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main execution function.
    """
    logger.info("Starting Variogram Analysis for GeoAnomalyMapper")
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Ensure output directories exist
    ensure_directories()
    
    # Define data sources
    datasets = {
        'Gravity Residual': OUTPUTS_DIR / 'gravity_residual.tif',
        'Magnetic': PROCESSED_DIR / 'magnetic' / 'magnetic_processed.tif'
    }
    
    # Check which files exist
    available_datasets = {}
    for name, path in datasets.items():
        if path.exists():
            available_datasets[name] = path
            logger.info(f"✓ Found: {name} at {path}")
        else:
            logger.warning(f"✗ Not found: {name} at {path}")
    
    if not available_datasets:
        logger.error("No raster files found. Please ensure data has been processed.")
        logger.error("Expected files:")
        for name, path in datasets.items():
            logger.error(f"  - {path}")
        sys.exit(1)
    
    # Analyze each available dataset
    results = {}
    for name, path in available_datasets.items():
        try:
            result = analyze_dataset(path, name, sample_size=5000, n_lags=20)
            results[name] = result
        except Exception as e:
            logger.error(f"Failed to analyze {name}: {e}", exc_info=True)
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("VARIOGRAM ANALYSIS RESULTS")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Samples analyzed: {result['n_samples']:,}")
        
        for model_type, model_result in result['models'].items():
            if model_result['success']:
                print(f"\n  {model_type.capitalize()} Model:")
                print(f"    Range:   {model_result['range']:.4f}° "
                      f"(~{model_result['range'] * 111:.1f} km)")
                print(f"    Nugget:  {model_result['nugget']:.6f}")
                print(f"    Sill:    {model_result['sill']:.6f}")
                print(f"    R²:      {model_result['r_squared']:.4f}")
            else:
                print(f"\n  {model_type.capitalize()} Model: FAILED")
    
    print("\n" + "="*70)
    print("RECOMMENDED BUFFER RADIUS:")
    print("="*70)
    
    # Calculate recommended buffer radius (average of best-fitting models)
    all_ranges = []
    for name, result in results.items():
        for model_type, model_result in result['models'].items():
            if model_result['success'] and model_result['r_squared'] > 0.5:
                all_ranges.append(model_result['range'])
    
    if all_ranges:
        recommended_range = np.median(all_ranges)
        print(f"\nRecommended buffer_radius: {recommended_range:.4f}° "
              f"(~{recommended_range * 111:.1f} km)")
        print(f"Based on median of {len(all_ranges)} well-fitted models")
        
        # Save results to file
        output_file = OUTPUTS_DIR / "variogram_analysis" / "recommended_buffer_radius.txt"
        with open(output_file, 'w') as f:
            f.write(f"Recommended buffer_radius: {recommended_range:.6f}\n")
            f.write(f"Approximate distance: {recommended_range * 111:.2f} km\n")
            f.write(f"\nBased on variogram analysis of:\n")
            for name in results.keys():
                f.write(f"  - {name}\n")
        logger.info(f"Saved recommendation to: {output_file}")
    else:
        print("\nWARNING: No well-fitted models found. Manual inspection recommended.")
    
    print("\n" + "="*70)
    print(f"Analysis complete. Plots saved to: {OUTPUTS_DIR / 'variogram_analysis'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
