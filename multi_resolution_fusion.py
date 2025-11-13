#!/usr/bin/env python3
"""
Multi-Resolution Data Fusion Pipeline

Combines geophysical datasets at their native resolutions to achieve maximum
effective resolution for subsurface anomaly detection:

- InSAR (Sentinel-1): 5-20m resolution for surface deformation
- Gravity (XGM2019e): ~4km resolution vs EIGEN-6C4 ~11km
- Magnetic (EMAG2v3): 2 arc-minute (~3.7km at equator)
- Regional gravity surveys: Sub-kilometer where available
- DEM/LiDAR: 10-30m for topographic corrections
- Seismic velocity: Variable resolution (typically 10-50km)

The pipeline uses adaptive resampling, uncertainty weighting, and spectral
fusion techniques to preserve fine-scale features while maintaining
statistical rigor.

Usage:
    python multi_resolution_fusion.py --region "lon_min,lat_min,lon_max,lat_max" --output fused_hires.tif
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as RioResampling
from scipy import ndimage, signal
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Configuration shim import for optional v2 features
from utils.config_shim import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs" / "multi_resolution"

# Data source configuration
class DataSource(Enum):
    """Available data sources with their native resolutions."""
    INSAR_SENTINEL1 = ("insar", 0.0001)  # ~10m
    GRAVITY_XGM2019E = ("gravity_hires", 0.036)  # ~4km
    GRAVITY_EGM2008 = ("gravity", 0.1)  # ~11km
    MAGNETIC_EMAG2 = ("magnetic", 0.033)  # ~3.7km
    DEM_SRTM30 = ("dem", 0.00083)  # ~30m
    SEISMIC_LITHO1 = ("seismic", 1.0)  # ~111km
    REGIONAL_GRAVITY = ("gravity_regional", 0.001)  # ~100m (where available)

@dataclass
class DataLayer:
    """Container for a data layer with metadata."""
    name: str
    data: np.ndarray
    resolution: float  # degrees
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    uncertainty: Optional[np.ndarray] = None
    weight: float = 1.0
    unit: str = ""
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def resolution_meters(self):
        """Approximate resolution in meters at equator."""
        return self.resolution * 111000


# ============================================================================
# ADAPTIVE RESAMPLING
# ============================================================================

class ResamplingStrategy:
    """Adaptive resampling strategies based on data characteristics."""
    
    @staticmethod
    def select_method(source_res: float, target_res: float, data_type: str) -> RioResampling:
        """
        Select optimal resampling method based on resolution change and data type.
        
        Args:
            source_res: Source resolution in degrees
            target_res: Target resolution in degrees
            data_type: 'continuous', 'categorical', or 'discrete'
        
        Returns:
            Rasterio resampling method
        """
        ratio = source_res / target_res
        
        if data_type == 'categorical':
            return RioResampling.nearest
        
        if ratio > 3:  # Upsampling significantly
            return RioResampling.cubic_spline
        elif ratio > 1.5:  # Moderate upsampling
            return RioResampling.cubic
        elif ratio < 0.5:  # Downsampling
            return RioResampling.average
        else:  # Similar resolution
            return RioResampling.bilinear
    
    @staticmethod
    def anti_alias_filter(data: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply anti-aliasing filter before downsampling.
        
        Args:
            data: Input array
            factor: Downsampling factor (source_res / target_res)
        
        Returns:
            Filtered array
        """
        if factor <= 1.0:
            return data
        
        # Gaussian filter with sigma proportional to downsampling
        sigma = factor / 2.0
        return ndimage.gaussian_filter(data, sigma=sigma)


def load_and_resample_adaptive(
    src_path: Path,
    target_bounds: Tuple[float, float, float, float],
    target_res: float,
    data_type: str = 'continuous',
    max_resolution: Optional[float] = None
) -> Optional[DataLayer]:
    """
    Load and adaptively resample raster to target grid.
    
    Args:
        src_path: Path to source raster
        target_bounds: (minx, miny, maxx, maxy)
        target_res: Target resolution in degrees
        data_type: Type of data for resampling method selection
        max_resolution: Limit resolution to avoid excessive memory use
    
    Returns:
        DataLayer object or None if file not found
    """
    if not src_path.exists():
        logger.warning(f"File not found: {src_path}")
        return None
    
    minx, miny, maxx, maxy = target_bounds
    
    # Limit resolution if needed
    if max_resolution and target_res < max_resolution:
        logger.info(f"Limiting resolution from {target_res}° to {max_resolution}°")
        target_res = max_resolution
    
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    with rasterio.open(src_path) as src:
        source_res = abs(src.transform.a)  # Assumes square pixels
        
        # Select optimal resampling method
        resampling = ResamplingStrategy.select_method(
            source_res, target_res, data_type
        )
        
        logger.info(f"Resampling {src_path.name}: {source_res:.6f}° → {target_res:.6f}° using {resampling.name}")
        
        # Create destination array
        dst_array = np.zeros((height, width), dtype=np.float32)
        
        # Reproject to target grid
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
        
        # Apply anti-aliasing if downsampling
        if source_res < target_res:
            factor = target_res / source_res
            dst_array = ResamplingStrategy.anti_alias_filter(dst_array, factor)
    
    return DataLayer(
        name=src_path.stem,
        data=dst_array,
        resolution=target_res,
        bounds=target_bounds
    )


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyModel:
    """Estimate uncertainty for different data sources."""
    
    # Typical uncertainties for each data source (relative units)
    SOURCE_UNCERTAINTY = {
        DataSource.INSAR_SENTINEL1: 0.05,  # ~5mm/year for deformation
        DataSource.GRAVITY_XGM2019E: 0.1,  # ~10% for gravity
        DataSource.GRAVITY_EGM2008: 0.15,
        DataSource.MAGNETIC_EMAG2: 0.08,
        DataSource.DEM_SRTM30: 0.1,
        DataSource.SEISMIC_LITHO1: 0.2,
        DataSource.REGIONAL_GRAVITY: 0.05,
    }
    
    @staticmethod
    def estimate_uncertainty(
        layer: DataLayer,
        source_type: DataSource
    ) -> np.ndarray:
        """
        Estimate pixel-wise uncertainty based on data characteristics.
        
        Uncertainty increases with:
        - Distance from data points (for interpolated data)
        - High gradient areas (less reliable)
        - Edge effects
        
        Returns:
            Uncertainty map (same shape as data)
        """
        data = layer.data
        base_uncertainty = UncertaintyModel.SOURCE_UNCERTAINTY.get(source_type, 0.1)
        
        # Start with base uncertainty
        uncertainty = np.full_like(data, base_uncertainty)
        
        # Increase uncertainty in high-gradient areas
        grad_y, grad_x = np.gradient(data)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = np.nan_to_num(gradient_mag, nan=0.0)
        
        # Normalize gradient
        if gradient_mag.max() > 0:
            gradient_norm = gradient_mag / np.nanpercentile(gradient_mag, 95)
            uncertainty += base_uncertainty * 0.5 * np.clip(gradient_norm, 0, 1)
        
        # Increase uncertainty at edges (assuming edge effects)
        edge_mask = np.zeros_like(data, dtype=bool)
        edge_width = max(1, int(0.05 * min(data.shape)))  # 5% edge
        edge_mask[:edge_width, :] = True
        edge_mask[-edge_width:, :] = True
        edge_mask[:, :edge_width] = True
        edge_mask[:, -edge_width:] = True
        uncertainty[edge_mask] *= 1.5
        
        # Set uncertainty to NaN where data is NaN
        uncertainty[np.isnan(data)] = np.nan
        
        return uncertainty.astype(np.float32)


# ============================================================================
# SPECTRAL FUSION
# ============================================================================

class SpectralFusion:
    """
    Combine datasets using frequency-domain fusion.
    
    Preserves high-frequency content from high-resolution sources
    and low-frequency content from more reliable low-resolution sources.
    """
    
    @staticmethod
    def highpass_filter(data: np.ndarray, cutoff_wavelength: float) -> np.ndarray:
        """
        Apply high-pass filter to extract fine-scale features.
        
        Args:
            data: Input array
            cutoff_wavelength: Wavelength cutoff in pixels
        
        Returns:
            High-pass filtered array
        """
        if cutoff_wavelength <= 0:
            return data
        
        # Gaussian high-pass: original - lowpass
        sigma = cutoff_wavelength / (2 * np.pi)
        lowpass = ndimage.gaussian_filter(data, sigma=sigma)
        highpass = data - lowpass
        
        return highpass
    
    @staticmethod
    def lowpass_filter(data: np.ndarray, cutoff_wavelength: float) -> np.ndarray:
        """
        Apply low-pass filter to extract broad-scale features.
        
        Args:
            data: Input array
            cutoff_wavelength: Wavelength cutoff in pixels
        
        Returns:
            Low-pass filtered array
        """
        sigma = cutoff_wavelength / (2 * np.pi)
        return ndimage.gaussian_filter(data, sigma=sigma)
    
    @staticmethod
    def fuse_spectral(
        high_res_layer: DataLayer,
        low_res_layer: DataLayer,
        transition_wavelength: float = 10.0
    ) -> DataLayer:
        """
        Fuse high and low resolution layers in frequency domain.
        
        Args:
            high_res_layer: High-resolution data (preserves fine details)
            low_res_layer: Low-resolution data (provides stable background)
            transition_wavelength: Transition between high/low freq in pixels
        
        Returns:
            Fused DataLayer
        """
        # Extract components
        high_freq = SpectralFusion.highpass_filter(
            high_res_layer.data, transition_wavelength
        )
        low_freq = SpectralFusion.lowpass_filter(
            low_res_layer.data, transition_wavelength
        )
        
        # Combine
        fused = high_freq + low_freq
        
        # Propagate NaNs from either source
        mask = np.isnan(high_res_layer.data) & np.isnan(low_res_layer.data)
        fused[mask] = np.nan
        
        return DataLayer(
            name=f"{high_res_layer.name}_spectral_fusion",
            data=fused,
            resolution=high_res_layer.resolution,
            bounds=high_res_layer.bounds,
            weight=(high_res_layer.weight + low_res_layer.weight) / 2
        )


# ============================================================================
# WEIGHTED FUSION
# ============================================================================

def fuse_weighted(
    layers: List[DataLayer],
    use_uncertainty: bool = True,
    dynamic: bool = False,
    target_res: Optional[float] = None
) -> DataLayer:
    """
    Perform weighted fusion of multiple DataLayer objects.

    Parameters
    ----------
    layers : List[DataLayer]
        List of data layers to be fused. All layers must share the same shape.
    use_uncertainty : bool, optional
        If True, incorporate per‑pixel uncertainty weighting when ``dynamic`` is False.
        Ignored when ``dynamic`` is True because dynamic weighting already accounts for uncertainty.
    dynamic : bool, optional
        When True, enable dynamic weighting based on data quality metrics (uncertainty
        and spatial resolution). This mode is activated via the ``GAM_DYNAMIC_WEIGHTING``
        configuration flag and is opt‑in only.
    target_res : float, optional
        Target resolution (in degrees) of the output grid. Required when ``dynamic`` is True
        because the resolution factor is computed relative to this value.

    Returns
    -------
    DataLayer
        A new DataLayer containing the fused data, optional fused uncertainty,
        and metadata inherited from the highest‑resolution input layer.
    """
    if not layers:
        raise ValueError("No layers provided for fusion")
    
    # All layers must have same shape
    ref_shape = layers[0].shape
    if not all(layer.shape == ref_shape for layer in layers):
        raise ValueError("All layers must have the same shape for fusion")
    
    # Normalize each layer to z-scores for fair comparison
    # Log which weighting mode is active
    if dynamic:
        logger.info("Dynamic weighting mode enabled")
    else:
        logger.info("Standard uncertainty‑weighted fusion mode")
    # If dynamic weighting is requested, compute per‑layer dynamic weights
    if dynamic:
        if target_res is None:
            raise ValueError("target_res must be provided when dynamic weighting is enabled")
    normalized_layers = []
    for layer in layers:
        data = layer.data.copy()
        
        # Robust normalization using median and MAD
        med = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - med))
        
        if mad > 0:
            z_score = (data - med) / (1.4826 * mad)
            z_score = np.clip(z_score, -6, 6)  # Clamp outliers
        else:
            z_score = np.zeros_like(data)
        
        normalized_layers.append(z_score)
    
    # Calculate weights
    if dynamic:
        # Dynamic weighting based on data quality (uncertainty & resolution)
        weights = []
        for layer in layers:
            # Base weight from uncertainty (inverse) if available
            if layer.uncertainty is not None:
                # Per-pixel inverse-uncertainty weighting
                base_w = 1.0 / (layer.uncertainty + 1e-6)
            else:
                # Fallback to scalar layer weight broadcast to raster shape
                base_w = np.full(layer.shape, layer.weight, dtype=np.float32)
            # Resolution factor: higher resolution (smaller degree) gets higher weight
            res_factor = target_res / layer.resolution if layer.resolution > 0 else 1.0
            w = base_w * res_factor
            weights.append(w.astype(np.float32))
    else:
        if use_uncertainty:
            weights = []
            for layer in layers:
                if layer.uncertainty is not None:
                    # Inverse uncertainty weighting
                    w = 1.0 / (layer.uncertainty + 1e-6)
                else:
                    w = np.full(layer.shape, layer.weight)
                weights.append(w)
        else:
            weights = [np.full(layer.shape, layer.weight) for layer in layers]
    
    # Stack arrays
    data_stack = np.stack(normalized_layers, axis=0)
    weight_stack = np.stack(weights, axis=0)
    
    # Weighted average (ignoring NaNs)
    with np.errstate(invalid='ignore', divide='ignore'):
        weighted_sum = np.nansum(data_stack * weight_stack, axis=0)
        weight_sum = np.nansum(weight_stack, axis=0)
        fused = weighted_sum / weight_sum
    # Where total weight is zero or invalid, set fused to NaN
    zero_w = (weight_sum == 0) | np.isnan(weight_sum)
    fused[zero_w] = np.nan
    
    # Set to NaN where all inputs are NaN
    all_nan = np.all(np.isnan(data_stack), axis=0)
    fused[all_nan] = np.nan
    
    # Estimate fused uncertainty (propagation of uncertainties)
    if use_uncertainty and any(layer.uncertainty is not None for layer in layers):
        # Use safe weight sum to avoid division by zero; invalid entries propagate as NaN
        weight_sum_safe = np.where(weight_sum == 0, np.nan, weight_sum)
        fused_uncertainty = np.sqrt(
            np.nansum(
                (weight_stack / weight_sum_safe) ** 2
                * np.stack(
                    [l.uncertainty if l.uncertainty is not None
                     else np.ones(l.shape, dtype=np.float32) for l in layers],
                    axis=0
                ) ** 2,
                axis=0
            )
        )
    else:
        fused_uncertainty = None
    
    return DataLayer(
        name="weighted_fusion",
        data=fused.astype(np.float32),
        resolution=min(layer.resolution for layer in layers),
        bounds=layers[0].bounds,
        uncertainty=fused_uncertainty,
        weight=1.0
    )


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_multi_resolution(
    bounds: Tuple[float, float, float, float],
    target_resolution: float = 0.001,  # ~100m
    output_name: str = "multi_res_fusion",
    data_sources: Optional[List[str]] = None
):
    """
    Process region using multi-resolution fusion.
    
    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max)
        target_resolution: Target resolution in degrees
        output_name: Output filename base
        data_sources: List of data sources to include (None = all available)
    """
    logger.info("=" * 70)
    logger.info("MULTI-RESOLUTION DATA FUSION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Region: {bounds}")
    logger.info(f"Target resolution: {target_resolution}° ({target_resolution * 111:.1f} km)")
    logger.info("")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define data paths (update these based on your data)
    data_paths = {
        'gravity': DATA_DIR / "outputs" / "final" / "fused_anomaly.tif",
        'gravity_hires': DATA_DIR / "raw" / "gravity" / "xgm2019e_gravity.tif",
        'magnetic': DATA_DIR / "raw" / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif",
        'insar': DATA_DIR / "raw" / "insar" / "sentinel1_velocity.tif",
        'dem': DATA_DIR / "raw" / "dem" / "srtm30.tif",
    }
    
    # Load available layers
    layers = []
    
    logger.info("Loading data layers...")
    logger.info("-" * 70)
    
    # High-resolution InSAR (if available)
    if data_paths.get('insar') and data_paths['insar'].exists():
        layer = load_and_resample_adaptive(
            data_paths['insar'],
            bounds,
            target_resolution,
            data_type='continuous'
        )
        if layer:
            layer.weight = 2.0  # High weight for direct measurements
            layer.unit = "mm/year"
            layer.uncertainty = UncertaintyModel.estimate_uncertainty(
                layer, DataSource.INSAR_SENTINEL1
            )
            layers.append(layer)
            logger.info(f"✓ InSAR (Sentinel-1): {layer.resolution_meters:.0f}m resolution, weight={layer.weight}")
    
    # High-resolution gravity (XGM2019e)
    if data_paths.get('gravity_hires') and data_paths['gravity_hires'].exists():
        layer = load_and_resample_adaptive(
            data_paths['gravity_hires'],
            bounds,
            target_resolution,
            data_type='continuous'
        )
        if layer:
            layer.weight = 1.5
            layer.unit = "mGal"
            layer.uncertainty = UncertaintyModel.estimate_uncertainty(
                layer, DataSource.GRAVITY_XGM2019E
            )
            layers.append(layer)
            logger.info(f"✓ Gravity (XGM2019e): {layer.resolution_meters:.0f}m resolution, weight={layer.weight}")
    
    # Standard gravity (EGM2008) - fallback
    elif data_paths.get('gravity') and data_paths['gravity'].exists():
        layer = load_and_resample_adaptive(
            data_paths['gravity'],
            bounds,
            target_resolution,
            data_type='continuous'
        )
        if layer:
            layer.weight = 1.0
            layer.unit = "σ units"
            layer.uncertainty = UncertaintyModel.estimate_uncertainty(
                layer, DataSource.GRAVITY_EGM2008
            )
            layers.append(layer)
            logger.info(f"✓ Gravity (EGM2008): {layer.resolution_meters:.0f}m resolution, weight={layer.weight}")
    
    # Magnetic
    if data_paths.get('magnetic') and data_paths['magnetic'].exists():
        layer = load_and_resample_adaptive(
            data_paths['magnetic'],
            bounds,
            target_resolution,
            data_type='continuous'
        )
        if layer:
            layer.weight = 1.0
            layer.unit = "nT"
            layer.uncertainty = UncertaintyModel.estimate_uncertainty(
                layer, DataSource.MAGNETIC_EMAG2
            )
            layers.append(layer)
            logger.info(f"✓ Magnetic (EMAG2): {layer.resolution_meters:.0f}m resolution, weight={layer.weight}")
    
    logger.info("-" * 70)
    logger.info(f"Total layers loaded: {len(layers)}")
    logger.info("")
    
    if not layers:
        logger.error("No data layers available! Please download data first.")
        return
    
    # Determine whether dynamic weighting is enabled via config flag
    dynamic_flag = get_config('GAM_DYNAMIC_WEIGHTING', 'false').lower() == 'true'
    if dynamic_flag:
        logger.info("GAM_DYNAMIC_WEIGHTING enabled – using dynamic weighting fusion")
    else:
        logger.info("GAM_DYNAMIC_WEIGHTING disabled – using standard weighted fusion")
    logger.info("Performing fusion...")
    if dynamic_flag:
        # Dynamic weighting mode
        fused_layer = fuse_weighted(
            layers,
            use_uncertainty=True,
            dynamic=True,
            target_res=target_resolution
        )
    else:
        # Standard mode (backward compatible)
        fused_layer = fuse_weighted(layers, use_uncertainty=True)
    
    # Save output
    output_tif = OUTPUT_DIR / f"{output_name}.tif"
    logger.info(f"Saving fused result: {output_tif}")
    
    height, width = fused_layer.shape
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform,
        nodata=np.nan,
        compress='DEFLATE',
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(fused_layer.data, 1)
        dst.set_band_description(1, "Multi-resolution fused anomaly (σ units)")
    
    # Generate statistics report
    valid_data = fused_layer.data[~np.isnan(fused_layer.data)]
    
    report = f"""
MULTI-RESOLUTION FUSION REPORT
{'=' * 70}

Region: {bounds}
Target Resolution: {target_resolution}° (~{target_resolution * 111:.1f} km)
Output Resolution: {fused_layer.resolution_meters:.0f} m

Data Layers:
{chr(10).join(f'  - {layer.name}: {layer.resolution_meters:.0f}m, weight={layer.weight}' for layer in layers)}

Statistics:
  - Valid pixels: {len(valid_data):,}
  - Mean: {np.mean(valid_data):.3f}
  - Std Dev: {np.std(valid_data):.3f}
  - Min: {np.min(valid_data):.3f}
  - Max: {np.max(valid_data):.3f}
  - 5th percentile: {np.percentile(valid_data, 5):.3f}
  - 95th percentile: {np.percentile(valid_data, 95):.3f}

Output:
  - Fused GeoTIFF: {output_tif}

Notes:
  - Values are normalized z-scores (σ units)
  - Higher absolute values indicate stronger anomalies
  - Negative values: density/magnetic deficits (potential voids)
  - Positive values: density/magnetic excesses (dense structures)
"""
    
    report_path = OUTPUT_DIR / f"{output_name}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("=" * 70)
    logger.info("FUSION COMPLETE")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-resolution geophysical data fusion pipeline"
    )
    
    # Accept bounds as 4 separate arguments (PowerShell-friendly)
    parser.add_argument(
        '--lon-min',
        type=float,
        default=-105.0,
        help='Minimum longitude (default: -105.0)'
    )
    parser.add_argument(
        '--lat-min',
        type=float,
        default=32.0,
        help='Minimum latitude (default: 32.0)'
    )
    parser.add_argument(
        '--lon-max',
        type=float,
        default=-104.0,
        help='Maximum longitude (default: -104.0)'
    )
    parser.add_argument(
        '--lat-max',
        type=float,
        default=33.0,
        help='Maximum latitude (default: 33.0)'
    )
    
    # Also support old comma-separated format for backwards compatibility
    parser.add_argument(
        '--region',
        type=str,
        help='DEPRECATED: Use --lon-min --lat-min --lon-max --lat-max instead. Region bounds: "lon_min,lat_min,lon_max,lat_max"',
        default=None
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.001,
        help='Target resolution in degrees (default: 0.001° ~ 100m)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='multi_res_fusion',
        help='Output filename base'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        help='Data sources to include (default: all available)'
    )
    
    args = parser.parse_args()
    
    # Parse bounds - support both new and old format
    if args.region is not None:
        # Old comma-separated format
        bounds_str = args.region.split(',')
        if len(bounds_str) != 4:
            logger.error("Invalid region format. Use: lon_min,lat_min,lon_max,lat_max")
            logger.error("Or use: --lon-min LON --lat-min LAT --lon-max LON --lat-max LAT")
            sys.exit(1)
        bounds = tuple(map(float, bounds_str))
    else:
        # New format with separate arguments
        bounds = (args.lon_min, args.lat_min, args.lon_max, args.lat_max)
    
    # Process
    process_multi_resolution(
        bounds,
        target_resolution=args.resolution,
        output_name=args.output,
        data_sources=args.sources
    )


if __name__ == "__main__":
    main()