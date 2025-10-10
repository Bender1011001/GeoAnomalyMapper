
#!/usr/bin/env python3
"""
Advanced Void Detection - Multi-Layer Probability Mapping

Combines multiple geophysical datasets to identify probable underground voids:
- Gravity anomalies (density deficits)
- InSAR subsidence (surface deformation)
- Lithology (karst-prone rock types)
- Seismic velocity (low-velocity zones)

Optimized for detecting voids at 20-300 feet (6-100m) depth.

Usage:
    python detect_voids.py --region "lat_min,lat_max,lon_min,lon_max" --output void_map.tif
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs" / "void_detection"

# Data paths - using processed data
GRAVITY_PATH = DATA_DIR / "processed" / "gravity" / "gravity_processed.tif"
MAGNETIC_PATH = DATA_DIR / "processed" / "magnetic" / "magnetic_processed.tif"
INSAR_PATH = DATA_DIR / "processed" / "insar" / "insar_processed.tif"
LITHOLOGY_PATH = DATA_DIR / "processed" / "lithology"  # Directory for lithology
DEM_PATH = DATA_DIR / "processed" / "dem" / "dem_processed.tif"

# Void detection thresholds
THRESHOLDS = {
    'gravity_negative': -1.0,      # σ units (negative anomaly = density deficit)
    'subsidence_rate': -5.0,       # mm/year (negative = sinking)
    'karst_lithology': [            # Karst-prone rock types
        'limestone', 'dolomite', 'marble', 'carbonate',
        'evaporite', 'gypsum', 'anhydrite', 'halite'
    ],
    'seismic_low_velocity': 3.0,   # km/s (low velocity = voids/fractures)
}

# Weighting for probability calculation
WEIGHTS = {
    'gravity': 0.35,      # Gravity anomalies
    'insar': 0.35,        # InSAR subsidence
    'lithology': 0.20,    # Rock type susceptibility
    'seismic': 0.10,      # Seismic velocity
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_and_resample(
    src_path: Path,
    target_bounds: Tuple[float, float, float, float],
    target_res: float,
    target_crs: str = "EPSG:4326"
) -> np.ndarray:
    """Load and resample raster to target grid."""
    if not src_path.exists():
        logger.warning(f"File not found: {src_path}")
        return None
    
    minx, miny, maxx, maxy = target_bounds
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    with rasterio.open(src_path) as src:
        # Create destination array
        dst_array = np.zeros((height, width), dtype=np.float32)
        
        # Reproject to target grid
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
    
    return dst_array


def score_gravity(data: np.ndarray) -> np.ndarray:
    """
    Score gravity anomalies for void probability.
    
    Negative anomalies indicate mass deficits (potential voids).
    Score: 0 (no void signature) to 1 (strong void signature)
    """
    if data is None:
        return None
    
    # Normalize: more negative = higher probability
    # Clamp between -6σ and 0σ
    score = np.clip(-data / 6.0, 0, 1)
    score[np.isnan(data)] = 0
    
    return score.astype(np.float32)


def score_insar(data: np.ndarray) -> np.ndarray:
    """
    Score InSAR subsidence for void probability.
    
    Subsidence (negative velocity) indicates ground settling over voids.
    Score: 0 (no subsidence) to 1 (strong subsidence)
    """
    if data is None:
        return None
    
    # Normalize: more negative = higher probability
    # -50 mm/year = maximum score
    score = np.clip(-data / 50.0, 0, 1)
    score[np.isnan(data)] = 0
    
    return score.astype(np.float32)


def score_lithology(data: np.ndarray, lithology_codes: Dict) -> np.ndarray:
    """
    Score lithology for karst susceptibility.
    
    Limestone, dolomite, evaporites = high void probability
    Score: 0 (non-karst) to 1 (highly susceptible)
    """
    if data is None:
        return None
    
    # Create binary mask for karst-prone rocks
    karst_mask = np.isin(data, list(lithology_codes.values()))
    score = karst_mask.astype(np.float32)
    
    return score


def score_seismic(data: np.ndarray) -> np.ndarray:
    """
    Score seismic velocity for void probability.
    
    Low velocity zones indicate fractured/void-rich rock.
    Score: 0 (normal velocity) to 1 (very low velocity)
    """
    if data is None:
        return None
    
    # Normalize: lower velocity = higher probability
    # 3 km/s = maximum score (very low for crustal rock)
    score = np.clip((5.0 - data) / 2.0, 0, 1)
    score[np.isnan(data)] = 0
    
    return score.astype(np.float32)


def calculate_void_probability(
    gravity_score: Optional[np.ndarray],
    insar_score: Optional[np.ndarray],
    lithology_score: Optional[np.ndarray],
    seismic_score: Optional[np.ndarray]
) -> np.ndarray:
    """
    Calculate overall void probability from weighted scores.
    
    Returns probability map: 0 (no void) to 1 (very likely void)
    """
    # Collect available scores
    scores = []
    weights = []
    
    if gravity_score is not None:
        scores.append(gravity_score)
        weights.append(WEIGHTS['gravity'])
    
    if insar_score is not None:
        scores.append(insar_score)
        weights.append(WEIGHTS['insar'])
    
    if lithology_score is not None:
        scores.append(lithology_score)
        weights.append(WEIGHTS['lithology'])
    
    if seismic_score is not None:
        scores.append(seismic_score)
        weights.append(WEIGHTS['seismic'])
    
    if not scores:
        logger.error("No data layers available for void detection!")
        return None
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Calculate weighted average
    probability = np.zeros_like(scores[0])
    for score, weight in zip(scores, weights):
        probability += score * weight
    
    return probability.astype(np.float32)


def identify_hotspots(
    probability: np.ndarray,
    threshold: float = 0.7,
    min_cluster_size: int = 5
) -> np.ndarray:
    """
    Identify high-probability void clusters.
    
    Returns labeled array of connected components above threshold.
    """
    from scipy import ndimage
    
    # Binary mask of high-probability pixels
    mask = probability >= threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Filter small clusters
    for label_id in range(1, num_features + 1):
        cluster_size = np.sum(labeled == label_id)
        if cluster_size < min_cluster_size:
            labeled[labeled == label_id] = 0
    
    return labeled


def visualize_void_probability(
    probability: np.ndarray,
    output_path: Path,
    bounds: Tuple[float, float, float, float],
    hotspots: Optional[np.ndarray] = None
):
    """Create publication-quality void probability map."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom colormap: white (low) -> yellow -> orange -> red (high)
    colors = ['white', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('void_prob', colors, N=n_bins)
    
    # Plot probability
    im = ax.imshow(
        probability,
        cmap=cmap,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        vmin=0,
        vmax=1,
        origin='upper',
        interpolation='bilinear'
    )
    
    # Overlay hotspot boundaries
    if hotspots is not None:
        from matplotlib import patches
        contours = plt.contour(
            hotspots,
            levels=[0.5],
            colors='blue',
            linewidths=2,
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
            origin='upper'
        )
    
    # Formatting
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title('Underground Void Probability Map\n(Higher values = greater likelihood)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Void Probability', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization: {output_path}")
    plt.close()


def write_geotiff(
    data: np.ndarray,
    output_path: Path,
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:4326"
):
    """Write probability map as GeoTIFF."""
    
    height, width = data.shape
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress='DEFLATE',
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "Void Probability (0-1)")
    
    logger.info(f"Saved GeoTIFF: {output_path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_region(
    bounds: Tuple[float, float, float, float],
    resolution: float = 0.001,  # ~100m at equator
    output_base: str = "void_probability"
):
    """
    Process region and generate void probability map.
    
    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max)
        resolution: Grid resolution in degrees
        output_base: Output filename base
    """
    
    logger.info(f"Processing region: {bounds}")
    logger.info(f"Resolution: {resolution}° (~{resolution * 111:.0f} km)")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and resample data layers
    logger.info("Loading gravity data...")
    gravity_data = load_and_resample(GRAVITY_PATH, bounds, resolution)
    
    logger.info("Loading magnetic data (if available)...")
    magnetic_data = load_and_resample(MAGNETIC_PATH, bounds, resolution)
    
    logger.info("Loading InSAR data (if available)...")
    insar_data = load_and_resample(INSAR_PATH, bounds, resolution)
    
    logger.info("Loading lithology data (if available)...")
    lithology_data = None  # TODO: Implement lithology loading from processed data
    
    logger.info("Loading seismic data (if available)...")
    seismic_data = None  # Seismic data not yet implemented
    
    # Score each layer
    logger.info("Calculating void probability scores...")
    gravity_score = score_gravity(gravity_data)
    magnetic_score = score_gravity(magnetic_data) if magnetic_data is not None else None  # Magnetic anomalies scored like gravity
    insar_score = score_insar(insar_data)
    lithology_score = score_lithology(lithology_data, {})
    seismic_score = score_seismic(seismic_data)
    
    # Combine gravity and magnetic if both available
    if gravity_score is not None and magnetic_score is not None:
        logger.info("Combining gravity and magnetic scores...")
        combined_geophysical = (gravity_score + magnetic_score) / 2.0
    elif gravity_score is not None:
        combined_geophysical = gravity_score
    elif magnetic_score is not None:
        combined_geophysical = magnetic_score
    else:
        combined_geophysical = None
    
    # Calculate combined probability
    probability = calculate_void_probability(
        combined_geophysical, insar_score, lithology_score, seismic_score
    )
    
    if probability is None:
        logger.error("Failed to calculate void probability")
        return
    
    # Identify hotspots
    logger.info("Identifying high-probability void zones...")
    hotspots = identify_hotspots(probability, threshold=0.7)
    num_hotspots = hotspots.max()
    logger.info(f"Found {num_hotspots} potential void clusters")
    
    # Save outputs
    output_tif = OUTPUT_DIR / f"{output_base}.tif"
    output_png = OUTPUT_DIR / f"{output_base}.png"
    
    write_geotiff(probability, output_tif, bounds)
    visualize_void_probability(probability, output_png, bounds, hotspots)
    
    # Generate report
    report_path = OUTPUT_DIR / f"{output_base}_report.txt"
    with open(report_path, 'w') as f:
        f.write("VOID DETECTION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Region: {bounds}\n")
        f.write(f"Resolution: {resolution}° (~{resolution * 111:.0f} km)\n\n")
        f.write(f"Data Layers Used:\n")
        f.write(f"  - Gravity: {'YES' if gravity_data is not None else 'NO'}\n")
        f.write(f"  - Magnetic: {'YES' if magnetic_data is not None else 'NO'}\n")
        f.write(f"  - InSAR: {'YES' if insar_data is not None else 'NO'}\n")
        f.write(f"  - Lithology: {'YES' if lithology_data is not None else 'NO'}\n")
        f.write(f"  - Seismic: {'YES' if seismic_data is not None else 'NO'}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - High-probability zones (>0.7): {num_hotspots}\n")
        f.write(f"  - Mean probability: {np.nanmean(probability):.3f}\n")
        f.write(f"  - Max probability: {np.nanmax(probability):.3f}\n\n")
        f.write(f"Outputs:\n")
        f.write(f"  - Probability map: {output_tif}\n")
        f.write(f"  - Visualization: {output_png}\n")
    
    logger.info(f"Report saved: {report_path}")
    logger.info("Void detection complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Detect underground voids using multi-layer geophysical analysis"
    )
    parser.add_argument(
        '--region',
        type=str,
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max" (default: Carlsbad Caverns area)',
        default="-105.0,32.0,-104.0,33.0"
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.001,
        help='Grid resolution in degrees (default: 0.001° ~ 100m)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='void_probability',
        help='Output filename base'
    )
    
    args = parser.parse_args()
    
    # Parse bounds
    bounds_str = args.region.split(',')
    if len(bounds_str) != 4:
        logger.error("Invalid region format. Use: lon_min,lat_min,lon_max,lat_max")
        sys.exit(1)
    
    bounds = tuple(map(float, bounds_str))
    
    # Process
    process_region(bounds, args.resolution, args.output)


if __name__ == "__main__":
    main()