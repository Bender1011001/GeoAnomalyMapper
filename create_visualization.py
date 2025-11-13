#!/usr/bin/env python3
"""
Create visualizations for multi-resolution fusion outputs
Automatically scales to actual data range for proper color display
"""

import logging
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling as RIOResampling
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid memory issues
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import simplekml
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_data_range(geotiff_path: Path) -> Tuple[float, float, float, float]:
    """Analyze GeoTIFF to determine optimal color scale range."""
    logger.info(f"Analyzing data range from: {geotiff_path}")
    
    with rasterio.open(geotiff_path) as src:
        data = src.read(1, masked=True)
    
    valid_data = data.compressed()  # Get non-masked values
    
    # Calculate percentiles for robust range determination
    p1, p5, p95, p99 = np.percentile(valid_data, [1, 5, 95, 99])
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    
    logger.info(f"Data statistics:")
    logger.info(f"  Mean: {mean:.3f}")
    logger.info(f"  Std: {std:.3f}")
    logger.info(f"  1st-99th percentile: {p1:.3f} to {p99:.3f}")
    logger.info(f"  5th-95th percentile: {p5:.3f} to {p95:.3f}")
    
    # Use 5th-95th percentile for color range (robust to outliers)
    # Expand slightly for better visualization
    vmin = p5 - 0.2 * abs(p5)
    vmax = p99 + 0.2 * abs(p99)
    
    # Make symmetric around zero for diverging colormap
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
    
    logger.info(f"Colormap range: {vmin:.3f} to {vmax:.3f}")
    
    return vmin, vmax, mean, std


def create_visualization_png(geotiff_path: Path, output_path: Path,
                             vmin: float, vmax: float, max_dimension: int = 8000) -> None:
    """Create a color-mapped PNG for overlay with intelligent downsampling."""
    logger.info("Creating visualization PNG...")
    
    with rasterio.open(geotiff_path) as src:
        # Check if we need to downsample
        height, width = src.height, src.width
        
        # Calculate downsample factor to keep largest dimension under max_dimension
        downsample_factor = max(1, max(height, width) // max_dimension)
        
        if downsample_factor > 1:
            logger.info(f"Downsampling by factor {downsample_factor} ({width}x{height} -> {width//downsample_factor}x{height//downsample_factor})")
            # Read with downsampling for memory efficiency
            data = src.read(
                1,
                out_shape=(height // downsample_factor, width // downsample_factor),
                resampling=RIOResampling.average,
                masked=True
            )
        else:
            data = src.read(1, masked=True)
    
    # Create diverging colormap (blue-white-red)
    cmap = plt.cm.RdBu_r  # Reversed: red for positive, blue for negative
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Apply colormap
    colored_data = cmap(norm(data.filled(np.nan)))
    
    # Set transparency based on magnitude
    alpha = np.ones_like(data.data, dtype=float)
    
    # Make weak signals more transparent
    magnitude = np.abs(data.data)
    max_mag = max(abs(vmin), abs(vmax))
    
    # Alpha scales from 0.2 (weak) to 1.0 (strong)
    alpha = 0.2 + 0.8 * (magnitude / max_mag)
    alpha[data.mask] = 0  # Fully transparent for nodata
    
    colored_data[:, :, 3] = alpha
    
    # Save as PNG with memory-efficient approach
    height, width = data.shape
    
    # Limit figure size to avoid memory issues
    max_fig_dim = 100  # inches
    dpi = min(100, int(8000 / max(height, width) * 100))  # Adaptive DPI
    figsize = (min(width / dpi, max_fig_dim), min(height / dpi, max_fig_dim))
    
    logger.info(f"Creating figure: {figsize[0]:.1f}x{figsize[1]:.1f} inches at {dpi} DPI")
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(colored_data, origin='upper', interpolation='bilinear')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close(fig)
    plt.clf()
    
    logger.info(f"Saved PNG: {output_path}")


def create_kmz(geotiff_path: Path, png_path: Path, output_path: Path,
               vmin: float, vmax: float, mean: float, std: float) -> None:
    """Create KMZ file for Google Earth."""
    logger.info("Creating KMZ for Google Earth...")
    
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
    
    # Create KML
    kml = simplekml.Kml()
    kml.document.name = f"Multi-Resolution Fusion: {geotiff_path.stem}"
    
    kml.document.description = f"""
Multi-Resolution Geophysical Anomaly Map

Data Statistics:
- Mean: {mean:.3f}σ
- Std Dev: {std:.3f}σ
- Display range: {vmin:.2f}σ to {vmax:.2f}σ

Color Scale:
- Dark Blue: Strong negative anomalies (potential voids, low density)
- Light Blue: Moderate negative anomalies
- White/Transparent: Near-zero anomalies
- Yellow/Orange: Moderate positive anomalies
- Red: Strong positive anomalies (dense structures)

Values are in standard deviation (σ) units from the regional mean.

Generated by GeoAnomalyMapper Multi-Resolution Fusion
"""
    
    # Add ground overlay
    ground = kml.newgroundoverlay(name='Anomaly Map')
    ground.icon.href = png_path.name
    ground.latlonbox.north = bounds.top
    ground.latlonbox.south = bounds.bottom
    ground.latlonbox.east = bounds.right
    ground.latlonbox.west = bounds.left
    
    # Set semi-transparent
    ground.color = simplekml.Color.changealphaint(180, simplekml.Color.white)
    
    # Create legend folder
    legend = kml.newfolder(name='Legend')
    
    # Add legend description
    legend_desc = f"""
COLOR LEGEND

Strong Negative: {vmin:.2f}σ
Moderate Negative: {vmin/2:.2f}σ
Zero: 0.00σ
Moderate Positive: {vmax/2:.2f}σ
Strong Positive: {vmax:.2f}σ

σ = standard deviations from regional mean
"""
    legend.description = legend_desc
    
    # Save as KMZ
    kml.savekmz(str(output_path), [str(png_path)])
    logger.info(f"Saved KMZ: {output_path}")


def create_preview_image(geotiff_path: Path, output_path: Path,
                         vmin: float, vmax: float, max_dimension: int = 4000) -> None:
    """Create a preview image with colorbar (downsampled for memory efficiency)."""
    logger.info("Creating preview image with colorbar...")
    
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        height, width = src.height, src.width
        
        # Downsample for preview
        downsample_factor = max(1, max(height, width) // max_dimension)
        
        if downsample_factor > 1:
            logger.info(f"Preview downsampling by factor {downsample_factor}")
            data = src.read(
                1,
                out_shape=(height // downsample_factor, width // downsample_factor),
                resampling=RIOResampling.average,
                masked=True
            )
        else:
            data = src.read(1, masked=True)
    
    # Create figure with colorbar
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data, cmap=cmap, norm=norm, 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Strength (σ units)', fontsize=12)
    
    # Set title and labels
    ax.set_title(f'Multi-Resolution Fusion: {geotiff_path.stem}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved preview: {output_path}")


def main():
    """Main visualization creation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create visualizations for multi-resolution fusion outputs"
    )
    parser.add_argument(
        'input_tif',
        type=str,
        help='Input GeoTIFF file to visualize'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: same as input)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_tif)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
    
    base_name = input_path.stem
    
    # Analyze data range
    vmin, vmax, mean, std = analyze_data_range(input_path)
    
    # Create outputs
    png_path = output_dir / f"{base_name}_overlay.png"
    create_visualization_png(input_path, png_path, vmin, vmax)
    
    kmz_path = output_dir / f"{base_name}.kmz"
    create_kmz(input_path, png_path, kmz_path, vmin, vmax, mean, std)
    
    preview_path = output_dir / f"{base_name}_preview.png"
    create_preview_image(input_path, preview_path, vmin, vmax)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VISUALIZATION FILES CREATED")
    print("=" * 70)
    print(f"\nInput: {input_path}")
    print(f"Data range: {vmin:.2f}σ to {vmax:.2f}σ")
    print(f"\nOutputs:")
    print(f"  1. Google Earth: {kmz_path}")
    print(f"     → Double-click to open in Google Earth")
    print(f"\n  2. Preview Image: {preview_path}")
    print(f"     → Quick view with colorbar")
    print(f"\n  3. Overlay PNG: {png_path}")
    print(f"     → For custom use")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()