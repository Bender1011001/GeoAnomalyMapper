#!/usr/bin/env python3
"""
Analyze and Visualize GeoAnomalyMapper Results

Generates summary statistics and PNG preview of the fused anomaly map.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import rasterio
from rasterio.plot import show

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def analyze_anomaly_data(geotiff_path: str):
    """Analyze the fused anomaly GeoTIFF and generate statistics and visualization."""
    logger.info(f"Analyzing {geotiff_path}")

    # Read the GeoTIFF
    with rasterio.open(geotiff_path) as src:
        data = src.read(1, masked=True)  # Read first band, mask nodata
        profile = src.profile
        bounds = src.bounds

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    logger.info(f"Bounds: {bounds}")

    # Flatten and remove NaN values for statistics
    flat_data = data.compressed()  # Remove masked values
    logger.info(f"Valid pixels: {len(flat_data)} / {data.size} ({100*len(flat_data)/data.size:.1f}%)")

    # Calculate statistics
    stats = {
        'min': float(np.min(flat_data)),
        'max': float(np.max(flat_data)),
        'mean': float(np.mean(flat_data)),
        'median': float(np.median(flat_data)),
        'std': float(np.std(flat_data)),
        'q25': float(np.percentile(flat_data, 25)),
        'q75': float(np.percentile(flat_data, 75)),
        'q95': float(np.percentile(flat_data, 95)),
        'q99': float(np.percentile(flat_data, 99)),
    }

    logger.info("Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.3f}")

    # Create output directory
    output_dir = Path(geotiff_path).parent
    output_dir.mkdir(exist_ok=True)

    # Generate histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram
    ax1.hist(flat_data, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Anomaly Value (σ)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Fused Anomaly Values')
    ax1.grid(True, alpha=0.3)

    # Add vertical lines for key percentiles
    for percentile, color, label in [
        (stats['q25'], 'orange', '25th percentile'),
        (stats['median'], 'red', 'Median'),
        (stats['q75'], 'orange', '75th percentile'),
        (stats['q95'], 'purple', '95th percentile'),
        (stats['q99'], 'darkred', '99th percentile')
    ]:
        ax1.axvline(percentile, color=color, linestyle='--', alpha=0.8, label=f'{label}: {percentile:.2f}')
    ax1.legend()

    # Global map preview
    # Create a custom colormap (blue for negative, red for positive anomalies)
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed
    norm = mcolors.TwoSlopeNorm(vmin=stats['min'], vcenter=0, vmax=stats['max'])

    # Show the data
    im = ax2.imshow(data, cmap=cmap, norm=norm, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    ax2.set_title('Global Fused Anomaly Map\n(Magnetic + Gravity)')
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', shrink=0.8)
    cbar.set_label('Anomaly Value (σ)')

    # Add some known reference points (major geological features)
    reference_sites = [
        (-104.4, 32.4, 'Carlsbad Caverns', 'red'),      # Carlsbad Caverns, NM
        (27.0, -27.0, 'Vredefort Dome', 'orange'),      # Vredefort impact structure
        (-6.6, 37.4, 'Rio Tinto', 'purple'),            # Rio Tinto mining district
        (35.5, 30.0, 'Dead Sea', 'green'),              # Dead Sea rift
        (-112.0, 36.0, 'Grand Canyon', 'brown'),        # Grand Canyon
        (139.7, 35.7, 'Mount Fuji', 'pink'),            # Mount Fuji
        (-155.3, 19.4, 'Kilauea', 'yellow'),            # Kilauea volcano
    ]

    for lon, lat, name, color in reference_sites:
        ax2.plot(lon, lat, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax2.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the figure
    output_png = output_dir / 'fused_anomaly_preview.png'
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    logger.info(f"Saved preview image: {output_png}")

    # Save statistics to text file
    stats_file = output_dir / 'anomaly_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("GeoAnomalyMapper Fused Anomaly Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input file: {geotiff_path}\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Valid pixels: {len(flat_data)} / {data.size} ({100*len(flat_data)/data.size:.1f}%)\n\n")

        f.write("Statistics:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value:.6f}\n")

        f.write("\nInterpretation:\n")
        f.write("- Values are in standard deviations (σ) from local median\n")
        f.write("- Positive values indicate higher-than-expected anomalies\n")
        f.write("- Negative values indicate lower-than-expected anomalies\n")
        f.write("- 95th percentile represents strong positive anomalies\n")
        f.write("- 5th percentile would represent strong negative anomalies\n")

    logger.info(f"Saved statistics: {stats_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("GEOANOMALYMAPPER RESULTS SUMMARY")
    print("="*60)
    print(f"Processed: {len(flat_data):,} valid pixels")
    print(f"Min anomaly: {stats['min']:.3f} σ")
    print(f"Max anomaly: {stats['max']:.3f} σ")
    print(f"Mean anomaly: {stats['mean']:.3f} σ")
    print(f"Median anomaly: {stats['median']:.3f} σ")
    print(f"Std deviation: {stats['std']:.3f} σ")
    print("\nTop anomaly hotspots (95th percentile and above):")
    threshold = stats['q95']
    hotspots = flat_data[flat_data >= threshold]
    print(f"  Count: {len(hotspots)} pixels ({100*len(hotspots)/len(flat_data):.2f}% of valid data)")
    print(f"  Threshold: {threshold:.3f} σ")
    print(f"  Max in hotspots: {np.max(hotspots):.3f} σ")

    print(f"\nPreview saved to: {output_png}")
    print(f"Statistics saved to: {stats_file}")
    print("="*60 + "\n")

    return stats

def main():
    """Main analysis function."""
    # Find the fused anomaly file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    geotiff_path = project_root / 'data' / 'outputs' / 'final' / 'fused_anomaly.tif'

    if not geotiff_path.exists():
        logger.error(f"Fused anomaly file not found: {geotiff_path}")
        sys.exit(1)

    analyze_anomaly_data(str(geotiff_path))

if __name__ == '__main__':
    main()