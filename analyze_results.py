#!/usr/bin/env python3
"""
Analyze and Visualize GeoAnomalyMapper Results

Generates summary statistics and PNG preview of the fused anomaly map.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio
from rasterio.transform import xy as transform_xy

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
        'q05': float(np.percentile(flat_data, 5)),
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
        (stats['q05'], 'teal', '5th percentile'),
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

    # Generate extended diagnostics and hotspot overlays
    diagnostics_path = output_dir / 'fused_anomaly_diagnostics.png'
    hotspot_csv = output_dir / 'anomaly_hotspots.csv'
    create_diagnostic_visuals(
        data,
        bounds,
        profile['transform'],
        stats,
        diagnostics_path,
        hotspot_csv
    )

    logger.info(f"Saved diagnostics image: {diagnostics_path}")
    logger.info(f"Saved hotspot summary: {hotspot_csv}")

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
        f.write("- 5th percentile represents strong negative anomalies\n")

        if hotspot_csv.exists():
            f.write("\nHotspot summary saved to anomaly_hotspots.csv\n")

    logger.info(f"Saved statistics: {stats_file}")

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
    print(f"  5th percentile (cold spots): {stats['q05']:.3f} σ")

    print(f"\nPreview saved to: {output_png}")
    print(f"Diagnostics saved to: {diagnostics_path}")
    print(f"Statistics saved to: {stats_file}")
    print("="*60 + "\n")

    return stats


def compute_lat_lon_vectors(transform, width, height):
    """Return longitude and latitude coordinate vectors for raster grid."""
    row_indices = np.arange(height)
    col_indices = np.arange(width)

    # Transform accepts sequences - compute along central column/row
    _, latitudes = transform_xy(
        transform,
        row_indices,
        np.full_like(row_indices, fill_value=width // 2),
        offset='center'
    )
    longitudes, _ = transform_xy(
        transform,
        np.full_like(col_indices, fill_value=height // 2),
        col_indices,
        offset='center'
    )

    return np.array(longitudes), np.array(latitudes)


def create_diagnostic_visuals(
    data: np.ma.MaskedArray,
    bounds,
    transform,
    stats: dict,
    output_path: Path,
    hotspot_csv: Path,
    top_n: int = 15
) -> None:
    """Create additional diagnostics and hotspot exports."""

    logger.info("Creating diagnostic visualizations...")

    cold_threshold = stats['q05']
    hot_threshold = stats['q95']

    cold_mask = np.ma.filled(data <= cold_threshold, False)
    hot_mask = np.ma.filled(data >= hot_threshold, False)

    longitudes, latitudes = compute_lat_lon_vectors(transform, data.shape[1], data.shape[0])
    zonal_mean = np.ma.mean(data, axis=1)
    meridional_mean = np.ma.mean(data, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=stats['min'], vcenter=0, vmax=stats['max'])

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    ax_map = axes[0, 0]
    base = ax_map.imshow(data, cmap=cmap, norm=norm, extent=extent)
    ax_map.set_title('Anomaly Map with Hot/Cold Spot Overlay')
    ax_map.set_xlabel('Longitude (°)')
    ax_map.set_ylabel('Latitude (°)')
    ax_map.grid(True, alpha=0.3)

    overlay = np.zeros((data.shape[0], data.shape[1], 4))
    overlay[..., 0] = hot_mask.astype(float)
    overlay[..., 2] = cold_mask.astype(float)
    overlay[..., 3] = (hot_mask | cold_mask) * 0.45
    ax_map.imshow(overlay, extent=extent, origin='upper')

    legend_handles = [
        Patch(facecolor=(1.0, 0.0, 0.0, 0.45), edgecolor='black', label=f'≥ {hot_threshold:.2f} σ'),
        Patch(facecolor=(0.0, 0.0, 1.0, 0.45), edgecolor='black', label=f'≤ {cold_threshold:.2f} σ')
    ]
    ax_map.legend(handles=legend_handles, loc='lower left')
    plt.colorbar(base, ax=ax_map, fraction=0.046, pad=0.04, label='Anomaly Value (σ)')

    ax_lat = axes[0, 1]
    ax_lat.plot(zonal_mean, latitudes)
    ax_lat.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_lat.set_title('Latitudinal Mean Profile')
    ax_lat.set_xlabel('Mean anomaly (σ)')
    ax_lat.set_ylabel('Latitude (°)')
    ax_lat.grid(True, alpha=0.3)

    ax_lon = axes[1, 0]
    ax_lon.plot(longitudes, meridional_mean)
    ax_lon.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_lon.set_title('Longitudinal Mean Profile')
    ax_lon.set_xlabel('Longitude (°)')
    ax_lon.set_ylabel('Mean anomaly (σ)')
    ax_lon.grid(True, alpha=0.3)

    sorted_values = np.sort(data.compressed())
    cumulative = np.linspace(0, 1, len(sorted_values), endpoint=False)
    ax_cdf = axes[1, 1]
    ax_cdf.plot(sorted_values, cumulative)
    for percentile, color in [
        (stats['q05'], 'teal'),
        (stats['median'], 'red'),
        (stats['q95'], 'purple')
    ]:
        ax_cdf.axvline(percentile, color=color, linestyle='--')
    ax_cdf.set_title('Cumulative Distribution of Anomaly Values')
    ax_cdf.set_xlabel('Anomaly value (σ)')
    ax_cdf.set_ylabel('Cumulative probability')
    ax_cdf.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    export_hotspots(data, transform, hotspot_csv, top_n=top_n)


def export_hotspots(
    data: np.ma.MaskedArray,
    transform,
    output_csv: Path,
    top_n: int = 15
) -> None:
    """Export top positive and negative anomaly locations to CSV."""

    logger.info("Extracting hotspot coordinates...")

    valid_mask = ~data.mask
    if not np.any(valid_mask):
        logger.warning("No valid data for hotspot extraction")
        return

    flat_indices = np.flatnonzero(valid_mask)
    flat_values = data.data.flat[flat_indices]

    if flat_indices.size == 0:
        logger.warning("No valid pixels found after masking")
        return

    top_count = min(top_n, flat_indices.size)

    top_pos_idx = flat_indices[np.argpartition(flat_values, -top_count)[-top_count:]]
    top_neg_idx = flat_indices[np.argpartition(flat_values, top_count)[:top_count]]

    hotspots: List[Tuple[str, float, float, float]] = []

    for rank, flat_idx in enumerate(np.flip(top_pos_idx[np.argsort(data.data.flat[top_pos_idx])]), start=1):
        row, col = np.unravel_index(flat_idx, data.shape)
        lon, lat = transform_xy(transform, row, col, offset='center')
        value = float(data.data[row, col])
        hotspots.append((f'positive_{rank}', value, lon, lat))

    for rank, flat_idx in enumerate(top_neg_idx[np.argsort(data.data.flat[top_neg_idx])], start=1):
        row, col = np.unravel_index(flat_idx, data.shape)
        lon, lat = transform_xy(transform, row, col, offset='center')
        value = float(data.data[row, col])
        hotspots.append((f'negative_{rank}', value, lon, lat))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'value_sigma', 'longitude_deg', 'latitude_deg'])
        for label, value, lon, lat in hotspots:
            writer.writerow([label, f"{value:.6f}", f"{lon:.6f}", f"{lat:.6f}"])


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