#!/usr/bin/env python3
"""
Validate Multi-Resolution Fusion Against Known Underground Features
Tests if detected anomalies match known caves, voids, and subsurface structures
"""

import logging
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known underground features in USA for validation
KNOWN_FEATURES = [
    # Caves and Cave Systems
    {
        'name': 'Carlsbad Caverns, NM',
        'lon': -104.4434,
        'lat': 32.1751,
        'type': 'cave_system',
        'expected': 'negative',  # Voids should show negative anomaly
        'description': 'Large limestone cave system with massive chambers',
        'verified': True
    },
    {
        'name': 'Mammoth Cave, KY',
        'lon': -86.1000,
        'lat': 37.1862,
        'type': 'cave_system',
        'expected': 'negative',
        'description': 'World\'s longest known cave system (400+ miles)',
        'verified': True
    },
    {
        'name': 'Lechuguilla Cave, NM',
        'lon': -104.4469,
        'lat': 32.1855,
        'type': 'cave_system',
        'expected': 'negative',
        'description': 'Deep cave system, 8th longest in world',
        'verified': True
    },
    {
        'name': 'Wind Cave, SD',
        'lon': -103.4789,
        'lat': 43.5571,
        'type': 'cave_system',
        'expected': 'negative',
        'description': 'Complex boxwork cave system',
        'verified': True
    },
    {
        'name': 'Jewel Cave, SD',
        'lon': -103.8290,
        'lat': 43.7306,
        'type': 'cave_system',
        'expected': 'negative',
        'description': 'Third longest cave in world',
        'verified': True
    },
    
    # Karst/Sinkhole Areas
    {
        'name': 'The Sinks, TN',
        'lon': -83.9397,
        'lat': 35.6556,
        'type': 'karst',
        'expected': 'negative',
        'description': 'Karst sinkhole and disappearing stream',
        'verified': True
    },
    {
        'name': 'Winter Park Sinkhole, FL',
        'lon': -81.3397,
        'lat': 28.6000,
        'type': 'sinkhole',
        'expected': 'negative',
        'description': 'Major collapse sinkhole (1981)',
        'verified': True
    },
    
    # Lava Tubes
    {
        'name': 'Lava Beds National Monument, CA',
        'lon': -121.5089,
        'lat': 41.7138,
        'type': 'lava_tube',
        'expected': 'negative',
        'description': 'Extensive lava tube cave system',
        'verified': True
    },
    {
        'name': 'Ape Cave, WA',
        'lon': -122.2053,
        'lat': 46.1103,
        'type': 'lava_tube',
        'expected': 'negative',
        'description': 'Third longest lava tube in North America',
        'verified': True
    },
    
    # Dense/Massive Structures (Positive Anomalies)
    {
        'name': 'Iron Range, MN',
        'lon': -92.5369,
        'lat': 47.5211,
        'type': 'iron_ore',
        'expected': 'positive',
        'description': 'Major iron ore deposits',
        'verified': True
    },
    {
        'name': 'Bingham Canyon Mine, UT',
        'lon': -112.1486,
        'lat': 40.5225,
        'type': 'copper_ore',
        'expected': 'positive',
        'description': 'Large open-pit copper mine',
        'verified': True
    },
    {
        'name': 'Sudbury Basin, ON (USA border)',
        'lon': -81.0,
        'lat': 46.5,
        'type': 'impact_crater',
        'expected': 'positive',
        'description': 'Meteorite impact with dense minerals',
        'verified': True
    },
    
    # Salt Domes/Diapirs
    {
        'name': 'Grand Saline Salt Dome, TX',
        'lon': -95.7094,
        'lat': 32.6718,
        'type': 'salt_dome',
        'expected': 'negative',
        'description': 'Large salt dome (lower density than surrounding rock)',
        'verified': True
    },
    
    # Underground Storage/Mines
    {
        'name': 'Strategic Petroleum Reserve, LA',
        'lon': -92.0369,
        'lat': 29.8969,
        'type': 'salt_cavern',
        'expected': 'negative',
        'description': 'Underground salt cavern oil storage',
        'verified': True
    },
]


def sample_anomaly_at_location(raster_path: Path, lon: float, lat: float, 
                                buffer_km: float = 2.0) -> Dict:
    """Sample anomaly value at a specific location with spatial averaging."""
    
    # Convert buffer from km to degrees (approximate)
    buffer_deg = buffer_km / 111.0  # 1 degree ≈ 111 km
    
    with rasterio.open(raster_path) as src:
        # Define window around point
        window = from_bounds(
            lon - buffer_deg, lat - buffer_deg,
            lon + buffer_deg, lat + buffer_deg,
            src.transform
        )
        
        # Read data in window
        try:
            data = src.read(1, window=window, masked=True)
        except ValueError:
            # Location outside raster bounds
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_pixels': 0,
                'in_bounds': False
            }
        
        # Calculate statistics
        valid_data = data.compressed()
        
        if len(valid_data) == 0:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_pixels': 0,
                'in_bounds': True
            }
        
        return {
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'min': np.min(valid_data),
            'max': np.max(valid_data),
            'valid_pixels': len(valid_data),
            'in_bounds': True
        }


def validate_features(raster_path: Path, features: List[Dict], 
                      buffer_km: float = 2.0) -> Dict:
    """Validate anomaly detection against known features."""
    
    logger.info(f"Validating {len(features)} known features...")
    logger.info(f"Sampling with {buffer_km} km buffer")
    
    results = []
    correct_detections = 0
    incorrect_detections = 0
    out_of_bounds = 0
    no_data = 0
    
    for feature in features:
        logger.info(f"Sampling: {feature['name']}")
        
        stats = sample_anomaly_at_location(
            raster_path,
            feature['lon'],
            feature['lat'],
            buffer_km
        )
        
        # Determine if detection matches expectation
        detected_correctly = None
        explanation = ""
        
        if not stats['in_bounds']:
            detected_correctly = None
            explanation = "Location outside map bounds"
            out_of_bounds += 1
        elif stats['valid_pixels'] == 0:
            detected_correctly = None
            explanation = "No valid data at location"
            no_data += 1
        else:
            mean_value = stats['mean']
            
            if feature['expected'] == 'negative':
                # Expect negative anomaly for voids/caves
                if mean_value < -0.3:  # Threshold for detection
                    detected_correctly = True
                    explanation = f"✓ Detected negative anomaly ({mean_value:.3f}σ)"
                    correct_detections += 1
                else:
                    detected_correctly = False
                    explanation = f"✗ Expected negative, got {mean_value:.3f}σ"
                    incorrect_detections += 1
                    
            elif feature['expected'] == 'positive':
                # Expect positive anomaly for dense structures
                if mean_value > 0.3:
                    detected_correctly = True
                    explanation = f"✓ Detected positive anomaly ({mean_value:.3f}σ)"
                    correct_detections += 1
                else:
                    detected_correctly = False
                    explanation = f"✗ Expected positive, got {mean_value:.3f}σ"
                    incorrect_detections += 1
        
        result = {
            'feature': feature,
            'stats': stats,
            'detected_correctly': detected_correctly,
            'explanation': explanation
        }
        results.append(result)
        
        logger.info(f"  {explanation}")
    
    # Calculate success rate
    testable = len(features) - out_of_bounds - no_data
    if testable > 0:
        success_rate = correct_detections / testable
    else:
        success_rate = 0.0
    
    return {
        'results': results,
        'summary': {
            'total_features': len(features),
            'testable': testable,
            'correct': correct_detections,
            'incorrect': incorrect_detections,
            'out_of_bounds': out_of_bounds,
            'no_data': no_data,
            'success_rate': success_rate
        }
    }


def generate_validation_report(validation: Dict, output_path: Path):
    """Generate detailed validation report."""
    
    summary = validation['summary']
    results = validation['results']
    
    report = f"""
VALIDATION REPORT: Multi-Resolution Fusion vs Known Features
{'=' * 70}

SUMMARY
-------
Total Features Tested: {summary['total_features']}
Testable (within bounds, has data): {summary['testable']}
Correct Detections: {summary['correct']}
Incorrect Detections: {summary['incorrect']}
Out of Bounds: {summary['out_of_bounds']}
No Data: {summary['no_data']}

SUCCESS RATE: {summary['success_rate']:.1%}

{'=' * 70}

DETAILED RESULTS
----------------

"""
    
    # Group by type
    by_type = {}
    for r in results:
        ftype = r['feature']['type']
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append(r)
    
    for ftype, type_results in sorted(by_type.items()):
        report += f"\n{ftype.upper().replace('_', ' ')}\n{'-' * 70}\n"
        
        for r in type_results:
            feature = r['feature']
            stats = r['stats']
            
            report += f"\n{feature['name']}\n"
            report += f"  Location: {feature['lat']:.4f}°N, {abs(feature['lon']):.4f}°W\n"
            report += f"  Type: {feature['type']}\n"
            report += f"  Expected: {feature['expected']} anomaly\n"
            report += f"  Description: {feature['description']}\n"
            
            if stats['in_bounds']:
                if stats['valid_pixels'] > 0:
                    report += f"  Measured Anomaly: {stats['mean']:.3f}σ (±{stats['std']:.3f})\n"
                    report += f"  Range: {stats['min']:.3f}σ to {stats['max']:.3f}σ\n"
                    report += f"  Valid Pixels: {stats['valid_pixels']}\n"
                else:
                    report += f"  Status: No valid data\n"
            else:
                report += f"  Status: Outside map bounds\n"
                
            report += f"  Result: {r['explanation']}\n"
    
    report += f"\n{'=' * 70}\n"
    report += "\nINTERPRETATION\n"
    report += "-" * 70 + "\n"
    report += f"""
A successful detection means the measured anomaly matches the expected signature:
- Caves/voids/karst: Should show NEGATIVE anomalies (< -0.3σ)
- Dense structures/ores: Should show POSITIVE anomalies (> +0.3σ)

Success rate of {summary['success_rate']:.1%} indicates the fusion pipeline is
{'performing well' if summary['success_rate'] > 0.7 else 'needs improvement'} at detecting known subsurface features.

NOTES:
- Values in sigma (σ) units = standard deviations from regional mean
- 2km buffer used for spatial averaging
- Threshold: |0.3σ| for detection significance
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Validation report saved: {output_path}")


def create_validation_map(validation: Dict, raster_path: Path, output_path: Path):
    """Create map showing validation results."""
    
    logger.info("Creating validation map...")
    
    # Read raster for context (downsampled)
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        height, width = src.height, src.width
        downsample = max(1, max(height, width) // 2000)
        
        if downsample > 1:
            data = src.read(
                1,
                out_shape=(height // downsample, width // downsample),
                resampling=rasterio.enums.Resampling.average,
                masked=True
            )
        else:
            data = src.read(1, masked=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot anomaly map
    im = ax.imshow(
        data,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        cmap='RdBu_r',
        vmin=-1.5,
        vmax=1.5,
        alpha=0.7,
        interpolation='bilinear'
    )
    
    # Plot validation points
    for r in validation['results']:
        feature = r['feature']
        
        if not r['stats']['in_bounds']:
            continue
            
        lon, lat = feature['lon'], feature['lat']
        
        if r['detected_correctly'] is True:
            color = 'lime'
            marker = 'o'
            size = 150
            label = '✓ Correct'
        elif r['detected_correctly'] is False:
            color = 'red'
            marker = 'x'
            size = 200
            label = '✗ Incorrect'
        else:
            color = 'gray'
            marker = 's'
            size = 100
            label = '? No Data'
        
        ax.scatter(lon, lat, c=color, marker=marker, s=size, 
                  edgecolors='black', linewidths=2, zorder=10,
                  label=label if lon == feature['lon'] else "")  # Label once
        
        # Add text label
        ax.text(lon, lat + 0.5, feature['name'].split(',')[0], 
               fontsize=8, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Strength (σ)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title('Validation: Known Features vs Detected Anomalies', 
                fontsize=14, fontweight='bold')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Validation map saved: {output_path}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate fusion results against known underground features"
    )
    parser.add_argument(
        'raster',
        type=str,
        help='Path to fused anomaly GeoTIFF'
    )
    parser.add_argument(
        '--buffer',
        type=float,
        default=2.0,
        help='Sampling buffer radius in km (default: 2.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for validation results'
    )
    
    args = parser.parse_args()
    
    raster_path = Path(args.raster)
    if not raster_path.exists():
        logger.error(f"Raster file not found: {raster_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = raster_path.parent
    
    # Run validation
    validation = validate_features(raster_path, KNOWN_FEATURES, args.buffer)
    
    # Generate report
    report_path = output_dir / f"{raster_path.stem}_validation_report.txt"
    generate_validation_report(validation, report_path)
    
    # Create map
    map_path = output_dir / f"{raster_path.stem}_validation_map.png"
    create_validation_map(validation, raster_path, map_path)
    
    # Print summary
    summary = validation['summary']
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nTested: {summary['testable']} features")
    print(f"Correct: {summary['correct']}")
    print(f"Incorrect: {summary['incorrect']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"\nReports:")
    print(f"  Text: {report_path}")
    print(f"  Map:  {map_path}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()