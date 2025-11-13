#!/usr/bin/env python3
"""
Validate multi-resolution fusion outputs against known underground features.

The script samples fused anomaly rasters near a curated list of caves, mines and
other subsurface structures. A feature counts as correctly detected when the
average anomaly within the sampling window exceeds a conservative threshold and
matches the expected sign (negative for voids, positive for dense structures).
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
from typing import List, Dict, Tuple, Optional
import json
from utils.config_shim import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DETECTION_THRESHOLD_SIGMA = 0.5  # Require at least ±0.5σ signal strength
MIN_VALID_PIXELS = 25            # Ignore samples with insufficient coverage


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
    insufficient_data = 0
    
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
        elif stats['valid_pixels'] < MIN_VALID_PIXELS:
            detected_correctly = None
            explanation = (
                f"Insufficient valid pixels ({stats['valid_pixels']} < {MIN_VALID_PIXELS})"
            )
            insufficient_data += 1
        else:
            mean_value = float(stats['mean'])
            expected_sign = feature.get('expected', 'any').lower()
            threshold = DETECTION_THRESHOLD_SIGMA

            if expected_sign == 'negative':
                meets_expectation = mean_value <= -threshold
                requirement = f"≤ -{threshold:.2f}σ"
                expectation_desc = 'negative'
            elif expected_sign == 'positive':
                meets_expectation = mean_value >= threshold
                requirement = f"≥ {threshold:.2f}σ"
                expectation_desc = 'positive'
            else:
                meets_expectation = abs(mean_value) >= threshold
                requirement = f"|σ| ≥ {threshold:.2f}"
                expectation_desc = 'positive or negative'

            detected_correctly = bool(meets_expectation)

            if detected_correctly:
                explanation = (
                    f"✓ Mean anomaly {mean_value:.3f}σ meets the {expectation_desc} "
                    f"requirement ({requirement})."
                )
                correct_detections += 1
            else:
                explanation = (
                    f"✗ Mean anomaly {mean_value:.3f}σ does not satisfy the expected "
                    f"{expectation_desc} requirement ({requirement})."
                )
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
    testable = len(features) - out_of_bounds - no_data - insufficient_data
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
            'insufficient_data': insufficient_data,
            'success_rate': success_rate
        }
    }


def generate_validation_report(validation: Dict, output_path: Path, enhanced_metrics: Optional[Dict] = None):
    """Generate detailed validation report.

    The default report structure and text are preserved to maintain docs/Pages compatibility.
    When enhanced_metrics is provided (opt-in), an additional section is appended at the end.
    """
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
Insufficient Data (<{MIN_VALID_PIXELS} valid pixels): {summary['insufficient_data']}

SUCCESS RATE: {summary['success_rate']:.1%}
Detection threshold: ±{DETECTION_THRESHOLD_SIGMA:.2f}σ
Minimum valid pixels: {MIN_VALID_PIXELS}

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
Detection criteria:
- Mean anomaly must exceed ±{DETECTION_THRESHOLD_SIGMA:.2f}σ within the sampling buffer.
- The anomaly sign must match the expected behaviour of the feature class.
- At least {MIN_VALID_PIXELS} valid pixels are required for a conclusive result.

Use the success rate as a coarse indication of how consistently the fused raster
captures known targets. Investigate locations flagged as incorrect or
insufficient data to understand whether additional preprocessing or data layers
are necessary.

NOTES:
- Values are expressed in standard deviation (σ) units relative to the regional mean.
- Buffers are applied in degrees converted from the requested kilometre radius.
"""
    # Append enhanced metrics if provided (opt-in, non-breaking)
    if enhanced_metrics:
        report += f"\n{'=' * 70}\n"
        report += "ENHANCED METRICS (opt-in)\n"
        report += "-" * 70 + "\n"
        # By-type metrics
        by_type = enhanced_metrics.get('by_type', {})
        if by_type:
            report += "Per-type performance:\n"
            for ftype, m in sorted(by_type.items()):
                testable = m.get('testable', 0)
                correct = m.get('correct', 0)
                success = (correct / testable) if testable else 0.0
                avg_abs = m.get('avg_abs_mean', float('nan'))
                report += f"  - {ftype}: testable={testable}, correct={correct}, success={success:.1%}, avg|mean|={avg_abs:.3f}σ\n"
        # Sigma bins
        bins = enhanced_metrics.get('sigma_bins', {})
        if bins:
            report += "Signal strength distribution (by |σ|):\n"
            report += f"  - ≥ 1.0σ: {bins.get('>=1.0', 0)}\n"
            report += f"  - ≥ 0.5σ and < 1.0σ: {bins.get('>=0.5', 0)}\n"
            report += f"  - < 0.5σ: {bins.get('<0.5', 0)}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Validation report saved: {output_path}")
    
def compute_enhanced_metrics(validation: Dict) -> Dict:
    """Compute optional enhanced validation metrics (opt-in only).
    
    Returns:
        Dict with:
          - by_type: {type: {testable, correct, avg_abs_mean}}
          - sigma_bins: counts by absolute sigma thresholds
    """
    by_type: Dict[str, Dict[str, float]] = {}
    sigma_bins = {'>=1.0': 0, '>=0.5': 0, '<0.5': 0}
    for r in validation.get('results', []):
        ftype = r['feature'].get('type', 'unknown')
        stats = r.get('stats', {})
        in_bounds = stats.get('in_bounds', False)
        valid_pixels = stats.get('valid_pixels', 0) or 0
        mean_val = stats.get('mean', float('nan'))
        if ftype not in by_type:
            by_type[ftype] = {'testable': 0, 'correct': 0, 'sum_abs_mean': 0.0}
        if in_bounds and valid_pixels >= MIN_VALID_PIXELS and not np.isnan(mean_val):
            by_type[ftype]['testable'] += 1
            if r.get('detected_correctly') is True:
                by_type[ftype]['correct'] += 1
            by_type[ftype]['sum_abs_mean'] += abs(float(mean_val))
            # Sigma distribution
            a = abs(float(mean_val))
            if a >= 1.0:
                sigma_bins['>=1.0'] += 1
            elif a >= 0.5:
                sigma_bins['>=0.5'] += 1
            else:
                sigma_bins['<0.5'] += 1
    # Finalize averages
    out_by_type: Dict[str, Dict[str, float]] = {}
    for ftype, m in by_type.items():
        t = int(m['testable'])
        avg_abs = (m['sum_abs_mean'] / t) if t > 0 else float('nan')
        out_by_type[ftype] = {
            'testable': t,
            'correct': int(m['correct']),
            'avg_abs_mean': float(avg_abs),
        }
    return {'by_type': out_by_type, 'sigma_bins': sigma_bins}

def to_python(obj):
    """Convert numpy scalars/arrays within nested structures to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


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
    labels_used = set()

    for r in validation['results']:
        feature = r['feature']

        if not r['stats']['in_bounds']:
            continue

        lon, lat = feature['lon'], feature['lat']

        if r['detected_correctly'] is True:
            color = 'lime'
            marker = 'o'
            size = 150
            label = 'Correct detection'
        elif r['detected_correctly'] is False:
            color = 'red'
            marker = 'x'
            size = 200
            label = 'Incorrect detection'
        else:
            if r['stats']['valid_pixels'] == 0:
                color = 'gray'
                label = 'No data'
            else:
                color = 'orange'
                label = 'Insufficient data'
            marker = 's'
            size = 120

        display_label = None if label in labels_used else label
        if display_label:
            labels_used.add(display_label)

        ax.scatter(
            lon,
            lat,
            c=color,
            marker=marker,
            s=size,
            edgecolors='black',
            linewidths=2,
            zorder=10,
            label=display_label,
        )
        
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
    # Opt-in features (non-breaking; defaults unchanged)
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Enable enhanced validation metrics (opt-in; default: disabled)'
    )
    parser.add_argument(
        '--json-report',
        action='store_true',
        help='Also write JSON report alongside text (opt-in; default: disabled)'
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
    
    # Determine opt-in feature flags from CLI or config_shim
    enhanced_flag = args.enhanced or str(get_config('GAM_VALIDATION_ENHANCED', 'false')).lower() == 'true'
    json_flag = args.json_report or str(get_config('GAM_VALIDATION_JSON', 'false')).lower() == 'true'
    logger.info(f"Validation mode: {'ENHANCED' if enhanced_flag else 'STANDARD'}")
    if json_flag:
        logger.info("JSON report: ENABLED")
    
    # Run validation
    validation = validate_features(raster_path, KNOWN_FEATURES, args.buffer)
    
    # Enhanced metrics (opt-in)
    metrics = compute_enhanced_metrics(validation) if enhanced_flag else None
    
    # Generate report (text always; structure unchanged; enhanced section appended only when enabled)
    report_path = output_dir / f"{raster_path.stem}_validation_report.txt"
    generate_validation_report(validation, report_path, metrics)
    
    # Optional JSON report (opt-in)
    if json_flag:
        json_path = output_dir / f"{raster_path.stem}_validation_report.json"
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(
                {'summary': to_python(validation['summary']),
                 'results': to_python(validation['results']),
                 'enhanced_metrics': to_python(metrics) if metrics else None},
                jf, indent=2
            )
        logger.info(f"Validation JSON saved: {json_path}")
    
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
    print(f"Out of Bounds: {summary['out_of_bounds']}")
    print(f"No Data: {summary['no_data']}")
    print(f"Insufficient Data: {summary['insufficient_data']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"\nReports:")
    print(f"  Text: {report_path}")
    print(f"  Map:  {map_path}")
    if json_flag:
        print(f"  JSON: {json_path}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()