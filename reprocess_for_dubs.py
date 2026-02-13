#!/usr/bin/env python3
"""
Reprocess USA Data for DUB (Deep Underground Base) Detection
=============================================================

This script reprocesses the gravity and magnetic data specifically for
detecting underground voids and artificial cavities.

Key changes from mineral detection:
1. PINN inversion runs in 'void' mode (looking for negative density)
2. Feature engineering emphasizes edge sharpness over magnitude
3. Classification trained on known DUB locations instead of MRDS
4. Output targets are ranked by void probability, not mineral potential

Usage:
    python reprocess_for_dubs.py [--full]
    
    --full: Run complete reprocessing including PINN inversion (slow, ~1 hour)
    Default: Skip PINN, use existing gravity residuals
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
OUTPUTS = PROJECT_ROOT / "data/outputs"
DUB_OUTPUT = OUTPUTS / "dub_detection"


def run_command(cmd, description):
    """Run a command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    elapsed = time.time() - start
    
    if result.stdout:
        for line in result.stdout.split('\n')[-30:]:  # Last 30 lines
            logger.info(f"  {line}")
    
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        if result.stderr:
            logger.error(result.stderr)
        return False
    
    logger.info(f"Completed in {elapsed:.1f}s")
    return True


def step_1_gravity_inversion_void_mode():
    """
    Run PINN gravity inversion in VOID mode.
    
    This is the key difference - instead of looking for dense ore bodies,
    we look for low-density voids (negative anomalies).
    """
    logger.info("Running PINN gravity inversion in VOID mode...")
    
    input_gravity = OUTPUTS / "usa_supervised/usa_gravity_residual.tif"
    output_density = DUB_OUTPUT / "void_density_contrast.tif"
    magnetic_guide = OUTPUTS / "usa_supervised/usa_magnetic_mosaic.tif"
    
    if not input_gravity.exists():
        logger.error(f"Input gravity file not found: {input_gravity}")
        return False
    
    DUB_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "pinn_gravity_inversion.py",
        "--input", str(input_gravity),
        "--output", str(output_density),
        "--mode", "void",  # KEY: Look for negative density anomalies
    ]
    
    if magnetic_guide.exists():
        cmd.extend(["--magnetic_guide", str(magnetic_guide)])
    
    return run_command(cmd, "PINN Gravity Inversion (Void Mode)")


def step_2_run_dub_detector():
    """
    Run the DUB detection pipeline on the processed data.
    """
    logger.info("Running DUB detector...")
    
    cmd = [
        sys.executable, "train_dub_detector.py",
        "--sensitivity", "medium"
    ]
    
    return run_command(cmd, "DUB Detection Pipeline")


def step_3_validate_results():
    """
    Run validation against known DUB locations and generate metrics.
    """
    logger.info("Validating DUB detection results...")
    
    candidates_file = DUB_OUTPUT / "dub_candidates.csv"
    known_dubs_file = PROJECT_ROOT / "data/known_dubs.csv"
    
    if not candidates_file.exists():
        logger.error("No candidates file found - detection may have failed")
        return False
    
    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree
    
    candidates = pd.read_csv(candidates_file)
    known_dubs = pd.read_csv(known_dubs_file)
    
    # Filter to USA
    usa_dubs = known_dubs[
        (known_dubs['lat'] >= 24.5) & (known_dubs['lat'] <= 49.5) &
        (known_dubs['lon'] >= -124.5) & (known_dubs['lon'] <= -66.5)
    ]
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  Total candidates: {len(candidates)}")
    logger.info(f"  Known USA DUBs: {len(usa_dubs)}")
    
    if len(candidates) == 0:
        logger.warning("No candidates to validate!")
        return True
    
    # Check how many known DUBs are detected
    cand_coords = candidates[['Latitude', 'Longitude']].values
    cand_tree = cKDTree(cand_coords)
    
    detected = 0
    for idx, row in usa_dubs.iterrows():
        known_coord = np.array([[row['lat'], row['lon']]])
        dist, _ = cand_tree.query(known_coord, k=1)
        if dist[0] * 111 < 15:  # Within 15 km
            detected += 1
            logger.info(f"  ✓ DETECTED: {row['name']}")
    
    detection_rate = detected / len(usa_dubs) if len(usa_dubs) > 0 else 0
    
    logger.info(f"\n  Detection Rate: {detected}/{len(usa_dubs)} ({detection_rate*100:.1f}%)")
    
    # Score distribution
    logger.info(f"\n  Score Distribution:")
    logger.info(f"    Min Score: {candidates['DUB_Score'].min():.4f}")
    logger.info(f"    Max Score: {candidates['DUB_Score'].max():.4f}")
    logger.info(f"    Mean Score: {candidates['DUB_Score'].mean():.4f}")
    logger.info(f"    Std Score: {candidates['DUB_Score'].std():.4f}")
    
    return True


def step_4_generate_report():
    """
    Generate a summary report of DUB detection results.
    """
    logger.info("Generating DUB detection report...")
    
    report_path = DUB_OUTPUT / "DUB_DETECTION_REPORT.md"
    
    import pandas as pd
    from datetime import datetime
    
    candidates_file = DUB_OUTPUT / "dub_candidates.csv"
    novel_file = DUB_OUTPUT / "dub_novel_candidates.csv"
    top100_file = DUB_OUTPUT / "dub_top100.csv"
    
    candidates = pd.read_csv(candidates_file) if candidates_file.exists() else pd.DataFrame()
    novel = pd.read_csv(novel_file) if novel_file.exists() else pd.DataFrame()
    
    report = f"""# DUB Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Candidates Detected**: {len(candidates)}
- **Novel Candidates (Unknown)**: {len(novel)}
- **Detection Threshold**: 95th percentile

## Methodology

This detection system uses physics-based analysis to identify subsurface voids:

1. **Gravity Analysis**: Underground voids create negative gravity anomalies
2. **Edge Detection**: Artificial structures have sharper boundaries than natural caves
3. **Magnetic Disturbance**: Large underground facilities may disturb local magnetic field
4. **Composite Scoring**: Multi-feature ensemble ranking

## Top 20 Candidates

| Rank | Latitude | Longitude | DUB Score | Void Strength | Area (km²) |
|------|----------|-----------|-----------|---------------|------------|
"""
    
    if not candidates.empty:
        for i, row in candidates.head(20).iterrows():
            report += f"| {i+1} | {row['Latitude']:.4f} | {row['Longitude']:.4f} | {row['DUB_Score']:.4f} | {row['Void_Strength']:.4f} | {row['Area_km2']:.2f} |\n"
    
    report += """

## Novel Candidates (Not Matching Known Facilities)

These are candidates that don't match any known declassified underground facility
within 15km. They may represent:

- Undisclosed facilities
- Natural geological features (caves, karst)
- Industrial underground structures
- False positives

**Further investigation recommended for high-scoring novel candidates.**

"""
    
    if not novel.empty:
        report += "### Top 10 Novel Candidates\n\n"
        report += "| Rank | Latitude | Longitude | DUB Score | Notes |\n"
        report += "|------|----------|-----------|-----------|-------|\n"
        
        for i, row in novel.head(10).iterrows():
            report += f"| {i+1} | {row['Latitude']:.4f} | {row['Longitude']:.4f} | {row['DUB_Score']:.4f} | Requires investigation |\n"
    
    report += """
## Files Generated

- `dub_probability.tif`: GeoTIFF probability map (higher = more likely void)
- `dub_candidates.csv`: All detected candidates with scores
- `dub_top100.csv`: Top 100 candidates by score
- `dub_novel_candidates.csv`: Candidates not matching known facilities

## Limitations

- Detection is based on publicly available gravity/magnetic survey data
- Deep facilities (>1km) may be difficult to detect
- Small facilities may fall below detection threshold
- Natural caves and karst formations may cause false positives

## Next Steps

1. Cross-reference high-scoring candidates with satellite imagery
2. Check for nearby infrastructure (roads, power lines, security perimeters)
3. Investigate anomalous patterns not matching natural geological features
4. Verify against historical records and land ownership data
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reprocess USA data for DUB detection")
    parser.add_argument("--full", action="store_true",
                        help="Run full reprocessing including PINN inversion (slow)")
    parser.add_argument("--skip-pinn", action="store_true",
                        help="Skip PINN inversion step")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("  GEOANOMALYMAPPER - DUB DETECTION REPROCESSING")
    logger.info("  Mode: Deep Underground Base Detection")
    logger.info("="*70)
    
    start_time = time.time()
    
    steps = [
        (step_1_gravity_inversion_void_mode, "PINN Inversion (Void Mode)", args.full and not args.skip_pinn),
        (step_2_run_dub_detector, "DUB Detection", True),
        (step_3_validate_results, "Validation", True),
        (step_4_generate_report, "Report Generation", True),
    ]
    
    results = {}
    
    for step_func, name, should_run in steps:
        if should_run:
            try:
                results[name] = step_func()
            except Exception as e:
                logger.error(f"Step '{name}' failed with error: {e}")
                results[name] = False
        else:
            logger.info(f"Skipping: {name}")
            results[name] = None
    
    # Summary
    elapsed = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("  REPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    
    for step, result in results.items():
        if result is None:
            status = "⏭ SKIPPED"
        elif result:
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"
        logger.info(f"  {step}: {status}")
    
    # Check outputs
    logger.info("\nOutput Files:")
    for f in sorted(DUB_OUTPUT.glob("*")):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name}: {size_kb:.1f} KB")
    
    return 0 if all(r is None or r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
