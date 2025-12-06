#!/usr/bin/env python3
"""
Phase 9: Intelligence Product Generation (KML Dossier).

Converts the raw target list into a formatted Google Earth KML file.
- Color-coded pins based on anomaly score.
- Metadata injection (Score, Rank, Coordinates).
- 'LookAt' camera positioning for immediate 3D inspection.
"""

import os
import argparse
import csv
import logging
from pathlib import Path
import simplekml # You may need to pip install simplekml

from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_kml(csv_path, output_path):
    kml = simplekml.Kml()
    
    # Define Styles
    style_high = simplekml.Style()
    style_high.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-stars.png'
    style_high.iconstyle.scale = 1.2

    style_med = simplekml.Style()
    style_med.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-blank.png'
    style_med.iconstyle.scale = 0.8

    targets = []
    
    # Read CSV
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(row)
    except Exception as e:
        logger.error(f"Could not read targets file: {e}")
        return

    logger.info(f"Processing {len(targets)} targets for dossier...")

    for i, t in enumerate(targets):
        try:
            lat = float(t['lat'])
            lon = float(t['lon'])
            score = float(t['score'])
            rank = i + 1
            
            # Create Point
            pnt = kml.newpoint(name=f"#{rank} Score: {score:.4f}", coords=[(lon, lat)])
            pnt.description = f"Rank: {rank}\nAnomaly Score: {score}\nLat: {lat}\nLon: {lon}"
            
            # Apply Style based on Score
            # Adjust these thresholds based on your data distribution
            if score > 0.5:
                pnt.style = style_high
            else:
                pnt.style = style_med

            # Add 'LookAt' for 3D orientation
            pnt.lookat.latitude = lat
            pnt.lookat.longitude = lon
            pnt.lookat.altitude = 0
            pnt.lookat.range = 2000 # Camera distance in meters
            pnt.lookat.tilt = 45    # 45 degree angle for 3D terrain view
            pnt.lookat.heading = 0
            
        except ValueError:
            continue

    kml.save(output_path)
    logger.info(f"Dossier saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Target Dossier Generator")
    parser.add_argument("--input", type=str, default=str(OUTPUTS_DIR / "targets_v1.csv"))
    parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "final_targets.kml"))
    args = parser.parse_args()

    # Need simplekml
    try:
        import simplekml
    except ImportError:
        logger.error("Missing library. Please run: pip install simplekml")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Target list not found: {input_path}")
        return

    create_kml(input_path, args.output)

if __name__ == "__main__":
    main()