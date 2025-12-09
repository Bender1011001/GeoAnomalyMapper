#!/usr/bin/env python3
"""
Validation Script for California Mineral Deposits.

This script validates a GeoAnomalyMapper probability raster against a curated list 
of 15 major known mineral deposits in California. It samples the raster at specific 
GPS coordinates and reports whether the model successfully detected the anomaly 
(Mass Excess/Positive Signal).

Usage:
    python validate_california.py path/to/probability_map.tif --threshold 1.5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- KNOWN CALIFORNIA MINERAL DEPOSITS ---
# Coordinates sourced from USGS MRDS, Mindat, and NASA records.
# 'expected': 'positive' means we expect a high probability (Mass Excess/Mineral Mode).
KNOWN_FEATURES = [
    # Rare Earths
    {
        'name': 'Mountain Pass Mine',
        'lat': 35.4769,
        'lon': -115.5333,
        'type': 'Rare Earths',
        'desc': 'Major carbonatite REE deposit (Bastnaesite).',
        'expected': 'positive'
    },
    # Borates
    {
        'name': 'Rio Tinto Boron Mine',
        'lat': 35.0429,
        'lon': -117.6793,
        'type': 'Borates',
        'desc': 'Largest open-pit borax mine in the world.',
        'expected': 'positive'
    },
    # Gold (Active/Historic Major)
    {
        'name': 'Mesquite Mine',
        'lat': 33.0603,
        'lon': -114.9944,
        'type': 'Gold',
        'desc': 'Large active open-pit gold mine (Glamis).',
        'expected': 'positive'
    },
    {
        'name': 'Castle Mountain Mine',
        'lat': 35.2811,
        'lon': -115.1025,
        'type': 'Gold',
        'desc': 'Epithermal gold deposit in San Bernardino County.',
        'expected': 'positive'
    },
    {
        'name': 'Soledad Mountain Mine',
        'lat': 34.9978,
        'lon': -118.1806,
        'type': 'Gold/Silver',
        'desc': 'Gold-silver deposit near Mojave.',
        'expected': 'positive'
    },
    {
        'name': 'Briggs Mine',
        'lat': 35.9375,
        'lon': -117.1850,
        'type': 'Gold',
        'desc': 'Carlin-type gold deposit in Inyo County.',
        'expected': 'positive'
    },
    {
        'name': 'Golden Queen Mine',
        'lat': 34.9869,
        'lon': -118.1889,
        'type': 'Gold/Silver',
        'desc': 'Historic and active production on Soledad Mountain.',
        'expected': 'positive'
    },
    {
        'name': 'McLaughlin Mine',
        'lat': 38.8381,
        'lon': -122.3639,
        'type': 'Gold',
        'desc': 'World-class epithermal deposit (Homestake), now a reserve.',
        'expected': 'positive'
    },
    {
        'name': 'Gold Run Mining District',
        'lat': 39.1808,
        'lon': -120.8558,
        'type': 'Gold',
        'desc': 'Historic hydraulic mining district in the Sierra Nevada.',
        'expected': 'positive'
    },
    {
        'name': 'Rand Mining District',
        'lat': 35.3500,
        'lon': -117.6500,
        'type': 'Gold/Silver/Tungsten',
        'desc': 'Includes the famous Yellow Aster mine.',
        'expected': 'positive'
    },
    # Lithium / Geothermal
    {
        'name': 'Salton Sea Geothermal Field',
        'lat': 33.1863,
        'lon': -115.5844,
        'type': 'Lithium/Geothermal',
        'desc': 'High-salinity brines rich in Lithium ("Lithium Valley").',
        'expected': 'positive' 
        # Note: Brines might show as mass deficit (void) or excess depending on host rock density.
        # In Mineral Mode, we usually look for the dense host structures.
    },
    # Iron / Base Metals
    {
        'name': 'Eagle Mountain Mine',
        'lat': 33.8647,
        'lon': -115.5203,
        'type': 'Iron',
        'desc': 'Massive iron ore skarn deposit.',
        'expected': 'positive'
    },
    {
        'name': 'Iron Mountain Mine',
        'lat': 40.6722,
        'lon': -122.5278,
        'type': 'Iron/Copper/Zinc',
        'desc': 'Massive sulfide deposit near Redding.',
        'expected': 'positive'
    },
    # Other Industrial
    {
        'name': 'Leviathan Mine',
        'lat': 38.7081,
        'lon': -119.6572,
        'type': 'Sulfur/Copper',
        'desc': 'Open-pit sulfur mine in Alpine County.',
        'expected': 'positive'
    },
    {
        'name': 'New Idria Mercury Mine',
        'lat': 36.4144,
        'lon': -120.6736,
        'type': 'Mercury',
        'desc': 'One of North America\'s largest mercury producers.',
        'expected': 'positive'
    }
]

def sample_at_location(src, lat, lon, buffer_pixels=3):
    """
    Sample the raster at a specific lat/lon with a small buffer window.
    Returns the mean value and max value within the window.
    """
    # Convert Lat/Lon to Row/Col
    try:
        row, col = src.index(lon, lat)
    except Exception:
        return None  # Out of bounds

    # Check bounds
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        return None

    # Define window
    window = rasterio.windows.Window(
        col - buffer_pixels, row - buffer_pixels, 
        buffer_pixels * 2 + 1, buffer_pixels * 2 + 1
    )
    
    # Read data
    try:
        data = src.read(1, window=window)
        # Mask nodata
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)
        
        if data.count() == 0:
            return None
            
        return {
            'mean': float(np.nanmean(data)),
            'max': float(np.nanmax(data))
        }
    except Exception:
        return None

def validate_features(raster_path, threshold):
    """
    Main validation loop.
    """
    logger.info(f"Validating against {len(KNOWN_FEATURES)} known California deposits...")
    logger.info(f"Raster: {raster_path}")
    logger.info(f"Detection Threshold: >= {threshold}")
    
    hits = 0
    misses = 0
    out_of_bounds = 0
    
    with rasterio.open(raster_path) as src:
        print(f"\n{'NAME':<30} | {'TYPE':<15} | {'SCORE (Max)':<12} | {'RESULT'}")
        print("-" * 75)
        
        for feature in KNOWN_FEATURES:
            stats = sample_at_location(src, feature['lat'], feature['lon'])
            
            if stats is None:
                out_of_bounds += 1
                result = "OOB (No Data)"
                score_display = "N/A"
            else:
                score = stats['max']
                score_display = f"{score:.4f}"
                
                # Check if it passes threshold
                if score >= threshold:
                    hits += 1
                    result = "✅ DETECTED"
                else:
                    misses += 1
                    result = "❌ MISSED"
            
            print(f"{feature['name']:<30} | {feature['type']:<15} | {score_display:<12} | {result}")

    # Summary
    total_valid = hits + misses
    if total_valid > 0:
        accuracy = (hits / total_valid) * 100
        print("\n" + "=" * 30)
        print("VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total Features Checked: {len(KNOWN_FEATURES)}")
        print(f"Out of Bounds:        {out_of_bounds}")
        print(f"Valid Test Cases:     {total_valid}")
        print(f"Successful Detections: {hits}")
        print(f"Missed Detections:     {misses}")
        print(f"Sensitivity (Recall):  {accuracy:.1f}%")
        print("=" * 30 + "\n")
    else:
        logger.warning("No features fell within the raster bounds. Check your --region arguments.")

def main():
    parser = argparse.ArgumentParser(description="Validate California Mineral Targets")
    parser.add_argument("raster", type=str, help="Path to the mineral probability GeoTIFF")
    parser.add_argument("--threshold", type=float, default=1.5, help="Anomaly score threshold for detection (default: 1.5)")
    
    args = parser.parse_args()
    
    raster_path = Path(args.raster)
    if not raster_path.exists():
        logger.error(f"File not found: {raster_path}")
        sys.exit(1)
        
    validate_features(raster_path, args.threshold)

if __name__ == "__main__":
    main()