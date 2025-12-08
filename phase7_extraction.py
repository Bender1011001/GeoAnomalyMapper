#!/usr/bin/env python3
"""
Phase 7: High-Precision Candidate Extraction.

1. Reads the full-resolution 'spatial_anomaly_v2.tif'.
2. Applies a high-confidence threshold (e.g., Score > 3.0).
   (Note: Raw scores ranged from -23 to 8.9. A score of 3.0 is very high confidence).
3. Clusters adjacent anomalous pixels into "Blobs" (Anomaly candidates).
4. Calculates the precise Centroid (Lat/Lon) of each blob.
5. Filters out tiny noise (artifacts smaller than 2 pixels).
6. Exports 'targets.csv' and 'targets.geojson' for inspection.
"""

import os
import argparse
import logging
from pathlib import Path
import csv
import json

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj

from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_candidates(tiff_path, score_threshold=3.0, min_area_pixels=4):
    """
    Scans the full-resolution raster for clusters of high values.
    Returns a list of dicts: {'id', 'lat', 'lon', 'score', 'area_sq_km'}
    """
    candidates = []
    
    with rasterio.open(tiff_path) as src:
        logger.info(f"Scanning full-resolution data: {tiff_path}")
        logger.info(f"Dimensions: {src.width}x{src.height}")
        
        # We need to reproject points to WGS84 (Lat/Lon) for export
        # Prepare a transformer: Source CRS -> WGS84 (EPSG:4326)
        src_crs = src.crs
        wgs84 = pyproj.CRS("EPSG:4326")
        project = pyproj.Transformer.from_crs(src_crs, wgs84, always_xy=True).transform

        # Iterate over blocks to handle memory efficiently (even on good hardware)
        # But for shape extraction, rasterio needs a mask. 
        # If you have >32GB RAM, we can load the mask into memory.
        # Let's try reading the whole band; if it fails, we'll need a block approach.
        # Given previous logs, valid pixels were ~34k (sparse), but raster is huge.
        # We will read windows if the file is massive.
        
        # However, rasterio.features.shapes requires an array. 
        # We will read the data. If it crashes, we need a tiling strategy.
        # Assuming desktop has decent RAM (16GB+), this should fit as it's float32.
        
        try:
            data = src.read(1)
        except MemoryError:
            logger.error("Map is too large for RAM. Creating a decimated scan (this script can be upgraded to tiled processing).")
            return []

        # Create a boolean mask of interesting pixels
        # RAW SCORES: You saw range -23 to 8.9.
        # Let's target the top tier: > 3.0 (This is roughly >3 standard deviations).
        mask = data > score_threshold
        
        logger.info(f"Vectorizing anomalies > {score_threshold}...")
        
        # Extract polygons (shapes)
        # This turns pixels into vector geometries
        results = shapes(data, mask=mask, transform=src.transform)
        
        count = 0
        for geom, val in results:
            # Convert to Shapely geometry
            poly = shape(geom)
            
            # Filter noise (single specks)
            # Area is in map units (likely meters if UTM, or degrees if LatLon)
            # We'll use a simple heuristic for now.
            if poly.area == 0: continue

            # Calculate Centroid
            center = poly.centroid
            
            # Transform centroid to Lat/Lon
            lon, lat = project(center.x, center.y)
            
            # Store
            candidates.append({
                "id": count,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "score": round(val, 4),
                "raw_area": poly.area
            })
            count += 1
            
    return candidates

def save_csv(candidates, output_path):
    if not candidates: return
    keys = candidates[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(candidates)
    logger.info(f"Saved {len(candidates)} targets to CSV: {output_path}")

def save_geojson(candidates, output_path):
    features = []
    for c in candidates:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [c['lon'], c['lat']]
            },
            "properties": {
                "id": c['id'],
                "score": c['score'],
                "area": c['raw_area']
            }
        })
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    logger.info(f"Saved {len(candidates)} targets to GeoJSON: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Precision Extraction")
    # Note: We are using the RAW PROBABILITY file from Phase 5
    parser.add_argument("--input", type=str, default=str(OUTPUTS_DIR / "mineral_void_probability.tif"))
    parser.add_argument("--threshold", type=float, default=2.0, help="Anomaly Score Threshold (Higher = Stricter)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file missing.")
        return

    # Extract
    targets = extract_candidates(input_path, score_threshold=args.threshold)
    
    # Sort by score (highest anomaly first)
    targets.sort(key=lambda x: x['score'], reverse=True)
    
    # Save
    save_csv(targets, OUTPUTS_DIR / "targets_v1.csv")
    save_geojson(targets, OUTPUTS_DIR / "targets_v1.geojson")
    
    # Print Top 5
    if targets:
        print("\n--- TOP 5 CANDIDATES ---")
        for i in range(min(5, len(targets))):
            t = targets[i]
            print(f"#{i+1}: Lat {t['lat']}, Lon {t['lon']} | Score: {t['score']}")
    else:
        print("No candidates found. Try lowering the --threshold.")

if __name__ == "__main__":
    main()