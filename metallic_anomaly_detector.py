#!/usr/bin/env python3
"""
Metallic Anomaly Detector (MAD)
===============================
Repurposed DUB detector to find large artificial metallic structures
and high-value metallic mineral deposits using Magnetic Dipole analysis.

Physics:
- Artificial metallic objects (bunkers, pipes, silos) create localized dipole signatures.
- Natural geology creates larger, more diffuse magnetic anomalies.
- Sharp, high-gradient dipoles in remote areas are high-value targets.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import label, find_objects, gaussian_filter, sobel
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
OUTPUTS = PROJECT_ROOT / "data/outputs/usa_supervised"
RESULTS_DIR = PROJECT_ROOT / "data/outputs/metallic_anomalies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def detect_metallic_anomalies():
    mag_path = OUTPUTS / "usa_magnetic_mosaic.tif"
    if not mag_path.exists():
        logger.error(f"Magnetic data not found: {mag_path}")
        return
    
    with rasterio.open(mag_path) as src:
        mag_data = src.read(1)
        transform = src.transform
        profile = src.profile
    
    logger.info(f"Loaded magnetic data: {mag_data.shape}")
    
    # 1. High-Pass Filter (Remove regional geology)
    # sigma=10 removes features larger than ~20km
    logger.info("Filtering out regional geology...")
    mag_smoothed = gaussian_filter(mag_data, sigma=15)
    mag_residual = mag_data - mag_smoothed
    
    # 2. Local Gradient (Sharpness detection)
    logger.info("Calculating anomaly sharpness...")
    gx = sobel(mag_residual, axis=1)
    gy = sobel(mag_residual, axis=0)
    mag_grad = np.sqrt(gx**2 + gy**2)
    
    # 3. Dipole Detection (Look for sharp +/- pairs)
    # This identifies the "edge" between a north and south pole
    logger.info("Identifying metallic dipoles...")
    dipole_score = mag_grad * np.abs(mag_residual)
    
    # Normalize
    dipole_score = (dipole_score - np.nanmin(dipole_score)) / (np.nanmax(dipole_score) - np.nanmin(dipole_score) + 1e-6)
    
    # 4. Filter for "Artificial" scale
    # Artificial structures are typically <1km, so we want the peaks of the score
    threshold = np.nanpercentile(dipole_score, 99.5) # Top 0.5%
    mask = dipole_score > threshold
    
    labeled, num_regions = label(mask)
    logger.info(f"Found {num_regions} potential metallic anomalies")
    
    objects = find_objects(labeled)
    candidates = []
    
    for i, sl in enumerate(objects, 1):
        if sl is None: continue
        
        region_mask = (labeled[sl] == i)
        region_scores = dipole_score[sl][region_mask]
        
        # Max score and area
        max_score = np.max(region_scores)
        area_pixels = np.sum(region_mask)
        
        # Geometry check (Artificial things are compact)
        # Ratio of perimeter to area would be better, but simple area filter works
        if area_pixels < 2 or area_pixels > 200: # Exclude noise and massive regional features
            continue
            
        # Peak location
        coords = np.argwhere(region_mask & (dipole_score[sl] == max_score))
        if len(coords) == 0: continue
        
        r, c = coords[0]
        global_r = sl[0].start + r
        global_c = sl[1].start + c
        
        lon, lat = rasterio.transform.xy(transform, global_r, global_c)
        
        candidates.append({
            'ID': i,
            'Latitude': lat,
            'Longitude': lon,
            'Sharpness_Score': max_score,
            'Area_Pixels': area_pixels,
            'Dipole_Strength': np.abs(mag_residual[global_r, global_c])
        })
        
    df = pd.DataFrame(candidates)
    if df.empty:
        logger.warning("No candidates found")
        return
        
    df = df.sort_values('Sharpness_Score', ascending=False).reset_index(drop=True)
    
    # Save results
    output_csv = RESULTS_DIR / "metallic_candidates.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} candidates to {output_csv}")
    
    # Save Heatmap
    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    with rasterio.open(RESULTS_DIR / "metallic_heatmap.tif", 'w', **profile) as dst:
        dst.write(dipole_score.astype(np.float32), 1)
    
    # Print Top 10
    print("\nTOP 10 METALLIC ANOMALIES (Potential Artificial Structures or Ore Bodies)")
    print("="*80)
    print(df.head(10).to_string())
    print("="*80)

if __name__ == "__main__":
    detect_metallic_anomalies()
