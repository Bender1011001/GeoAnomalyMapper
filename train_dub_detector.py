#!/usr/bin/env python3
"""
Deep Underground Base (DUB) Detection System
=============================================
Repurposed GeoAnomalyMapper for detecting subsurface voids and underground facilities.

Physics Basis:
- Underground cavities/tunnels create NEGATIVE gravity anomalies
- Large underground facilities disrupt local magnetic field patterns
- Combined gravity-magnetic analysis can isolate artificial voids from natural caves

Training Data Sources:
- Declassified FOIA documents (Area 51, Cheyenne Mountain, etc.)
- Official DoD facility locations
- Nuclear test site tunnel networks
- Known missile silo fields

Key Differences from Mineral Detection:
1. Looking for NEGATIVE density anomalies (voids) vs positive (ore bodies)
2. Focus on sharp, geometric anomalies (artificial) vs diffuse (natural)
3. Correlation with infrastructure (roads, power lines) is POSITIVE not negative
4. Ignoring natural caves/karst by filtering known geological areas

Author: GeoAnomalyMapper Team
Target: Deep Underground Bases Detection
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

# Project imports
from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "data/outputs"
DUB_TRAINING_FILE = DATA_DIR / "known_dubs.csv"

# USA Continental Bounds
USA_BOUNDS = {
    'lat_min': 24.5,
    'lat_max': 49.5,
    'lon_min': -124.5,
    'lon_max': -66.5
}

# DUB Detection Parameters
DUB_CONFIG = {
    # Physics parameters for void detection
    'target_mode': 'void',  # NEGATIVE density anomalies
    'depth_range_m': (100, 1000),  # Expected depth range of DUBs
    'min_anomaly_size_km2': 0.1,  # Minimum anomaly footprint
    'max_anomaly_size_km2': 50.0,  # Maximum (larger = not a DUB)
    
    # Detection thresholds (for NEGATIVE values, more negative = stronger void)
    'gravity_threshold_mgal': -0.5,  # Gravity anomaly threshold
    'edge_sharpness_threshold': 0.3,  # How sharp the edges are (artificial caves are sharp)
    
    # Filtering
    'exclude_known_karst': True,  # Exclude areas with natural caves
    'require_infrastructure': False,  # Require nearby roads/power (not always visible)
    'exclude_ocean': True,
}


# ==========================================
# Data Loading
# ==========================================
def load_dub_training_data():
    """Load known DUB locations for training."""
    if not DUB_TRAINING_FILE.exists():
        logger.error(f"DUB training file not found: {DUB_TRAINING_FILE}")
        return None
    
    df = pd.read_csv(DUB_TRAINING_FILE)
    logger.info(f"Loaded {len(df)} known/suspected DUB locations")
    
    # Filter to USA only for now
    usa_mask = (
        (df['lat'] >= USA_BOUNDS['lat_min']) &
        (df['lat'] <= USA_BOUNDS['lat_max']) &
        (df['lon'] >= USA_BOUNDS['lon_min']) &
        (df['lon'] <= USA_BOUNDS['lon_max'])
    )
    usa_dubs = df[usa_mask].copy()
    logger.info(f"Found {len(usa_dubs)} DUB locations within USA bounds")
    
    return usa_dubs


def load_gravity_data():
    """Load gravity anomaly data for USA."""
    gravity_path = OUTPUTS / "usa_supervised/usa_gravity_mosaic.tif"
    residual_path = OUTPUTS / "usa_supervised/usa_gravity_residual.tif"
    
    # Prefer residual (regional trend removed)
    use_path = residual_path if residual_path.exists() else gravity_path
    
    if not use_path.exists():
        logger.error(f"Gravity data not found: {use_path}")
        return None, None, None
    
    with rasterio.open(use_path) as src:
        data = src.read(1)
        transform = src.transform
        profile = src.profile
        
    logger.info(f"Loaded gravity data: {data.shape}")
    logger.info(f"  Range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f} mGal")
    
    return data, transform, profile


def load_magnetic_data():
    """Load magnetic anomaly data for USA."""
    mag_path = OUTPUTS / "usa_supervised/usa_magnetic_mosaic.tif"
    
    if not mag_path.exists():
        logger.warning(f"Magnetic data not found: {mag_path}")
        return None, None
    
    with rasterio.open(mag_path) as src:
        data = src.read(1)
        transform = src.transform
        
    logger.info(f"Loaded magnetic data: {data.shape}")
    return data, transform


# ==========================================
# PINN-Based Void Detection
# ==========================================
def run_pinn_void_inversion(gravity_path, magnetic_path=None, output_path=None):
    """
    Run the PINN gravity inversion in VOID mode.
    
    This is the core physics-informed approach:
    - PINN learns the inverse mapping from gravity anomaly to density contrast
    - In 'void' mode, it's biased to find NEGATIVE density (cavities/tunnels)
    - The physics layer ensures geophysically plausible solutions
    
    Returns:
        Path to the inverted density contrast map
    """
    from pinn_gravity_inversion import invert_gravity
    
    if output_path is None:
        output_path = OUTPUTS / "dub_detection" / "pinn_void_density.tif"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Running PINN gravity inversion in VOID mode...")
    logger.info(f"  Input gravity: {gravity_path}")
    logger.info(f"  Output density: {output_path}")

    # Use our freshly trained Synthetic DUB Expert
    model_path = PROJECT_ROOT / "synthetic_dub_pinn.pth"
    if not model_path.exists():
        logger.warning(f"Synthetic DUB model not found at {model_path}! Using untrained PINN (slower/worse).")
        checkpoint = None
    else:
        logger.info(f"Using Synthetic DUB Expert Model: {model_path}")
        checkpoint = str(model_path)
    
    try:
        invert_gravity(
            tif_path=str(gravity_path),
            output_path=str(output_path),
            target_mode='void',  # KEY: Look for negative density anomalies
            magnetic_guide_path=str(magnetic_path) if magnetic_path else None,
            checkpoint_path=checkpoint
        )
        logger.info("PINN inversion complete!")
        return output_path
    except Exception as e:
        logger.error(f"PINN inversion failed: {e}")
        return None


def load_pinn_density_output():
    """Load the PINN-inverted density contrast map."""
    pinn_output = OUTPUTS / "dub_detection" / "pinn_void_density.tif"
    
    if not pinn_output.exists():
        logger.warning(f"PINN output not found: {pinn_output}")
        return None, None, None
    
    with rasterio.open(pinn_output) as src:
        data = src.read(1)
        transform = src.transform
        profile = src.profile
    
    logger.info(f"Loaded PINN density output: {data.shape}")
    logger.info(f"  Range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f} kg/m³")
    
    # For void detection, we want NEGATIVE density (voids = less mass)
    # The model outputs density contrast, negative = void
    void_indicator = -data  # Flip: positive = stronger void
    
    return void_indicator, transform, profile


# ==========================================
# Feature Engineering for DUB Detection
# ==========================================
def calculate_void_indicators(gravity_data, magnetic_data=None, pinn_density=None):
    """
    Calculate features that indicate subsurface voids.
    
    PRIMARY: Uses PINN-inverted density contrast if available
    FALLBACK: Uses simple feature engineering on raw gravity
    
    The PINN approach is superior because:
    1. It solves the inverse problem with physics constraints
    2. It accounts for depth and source geometry
    3. It produces actual density estimates, not just anomaly indicators
    """
    features = {}
    
    # ====== PRIMARY: PINN-based features ======
    if pinn_density is not None:
        logger.info("Using PINN-inverted density for void detection")
        
        pinn_normalized = np.nan_to_num(pinn_density, nan=0.0)
        
        # PINN void strength (already inverted: positive = void)
        features['void_strength'] = np.clip(pinn_normalized, 0, None)
        
        # Normalize to 0-1 range
        pinn_max = np.nanmax(features['void_strength'])
        if pinn_max > 0:
            features['void_strength_normalized'] = features['void_strength'] / pinn_max
        else:
            features['void_strength_normalized'] = features['void_strength']
        
        # PINN-based edge detection (sharp void boundaries)
        sobel_x = sobel(pinn_normalized, axis=1)
        sobel_y = sobel(pinn_normalized, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['pinn_edge_sharpness'] = edge_magnitude / (np.nanmax(edge_magnitude) + 1e-6)
        
        # Local PINN anomaly (void relative to local average)
        smoothed = gaussian_filter(pinn_normalized, sigma=5)
        local_anomaly = pinn_normalized - smoothed
        features['pinn_local_anomaly'] = np.clip(local_anomaly, 0, None)
        local_max = np.nanmax(features['pinn_local_anomaly'])
        if local_max > 0:
            features['pinn_local_anomaly'] /= local_max
        
    # ====== FALLBACK: Raw gravity features ======
    else:
        logger.warning("PINN output not available - using raw gravity features (less accurate)")
    
    # Always calculate gravity-based features (for comparison or fallback)
    grav_normalized = np.nan_to_num(gravity_data.copy(), nan=0.0)
    
    # Raw gravity void indicator (negative gravity = void)
    raw_void = -grav_normalized
    features['raw_void_strength'] = np.clip(raw_void, 0, None)
    
    # Edge sharpness from gravity
    sobel_x = sobel(grav_normalized, axis=1)
    sobel_y = sobel(grav_normalized, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    features['edge_sharpness'] = edge_magnitude / (np.nanmax(edge_magnitude) + 1e-6)
    
    # Local gravity anomaly
    smoothed = gaussian_filter(grav_normalized, sigma=10)
    local_anomaly = -(grav_normalized - smoothed)  
    features['local_void_anomaly'] = np.clip(local_anomaly, 0, None)
    
    # ====== Magnetic features (if available) ======
    if magnetic_data is not None:
        mag_normalized = np.nan_to_num(magnetic_data, nan=0.0)
        
        # Magnetic disturbance magnitude
        mag_smoothed = gaussian_filter(mag_normalized, sigma=10)
        mag_disturbance = np.abs(mag_normalized - mag_smoothed)
        features['magnetic_disturbance'] = mag_disturbance / (np.nanmax(mag_disturbance) + 1e-6)
        
        # Gravity-Magnetic decorrelation
        mag_edges = np.sqrt(sobel(mag_normalized, axis=0)**2 + sobel(mag_normalized, axis=1)**2)
        grav_mag_ratio = (edge_magnitude + 1e-6) / (mag_edges + 1e-6)
        features['grav_mag_decorrelation'] = np.clip(grav_mag_ratio, 0, 10) / 10.0
    
    # ====== Composite DUB Score ======
    if pinn_density is not None:
        # PINN-weighted composite (trust the physics-informed model more)
        dub_score = (
            0.50 * features['void_strength_normalized'] +
            0.20 * features['pinn_edge_sharpness'] +
            0.15 * features['pinn_local_anomaly'] +
            0.15 * features.get('magnetic_disturbance', np.zeros_like(gravity_data))
        )
    else:
        # Fallback composite (less reliable)
        raw_max = np.nanmax(features['raw_void_strength'])
        if raw_max > 0:
            features['raw_void_strength'] /= raw_max
        local_max = np.nanmax(features['local_void_anomaly'])
        if local_max > 0:
            features['local_void_anomaly'] /= local_max
            
        dub_score = (
            0.35 * features['raw_void_strength'] +
            0.25 * features['edge_sharpness'] +
            0.25 * features['local_void_anomaly'] +
            0.15 * features.get('magnetic_disturbance', np.zeros_like(gravity_data))
        )
    
    features['dub_composite_score'] = dub_score
    
    return features


# ==========================================
# DUB Detection Pipeline
# ==========================================
def extract_dub_candidates(features, transform, threshold_percentile=95):
    """
    Extract candidate DUB locations from feature maps.
    
    Uses the composite DUB score to identify anomalous regions.
    """
    score_map = features['dub_composite_score']
    
    # Threshold at given percentile
    threshold = np.nanpercentile(score_map, threshold_percentile)
    logger.info(f"Detection threshold (p{threshold_percentile}): {threshold:.4f}")
    
    # Binary mask of candidates
    mask = score_map > threshold
    mask = np.nan_to_num(mask, nan=False).astype(bool)
    
    if not np.any(mask):
        logger.warning("No candidates found above threshold")
        return pd.DataFrame()
    
    # Find connected regions
    labeled_array, num_regions = label(mask)
    logger.info(f"Found {num_regions} candidate regions")
    
    objects = find_objects(labeled_array)
    
    candidates = []
    
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        
        # Get region properties
        region_mask = (labeled_array[sl] == i)
        region_scores = score_map[sl][region_mask]
        
        if len(region_scores) == 0:
            continue
        
        # Calculate region statistics
        max_score = np.max(region_scores)
        mean_score = np.mean(region_scores)
        area_pixels = np.sum(region_mask)
        
        # Find peak location (highest score in region)
        local_coords = np.argwhere(region_mask & (score_map[sl] == max_score))
        if len(local_coords) == 0:
            continue
        
        local_r, local_c = local_coords[0]
        global_r = sl[0].start + local_r
        global_c = sl[1].start + local_c
        
        # Convert to lat/lon
        x, y = rasterio.transform.xy(transform, global_r, global_c, offset='center')
        
        # Get individual feature values at peak
        # Handle both PINN mode (void_strength) and fallback mode (raw_void_strength)
        void_strength = features.get('void_strength', features.get('raw_void_strength', np.zeros_like(score_map)))[global_r, global_c]
        edge_sharpness = features['edge_sharpness'][global_r, global_c]
        local_anomaly = features['local_void_anomaly'][global_r, global_c]
        mag_disturbance = features.get('magnetic_disturbance', np.zeros_like(score_map))[global_r, global_c]
        
        # Calculate pixel size in km^2 (rough approximation)
        pixel_size_deg = abs(transform[0])
        pixel_size_km = pixel_size_deg * 111.0  # degrees to km
        area_km2 = area_pixels * (pixel_size_km ** 2)
        
        candidates.append({
            'Region_ID': i,
            'Latitude': y,  # Note: for geographic CRS, y = lat
            'Longitude': x,  # x = lon
            'DUB_Score': max_score,
            'Mean_Score': mean_score,
            'Void_Strength': void_strength,
            'Edge_Sharpness': edge_sharpness,
            'Local_Anomaly': local_anomaly,
            'Magnetic_Disturbance': mag_disturbance,
            'Area_km2': area_km2,
            'Area_Pixels': area_pixels
        })
    
    df = pd.DataFrame(candidates)
    
    if not df.empty:
        # Sort by DUB score descending
        df = df.sort_values('DUB_Score', ascending=False).reset_index(drop=True)
        
        # Apply size filters
        df = df[
            (df['Area_km2'] >= DUB_CONFIG['min_anomaly_size_km2']) &
            (df['Area_km2'] <= DUB_CONFIG['max_anomaly_size_km2'])
        ]
        
        logger.info(f"After size filtering: {len(df)} candidates")
    
    return df


def validate_against_known_dubs(candidates, known_dubs, distance_threshold_km=10.0):
    """
    Validate candidate detections against known DUB locations.
    
    This is our ground truth check - if we find known facilities, we're on track.
    """
    if candidates.empty or known_dubs is None or known_dubs.empty:
        return candidates, {}
    
    # Build spatial index for candidates
    cand_coords = candidates[['Latitude', 'Longitude']].values
    known_coords = known_dubs[['lat', 'lon']].values
    
    # For each known DUB, find nearest candidate
    results = {
        'detected': [],
        'missed': [],
        'detection_rate': 0.0
    }
    
    degree_threshold = distance_threshold_km / 111.0  # Convert km to degrees
    
    if len(cand_coords) > 0:
        cand_tree = cKDTree(cand_coords)
        
        for idx, row in known_dubs.iterrows():
            known_coord = np.array([[row['lat'], row['lon']]])
            distance, nearest_idx = cand_tree.query(known_coord, k=1)
            
            if distance[0] < degree_threshold:
                results['detected'].append({
                    'name': row['name'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'matched_candidate_id': candidates.iloc[nearest_idx[0]]['Region_ID'],
                    'distance_km': distance[0] * 111.0,
                    'candidate_score': candidates.iloc[nearest_idx[0]]['DUB_Score']
                })
            else:
                results['missed'].append({
                    'name': row['name'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'nearest_distance_km': distance[0] * 111.0 if len(cand_coords) > 0 else float('inf')
                })
    else:
        for idx, row in known_dubs.iterrows():
            results['missed'].append({
                'name': row['name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'nearest_distance_km': float('inf')
            })
    
    results['detection_rate'] = len(results['detected']) / len(known_dubs)
    
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION AGAINST KNOWN DUBs")
    logger.info(f"{'='*60}")
    logger.info(f"Total Known DUBs in USA: {len(known_dubs)}")
    logger.info(f"DETECTED: {len(results['detected'])} ({results['detection_rate']*100:.1f}%)")
    logger.info(f"MISSED: {len(results['missed'])}")
    
    if results['detected']:
        logger.info("\nSuccessfully Detected:")
        for det in results['detected']:
            logger.info(f"  ✓ {det['name']} (distance: {det['distance_km']:.1f} km, score: {det['candidate_score']:.3f})")
    
    if results['missed']:
        logger.info("\nMissed Facilities:")
        for miss in results['missed']:
            logger.info(f"  ✗ {miss['name']} (nearest candidate: {miss['nearest_distance_km']:.1f} km)")
    
    # Add validation column to candidates
    candidates = candidates.copy()
    detected_ids = [d['matched_candidate_id'] for d in results['detected']]
    candidates['Matches_Known_DUB'] = candidates['Region_ID'].isin(detected_ids)
    
    return candidates, results


# ==========================================
# Output and Visualization
# ==========================================
def save_dub_probability_map(features, profile, output_path):
    """Save the DUB probability/score map as GeoTIFF."""
    score_map = features['dub_composite_score'].astype(np.float32)
    
    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(score_map, 1)
        dst.set_band_description(1, "DUB Detection Score")
    
    logger.info(f"Saved DUB probability map to: {output_path}")


def save_candidates(candidates, output_path):
    """Save candidate DUB locations to CSV."""
    candidates.to_csv(output_path, index=False)
    logger.info(f"Saved {len(candidates)} DUB candidates to: {output_path}")


# ==========================================
# Main Pipeline
# ==========================================
def run_dub_detection(threshold_percentile=95, validate=True, run_pinn=True):
    """
    Main DUB detection pipeline.
    
    1. Run PINN gravity inversion in VOID mode (if enabled)
    2. Load gravity and magnetic data
    3. Calculate void indicators (PINN-based if available)
    4. Extract candidate locations
    5. Validate against known DUBs
    6. Save results
    """
    logger.info("="*60)
    logger.info("DEEP UNDERGROUND BASE (DUB) DETECTION SYSTEM")
    logger.info("="*60)
    logger.info(f"Mode: {DUB_CONFIG['target_mode'].upper()}")
    logger.info(f"Looking for: Negative density anomalies (voids/cavities)")
    logger.info(f"Using PINN: {run_pinn}")
    
    # Load gravity data
    gravity_data, transform, profile = load_gravity_data()
    if gravity_data is None:
        logger.error("Cannot proceed without gravity data")
        return None, None
    
    magnetic_data, _ = load_magnetic_data()
    
    # Get gravity file path for PINN
    gravity_path = OUTPUTS / "usa_supervised/usa_gravity_residual.tif"
    if not gravity_path.exists():
        gravity_path = OUTPUTS / "usa_supervised/usa_gravity_mosaic.tif"
    
    magnetic_path = OUTPUTS / "usa_supervised/usa_magnetic_mosaic.tif"
    if not magnetic_path.exists():
        magnetic_path = None
    
    # Run PINN inversion in VOID mode
    pinn_density = None
    if run_pinn:
        logger.info("\n" + "="*60)
        logger.info("RUNNING PINN GRAVITY INVERSION (VOID MODE)")
        logger.info("="*60)
        
        pinn_output_path = run_pinn_void_inversion(
            gravity_path=gravity_path,
            magnetic_path=magnetic_path
        )
        
        if pinn_output_path and pinn_output_path.exists():
            pinn_density, pinn_transform, _ = load_pinn_density_output()
            logger.info("PINN inversion successful - using physics-informed density map")
        else:
            logger.warning("PINN inversion failed - falling back to simple feature engineering")
    else:
        # Try to load existing PINN output
        pinn_density, pinn_transform, _ = load_pinn_density_output()
        if pinn_density is not None:
            logger.info("Loaded existing PINN output")
    
    # Calculate DUB features
    logger.info("\n" + "="*60)
    logger.info("CALCULATING VOID DETECTION FEATURES")
    logger.info("="*60)
    
    features = calculate_void_indicators(gravity_data, magnetic_data, pinn_density)
    
    # Log feature statistics
    for name, feat in features.items():
        logger.info(f"  {name}: min={np.nanmin(feat):.4f}, max={np.nanmax(feat):.4f}, mean={np.nanmean(feat):.4f}")
    
    # Extract candidates
    logger.info(f"\nExtracting candidates (threshold: p{threshold_percentile})...")
    candidates = extract_dub_candidates(features, transform, threshold_percentile)
    
    if candidates.empty:
        logger.warning("No DUB candidates found!")
        return candidates, None
    
    logger.info(f"Found {len(candidates)} DUB candidates")
    
    # Validate if requested
    validation_results = None
    if validate:
        known_dubs = load_dub_training_data()
        if known_dubs is not None:
            candidates, validation_results = validate_against_known_dubs(candidates, known_dubs)
    
    # Save outputs
    output_dir = OUTPUTS / "dub_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_dub_probability_map(features, profile, output_dir / "dub_probability.tif")
    save_candidates(candidates, output_dir / "dub_candidates.csv")
    
    # Save top candidates separately
    if len(candidates) > 0:
        top_candidates = candidates.head(100)
        save_candidates(top_candidates, output_dir / "dub_top100.csv")
        
        # Save novel candidates (not matching known DUBs)
        novel = candidates[~candidates.get('Matches_Known_DUB', False)]
        save_candidates(novel, output_dir / "dub_novel_candidates.csv")
        logger.info(f"Found {len(novel)} NOVEL DUB candidates (not matching known facilities)")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DUB DETECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total candidates: {len(candidates)}")
    logger.info(f"Method: {'PINN Physics-Informed' if pinn_density is not None else 'Simple Feature Engineering'}")
    
    if validation_results:
        logger.info(f"Known DUBs detected: {len(validation_results['detected'])}/{len(validation_results['detected'])+len(validation_results['missed'])}")
        logger.info(f"Detection rate: {validation_results['detection_rate']*100:.1f}%")
    
    logger.info(f"\nTop 10 candidates by DUB Score:")
    if not candidates.empty:
        top10 = candidates.head(10)[['Region_ID', 'Latitude', 'Longitude', 'DUB_Score', 'Void_Strength', 'Area_km2']]
        for idx, row in top10.iterrows():
            logger.info(f"  {int(row['Region_ID']):4d}: ({row['Latitude']:.4f}, {row['Longitude']:.4f}) "
                       f"Score: {row['DUB_Score']:.3f}, Void: {row['Void_Strength']:.3f}, Area: {row['Area_km2']:.2f} km²")
    
    return candidates, validation_results


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Deep Underground Base Detection System")
    parser.add_argument("--threshold", type=int, default=95,
                        help="Detection threshold percentile (default: 95)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation against known DUBs")
    parser.add_argument("--sensitivity", choices=['low', 'medium', 'high'], default='medium',
                        help="Detection sensitivity (affects threshold)")
    parser.add_argument("--run-pinn", action="store_true", default=True,
                        help="Run PINN gravity inversion in void mode (default: True)")
    parser.add_argument("--skip-pinn", action="store_true",
                        help="Skip PINN, use existing output or fallback to simple features")
    parser.add_argument("--pinn-only", action="store_true",
                        help="Only run PINN inversion, skip detection")
    
    args = parser.parse_args()
    
    # Adjust threshold based on sensitivity
    sensitivity_map = {
        'low': 98,
        'medium': 95,
        'high': 90
    }
    threshold = sensitivity_map.get(args.sensitivity, args.threshold)
    
    # Handle PINN-only mode
    if args.pinn_only:
        gravity_path = OUTPUTS / "usa_supervised/usa_gravity_residual.tif"
        if not gravity_path.exists():
            gravity_path = OUTPUTS / "usa_supervised/usa_gravity_mosaic.tif"
        
        magnetic_path = OUTPUTS / "usa_supervised/usa_magnetic_mosaic.tif"
        if not magnetic_path.exists():
            magnetic_path = None
        
        result = run_pinn_void_inversion(gravity_path, magnetic_path)
        if result:
            logger.info(f"\n✅ PINN inversion complete: {result}")
            return 0
        else:
            logger.error("\n❌ PINN inversion failed")
            return 1
    
    candidates, results = run_dub_detection(
        threshold_percentile=threshold,
        validate=not args.no_validate,
        run_pinn=not args.skip_pinn
    )
    
    if candidates is not None and not candidates.empty:
        logger.info("\n✅ DUB detection complete!")
        return 0
    else:
        logger.error("\n❌ DUB detection failed or found no candidates")
        return 1


if __name__ == "__main__":
    sys.exit(main())
