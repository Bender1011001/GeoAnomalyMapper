#!/usr/bin/env python3
"""
Phase 2 Validation: Confidence Scoring and MRDS Cross-Reference
Generates confidence scores for targets based on:
1. Distance from training data (novelty)
2. Density contrast strength
3. Distance from tile edges
4. Proximity to known MRDS deposits (geological plausibility)
"""

import argparse
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

def load_mrds_data(mrds_path):
    """Load and clean MRDS data."""
    print(f"Loading MRDS data from {mrds_path}...")
    df = pd.read_csv(mrds_path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Find lat/lon columns
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c), None)
    
    if not lat_col or not lon_col:
        print(f"Available columns: {list(df.columns)[:20]}")
        raise ValueError("Could not find lat/lon columns in MRDS data")
    
    # Filter valid coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
    df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Filter to CONUS
    df = df[(df['latitude'] >= 24) & (df['latitude'] <= 50) &
            (df['longitude'] >= -125) & (df['longitude'] <= -66)]
    
    print(f"Loaded {len(df)} MRDS deposits in CONUS")
    return df

def compute_confidence_scores(targets, mrds_df):
    """Compute composite confidence score for each target."""
    print("\nComputing confidence scores...")
    
    # Build KD-Tree for MRDS deposits
    mrds_coords = np.column_stack([mrds_df['latitude'], mrds_df['longitude']])
    mrds_tree = cKDTree(mrds_coords)
    
    target_coords = np.column_stack([targets['Latitude'], targets['Longitude']])
    
    # Distance to nearest MRDS deposit
    dists_mrds, _ = mrds_tree.query(target_coords, k=1)
    dists_mrds_km = dists_mrds * 111.0  # Approximate degrees to km
    
    targets['Dist_to_MRDS_km'] = dists_mrds_km
    
    # Scoring components (0-1 scale, higher = better)
    
    # 1. Density Score (normalized by max)
    max_density = targets['Density_Contrast'].max()
    targets['Score_Density'] = targets['Density_Contrast'] / max_density
    
    # 2. Novelty Score (inverse of training proximity - prefer novel discoveries)
    # From audited data: Dist_to_Training_km
    if 'Dist_to_Training_km' in targets.columns:
        # Score higher for targets further from training (more novel)
        max_train_dist = targets['Dist_to_Training_km'].max()
        targets['Score_Novelty'] = np.clip(targets['Dist_to_Training_km'] / 500, 0, 1)
    else:
        targets['Score_Novelty'] = 0.5  # Default
    
    # 3. Edge Distance Score (inverse - prefer targets away from edges)
    # Check if pixel coordinates exist
    if 'Pixel_X' in targets.columns and 'Pixel_Y' in targets.columns:
        # Distance from 512px tile edges
        tile_size = 512
        dist_x = np.minimum(targets['Pixel_X'] % tile_size, tile_size - (targets['Pixel_X'] % tile_size))
        dist_y = np.minimum(targets['Pixel_Y'] % tile_size, tile_size - (targets['Pixel_Y'] % tile_size))
        edge_dist = np.minimum(dist_x, dist_y)
        targets['Score_EdgeDist'] = np.clip(edge_dist / 64, 0, 1)
    else:
        targets['Score_EdgeDist'] = 0.5  # Default
    
    # 4. MRDS Proximity Score (prefer targets NEAR known deposits - geological plausibility)
    # But not TOO close (would be redundant). Sweet spot: 10-100km
    # Score = 1 if 10-50km, 0.5 if <10km or 50-100km, 0 if >100km
    targets['Score_MRDS'] = np.where(
        (dists_mrds_km >= 10) & (dists_mrds_km <= 50), 1.0,
        np.where(
            (dists_mrds_km < 10), 0.5,  # Near known deposit
            np.where(
                (dists_mrds_km > 50) & (dists_mrds_km <= 100), 0.5,
                0.2  # Very far from any known deposit
            )
        )
    )
    
    # Composite Score (weighted average)
    weights = {
        'density': 0.30,
        'novelty': 0.20,
        'edge_dist': 0.20,
        'mrds': 0.30
    }
    
    targets['Confidence_Score'] = (
        weights['density'] * targets['Score_Density'] +
        weights['novelty'] * targets['Score_Novelty'] +
        weights['edge_dist'] * targets['Score_EdgeDist'] +
        weights['mrds'] * targets['Score_MRDS']
    )
    
    # Tier classification
    targets['Tier'] = pd.cut(
        targets['Confidence_Score'],
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Tier 3', 'Tier 2', 'Tier 1']
    )
    
    return targets

def analyze_top_targets(targets, mrds_df, top_n=50):
    """Analyze top N targets against MRDS."""
    print(f"\n{'='*60}")
    print(f"TOP {top_n} TARGETS ANALYSIS")
    print('='*60)
    
    top = targets.nlargest(top_n, 'Confidence_Score')
    
    # MRDS hit rate (within 50km of known deposit)
    hits = (top['Dist_to_MRDS_km'] < 50).sum()
    hit_rate = hits / top_n * 100
    
    print(f"\nMRDS Correlation (within 50km):")
    print(f"  Hits: {hits}/{top_n} ({hit_rate:.1f}%)")
    
    # Tier distribution
    print(f"\nTier Distribution:")
    print(top['Tier'].value_counts().to_string())
    
    # Summary stats
    print(f"\nTop 20 Targets:")
    display_cols = ['Latitude', 'Longitude', 'Density_Contrast', 'Confidence_Score', 'Tier', 'Dist_to_MRDS_km']
    existing_cols = [c for c in display_cols if c in top.columns]
    print(top.head(20)[existing_cols].to_string(index=False))
    
    return top

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Validation")
    parser.add_argument("--targets", default="data/outputs/usa_targets_audited.csv")
    parser.add_argument("--mrds", default="validation_results/mrds_data/mrds.csv")
    parser.add_argument("--output", default="data/outputs/usa_targets_scored.csv")
    
    args = parser.parse_args()
    
    # Load data
    if not Path(args.targets).exists():
        print(f"Targets file not found: {args.targets}")
        return
    
    targets = pd.read_csv(args.targets)
    print(f"Loaded {len(targets)} targets")
    
    # Load MRDS
    if Path(args.mrds).exists():
        mrds_df = load_mrds_data(args.mrds)
    else:
        print(f"MRDS file not found: {args.mrds}")
        print("Skipping MRDS cross-reference...")
        mrds_df = pd.DataFrame(columns=['latitude', 'longitude'])
    
    # Compute scores
    targets = compute_confidence_scores(targets, mrds_df)
    
    # Analyze top targets
    top_targets = analyze_top_targets(targets, mrds_df, top_n=50)
    
    # Save results
    targets_sorted = targets.sort_values('Confidence_Score', ascending=False)
    targets_sorted.to_csv(args.output, index=False)
    print(f"\nSaved scored targets to {args.output}")
    
    # Save top 50
    top_output = Path(args.output).with_name("usa_top50_targets.csv")
    top_targets.to_csv(top_output, index=False)
    print(f"Saved top 50 targets to {top_output}")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    print(f"Total targets: {len(targets)}")
    print(f"Tier 1 (High Confidence): {(targets['Tier'] == 'Tier 1').sum()}")
    print(f"Tier 2 (Medium Confidence): {(targets['Tier'] == 'Tier 2').sum()}")
    print(f"Tier 3 (Low Confidence): {(targets['Tier'] == 'Tier 3').sum()}")

if __name__ == "__main__":
    main()
