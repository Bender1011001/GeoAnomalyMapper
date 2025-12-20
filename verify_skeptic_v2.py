#!/usr/bin/env python3
"""
GeoAnomalyMapper Forensic Audit Script (verify_skeptic_v2.py)
------------------------------------------------------------
Implements the "Skeptic" validation strategy to detect:
1. Topography correlation ("The Altimeter Trap")
2. Tile edge artifacts ("The Border Effect")
3. Physical realism violations ("The Ghost in the Machine")
4. Data leakage ("The Memory Game")
5. Road proximity bias ("Accessibility Bias")

Usage:
    python verify_skeptic_v2.py --dem data/World_e-Atlas-UCSD_SRTM30-plus_v8.tif --roads data/ne_10m_roads/ne_10m_roads.shp
"""

import argparse
import logging
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try importing geopandas for roads check
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    logger.warning("geopandas not installed. Roads proximity check will be skipped.")


def check_topography_trap(targets, dem_path):
    """
    Hypothesis: The model is just an altimeter.
    Test: Correlate Model Probability/Density with Elevation.
    """
    logger.info("\n" + "="*60)
    logger.info("1. THE 'TOPOGRAPHY TRAP' INVESTIGATION")
    logger.info("="*60)
    
    if not dem_path or not Path(dem_path).exists():
        logger.warning("No DEM provided or file not found. Skipping Topography Check.")
        logger.warning("   (To fix: Provide path to elevation raster using --dem)")
        return None
    
    try:
        elevations = []
        
        with rasterio.open(dem_path) as src:
            # Sample elevation at every target location
            coords = [(x, y) for x, y in zip(targets['Longitude'], targets['Latitude'])]
            
            # Use sample generator
            for i, val in enumerate(src.sample(coords)):
                elev = val[0]
                # Filter nodata (usually -9999 or extremely high/low)
                if elev > -500 and elev < 9000:  # Valid Earth elevations
                    elevations.append(elev)
                else:
                    elevations.append(np.nan)
        
        targets['Elevation'] = elevations
        valid_mask = ~np.isnan(targets['Elevation'])
        target_subset = targets[valid_mask].copy()
        
        if len(target_subset) < 10:
            logger.warning("Not enough targets with valid elevation data.")
            return None

        # Correlation Checks
        # Note: Model_Probability may not exist, use Density_Contrast which is available
        corr_dens = target_subset['Elevation'].corr(target_subset['Density_Contrast'])
        
        logger.info(f"Analyzed {len(target_subset)} targets with valid elevation.")
        logger.info(f"Elevation Range: {target_subset['Elevation'].min():.0f}m to {target_subset['Elevation'].max():.0f}m")
        logger.info(f"Mean Elevation: {target_subset['Elevation'].mean():.0f}m")
        logger.info("")
        logger.info(f"Correlation (Elevation vs Density Contrast): {corr_dens:.4f}")
        
        if abs(corr_dens) > 0.6:
            logger.warning("\n⚠️  FAIL: High correlation with topography detected!")
            logger.warning("   The model might just be mapping mountain ranges.")
            return False
        elif abs(corr_dens) > 0.4:
            logger.warning("\n⚠️  WARN: Moderate correlation with topography.")
            logger.warning("   This is expected for mineral deposits, but warrants scrutiny.")
            return True
        else:
            logger.info("\n✅ PASS: Low correlation with topography. Model is likely looking at subsurface.")
            return True
            
    except Exception as e:
        logger.error(f"Topography check failed: {e}")
        return None


def check_edge_artifacts(targets, ref_raster_path, tile_sizes=[2048, 512]):
    """
    Hypothesis: Targets cluster at tile edges.
    Test: Calculate pixel modulo coordinates for both PINN training (512) and inference (2048) tiles.
    """
    logger.info("\n" + "="*60)
    logger.info("2. THE 'BORDER & EDGE' ARTIFACT INVESTIGATION")
    logger.info("="*60)
    
    if not Path(ref_raster_path).exists():
        logger.error(f"Reference raster not found at {ref_raster_path}. Cannot check edge artifacts.")
        return None

    try:
        with rasterio.open(ref_raster_path) as src:
            transform = src.transform
            
        # Convert Lat/Lon to Pixel Coords
        xs = []
        ys = []
        
        for lat, lon in zip(targets['Latitude'], targets['Longitude']):
            row, col = rasterio.transform.rowcol(transform, lon, lat)
            xs.append(col)
            ys.append(row)
            
        targets['Pixel_X'] = xs
        targets['Pixel_Y'] = ys
        
        all_pass = True
        
        for tile_size in tile_sizes:
            # Calculate distance to nearest tile boundary
            dist_x = [min(x % tile_size, tile_size - (x % tile_size)) for x in xs]
            dist_y = [min(y % tile_size, tile_size - (y % tile_size)) for y in ys]
            
            dist_edge = [min(dx, dy) for dx, dy in zip(dist_x, dist_y)]
            
            # Define "Edge" thresholds
            edge_threshold_strict = 16  # pixels
            edge_threshold_loose = 32   # pixels
            
            on_edge_strict = sum(1 for d in dist_edge if d < edge_threshold_strict)
            on_edge_loose = sum(1 for d in dist_edge if d < edge_threshold_loose)
            
            pct_edge_strict = (on_edge_strict / len(targets)) * 100
            pct_edge_loose = (on_edge_loose / len(targets)) * 100
            
            logger.info(f"\nTile Size: {tile_size}x{tile_size}")
            logger.info(f"  Targets within {edge_threshold_strict}px of edge: {on_edge_strict}/{len(targets)} ({pct_edge_strict:.1f}%)")
            logger.info(f"  Targets within {edge_threshold_loose}px of edge: {on_edge_loose}/{len(targets)} ({pct_edge_loose:.1f}%)")
            
            # Expected by random chance: (2 * threshold / tile_size) * 100%
            expected_random = (2 * edge_threshold_loose / tile_size) * 100
            logger.info(f"  (Random expectation: ~{expected_random:.1f}%)")
            
            if pct_edge_loose > 25.0:
                logger.warning(f"  ⚠️  FAIL: {pct_edge_loose:.1f}% of targets are near {tile_size}px tile borders.")
                all_pass = False
            else:
                logger.info(f"  ✅ PASS for {tile_size}px tiles.")
        
        if all_pass:
            logger.info("\n✅ OVERALL PASS: Targets are well-distributed relative to tile grids.")
        else:
            logger.warning("\n⚠️  FAIL: Potential tiling artifacts detected.")
            
        return all_pass
            
    except Exception as e:
        logger.error(f"Edge check failed: {e}")
        return None


def check_physics_realism(targets):
    """
    Hypothesis: Density values are exploding to minimize loss.
    Test: Check range of Density_Contrast.
    """
    logger.info("\n" + "="*60)
    logger.info("3. THE 'GHOST IN THE MACHINE' (PHYSICS CHECK)")
    logger.info("="*60)
    
    if 'Density_Contrast' not in targets.columns:
        logger.warning("Density_Contrast column missing. Skipping.")
        return None
        
    vals = targets['Density_Contrast']
    min_v, max_v, mean_v, std_v = vals.min(), vals.max(), vals.mean(), vals.std()
    
    logger.info(f"Density Contrast Statistics (kg/m³):")
    logger.info(f"  Min:  {min_v:.2f}")
    logger.info(f"  Max:  {max_v:.2f}")
    logger.info(f"  Mean: {mean_v:.2f}")
    logger.info(f"  Std:  {std_v:.2f}")
    
    # Physical limits
    # 500 kg/m³ = 0.5 g/cm³ (typical ore contrast)
    # 1000 kg/m³ = 1.0 g/cm³ (massive sulfide)
    # 2000 kg/m³ = unrealistic for km-scale voxels
    
    threshold_physical = 1500.0  # kg/m³
    threshold_extreme = 3000.0   # kg/m³
    
    outliers_physical = vals[abs(vals) > threshold_physical]
    outliers_extreme = vals[abs(vals) > threshold_extreme]
    
    pct_physical = (len(outliers_physical) / len(targets)) * 100
    pct_extreme = (len(outliers_extreme) / len(targets)) * 100
    
    logger.info(f"\nPhysically questionable (|ρ| > {threshold_physical} kg/m³): {len(outliers_physical)} ({pct_physical:.1f}%)")
    logger.info(f"Extremely unrealistic (|ρ| > {threshold_extreme} kg/m³): {len(outliers_extreme)} ({pct_extreme:.1f}%)")
    
    if pct_extreme > 5.0:
        logger.warning(f"\n⚠️  FAIL: {pct_extreme:.1f}% of targets have impossible density contrast.")
        logger.warning("   The model is 'hallucinating' mass to fit the gravity field.")
        return False
    elif pct_physical > 20.0:
        logger.warning(f"\n⚠️  WARN: {pct_physical:.1f}% of targets have high (but possible) contrast.")
        return True
    else:
        logger.info("\n✅ PASS: Density contrasts are within physical limits.")
        return True


def check_leakage(targets, training_csv):
    """
    Hypothesis: The model targets are just the training data returned.
    Test: Distance check validation.
    """
    logger.info("\n" + "="*60)
    logger.info("4. THE 'MEMORY GAME' (LEAKAGE CHECK)")
    logger.info("="*60)
    
    if not Path(training_csv).exists():
        logger.warning(f"Training data not found at {training_csv}. Skipping leakage check.")
        return None
    
    train_df = pd.read_csv(training_csv)
    train_df.columns = [c.lower() for c in train_df.columns]
    
    lat_col = next((c for c in train_df.columns if 'lat' in c), None)
    lon_col = next((c for c in train_df.columns if 'lon' in c), None)
    
    if not lat_col or not lon_col:
        logger.error("Could not find lat/lon columns in training data.")
        return None
    
    # Build tree of training data
    train_coords = np.column_stack([train_df[lat_col], train_df[lon_col]])
    tree = cKDTree(train_coords)
    
    target_coords = np.column_stack([targets['Latitude'], targets['Longitude']])
    dists, indices = tree.query(target_coords, k=1)
    
    # Convert degrees to approximate km (at ~40N)
    dists_km = dists * 111.0
    
    # Thresholds
    exact_match_thresh = 1.0   # km
    near_match_thresh = 10.0   # km
    
    exact_count = np.sum(dists_km < exact_match_thresh)
    near_count = np.sum(dists_km < near_match_thresh)
    
    pct_exact = (exact_count / len(targets)) * 100
    pct_near = (near_count / len(targets)) * 100
    
    logger.info(f"Training Sites: {len(train_df)}")
    logger.info(f"Model Targets:  {len(targets)}")
    logger.info("")
    logger.info(f"Exact matches (<{exact_match_thresh}km from training site): {exact_count} ({pct_exact:.1f}%)")
    logger.info(f"Near matches  (<{near_match_thresh}km from training site): {near_count} ({pct_near:.1f}%)")
    
    targets['Is_Training_Site'] = dists_km < near_match_thresh
    targets['Dist_to_Training_km'] = dists_km
    
    novel_count = len(targets) - near_count
    logger.info(f"Novel/Discovery candidates (>{near_match_thresh}km from any training): {novel_count} ({100-pct_near:.1f}%)")
    
    if pct_exact > 80.0:
        logger.warning("\n⚠️  NOTE: High 'Training Recall'. Most targets are memorized training data.")
        logger.warning("   This proves the model works as a memory engine, NOT generalization.")
    elif pct_near < 30.0:
        logger.info("\nℹ️  NOTE: High 'Novelty Rate'. Most targets are NOT in the training set.")
        logger.info("   These are potential new discoveries (or false positives).")
    else:
        logger.info("\nℹ️  Mixed: Model found both training sites and new candidates.")
    
    return True


def check_road_proximity(targets, roads_path):
    """
    Hypothesis: Model favors accessible areas near roads.
    Test: Calculate distance from targets to nearest road.
    """
    logger.info("\n" + "="*60)
    logger.info("5. THE 'ACCESSIBILITY BIAS' (ROADS CHECK)")
    logger.info("="*60)
    
    if not HAS_GEOPANDAS:
        logger.warning("geopandas not installed. Skipping roads check.")
        logger.warning("   Install with: pip install geopandas")
        return None
    
    if not Path(roads_path).exists():
        logger.warning(f"Roads shapefile not found at {roads_path}. Skipping.")
        return None
    
    try:
        logger.info("Loading roads shapefile (this may take a moment)...")
        roads = gpd.read_file(roads_path)
        
        # Filter to CONUS approximate bounds
        conus_bounds = (-125, 24, -66, 50)  # (minx, miny, maxx, maxy)
        roads_conus = roads.cx[conus_bounds[0]:conus_bounds[2], conus_bounds[1]:conus_bounds[3]]
        
        logger.info(f"Loaded {len(roads_conus)} road segments in CONUS.")
        
        if len(roads_conus) == 0:
            logger.warning("No roads found in CONUS bounds. Check shapefile.")
            return None
        
        # Create target points
        target_points = gpd.GeoDataFrame(
            targets,
            geometry=[Point(lon, lat) for lon, lat in zip(targets['Longitude'], targets['Latitude'])],
            crs="EPSG:4326"
        )
        
        # Union all roads into a single geometry for distance calc
        logger.info("Computing distances to nearest road...")
        roads_union = roads_conus.unary_union
        
        # Calculate distances (in degrees, then convert to km)
        distances_deg = target_points.geometry.distance(roads_union)
        distances_km = distances_deg * 111.0  # Approximate conversion
        
        targets['Dist_to_Road_km'] = distances_km.values
        
        # Stats
        mean_dist = distances_km.mean()
        median_dist = distances_km.median()
        
        near_road_thresh = 10.0  # km
        near_road_count = np.sum(distances_km < near_road_thresh)
        pct_near_road = (near_road_count / len(targets)) * 100
        
        logger.info(f"\nDistance to Nearest Road:")
        logger.info(f"  Mean:   {mean_dist:.1f} km")
        logger.info(f"  Median: {median_dist:.1f} km")
        logger.info(f"  Min:    {distances_km.min():.1f} km")
        logger.info(f"  Max:    {distances_km.max():.1f} km")
        logger.info("")
        logger.info(f"Targets within {near_road_thresh}km of a road: {near_road_count}/{len(targets)} ({pct_near_road:.1f}%)")
        
        # To interpret: We need a baseline. What % of CONUS is within 10km of a road?
        # Roughly 60-70% of CONUS is near roads. If model has >90%, it's biased.
        
        if pct_near_road > 90.0:
            logger.warning("\n⚠️  WARN: Model heavily favors road-accessible areas.")
            logger.warning("   This may reflect sampling bias in training data.")
            return False
        else:
            logger.info("\n✅ PASS: Road proximity is reasonable.")
            return True
        
    except Exception as e:
        logger.error(f"Roads check failed: {e}")
        return None


def generate_summary(results, targets):
    """Generate final summary with pass/fail verdicts."""
    logger.info("\n" + "="*60)
    logger.info("FORENSIC AUDIT SUMMARY")
    logger.info("="*60)
    
    total_checks = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        total_checks += 1
        if result is True:
            passed += 1
            status = "✅ PASS"
        elif result is False:
            failed += 1
            status = "❌ FAIL"
        else:
            skipped += 1
            status = "⏭️  SKIP"
        logger.info(f"  {name}: {status}")
    
    logger.info("")
    logger.info(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        logger.info("\n🎉 MODEL PASSED FORENSIC AUDIT")
    elif failed > 0:
        logger.warning("\n⚠️  MODEL HAS ISSUES - REVIEW REQUIRED")
    
    # Save audited targets
    output_path = 'data/outputs/usa_targets_audited.csv'
    targets.to_csv(output_path, index=False)
    logger.info(f"\nSaved audited targets to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Forensic Audit of GeoAnomalyMapper")
    parser.add_argument("--targets", default="data/outputs/usa_targets.csv", help="Targets CSV")
    parser.add_argument("--training", default="data/usgs_goldilocks.csv", help="Training Data CSV")
    parser.add_argument("--gravity", default="data/outputs/usa_supervised/usa_gravity_mosaic.tif", help="Gravity Mosaic (for tile geometry)")
    parser.add_argument("--dem", default="data/World_e-Atlas-UCSD_SRTM30-plus_v8.tif", help="Digital Elevation Model")
    parser.add_argument("--roads", default="data/ne_10m_roads/ne_10m_roads.shp", help="Roads Shapefile")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("GEOANOMALYMAPPER FORENSIC AUDIT")
    logger.info("Chief Scientific Auditor Mode")
    logger.info("="*60)
    
    if not Path(args.targets).exists():
        logger.error(f"Targets file not found: {args.targets}")
        return
        
    targets = pd.read_csv(args.targets)
    logger.info(f"Loaded {len(targets)} targets from {args.targets}")
    
    # Run all checks
    results = {}
    
    results['Physics Realism'] = check_physics_realism(targets)
    results['Tile Edge Artifacts'] = check_edge_artifacts(targets, args.gravity)
    results['Data Leakage'] = check_leakage(targets, args.training)
    results['Topography Correlation'] = check_topography_trap(targets, args.dem)
    results['Road Proximity Bias'] = check_road_proximity(targets, args.roads)
    
    # Generate summary
    generate_summary(results, targets)


if __name__ == "__main__":
    main()
