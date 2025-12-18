
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Skeptic Verification Process...")
    
    # 1. Load Data
    try:
        targets = pd.read_csv('data/outputs/usa_targets.csv')
        # Load full MRDS
        mrds = pd.read_csv('data/usgs_mrds_full.csv', low_memory=False)
        
        # Normalize columns immediately
        mrds.columns = [c.lower() for c in mrds.columns]

        logger.info(f"Loaded {len(targets)} model targets and {len(mrds)} MRDS deposits.")
        
        # STRICT FILTERING: Only Producers/Past Producers
        if 'dev_stat' in mrds.columns:
            # Standardize case
            mrds['dev_stat'] = mrds['dev_stat'].astype(str).str.title()
            
            # Filter
            mrds = mrds[mrds['dev_stat'].isin(['Producer', 'Past Producer'])]
            logger.info(f"Filtered to {len(mrds)} PRODUCERS / PAST PRODUCERS.")
        else:
            logger.warning("Warning: 'dev_stat' column missing. Using all records.")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Normalize column names for MRDS
    mrds.columns = [c.lower() for c in mrds.columns]
    
    # Identify lat/lon columns
    lat_col = next((c for c in mrds.columns if 'lat' in c or 'y_coord' in c), None)
    lon_col = next((c for c in mrds.columns if 'lon' in c or 'x_coord' in c), None)
    
    if not lat_col or not lon_col:
        logger.error(f"Could not identify lat/lon columns in MRDS. Found: {mrds.columns.tolist()}")
        return
        
    logger.info(f"Using MRDS columns: {lat_col}, {lon_col}")

    # 2. Define Nevada Region
    nv_lat_min, nv_lat_max = 35.0, 42.0
    nv_lon_min, nv_lon_max = -120.0, -114.0
    
    # Filter Model Targets to Nevada
    nv_targets = targets[
        (targets['Latitude'] >= nv_lat_min) & (targets['Latitude'] <= nv_lat_max) &
        (targets['Longitude'] >= nv_lon_min) & (targets['Longitude'] <= nv_lon_max)
    ].copy()
    
    logger.info(f"Filtered to {len(nv_targets)} targets in Nevada box.")

    # Filter MRDS to Nevada Box (for speed and consistency)
    nv_mrds = mrds[
        (mrds[lat_col] >= nv_lat_min) & (mrds[lat_col] <= nv_lat_max) &
        (mrds[lon_col] >= nv_lon_min) & (mrds[lon_col] <= nv_lon_max)
    ].copy()
    
    logger.info(f"Filtered to {len(nv_mrds)} known deposits in Nevada box.")

    if len(nv_mrds) == 0:
        logger.error("No MRDS deposits found in Nevada box. Check coordinates/filtering.")
        return

    # Build Tree for Fast lookup
    mrds_coords = nv_mrds[[lat_col, lon_col]].values
    mrds_tree = cKDTree(mrds_coords)
    
    # Helper for distance (approximate, degrees to km)
    # 1 deg lat ~ 111 km. 1 deg lon at 38N ~ 88 km.
    # Using simple Euclidean on degrees for rough 10km check or strictly converting?
    # The original script heavily implied specific boolean logic.
    # Let's use Haversine or simple approximation if used in original.
    # Original used: `haversine` or similar.
    # To match original metric: "matches_within_10km"
    
    # Let's use a robust Haversine for validation
    def get_hits(points, tree, threshold_degrees=0.1): 
        # 0.1 deg is approx 11km. 10km is ~0.09 degrees.
        # But wait, KDTree is euclidean. 
        # Proper way: Query ball point with r.
        # Let's stick to the 72.5% methodology: "nearest distance < 10km"
        # We need to simulate exactly that.
        pass

    # Actually, let's just do it explicitly for every point to be safe and accurate.
    # Using KDTree query.
    
    # 3. Monte Carlo Simulation
    logger.info("\n--- Running Monte Carlo Simulation (Base Rate) ---")
    n_iterations = 1000
    n_points = 40 # Determine exactly how many points the model found in Nevada?
    # User said "40 generated targets". Let's verify.
    actual_n_targets = len(nv_targets)
    if actual_n_targets != 40:
        logger.warning(f"Note: Model found {actual_n_targets} targets, but plan said 40. Using {actual_n_targets} for simulation to match.")
        n_points = actual_n_targets
    
    hit_counts = []
    
    # Pre-converting MRDS to radians for BallTree/Haversine if we wanted strictness,
    # but for 10km tolerance in Nevada (mid-latitude), a degree distance check is common in these quick scripts.
    # However, to be rigorous (Skeptic Mode), we MUST use km.
    # We'll use cKDTree on projected coords or simple haversine check loop.
    # Loops are slow for 1000 iter. 
    # Let's project to simple local grid or just use Haversine vectorization.
    
    # Vectorized Haversine Check
    # We want: P(random_point is within 10km of ANY mrds_point)
    # Invert: Coverage area of MRDS points.
    # Better: Generate random points, check distance to nearest neighbor.
    
    # Let's use the actual 10km threshold. 
    # Converting MRDS coords to (lat, lon) arrays.
    dep_lats = nv_mrds[lat_col].values
    dep_lons = nv_mrds[lon_col].values
    
    def count_hits_vectorized(rand_lats, rand_lons):
        # This is n_rand x n_deposits. 40 x 10000? Might be big.
        # Nevada MRDS is likely huge (thousands).
        # KDTree is needed.
        # degrees to km conversion factor (approx at 38N): 111km lat, 88km lon.
        # dist_sq = ((dlat * 111)**2 + (dlon * 88)**2)
        # 10km squared = 100.
        
        # Scaling to "approx km space"
        coords_km = np.column_stack([
            nv_mrds[lat_col].values * 111.0,
            nv_mrds[lon_col].values * 88.0
        ])
        tree_km = cKDTree(coords_km)
        
        q_coords_km = np.column_stack([
            rand_lats * 111.0,
            rand_lons * 88.0
        ])
        
        dists, idxs = tree_km.query(q_coords_km, k=1)
        return np.sum(dists < 10.0)

    # Prepare MRDS tree once
    coords_km = np.column_stack([
        nv_mrds[lat_col].values * 111.0,
        nv_mrds[lon_col].values * 88.0
    ])
    mrds_tree_km = cKDTree(coords_km)

    for i in range(n_iterations):
        # Generate random points
        rand_lats = np.random.uniform(nv_lat_min, nv_lat_max, n_points)
        rand_lons = np.random.uniform(nv_lon_min, nv_lon_max, n_points)
        
        # Check hits
        q_coords_km = np.column_stack([rand_lats * 111.0, rand_lons * 88.0])
        dists, _ = mrds_tree_km.query(q_coords_km, k=1)
        hits = np.sum(dists < 10.0)
        hit_counts.append(hits)
        
    hit_counts = np.array(hit_counts)
    hit_percentages = (hit_counts / n_points) * 100
    
    mean_rate = np.mean(hit_percentages)
    median_rate = np.median(hit_percentages)
    p95 = np.percentile(hit_percentages, 95)
    
    logger.info(f"Simulation Results ({n_iterations} runs):")
    logger.info(f"Mean Random Hit Rate: {mean_rate:.2f}%")
    logger.info(f"Median Random Hit Rate: {median_rate:.2f}%")
    logger.info(f"95th Percentile: {p95:.2f}%")
    
    # 4. Analyze Misses (Model Targets Analysis)
    logger.info("\n--- Analyzing Model Misses ---")
    
    # Check model targets against the same tree
    model_coords_km = np.column_stack([
        nv_targets['Latitude'].values * 111.0,
        nv_targets['Longitude'].values * 88.0
    ])
    
    dists, _ = mrds_tree_km.query(model_coords_km, k=1)
    
    nv_targets['Dist_to_Nearest_MRDS_km'] = dists
    nv_targets['Hit'] = dists < 10.0
    
    real_hit_rate = nv_targets['Hit'].mean() * 100
    logger.info(f"Re-calculated Model Hit Rate: {real_hit_rate:.2f}% ({nv_targets['Hit'].sum()}/{len(nv_targets)})")
    
    misses = nv_targets[~nv_targets['Hit']]
    logger.info(f"Found {len(misses)} Misses (>10km from deposit):")
    
    if len(misses) > 0:
        print(misses[['Latitude', 'Longitude', 'Density_Contrast', 'Dist_to_Nearest_MRDS_km']].to_string(index=False))
        # Save misses to file
        misses.to_csv('data/outputs/nevada_misses.csv', index=False)
        logger.info("Saved misses to data/outputs/nevada_misses.csv")
    else:
        logger.info("No misses found! All targets are near deposits.")

if __name__ == "__main__":
    main()
