import argparse
import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio import enums
from rasterio import warp

# Try importing project paths, fallback to local if missing
try:
    from project_paths import OUTPUTS_DIR
except ImportError:
    OUTPUTS_DIR = Path("outputs")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_mining_results(
    belief_map_path: Path,
    density_map_path: Path,
    validation_features_path: Path,
    output_report_path: Path,
    belief_threshold: float = 0.7,
    density_threshold: float = 100.0, # kg/m^3 positive contrast
) -> None:
    """
    Validates mineral exploration results against known features.
    
    This function checks for spatial overlap between high-belief/high-density
    anomalies (mass excess) and known mineral deposits or mining sites.
    """
    logger.info(f"Starting Mining Validation against {validation_features_path.name}...")

    if not belief_map_path.exists() or not density_map_path.exists():
        logger.error("Missing required input maps (Belief or Density). Skipping validation.")
        return

    if not validation_features_path.exists():
        logger.warning(f"Validation features not found at {validation_features_path}. Skipping spatial analysis.")
        # Create a dummy report
        output_report_path.write_text("Validation skipped: Missing known features file.")
        return

    # 1. Load Belief Map (Master Grid)
    with rasterio.open(belief_map_path) as src:
        belief_data = src.read(1)
        profile = src.profile
        
    # 2. Load Density Map and resample to match belief map
    with rasterio.open(density_map_path) as src:
        density_data = np.empty_like(belief_data, dtype=np.float32)
        warp.reproject(
            source=rasterio.band(src, 1),
            destination=density_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile['transform'],
            dst_crs=profile['crs'],
            resampling=enums.Resampling.bilinear,
            dst_nodata=np.nan
        )

    # 3. Identify Mineral Candidates (Mass Excess)
    # We look for high belief AND positive density contrast above threshold
    
    # Note: We assume 'Reinforced' belief in detect_voids.py now represents 'Mineral' in mineral mode.
    # We also check for positive density contrast (mass excess).
    
    # Mask NaNs
    valid_mask = ~np.isnan(belief_data) & ~np.isnan(density_data)
    
    # Candidate mask: High Belief AND High Positive Density
    candidate_mask = (belief_data > belief_threshold) & \
                     (density_data > density_threshold) & \
                     valid_mask

    num_candidates = np.sum(candidate_mask)
    logger.info(f"Identified {num_candidates} pixels as high-confidence mineral candidates.")

    # 4. Spatial Overlap Analysis (Simplified for this script)
    # In a real scenario, this would involve vector operations (e.g., using geopandas/shapely)
    # For simplicity, we check if any candidate pixel overlaps with the bounding box of the validation features.
    
    # Read validation features (assuming it's a simple raster mask or similar)
    # For this implementation, we will assume validation_features_path points to a raster
    # where 1 indicates a known mineral deposit/mine.
    try:
        with rasterio.open(validation_features_path) as src:
            # Reproject/resample validation features to match belief map grid
            validation_data = np.empty_like(belief_data, dtype=np.float32)
            warp.reproject(
                source=rasterio.band(src, 1),
                destination=validation_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=enums.Resampling.nearest,
                dst_nodata=0
            )
            # Treat any non-zero value as a known feature
            known_feature_mask = (validation_data > 0) & valid_mask
            num_known_features = np.sum(known_feature_mask)
            logger.info(f"Found {num_known_features} pixels of known mineral features.")

            # Calculate overlap
            overlap_mask = candidate_mask & known_feature_mask
            num_overlap = np.sum(overlap_mask)
            
            if num_candidates > 0:
                overlap_percentage = (num_overlap / num_candidates) * 100
            else:
                overlap_percentage = 0.0

            # 5. Generate Report
            report_content = f"""
Mining Validation Report
=======================
Belief Map: {belief_map_path.name}
Density Map: {density_map_path.name}
Validation Features: {validation_features_path.name}

Parameters:
  Belief Threshold: > {belief_threshold}
  Density Threshold: > {density_threshold} kg/m^3

Results:
  Total High-Confidence Mineral Candidates (Pixels): {num_candidates}
  Total Known Mineral Features (Pixels): {num_known_features}
  Overlapping Pixels (True Positives): {num_overlap}
  Overlap Percentage (Candidates validated): {overlap_percentage:.2f}%

Conclusion: {'SUCCESS' if overlap_percentage > 10 else 'NEEDS REVIEW'}
"""
            output_report_path.write_text(report_content)
            logger.info(f"Validation complete. Report saved to {output_report_path}")

    except Exception as e:
        logger.error(f"Error during spatial analysis: {e}")
        output_report_path.write_text(f"Validation failed due to error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 9: Mineral Exploration Validation")
    parser.add_argument("--prefix", required=True, help="Output file prefix used in workflow (e.g., 'outputs/test_tile')")
    parser.add_argument("--belief-thresh", type=float, default=0.7, help="Belief threshold for candidate selection")
    parser.add_argument("--density-thresh", type=float, default=100.0, help="Density contrast threshold (kg/m^3)")
    args = parser.parse_args()

    prefix = args.prefix
    
    # Input paths based on workflow output naming convention
    belief_path = Path(f"{prefix}_fused_belief_reinforced.tif")
    density_path = Path(f"{prefix}_density_model.tif")
    
    # Placeholder for actual validation data path (e.g., known mine locations)
    # This file must be prepared separately, e.g., a raster mask of known deposits.
    validation_path = OUTPUTS_DIR / "validation" / "known_mineral_deposits_mask.tif"
    
    # Output report path
    report_path = Path(f"{prefix}_mining_validation_report.txt")

    validate_mining_results(
        belief_map_path=belief_path,
        density_map_path=density_path,
        validation_features_path=validation_path,
        output_report_path=report_path,
        belief_threshold=args.belief_thresh,
        density_threshold=args.density_thresh,
    )

if __name__ == "__main__":
    main()