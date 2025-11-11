#!/usr/bin/env python3
"""
GeoAnomalyMapper - One-Click Processing Pipeline
=================================================
Automates the full data processing workflow:
1. Processes all raw data (DEMs, gravity, magnetic)
2. Detects void probabilities
3. Converts the final probability map to GeoJSON "pins"
4. Creates a final visualization map

Usage:
    python one-click-pipeline.py --region="lon_min,lat_min,lon_max,lat_max"
    
Example:
    python one-click-pipeline.py --region="-119.0,33.0,-117.0,35.0"
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path

# --- Dependencies for GeoJSON conversion ---
# You may need to install these:
# pip install gdal rasterio geojson shapely
try:
    from osgeo import gdal
    import rasterio
    import geojson
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except ImportError:
    print("Error: Missing required packages. Please run:")
    print("pip install gdal rasterio geojson shapely")
    sys.exit(1)
# -----------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Paths ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

SCRIPT_PATHS = {
    "process": SCRIPT_DIR / "process_data.py",
    "detect": SCRIPT_DIR / "detect_voids.py",
    "visualize": SCRIPT_DIR / "create_visualization.py"
}

OUTPUT_PATHS = {
    "probability_tif": PROJECT_DIR / "data" / "outputs" / "void_detection" / "void_probability.tif",
    "geojson_pins": PROJECT_DIR / "docs" / "data" / "voids.geojson",
    "final_viz_kmz": PROJECT_DIR / "data" / "outputs" / "final_visualization" / "final_anomaly_map.kmz"
}
# ---------------------

def run_step(step_name: str, cmd_args: list, log_file: Path):
    """Executes a processing step as a subprocess and logs its output."""
    logging.info(f"--- Starting: {step_name} ---")
    
    # Ensure the log file directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f_log:
        try:
            # Run the subprocess
            process = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            # Log STDOUT and STDERR
            f_log.write("--- STDOUT ---\n")
            f_log.write(process.stdout)
            logging.info(f"STDOUT:\n{process.stdout}")
            
            if process.stderr:
                f_log.write("\n--- STDERR ---\n")
                f_log.write(process.stderr)
                logging.info(f"STDERR:\n{process.stderr}")
                
            logging.info(f"✓ SUCCESS: {step_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            # Log failure
            logging.error(f"!!! FAILED: {step_name} !!!")
            logging.error(f"Return Code: {e.returncode}")
            
            f_log.write("--- STDOUT ---\n")
            f_log.write(e.stdout)
            logging.error(f"STDOUT:\n{e.stdout}")
            
            f_log.write("\n--- STDERR ---\n")
            f_log.write(e.stderr)
            logging.error(f"STDERR:\n{e.stderr}")
            
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during '{step_name}': {e}")
            f_log.write(f"An unexpected error occurred: {e}")
            return False

def convert_tif_to_geojson(
    input_tif: Path,
    output_geojson: Path,
    threshold: float = 0.7
):
    """
    Converts a probability GeoTIFF to GeoJSON "pins" (polygons).
    Finds all pixels above the threshold and converts them to polygons.
    """
    logging.info(f"--- Starting: Step 3: Convert TIF to GeoJSON Pins ---")
    
    if not input_tif.exists():
        logging.error(f"Input file not found, skipping GeoJSON conversion: {input_tif}")
        return False
    
    try:
        # Open the probability raster
        with rasterio.open(input_tif) as src:
            image = src.read(1)
            # Create a mask where pixels are *above* the threshold
            mask = (image >= threshold).astype('uint8')
            
            # Check if any pixels met the threshold
            if mask.sum() == 0:
                logging.warning(f"No pixels found above threshold ({threshold}).")
                logging.warning("GeoJSON file will be empty.")
                features = []
            else:
                # Extract shapes (polygons) from the raster mask
                # This returns a generator of (geometry, value) pairs
                shapes_gen = rasterio.features.shapes(
                    mask,
                    mask=mask,
                    transform=src.transform,
                    connectivity=8  # Use 8-way connectivity
                )
                
                # Get all geometries where the value is 1 (i.e., above threshold)
                geometries = [shape(geom) for geom, val in shapes_gen if val == 1]
                
                if not geometries:
                    logging.warning("Raster shapes were generated but resulted in no geometries.")
                    features = []
                else:
                    logging.info(f"Found {len(geometries)} potential anomaly zones.")
                    
                    # To simplify, we can merge all overlapping polygons into one MultiPolygon
                    # For individual pins, we would iterate and create a Feature for each.
                    # Let's create individual pins.
                    features = []
                    for geom in geometries:
                        # Calculate centroid for a simple pin location
                        centroid = geom.centroid
                        
                        # Create a GeoJSON feature for each polygon
                        features.append(geojson.Feature(
                            geometry=mapping(geom),
                            properties={
                                "description": "High Probability Anomaly Zone",
                                "probability_threshold": threshold,
                                "latitude": centroid.y,
                                "longitude": centroid.x
                            }
                        ))

        # Create the final GeoJSON FeatureCollection
        feature_collection = geojson.FeatureCollection(features)
        
        # Ensure the output directory exists
        output_geojson.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the GeoJSON file
        with open(output_geojson, 'w') as f:
            geojson.dump(feature_collection, f, indent=2)
            
        logging.info(f"✓ SUCCESS: Step 3: Saved {len(features)} pins to {output_geojson}")
        return True
        
    except Exception as e:
        logging.error(f"!!! FAILED: Step 3: GeoJSON Conversion !!!")
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(
        description="GeoAnomalyMapper One-Click Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
    )
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip the final visualization step (for speed)'
    )
    
    args = parser.parse_args()
    
    # Define log paths
    LOG_DIR = PROJECT_DIR / "logs"
    LOG_STEP1 = LOG_DIR / "step1_process_data.log"
    LOG_STEP2 = LOG_DIR / "step2_detect_voids.log"
    LOG_STEP4 = LOG_DIR / "step4_visualize.log"
    
    logging.info("=" * 65)
    logging.info("       STARTING GEOANOMALYMAPPER PIPELINE")
    logging.info("=" * 65)
    logging.info(f"Processing region: {args.region}")
    
    logging.warning("!!! IMPORTANT !!!")
    logging.warning("This script AUTOMATES PROCESSING, not downloading.")
    logging.warning("Please ensure you have MANUALLY downloaded raw data")
    logging.warning("(gravity, magnetic, DEM) into the 'data/raw/' directory.")
    logging.warning("This script will attempt to unzip and merge DEM tiles.")
    logging.warning("-------------------------------------------------")
    
    # --- Step 1: Process Raw Data ---
    # **** THIS IS THE FIX ****
    # We now pass the argument as f"--region={args.region}"
    # This combines it into one string (e.g., "--region=-119.0...")
    # This prevents PowerShell/sh from misinterpreting the negative number.
    cmd_step1 = [
        sys.executable, 
        str(SCRIPT_PATHS["process"]), 
        f"--region={args.region}"
    ]
    if not run_step("Step 1: Process Raw Data (with auto-DEM merge)", cmd_step1, LOG_STEP1):
        logging.error("Exiting due to failure in Step 1.")
        sys.exit(1)
        
    # --- Step 2: Detect Voids ---
    # **** THIS IS THE FIX ****
    cmd_step2 = [
        sys.executable,
        str(SCRIPT_PATHS["detect"]),
        f"--region={args.region}"
    ]
    if not run_step("Step 2: Detect Void Probabilities", cmd_step2, LOG_STEP2):
        logging.error("Exiting due to failure in Step 2.")
        sys.exit(1)
        
    # --- Step 3: Convert TIF to GeoJSON Pins ---
    if not convert_tif_to_geojson(OUTPUT_PATHS["probability_tif"], OUTPUT_PATHS["geojson_pins"]):
        logging.error("Exiting due to failure in Step 3.")
        sys.exit(1)
        
    # --- Step 4: Final Visualization ---
    if not args.skip_viz:
        # Create output dir for visualization
        OUTPUT_PATHS["final_viz_kmz"].parent.mkdir(parents=True, exist_ok=True)
        
        # **** THIS IS THE FIX (for consistency) ****
        cmd_step4 = [
            sys.executable,
            str(SCRIPT_PATHS["visualize"]),
            str(OUTPUT_PATHS["probability_tif"]),
            f"--output-dir={OUTPUT_PATHS['final_viz_kmz'].parent}"
        ]
        if not run_step("Step 4: Create Final Visualization (KMZ)", cmd_step4, LOG_STEP4):
            logging.warning("Pipeline completed, but final visualization failed.")
    else:
        logging.info("Skipping final visualization step as requested.")
        
    logging.info("=" * 65)
    logging.info("     GEOANOMALYMAPPER PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("=" * 65)
    logging.info(f"Your 'pins' are ready: {OUTPUT_PATHS['geojson_pins']}")
    logging.info("You can now commit and push this file to update your GitHub Pages globe:")
    logging.info(f"  git add {os.path.relpath(OUTPUT_PATHS['geojson_pins'])}")
    logging.info('  git commit -m "Update anomaly pins"')
    logging.info("  git push")
    logging.info("=" * 65)

if __name__ == "__main__":
    main()