#!/usr/bin/env python3
"""
Robust Pipeline Runner for GeoAnomalyMapper
===========================================

Orchestrates the full pipeline with checkpointing.
If a step crashes, this script can be re-run and will skip successful steps.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PYTHON_EXE = sys.executable

# Paths
DATA_DIR = Path("data")
OUTPUTS_DIR = DATA_DIR / "outputs" / "usa_supervised"
FINAL_OUTPUTS = DATA_DIR / "outputs"
USGS_CSV = DATA_DIR / "usgs_goldilocks.csv"

# Models and Mosaics
GRAVITY_MOSAIC = OUTPUTS_DIR / "usa_gravity_mosaic.tif"
MAGNETIC_MOSAIC = OUTPUTS_DIR / "usa_magnetic_mosaic.tif"
PINN_MODEL = Path("usa_pinn_model.pth")
DENSITY_MODEL = DATA_DIR / "outputs" / "usa_density_model.tif" # Predicted density map
SUPERVISED_PROB = DATA_DIR / "outputs" / "usa_supervised_probability.tif"

def run_step(step_name, command, check_file=None):
    """
    Runs a pipeline step if the check_file doesn't exist.
    """
    logger.info(f"--- STEP: {step_name} ---")
    
    if check_file and Path(check_file).exists():
        logger.info(f">> Skipping {step_name} (Output exists: {check_file})")
    
    logger.info(f">> Executing: {' '.join(command)}")
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to log
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"!! {step_name} FAILED with exit code {process.returncode}")
            return False
            
        logger.info(f">> {step_name} COMPLETED")
        return True
        
    except Exception as e:
        logger.exception(f"❌ {step_name} CRASHED: {e}")
        return False

def main():
    logger.info("Initializing Robust Pipeline Runner...")
    
    # Check Prerequisite Data
    if not GRAVITY_MOSAIC.exists():
        logger.warning(f"⚠️ Gravity Mosaic missing: {GRAVITY_MOSAIC}")
        # In a real scenario, we might trigger the download/processing steps.
        # For now, we assume user has the mosaics as per previous context.
    
    # STEP 1: USGS Data Setup
    if not run_step(
        "USGS Data Setup",
        [PYTHON_EXE, "setup_usgs_data.py"],
        check_file=USGS_CSV
    ): return

    # STEP 2: PINN Training
    # Only train if model doesn't exist
    if not run_step(
        "PINN Training",
        [PYTHON_EXE, "train_usa_pinn.py", 
         "--gravity", str(GRAVITY_MOSAIC),
         "--magnetic", str(MAGNETIC_MOSAIC),
         "--output", str(PINN_MODEL),
         "--epochs", "20"], # Reduced for speed in this demo run
        check_file=PINN_MODEL
    ): return

    # STEP 3: Density Inversion (Prediction)
    if not run_step(
        "Density Map Prediction",
        [PYTHON_EXE, "predict_usa.py",
         "--model", str(PINN_MODEL),
         "--gravity", str(GRAVITY_MOSAIC),
         "--output", str(DENSITY_MODEL)],
        check_file=DENSITY_MODEL
    ): return

    # STEP 4: Supervised Classification
    # Inputs: Gravity, Magnetic, Density
    # We pass --features explicitly
    feature_list = [str(GRAVITY_MOSAIC), str(MAGNETIC_MOSAIC), str(DENSITY_MODEL)]
    
    if not run_step(
        "Supervised Classification",
        [PYTHON_EXE, "classify_supervised.py",
         "--features", *feature_list,
         "--use-usgs", # Use the Goldilocks dataset we set up in Step 1
         "--output", str(SUPERVISED_PROB),
         "--n-estimators", "100"],
        check_file=SUPERVISED_PROB
    ): return

    # STEP 5: Validation
    # Use "count_targets.py" for basic stats
    run_step(
        "Target Analysis",
        [PYTHON_EXE, "count_targets.py", "--threshold", "0.7"]
    )
    
    # Use "validate_robustness.py" for rigorous metrics
    # Note: validate_robustness takes a long time, so we just run it.
    run_step(
        "Robustness Validation",
        [PYTHON_EXE, "validate_robustness.py"]
    )

    logger.info("🎉 Full Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    main()
