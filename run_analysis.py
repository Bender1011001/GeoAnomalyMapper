#!/usr/bin/env python3
"""
Master script to run the GeoAnomalyMapper analysis workflow.

This script orchestrates the execution of the following steps:
1.  **Spatial Analysis (Phase 8):** Generates the initial anomaly map using a Convolutional Autoencoder.
2.  **Land Masking (Phase 8b):** Masks out ocean/water pixels from the anomaly map.
3.  **Visualization (Phase 9):** Generates HTML, PNG, and KMZ visualizations from the masked anomaly map.

Usage:
    python run_analysis.py [--config config.yaml]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.config import load_config
from project_paths import OUTPUTS_DIR

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_step(script_name: str, args: list, description: str) -> None:
    """
    Executes a Python script as a subprocess with error handling.

    Args:
        script_name: The name of the script to run (relative to project root).
        args: List of command-line arguments to pass to the script.
        description: A human-readable description of the step for logging.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    logger.info(f"--- Starting {description} ({script_name}) ---")
    
    cmd = [sys.executable, script_name] + args
    
    try:
        # Execute the command and wait for it to complete
        # check=True raises CalledProcessError if exit code is non-zero
        subprocess.run(cmd, check=True)
        logger.info(f"--- Finished {description} ---\n")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Step failed: {description}")
        logger.error(f"Command '{' '.join(cmd)}' returned non-zero exit status {e.returncode}.")
        sys.exit(e.returncode)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full GeoAnomalyMapper analysis workflow.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Initializing workflow with config: {config_path}")

    # Load config to determine filenames for chaining steps
    try:
        config = load_config(str(config_path))
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        sys.exit(1)

    # --- Determine Paths for Chaining ---
    # 1. Get the base anomaly map filename from config (Phase 8 output)
    anomaly_filename = config.get('output', {}).get('anomaly_map', 'spatial_anomaly_v1.tif')
    
    # 2. Calculate the expected masked filename (Phase 8b output / Phase 9 input)
    # Logic matches phase8b_land_mask.py: stem + "_masked" + suffix
    p = Path(anomaly_filename)
    masked_filename = f"{p.stem}_masked{p.suffix}"
    
    # Resolve full path relative to OUTPUTS_DIR for clarity
    masked_path = OUTPUTS_DIR / masked_filename

    # --- Step 1: Spatial Analysis ---
    # Runs phase8_spatial_analysis.py
    # Output: defined in config['output']['anomaly_map']
    run_step(
        "phase8_spatial_analysis.py",
        ["--config", str(config_path)],
        "Step 1: Spatial Analysis"
    )

    # --- Step 2: Land Masking ---
    # Runs phase8b_land_mask.py
    # Input: implicitly config['output']['anomaly_map'] (or we could pass --input)
    # Output: implicitly input_stem + "_masked" (or we could pass --output)
    run_step(
        "phase8b_land_mask.py",
        ["--config", str(config_path)],
        "Step 2: Land Masking"
    )

    # --- Step 3: Visualization ---
    # Runs phase9_visualization.py
    # Input: We EXPLICITLY pass the masked file from Step 2
    run_step(
        "phase9_visualization.py",
        ["--config", str(config_path), "--input", str(masked_path)],
        "Step 3: Visualization"
    )

    logger.info("=== Workflow Completed Successfully ===")
    logger.info(f"Final products generated in: {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()