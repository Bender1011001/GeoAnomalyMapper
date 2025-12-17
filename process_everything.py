import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("full_processing.log")
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PYTHON_EXE = sys.executable
DATA_DIR = Path("data")
RAW_GRAVITY = DATA_DIR / "raw/gravity/gravity_anomaly.tif"
RAW_MAGNETIC = DATA_DIR / "raw/magnetic/EMAG2_V3_20170530_Sealevel.tif"

OUTPUT_DIR = DATA_DIR / "outputs/usa_supervised"
GRAVITY_MOSAIC = OUTPUT_DIR / "usa_gravity_mosaic.tif"
MAGNETIC_MOSAIC = OUTPUT_DIR / "usa_magnetic_mosaic.tif"

# USA CONUS EXTENT (Approx)
# West, South, East, North
EXTENT = "-125 24 -66 50"
RESOLUTION = "0.01 0.01" # Approx 1km

def run_cmd(cmd):
    logger.info(f"▶️ Executing: {cmd}")
    try:
        ret = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # logger.info(ret.stdout) # Don't spam log with GDAL output unless needed
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {e.output}")
        raise

def prep_data():
    logger.info("--- PHASE 1: Data Preparation ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Gravity
    if not GRAVITY_MOSAIC.exists():
        logger.info("Processing Gravity Data...")
        if not RAW_GRAVITY.exists():
            logger.error(f"Missing Raw Gravity: {RAW_GRAVITY}")
            return False
            
        # gdalwarp to clip and resample
        cmd = f'gdalwarp -t_srs EPSG:4326 -te {EXTENT} -tr {RESOLUTION} -r cubic -dstnodata -9999 "{RAW_GRAVITY}" "{GRAVITY_MOSAIC}"'
        run_cmd(cmd)
        logger.info("✅ Gravity Mosaic Created")
    else:
        logger.info("✅ Gravity Mosaic exists")

    # 2. Magnetic
    if not MAGNETIC_MOSAIC.exists():
        logger.info("Processing Magnetic Data...")
        if not RAW_MAGNETIC.exists():
            logger.error(f"Missing Raw Magnetic: {RAW_MAGNETIC}")
            return False
            
        cmd = f'gdalwarp -t_srs EPSG:4326 -te {EXTENT} -tr {RESOLUTION} -r cubic -dstnodata -9999 "{RAW_MAGNETIC}" "{MAGNETIC_MOSAIC}"'
        run_cmd(cmd)
        logger.info("✅ Magnetic Mosaic Created")
    else:
        logger.info("✅ Magnetic Mosaic exists")
        
    return True

def run_pipeline():
    logger.info("--- PHASE 2: Pipeline Execution ---")
    
    # Run the robust pipeline script
    # This handles Training, Inference, Classification
    cmd = [PYTHON_EXE, "run_robust_pipeline.py"]
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Use Popen to stream output to our log
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='') # Stream to console
        
    process.wait()
    
    if process.returncode != 0:
        logger.error("❌ Pipeline failed.")
        return False
        
    logger.info("✅ Pipeline completed.")
    return True

def post_process():
    logger.info("--- PHASE 3: Extraction & Visualization ---")
    
    # The pipeline outputs 'usa_supervised_probability.tif' in 'data/outputs'
    # Wait, run_robust_pipeline.py lines 33/41 says:
    # DATA_DIR = Path("data")
    # OUTPUTS_DIR = DATA_DIR / "outputs" / "usa_supervised"
    # SUPERVISED_PROB = DATA_DIR / "outputs" / "usa_supervised_probability.tif"
    
    prob_map = DATA_DIR / "outputs/usa_supervised_probability.tif"
    target_csv = DATA_DIR / "outputs/usa_targets.csv"
    graded_csv = DATA_DIR / "outputs/high_value_targets_usa.csv"
    
    if not prob_map.exists():
        logger.error(f"❌ Probability map missing: {prob_map}")
        return False
        
    # 1. Extract
    logger.info("Extracting Targets...")
    cmd1 = [PYTHON_EXE, "extract_targets.py", str(prob_map), "--output", str(target_csv), "--threshold", "0.8"]
    subprocess.run(cmd1, check=True)
    
    # 2. Grade
    logger.info("Grading Targets...")
    cmd2 = [PYTHON_EXE, "filter_and_grade_targets.py", str(target_csv), "--known", "data/usgs_goldilocks.csv"]
    subprocess.run(cmd2, check=True)
    
    # 3. Plot
    logger.info("Plotting...")
    cmd3 = [PYTHON_EXE, "plot_targets.py", str(graded_csv), "--tif", str(prob_map)]
    subprocess.run(cmd3, check=True)
    
    return True

if __name__ == "__main__":
    logger.info("Starting PROCESS EVERYTHING...")
    
    if prep_data():
        if run_pipeline():
            post_process()
            
    logger.info("FINISH.")
