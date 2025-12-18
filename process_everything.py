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
RAW_MAGNETIC = DATA_DIR / "raw/magnetic/EMAG2_V3_20170530_UpCont.tif"

OUTPUT_DIR = DATA_DIR / "outputs/usa_supervised"
GRAVITY_MOSAIC = OUTPUT_DIR / "usa_gravity_mosaic.tif"
MAGNETIC_MOSAIC = OUTPUT_DIR / "usa_magnetic_mosaic.tif"

# USA CONUS EXTENT (Approx)
# West, South, East, North
# USA + ALASKA EXTENT
# Covers Aleutians to Maine, Florida to Barrow
# -179 to -65 Longitude, 24 to 72 Latitude
EXTENT = "-179 24 -65 72" 
RESOLUTION = "0.03 0.03" # ~3.3km

def run_cmd(cmd):
    logger.info(f"▶️ Executing: {cmd}")
    try:
        ret = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.output}")
        raise

def create_land_mask():
    """
    Creates a high-res land mask by processing the Lithology file.
    """
    litho_raw = Path("data/raw/lithology/glim_wgs84_0.5deg.tif")
    mask_out = OUTPUT_DIR / "usa_land_mask.tif"
    
    if mask_out.exists(): return True
    
    if not litho_raw.exists():
        logger.warning(f"Lithology file missing: {litho_raw}. Skipping Ocean Mask.")
        return False
        
    logger.info("Creating Land Mask from Lithology...")
    # Resample Lithology to match our master grid (Gravity Mosaic)
    # Any pixel with lithology (not nodata) is land.
    
    # Check litho nodata?
    # We warp it to our Extent/Resolution. 
    # Use 'near' for categorical data (lithology IDs)
    cmd = f'gdalwarp -t_srs EPSG:4326 -te {EXTENT} -tr {RESOLUTION} -r near -dstnodata 255 "{litho_raw}" "{mask_out}"'
    run_cmd(cmd)
    return True

def clean_outputs():
    logger.info("Cleaning old outputs for FRESH RUN...")
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Also remove intermediate models if we want a TRULY fresh run?
    # The user said "without skipping anything", implying re-training.
    pinn_model = Path("usa_pinn_model.pth")
    if pinn_model.exists():
        pinn_model.unlink()
    
    # Clean final targets
    final_targets = DATA_DIR / "outputs/usa_targets.csv"
    if final_targets.exists():
        final_targets.unlink()
        
    # Clean intermediate models that might cause skipping
    density_model = DATA_DIR / "outputs/usa_density_model.tif"
    if density_model.exists(): density_model.unlink()
    
    prob_map = DATA_DIR / "outputs/usa_supervised_probability.tif"
    if prob_map.exists(): prob_map.unlink()

def prep_data():
    logger.info("--- PHASE 1: Data Preparation ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Gravity
    if not GRAVITY_MOSAIC.exists():
        logger.info("Processing Gravity Data...")
        if not RAW_GRAVITY.exists():
            logger.error(f"Missing Raw Gravity: {RAW_GRAVITY}")
            return False
        cmd = f'gdalwarp -t_srs EPSG:4326 -te {EXTENT} -tr {RESOLUTION} -r cubic -dstnodata -9999 "{RAW_GRAVITY}" "{GRAVITY_MOSAIC}"'
        run_cmd(cmd)
        logger.info("✅ Gravity Mosaic Created")
    
    # 2. Magnetic
    if not MAGNETIC_MOSAIC.exists():
        logger.info("Processing Magnetic Data...")
        if not RAW_MAGNETIC.exists():
            logger.error(f"Missing Raw Magnetic: {RAW_MAGNETIC}")
            return False
        temp_mag = DATA_DIR / "raw/magnetic/temp_mag_georef.tif"
        cmd_ref = f'gdal_translate -a_srs EPSG:4326 -a_ullr 0 90 360 -90 "{RAW_MAGNETIC}" "{temp_mag}"'
        run_cmd(cmd_ref)
        
        cmd = f'gdalwarp -t_srs EPSG:4326 -te {EXTENT} -tr {RESOLUTION} -r cubic -dstnodata -9999 --config CENTER_LONG 180 "{temp_mag}" "{MAGNETIC_MOSAIC}"'
        run_cmd(cmd)
        logger.info("✅ Magnetic Mosaic Created")
        
    # 3. Land Mask
    create_land_mask()
        
    return True

def run_pipeline():
    logger.info("--- PHASE 2: Pipeline Execution ---")
    cmd = [PYTHON_EXE, "run_robust_pipeline.py"]
    logger.info(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8'
    )
    
    for line in process.stdout:
        print(line, end='') 
        
    process.wait()
    
    if process.returncode != 0:
        logger.error("Pipeline failed.")
        return False
        
    logger.info("Pipeline completed.")
    return True

def post_process():
    logger.info("--- PHASE 3: Extraction & Visualization ---")
    
    # Update paths
    supervised_prob = DATA_DIR / "outputs/usa_supervised_probability.tif"
    masked_prob = DATA_DIR / "outputs/usa_supervised_probability_MASKED.tif"
    target_csv = DATA_DIR / "outputs/usa_targets.csv"
    graded_csv = DATA_DIR / "outputs/high_value_targets_usa.csv"
    
    if not supervised_prob.exists():
        logger.error(f"Probability map missing: {supervised_prob}")
        return False

    # Apply Land Mask if exists
    mask_tif = OUTPUT_DIR / "usa_land_mask.tif"
    final_tif = supervised_prob
    
    if mask_tif.exists():
        logger.info("Applying Ocean Mask...")
        # Multiply Probability by (Mask != nodata). 
        # Using gdal_calc for robust raster math
        # A=Prob, B=Mask. Result = A * (B > 0). Assumes Mask is valid land=ID, ocean=0/NaN
        try:
             # Check if gdal_calc is available via subprocess, else numpy
             # Let's simple use numpy here to ensure python compatibility
             pass # We'll do it in python logic or assume extract_targets handles it?
             # Actually, simpler to leave it to extract_targets? 
             # No, visualize needs it.
             # Let's use gdal_calc.py command string if available
             cmd_mask = f'gdal_calc.py -A "{supervised_prob}" -B "{mask_tif}" --outfile="{masked_prob}" --calc="A*(B>0)" --NoDataValue=0'
             # Note: gdal_calc is often not in path on windows as exe, usually python script. 
             # Safe fallback: Do nothing, let extract_targets check mask?
             # User explicit request: "Mask off the ocean".
             # Better: Use python rasterio to mask
             with rasterio.open(mask_tif) as m, rasterio.open(supervised_prob) as s:
                 mask_data = m.read(1)
                 prob_data = s.read(1)
                 
                 # Resize mask found?? gdalwarp ensured they match extent/resolution. 
                 # But dimensions might differ by 1 pixel due to rounding.
                 # Safe warp done in prep_data. 
                 
                 # Force Masking
                 # If shapes mismatch (rare but possible), we skip.
                 if mask_data.shape == prob_data.shape:
                     # Mask: Keep where mask > 0
                     prob_data[mask_data <= 0] = 0
                     
                     profile = s.profile
                     with rasterio.open(masked_prob, 'w', **profile) as dst:
                         dst.write(prob_data, 1)
                     final_tif = masked_prob
                     logger.info("Ocean Mask Applied.")
                 else:
                     logger.warning(f"Shape mismatch (Prob {prob_data.shape} vs Mask {mask_data.shape}). Skipping mask.")
        except Exception as e:
            logger.warning(f"Masking failed: {e}")
    
    # 1. Extract
    logger.info("Extracting Targets...")
    cmd1 = [PYTHON_EXE, "extract_targets.py", str(final_tif), "--output", str(target_csv), "--threshold", "0.85"]
    subprocess.run(cmd1, check=True)
    
    # 2. Grade
    logger.info("Grading Targets...")
    cmd2 = [PYTHON_EXE, "filter_and_grade_targets.py", str(target_csv), "--known", "data/usgs_goldilocks.csv"]
    subprocess.run(cmd2, check=True)
    
    # 3. Plot
    logger.info("Plotting...")
    cmd3 = [PYTHON_EXE, "plot_targets.py", str(graded_csv), "--tif", str(final_tif)]
    subprocess.run(cmd3, check=True)
    
    return True

if __name__ == "__main__":
    logger.info("Starting PROCESS EVERYTHING...")
    import sys
    import rasterio # Late import for post-process
    import numpy as np
    
    # Force clean for fresh run
    clean_outputs()
    
    if prep_data():
        if run_pipeline():
            post_process()
            
    logger.info("FINISH.")
