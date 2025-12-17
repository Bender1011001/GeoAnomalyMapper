import os
import subprocess
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw/insar/seasonal_usa")
PROCESSED_DIR = Path("data/processed/insar/mosaics")
SEASONS = ["winter", "spring", "summer", "fall"]
METRIC = "COH12" # 12-day coherence
POL = "vv"

def build_mosaics():
    if not RAW_DIR.exists():
        print(f"Error: Raw data directory not found: {RAW_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for GDAL
    try:
        subprocess.run(["gdalbuildvrt", "--version"], check=True, stdout=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: GDAL tools not found. Please install GDAL (e.g., 'conda install gdal').")
        return

    for season in SEASONS:
        print(f"Building mosaic for {season}...")
        
        # Define output filename
        output_vrt = PROCESSED_DIR / f"usa_{season}_{POL}_{METRIC}.vrt"
        
        # Find all tiles for this season
        # Pattern: */*_winter_vv_COH12.tif
        pattern = f"*_{season}_{POL}_{METRIC}.tif"
        files = list(RAW_DIR.rglob(pattern))
        
        if not files:
            print(f"  Warning: No files found for {season}")
            continue
            
        print(f"  Found {len(files)} tiles.")
        
        # Build file list text file (to avoid command line length limits)
        list_file = PROCESSED_DIR / f"{season}_filelist.txt"
        with open(list_file, "w") as f:
            for path in files:
                f.write(str(path.absolute()) + "\n")
        
        # Run gdalbuildvrt
        try:
            cmd = [
                "gdalbuildvrt",
                "-input_file_list", str(list_file),
                str(output_vrt)
            ]
            subprocess.run(cmd, check=True)
            print(f"  Created: {output_vrt}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed to build VRT for {season}: {e}")
        finally:
            # Clean up list file
            if list_file.exists():
                list_file.unlink()

    print("\n--- Mosaic Build Complete ---")
    print(f"VRTs saved to: {PROCESSED_DIR.resolve()}")

if __name__ == "__main__":
    build_mosaics()