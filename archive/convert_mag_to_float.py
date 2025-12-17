import rasterio
import numpy as np
from pathlib import Path
import logging
import project_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_rgb_to_float():
    """
    Converts the RGB EMAG2 source file to a single-band float grayscale GeoTIFF.
    This is a fallback because the source data appears to be an RGB visualization
    rather than the raw float grid.
    """
    input_path = project_paths.RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff.tif"
    output_path = project_paths.RAW_DIR / "emag2" / "EMAG2_V3_SeaLevel_DataTiff_Float.tif"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    logger.info(f"Reading RGB source: {input_path}")
    
    try:
        with rasterio.open(input_path) as src:
            # Read all 3 bands
            if src.count < 3:
                logger.error(f"Source file has {src.count} bands, expected at least 3 for RGB.")
                return False
            
            # Read RGB bands
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
            
            # Convert to grayscale using standard luminance formula
            # Y = 0.299 R + 0.587 G + 0.114 B
            logger.info("Converting RGB to Grayscale Float...")
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Update profile for single band float output
            profile = src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'compress': 'DEFLATE',
                'photometric': 'MINISBLACK' # Grayscale
            })
            
            # Write output
            logger.info(f"Writing float output to: {output_path}")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(gray, 1)
                
            logger.info("Conversion complete.")
            return True

    except Exception as e:
        logger.error(f"Failed to convert raster: {e}")
        return False

if __name__ == "__main__":
    convert_rgb_to_float()