import os
import argparse
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import torch
from tqdm import tqdm

from pinn_gravity_inversion import DensityUNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_region(model_path, grav_path, mag_path, litho_path, output_path, bounds, tile_size=2048, overlap=256):
    """
    Runs inference on a specific geographic region using the Global Model.
    bounds: (west, south, east, north)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Predicting on {device} for bounds: {bounds}")
    
    # Load Model (Residual)
    # Model predicts Delta Density (small range)
    model = DensityUNet(max_density=1.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with rasterio.open(grav_path) as grav_src, \
         rasterio.open(mag_path) as mag_src, \
         rasterio.open(litho_path) as litho_src:
        
        # Calculate pixel window for the region
        west, south, east, north = bounds
        
        # Get indices (row, col)
        # rasterio.index(x, y) -> (lon, lat)
        row_min, col_min = grav_src.index(west, north) 
        row_max, col_max = grav_src.index(east, south)
        
        # Handle wraparound or edge cases if needed (assuming simple consistent grid here)
        # Global grid might be 0-360 or -180/180.
        # Ensure indices are valid
        r_start = max(0, row_min)
        c_start = max(0, col_min)
        r_end = min(grav_src.height, row_max)
        c_end = min(grav_src.width, col_max)
        
        region_height = r_end - r_start
        region_width = c_end - c_start
        
        logger.info(f"Region Dimensions: {region_width}x{region_height}")
        
        # Create Output Profile
        # Transform needs to be adjusted to the new window
        region_window = Window(c_start, r_start, region_width, region_height)
        region_transform = grav_src.window_transform(region_window)
        
        profile = grav_src.profile.copy()
        profile.update(
            height=region_height,
            width=region_width,
            transform=region_transform,
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            
            step = tile_size
            total_tiles = (region_height // step + 1) * (region_width // step + 1)
            pbar = tqdm(total=total_tiles, desc="Predicting Tiles")
            
            # Iterate relative to the Region Window
            for row in range(0, region_height, step):
                for col in range(0, region_width, step):
                    
                    # 1. Define 'Logical' Window (relative to region)
                    # We want to fill [row : row+step, col : col+step] in output
                    
                    # 2. Define 'Read' Window (with overlap, relative to GLOBAL source)
                    # Coordinates in global source = (r_start + row - overlap)
                    
                    global_r = r_start + row
                    global_c = c_start + col
                    
                    # Offsets for overlap
                    off_r = max(r_start, global_r - overlap)
                    off_c = max(c_start, global_c - overlap)
                    
                    # Lengths (clipped to region bounds to avoid reading outside input if close to edge)
                    # Actually, we should clip to Source Bounds
                    max_r = min(r_end, global_r + step + overlap)
                    max_c = min(c_end, global_c + step + overlap)
                    
                    read_h = max_r - off_r
                    read_w = max_c - off_c
                    
                    # Create Window for reading
                    read_window = Window(off_c, off_r, read_w, read_h)
                    
                    # READ DATA
                    g_data = grav_src.read(1, window=read_window)
                    
                    # Skip if empty
                    if g_data.size == 0: continue
                    
                    # Read Mag & Litho (resampled to match gravity window)
                    # Get bounds of this read window to query others
                    win_bounds = grav_src.window_bounds(read_window)
                    
                    m_win = mag_src.window(*win_bounds)
                    m_data = mag_src.read(1, window=m_win, out_shape=(read_h, read_w), resampling=rasterio.enums.Resampling.bilinear)
                    
                    l_win = litho_src.window(*win_bounds)
                    l_data = litho_src.read(1, window=l_win, out_shape=(read_h, read_w), resampling=rasterio.enums.Resampling.nearest)
                    
                    # PREPARE INPUT
                    g_norm = np.nan_to_num(g_data / 200.0, nan=0.0)
                    # m_norm = np.nan_to_num(m_data / 1000.0, nan=0.0) 
                    # Note: Model trained with m_norm? Yes. 
                    # But wait, does the model take Mag? Yes.
                    # Wait, DenseUNet input channels?
                    # In train_global_pinn.py: `g_t, m_t, l_t` returned, but `model(grav)` logic used?
                    # Let's check train_global_pinn.py...
                    # LINE 183: `pred_delta = model(grav)` in loop.
                    # The UNet ONLY takes Gravity as input! Magnetic is used for LOSS weighting.
                    # Lithology is added to output.
                    # SO: Inference only needs Gravity input, but needs Lithology for sum.
                    
                    inp = torch.from_numpy(g_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_delta_t = model(inp)
                        
                    pred_delta = pred_delta_t.squeeze().cpu().numpy()
                    
                    # COMBINE: Total = Lithology + Delta
                    # Lithology prior might have NaNs -> default 2.67
                    l_clean = np.nan_to_num(l_data, nan=2.67)
                    
                    total_density = l_clean + pred_delta
                    
                    # CROP MARGINS
                    # We read from [off_r, off_c]
                    # We want to write to [global_r, global_c]
                    
                    # Delta within the read chunk
                    valid_r_start = global_r - off_r
                    valid_c_start = global_c - off_c
                    
                    valid_h = min(step, region_height - row)
                    valid_w = min(step, region_width - col)
                    
                    crop = total_density[valid_r_start : valid_r_start + valid_h, 
                                         valid_c_start : valid_c_start + valid_w]
                    
                    # WRITE
                    dst.write(crop.astype(np.float32), 1, window=Window(col, row, valid_w, valid_h))
                    pbar.update(1)
            
            pbar.close()
            
    logger.info(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="global_pinn_model.pth")
    # Default Bounds: Continental USA
    parser.add_argument("--bounds", nargs=4, type=float, default=[-125.0, 24.0, -66.0, 50.0], help="West South East North")
    parser.add_argument("--output", default="D:/Geo_data/usa_prediction_200m.tif")
    
    args = parser.parse_args()
    
    predict_region(
        args.model, 
        r"D:\Geo_data\global_gravity_200m.tif",
        r"D:\Geo_data\global_magnetics_2arcmin.tif",
        r"D:\Geo_data\global_lithology_density.tif",
        args.output,
        args.bounds
    )
