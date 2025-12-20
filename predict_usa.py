import os
import argparse
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm

from pinn_gravity_inversion import DensityUNet, GravityPhysicsLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_usa(model_path, gravity_path, output_path, tile_size=2048, overlap=256, edge_crop=64):
    """
    Runs inference on the massive mosaic using a sliding window approach.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = DensityUNet(max_density=800.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with rasterio.open(gravity_path) as src:
        h, w = src.shape
        profile = src.profile.copy()
        
        profile.update(dtype=rasterio.float32, count=1, compress='deflate', tiled=False)
        
        # Open output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            
            # Sliding window loops
            # We step by tile_size, but read tile_size + overlap check
            # Actually, standard Overlap-Tile strategy:
            # Read (Tile + Margin), Predict, Crop Margin, Write Tile.
            
            step = tile_size
            
            # Initialize weight accumulator for blending
            weight_accumulator = np.zeros((h, w), dtype=np.float32)
            
            pbar = tqdm(total=(h//step + 1) * (w//step + 1), desc="Predicting Tiles")
            
            for row in range(0, h, step):
                for col in range(0, w, step):
                    
                    # Define Read Window (with padding/overlap)
                    # We want to predict for [row : row+step, col : col+step]
                    # But input needs context.
                    
                    # Pad read window
                    r_start = max(0, row - overlap)
                    c_start = max(0, col - overlap)
                    r_end = min(h, row + step + overlap)
                    c_end = min(w, col + step + overlap)
                    
                    read_h = r_end - r_start
                    read_w = c_end - c_start
                    
                    window = Window(c_start, r_start, read_w, read_h)
                    
                    data = src.read(1, window=window)
                    
                    # Handle NaNs
                    # Normalization (MATCHING UPDATED TRAINING SCRIPT)
                    # Uses global scaling: val / 300.0 (expanded from 100.0)
                    norm_data = data / 300.0
                    norm_data = np.nan_to_num(norm_data, nan=0.0)
                    
                    # To Tensor
                    inp = torch.from_numpy(norm_data).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out = model(inp)
                        
                    res = out.squeeze().cpu().numpy()
                    
                    # Crop Output to match the valid Write Window
                    # Calculate offsets relative to the read window
                    valid_r_start = row - r_start
                    valid_c_start = col - c_start
                    valid_r_end = valid_r_start + min(step, h - row)
                    valid_c_end = valid_c_start + min(step, w - col)
                    
                    # Crop
                    crop = res[valid_r_start:valid_r_end, valid_c_start:valid_c_end]

                    # Simple direct write (blending was causing issues)
                    write_window = Window(col, row, crop.shape[1], crop.shape[0])
                    dst.write(crop.astype(np.float32), 1, window=write_window)
                    
                    pbar.update(1)
            
    logger.info(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="usa_pinn_model.pth")
    parser.add_argument("--gravity", default="data/outputs/usa_supervised/usa_gravity_mosaic.tif")
    parser.add_argument("--output", default="data/outputs/usa_density_model.tif")
    args = parser.parse_args()
    
    if os.path.exists(args.model) and os.path.exists(args.gravity):
        predict_usa(args.model, args.gravity, args.output)
    else:
        print("Inputs not found.")
