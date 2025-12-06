#!/usr/bin/env python3
"""
Phase 8: Spatial Deep Learning for Anomaly Detection (Convolutional Autoencoder).

This script:
1. Loads the 3-band geospatial map (Gravity, Belief, InSAR) into RAM.
2. Slices the map into 64x64 pixel "chips".
3. Trains a Convolutional Autoencoder (CAE) to reconstruct "Normal" geology.
4. Measures the "Reconstruction Error" (MSE) for every chip.
5. High error = The AI doesn't recognize this pattern = ANOMALY.
6. Re-assembles a new "Spatial Anomaly Map".

Optimized for: NVIDIA RTX 4060 Ti (or any CUDA GPU).
"""

import os
import argparse
import logging
from pathlib import Path
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from project_paths import OUTPUTS_DIR, PROCESSED_DIR, DATA_DIR
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Define the Neural Network ---
class SpatialAutoencoder(nn.Module):
    def __init__(self):
        super(SpatialAutoencoder, self).__init__()
        
        # Encoder (Compresses the map chip into a tiny 'code')
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU()
        )
        
        # Decoder (Tries to rebuild the map from the code)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # 32 -> 64
            nn.Sigmoid() # Forces output between 0.0 and 1.0 (matching our normalized data)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 2. Data Loader ---
class GeoChipDataset(Dataset):
    def __init__(self, data_array, chip_size=64, stride=32):
        """
        Slices the massive numpy array into overlapping chips.
        stride=32 means 50% overlap, which improves coverage.
        """
        self.chips = []
        self.coords = []
        c, h, w = data_array.shape
        
        # Slide a window across the map
        for y in range(0, h - chip_size, stride):
            for x in range(0, w - chip_size, stride):
                # Extract chip
                chip = data_array[:, y:y+chip_size, x:x+chip_size]
                
                # Only keep chips with valid data (no NaNs)
                if not np.isnan(chip).any():
                    self.chips.append(chip)
                    self.coords.append((y, x))
                    
        self.chips = np.array(self.chips)
        logger.info(f"Created {len(self.chips)} training chips from the map.")

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        # Convert to Float tensor
        return torch.from_numpy(self.chips[idx]).float()

def prepare_data(config):
    """Loads all feature rasters into a single normalized 3-band array."""
    logger.info("Loading rasters into memory...")
    
    inputs = config['data']['inputs']
    
    # Construct paths relative to DATA_DIR
    paths = [
        str(DATA_DIR / inputs['gravity']),
        str(DATA_DIR / inputs['belief']),
        str(DATA_DIR / inputs['insar'])
    ]
    
    # Check fallback paths if primary missing (specifically for gravity as per original logic)
    if not Path(paths[0]).exists():
        # Try fallback location if configured path doesn't exist
        fallback = OUTPUTS_DIR / "gravity" / "xgm2019e_gravdist_box_mgal.tif"
        if fallback.exists():
            paths[0] = str(fallback)
            logger.info(f"Using fallback gravity file: {paths[0]}")
        else:
            logger.warning(f"Primary gravity file missing and fallback not found: {paths[0]}")
    
    loaded_layers = []
    ref_profile = None

    # Load Reference
    with rasterio.open(paths[0]) as src:
        ref_profile = src.profile
        ref_shape = (src.height, src.width)
    
    for p in paths:
        with rasterio.open(p) as src:
            # For simplicity in this script, we assume they are pre-aligned (Phase 5 aligned them).
            # We read the first band.
            # We resize on the fly if needed (brute force alignment for the script).
            arr = src.read(1, out_shape=ref_shape)
            
            # Normalize to 0-1
            arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
            arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-6)
            loaded_layers.append(arr_norm)
            
    # Stack into (Channels, Height, Width) -> (3, H, W)
    stacked = np.stack(loaded_layers)
    return stacked, ref_profile

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Spatial Deep Learning for Anomaly Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Extract config parameters
    model_config = config['model']
    chip_size = model_config['chip_size']
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']
    learning_rate = model_config['learning_rate']
    stride = model_config['stride']
    
    # Setup device
    if config['training']['device'] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['training']['device'])
        
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"
    logger.info(f"Initializing Phase 8 on {device} ({device_name})")
    
    # 1. Load Data
    try:
        data_stack, profile = prepare_data(config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Please ensure Phase 1-4 outputs exist.")
        return

    # 2. Chip the Map
    dataset = GeoChipDataset(data_stack, chip_size=chip_size, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # 3. Initialize Model
    model = SpatialAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 4. Train
    logger.info(f"Training Autoencoder for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Time: {time.time()-start_time:.1f}s")

    # 5. Inference (Generate Anomaly Map)
    logger.info("Generating Spatial Anomaly Map...")
    model.eval()
    
    # We create a blank map for the errors
    # We will stitch the chips back together.
    # Since we have overlap (stride 32), we average the errors.
    
    c, h, w = data_stack.shape
    error_map = np.zeros((h, w), dtype=np.float32)
    counts_map = np.zeros((h, w), dtype=np.float32)
    
    # Re-iterate (no shuffle)
    # We use the dataset's internal list to preserve coordinates
    # Doing this manually to map back to (y, x)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.chips), batch_size), desc="Inference"):
            # Get batch of chips
            batch_chips = dataset.chips[i : i+batch_size]
            batch_coords = dataset.coords[i : i+batch_size]
            
            # Convert to tensor
            input_tensor = torch.from_numpy(batch_chips).float().to(device)
            
            # Predict
            reconstruction = model(input_tensor)
            
            # Calculate Error (MSE per pixel)
            # |Input - Output|^2
            # Sum errors across channels (Gravity+Belief+InSAR)
            diff = (input_tensor - reconstruction) ** 2
            error_batch = torch.sum(diff, dim=1).cpu().numpy() # Shape: (Batch, 64, 64)
            
            # Stitch back
            for j, (y, x) in enumerate(batch_coords):
                err_chip = error_batch[j]
                error_map[y:y+chip_size, x:x+chip_size] += err_chip
                counts_map[y:y+chip_size, x:x+chip_size] += 1

    # Average the overlaps
    # Avoid divide by zero
    valid_mask = counts_map > 0
    error_map[valid_mask] /= counts_map[valid_mask]
    
    # Normalize result 0-1
    e_min, e_max = error_map.min(), error_map.max()
    final_map = (error_map - e_min) / (e_max - e_min + 1e-6)

    # 6. Save
    out_filename = config['output']['anomaly_map']
    out_path = OUTPUTS_DIR / out_filename
    
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(final_map.astype(rasterio.float32), 1)
        
    logger.info(f"Spatial analysis complete. Saved to: {out_path}")
    logger.info("This map highlights areas where the GEOMETRY of the geology is unnatural.")

if __name__ == "__main__":
    main()