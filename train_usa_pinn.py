import os
import time
import logging
import argparse
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pinn_gravity_inversion import DensityUNet, GravityPhysicsLayer, DEFAULT_CONFIG
from loss_functions import StructureGuidedTVLoss, calculate_weights_from_magnetic_gradient

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class USADataset:
    """
    Handles sampling of patches from the massive USA mosaics.
    """
    def __init__(self, gravity_path, magnetic_path, patch_size=512, batches_per_epoch=100):
        self.grav_src = rasterio.open(gravity_path)
        self.mag_src = rasterio.open(magnetic_path)
        self.patch_size = patch_size
        self.batches = batches_per_epoch
        
        self.grav_h, self.grav_w = self.grav_src.shape
        self.mag_h, self.mag_w = self.mag_src.shape
        
        # Determine strict bounds to avoid OOB
        # Using gravity as the master coordinate system
        self.max_y = self.grav_h - patch_size
        self.max_x = self.grav_w - patch_size
        
        # Calculate pixel size (assume consistent)
        self.pixel_size = self.grav_src.transform[0] 
        # Check if geographic; if so, approximate meter conversion at center lat (approx 38N)
        # 1 deg lat approx 111km. 
        # If transform is small (e.g. 0.01), it's degrees.
        if self.pixel_size < 1.0: 
            # Approximate degrees to meters (CONUS center ~39N)
            self.pixel_size_m = self.pixel_size * 111320 * 0.77 # rough avg for long
        else:
            self.pixel_size_m = self.pixel_size

    def sample_batch(self, batch_size=1):
        """
        Returns a batch of (gravity_patch, magnetic_patch) tensors.
        """
        grav_batch = []
        mag_batch = []
        
        for _ in range(batch_size):
            valid = False
            attempts = 0
            while not valid and attempts < 10:
                # Random top-left
                row = random.randint(0, self.max_y)
                col = random.randint(0, self.max_x)
                
                window = Window(col, row, self.patch_size, self.patch_size)
                
                # Read Gravity
                g_data = self.grav_src.read(1, window=window)
                
                # Check validity (simple check: not too many NaNs)
                if np.isnan(g_data).mean() > 0.5:
                    attempts += 1
                    continue
                    
                # Read Magnetic
                # Note: If magnetic transform differs, we need to map coordinates.
                # For this implementation, we assume they are coregistered or we use the same window
                # If they are DIFFERENT dimensions/CRS, this will break.
                # The file listing showed diff dimensions: Grav (1921, 11645), Mag (5356, 10715)
                # CRITCAL: We must map coordinates.
                
                # Get bounds of gravity window
                g_win_bounds = self.grav_src.window_bounds(window)
                # g_win_bounds = (left, bottom, right, top)
                
                # Convert to magnetic window
                m_window = self.mag_src.window(g_win_bounds[0], g_win_bounds[1], g_win_bounds[2], g_win_bounds[3])
                
                # Read Magnetic (resample to match gravity patch size)
                m_data = self.mag_src.read(
                    1, 
                    window=m_window,
                    out_shape=(self.patch_size, self.patch_size), # Force resize to match gravity patch
                    resampling=rasterio.enums.Resampling.bilinear
                )

                # Preprocess
                # Gravity Normalization (Global Physical Normalization)
                # Research-backed: Scale by fixed global constant to preserve physical magnitude
                GLOBAL_MAX_GRAVITY = 100.0 # mGal
                GLOBAL_MAX_MAGNETIC = 1000.0 # nT
                
                # Global scaling
                g_norm = g_data / GLOBAL_MAX_GRAVITY
                m_norm = m_data / GLOBAL_MAX_MAGNETIC
                
                # Replace NaNs with 0 (implies no signal)
                g_norm = np.nan_to_num(g_norm, nan=0.0)
                # m_norm already scaled above, just handle NaNs
                m_norm = np.nan_to_num(m_norm, nan=0.0)
                
                grav_batch.append(g_norm)
                mag_batch.append(m_norm)
                valid = True
                
            if not valid:
                # Fallback to zeros if failing to find valid data
                grav_batch.append(np.zeros((self.patch_size, self.patch_size)))
                mag_batch.append(np.zeros((self.patch_size, self.patch_size)))

        # Stack
        grav_tensor = torch.from_numpy(np.stack(grav_batch)).float().unsqueeze(1) # (B, 1, H, W)
        mag_tensor = torch.from_numpy(np.stack(mag_batch)).float().unsqueeze(1)
        
        return grav_tensor, mag_tensor
        
    def close(self):
        self.grav_src.close()
        self.mag_src.close()

def train_usa(grav_path, mag_path, output_model_path, epochs=100, patch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Dataset
    dataset = USADataset(grav_path, mag_path, patch_size=patch_size)
    logger.info(f"Pixel size estimated: {dataset.pixel_size_m:.1f}m")
    
    # Model
    model = DensityUNet(max_density=800.0).to(device)
    
    # Physics
    # Note: Depth estimate 200m might be shallow for continental scale, but we stick to consistent params
    physics = GravityPhysicsLayer(dataset.pixel_size_m, mean_depth=200.0).to(device)
    
    # Loss
    sg_tv_loss = StructureGuidedTVLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')
    
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Training Steps per Epoch
        steps = dataset.batches
        pbar = tqdm(range(steps), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for _ in pbar:
            grav, mag = dataset.sample_batch(batch_size=4) # Batch size 4 fitting in VRAM?
            grav = grav.to(device)
            mag = mag.to(device)
            
            optimizer.zero_grad()
            
            # Disable AMP for stability
            # with autocast('cuda'): 
            
            # Forward
            pred_density = model(grav)
            
            # Physics Re-forward
            pred_gravity = physics(pred_density)
            
            # Losses
            loss_mse = F.mse_loss(pred_gravity, grav)
            
            # Structure weights
            weights = calculate_weights_from_magnetic_gradient(mag, beta=1.5)
            loss_reg = sg_tv_loss(pred_density, weights) / pred_density.numel()
            
            loss_sparsity = torch.mean(torch.abs(pred_density))
            
            loss = 10.0 * loss_mse + 0.1 * loss_reg + 0.001 * loss_sparsity
            
            # Check for NaN
            if torch.isnan(loss):
                # Zero out gradients and skip step?
                optimizer.zero_grad()
                continue
                
            # scale(loss).backward() -> loss.backward()
            # scaler.step(optimizer) -> optimizer.step()
            loss.backward()
            optimizer.step()
            # scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        logger.info(f"Epoch {epoch+1} Loss: {epoch_loss/steps:.4f}")
        
    # Save Model
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Model saved to {output_model_path}")
    dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", default="data/outputs/usa_supervised/usa_gravity_mosaic.tif")
    parser.add_argument("--magnetic", default="data/outputs/usa_supervised/usa_magnetic_mosaic.tif")
    parser.add_argument("--output", default="usa_pinn_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if os.path.exists(args.gravity) and os.path.exists(args.magnetic):
        train_usa(args.gravity, args.magnetic, args.output, args.epochs)
    else:
        print("Mosaics not found. Skipping training.")
