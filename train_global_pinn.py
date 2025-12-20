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

class GlobalDataset:
    """
    Handles sampling of patches from massive global mosaics (Gravity, Magnetic, Lithology).
    """
    def __init__(self, gravity_path, magnetic_path, lithology_path=None, patch_size=512, batches_per_epoch=100):
        self.grav_src = rasterio.open(gravity_path)
        self.mag_src = rasterio.open(magnetic_path)
        
        self.litho_src = None
        if lithology_path and os.path.exists(lithology_path):
            self.litho_src = rasterio.open(lithology_path)
            logger.info("Lithology prior enabled.")
        else:
            logger.info("Lithology prior NOT found or disabled.")

        self.patch_size = patch_size
        self.batches = batches_per_epoch
        
        self.grav_h, self.grav_w = self.grav_src.shape
        
        # Determine strict bounds
        self.max_y = self.grav_h - patch_size
        self.max_x = self.grav_w - patch_size
        
        # Calculate pixel size (GGMplus is 7.2 arcsec usually ~220m)
        self.pixel_size = self.grav_src.transform[0] 
        if self.pixel_size < 1.0: 
            self.pixel_size_m = self.pixel_size * 111320 * 0.7 # cos(45) approx
        else:
            self.pixel_size_m = self.pixel_size

    def sample_batch(self, batch_size=1):
        """
        REGIONAL SAMPLING: Pick a large block and sample batch from it.
        Significantly faster on HDDs.
        """
        grav_batch = []
        mag_batch = []
        litho_batch = []
        
        # Pick a regional 'mother' window (e.g. 4x patch size) to sample from
        # This keeps disk head seeks close together
        mother_size = self.patch_size * 4
        valid_mother = False
        while not valid_mother:
            m_row = random.randint(0, self.grav_h - mother_size)
            m_col = random.randint(0, self.grav_w - mother_size)
            m_window = Window(m_col, m_row, mother_size, mother_size)
            
            # Fast check on mother block (low res read)
            check_data = self.grav_src.read(1, window=m_window, out_shape=(16, 16))
            if (check_data < -9000).mean() < 0.2 and not np.isnan(check_data).all():
                valid_mother = True

        for _ in range(batch_size):
            valid = False
            attempts = 0
            while not valid and attempts < 15:
                # Sample within mother block
                row_off = random.randint(0, mother_size - self.patch_size)
                col_off = random.randint(0, mother_size - self.patch_size)
                
                window = Window(m_col + col_off, m_row + row_off, self.patch_size, self.patch_size)
                
                # 1. Read Gravity
                g_data = self.grav_src.read(1, window=window)
                if (g_data < -9000).mean() > 0.1 or np.isnan(g_data).mean() > 0.1:
                    attempts += 1
                    continue
                    
                # 2. Read Magnetic
                g_bounds = self.grav_src.window_bounds(window)
                mag_win = self.mag_src.window(*g_bounds)
                m_data = self.mag_src.read(1, window=mag_win, out_shape=(self.patch_size, self.patch_size))
                
                # 3. Read Lithology
                if self.litho_src:
                    l_win = self.litho_src.window(*g_bounds)
                    l_data = self.litho_src.read(1, window=l_win, out_shape=(self.patch_size, self.patch_size))
                    l_val = np.nan_to_num(l_data, nan=2.67)
                else:
                    l_val = np.full_like(g_data, 2.67)

                # Preprocessing
                g_norm = np.nan_to_num(g_data / 200.0, nan=0.0)
                m_norm = np.nan_to_num(m_data / 1000.0, nan=0.0)
                
                grav_batch.append(g_norm)
                mag_batch.append(m_norm)
                litho_batch.append(l_val)
                valid = True
                
                # Augmentation
                if np.random.random() > 0.5:
                    grav_batch[-1] = np.flip(grav_batch[-1], axis=0).copy()
                    mag_batch[-1] = np.flip(mag_batch[-1], axis=0).copy()
                    litho_batch[-1] = np.flip(litho_batch[-1], axis=0).copy()
            
            if not valid:
                patch = np.zeros((self.patch_size, self.patch_size))
                grav_batch.append(patch)
                mag_batch.append(patch)
                litho_batch.append(patch + 2.67)

        return (torch.from_numpy(np.stack(grav_batch)).float().unsqueeze(1),
                torch.from_numpy(np.stack(mag_batch)).float().unsqueeze(1),
                torch.from_numpy(np.stack(litho_batch)).float().unsqueeze(1))

    def close(self):
        self.grav_src.close()
        self.mag_src.close()
        if self.litho_src: self.litho_src.close()

def train_global(grav_path, mag_path, litho_path, output_model_path, epochs=100, patch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device} - GLOBAL MODE (Residual Learning)")
    
    train_dataset = GlobalDataset(grav_path, mag_path, litho_path, patch_size=patch_size, batches_per_epoch=200)
    val_dataset = GlobalDataset(grav_path, mag_path, litho_path, patch_size=patch_size, batches_per_epoch=50)
    logger.info(f"Pixel size: {train_dataset.pixel_size_m:.1f}m")
    
    # Model: Predicts density ADJUSTMENT (delta)
    # Using small max_density constraint for the delta
    model = DensityUNet(max_density=1.0).to(device) 
    
    # Physics
    physics = GravityPhysicsLayer(train_dataset.pixel_size_m, mean_depth=200.0).to(device)
    
    # Loss
    sg_tv_loss = StructureGuidedTVLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        # ============ TRAINING PHASE ============
        model.train()
        train_loss = 0
        pbar = tqdm(range(train_dataset.batches), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ascii=True)
        
        for _ in pbar:
            grav, mag, litho = train_dataset.sample_batch(batch_size=8)
            grav = grav.to(device)
            mag = mag.to(device)
            litho = litho.to(device)
            
            optimizer.zero_grad()
            
            # Predict residual density anomaly
            pred_delta = model(grav) 
            
            # Total Density = Lithology Prior + Delta
            total_density = litho + pred_delta
            
            # Gravity Physics: Forward model on density CONTRAST relative to 2.67
            # Litho is absolute density (e.g. 2.8). Background is 2.67.
            # Contrast = (Litho + Delta) - 2.67
            input_contrast = (total_density - 2.67)
            pred_gravity = physics(input_contrast)
            
            loss_mse = F.mse_loss(pred_gravity, grav)
            
            # Regularize: Structure Guide the DELTA
            weights = calculate_weights_from_magnetic_gradient(mag, beta=1.5)
            loss_reg = sg_tv_loss(pred_delta, weights) / pred_delta.numel()
            
            # Sparsity on delta (we want to trust lithology mostly)
            loss_sparsity = torch.mean(torch.abs(pred_delta))
            
            loss = 10.0 * loss_mse + 0.1 * loss_reg + 0.01 * loss_sparsity
            
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / train_dataset.batches
        
        # ============ VALIDATION PHASE ============
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_dataset.batches):
                grav, mag, litho = val_dataset.sample_batch(batch_size=8)
                grav = grav.to(device)
                mag = mag.to(device)
                litho = litho.to(device)
                
                pred_delta = model(grav)
                pred_gravity = physics((litho + pred_delta) - 2.67)
                
                loss_mse = F.mse_loss(pred_gravity, grav)
                weights = calculate_weights_from_magnetic_gradient(mag, beta=1.5)
                loss_reg = sg_tv_loss(pred_delta, weights) / pred_delta.numel()
                loss_sparsity = torch.mean(torch.abs(pred_delta))
                
                loss = 10.0 * loss_mse + 0.1 * loss_reg + 0.01 * loss_sparsity
                val_loss += loss.item()
                
        avg_val_loss = val_loss / val_dataset.batches
        logger.info(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("EARLY STOPPING")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Model saved to {output_model_path}")
    
    train_dataset.close()
    val_dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", default=r"D:\Geo_data\global_gravity_200m.tif")
    parser.add_argument("--magnetic", default=r"D:\Geo_data\global_magnetics_2arcmin.tif")
    parser.add_argument("--lithology", default=r"D:\Geo_data\global_lithology_density.tif")
    parser.add_argument("--output", default="global_pinn_model.pth")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    if os.path.exists(args.gravity):
        train_global(args.gravity, args.magnetic, args.lithology, args.output, args.epochs)
    else:
        print("Mosaics not found. Skipping training.")
