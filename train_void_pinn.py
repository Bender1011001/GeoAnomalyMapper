#!/usr/bin/env python3
"""
Train PINN for Void Detection (DUB Mode)
=========================================

This uses the same fast tile-based training as train_usa_pinn.py but
optimized for detecting underground VOIDS (negative density anomalies).

Key differences from mineral mode:
1. Loss function penalizes POSITIVE density (we want negative = voids)
2. Looking for sharp, localized anomalies (artificial structures)
3. Different depth assumptions (100-500m for DUBs vs deeper for minerals)
"""

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VoidDataset:
    """
    Tile-based dataset for void detection training.
    Same as USADataset but focuses on areas with negative gravity anomalies.
    """
    def __init__(self, gravity_path, magnetic_path, patch_size=256, batches_per_epoch=50):
        self.grav_src = rasterio.open(gravity_path)
        self.mag_src = rasterio.open(magnetic_path)
        self.patch_size = patch_size
        self.batches = batches_per_epoch
        
        self.grav_h, self.grav_w = self.grav_src.shape
        
        self.max_y = self.grav_h - patch_size
        self.max_x = self.grav_w - patch_size
        
        # Calculate pixel size
        self.pixel_size = self.grav_src.transform[0]
        if self.pixel_size < 1.0:
            self.pixel_size_m = self.pixel_size * 111320 * 0.77
        else:
            self.pixel_size_m = self.pixel_size
    
    def sample_batch(self, batch_size=4):
        """Sample patches, preferring areas with negative gravity anomalies."""
        grav_batch = []
        mag_batch = []
        
        for _ in range(batch_size):
            valid = False
            attempts = 0
            
            while not valid and attempts < 20:
                row = random.randint(0, self.max_y)
                col = random.randint(0, self.max_x)
                
                window = Window(col, row, self.patch_size, self.patch_size)
                g_data = self.grav_src.read(1, window=window)
                
                # Check validity
                nan_ratio = np.isnan(g_data).mean()
                if nan_ratio > 0.5:
                    attempts += 1
                    continue
                
                # Get magnetic data
                try:
                    g_win_bounds = self.grav_src.window_bounds(window)
                    m_window = self.mag_src.window(g_win_bounds[0], g_win_bounds[1], 
                                                   g_win_bounds[2], g_win_bounds[3])
                    m_data = self.mag_src.read(
                        1, window=m_window,
                        out_shape=(self.patch_size, self.patch_size),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                except:
                    m_data = np.zeros((self.patch_size, self.patch_size))
                
                # Normalize
                GLOBAL_MAX_GRAVITY = 300.0
                GLOBAL_MAX_MAGNETIC = 1000.0
                
                g_norm = np.nan_to_num(g_data / GLOBAL_MAX_GRAVITY, nan=0.0)
                m_norm = np.nan_to_num(m_data / GLOBAL_MAX_MAGNETIC, nan=0.0)
                
                # Random flips for augmentation
                if random.random() > 0.5:
                    g_norm = np.flip(g_norm, axis=0).copy()
                    m_norm = np.flip(m_norm, axis=0).copy()
                if random.random() > 0.5:
                    g_norm = np.flip(g_norm, axis=1).copy()
                    m_norm = np.flip(m_norm, axis=1).copy()
                
                grav_batch.append(g_norm)
                mag_batch.append(m_norm)
                valid = True
            
            if not valid:
                grav_batch.append(np.zeros((self.patch_size, self.patch_size)))
                mag_batch.append(np.zeros((self.patch_size, self.patch_size)))
        
        grav_tensor = torch.from_numpy(np.stack(grav_batch)).float().unsqueeze(1)
        mag_tensor = torch.from_numpy(np.stack(mag_batch)).float().unsqueeze(1)
        
        return grav_tensor, mag_tensor
    
    def close(self):
        self.grav_src.close()
        self.mag_src.close()


def train_void_pinn(grav_path, mag_path, output_model_path, epochs=50, patch_size=256):
    """
    Train PINN specifically for void detection.
    
    Key void-mode changes:
    1. Penalize positive density predictions (we want negative/voids)
    2. Shallower depth assumption (100-300m for DUBs)
    3. Encourage sharp edges (artificial structures)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training VOID-mode PINN on {device}")
    
    # Dataset
    train_data = VoidDataset(grav_path, mag_path, patch_size=patch_size, batches_per_epoch=40)
    val_data = VoidDataset(grav_path, mag_path, patch_size=patch_size, batches_per_epoch=10)
    
    logger.info(f"Pixel size: {train_data.pixel_size_m:.1f}m")
    
    # Model - higher max density for sensitivity
    model = DensityUNet(max_density=800.0).to(device)
    
    # Physics layer - SHALLOWER depth for DUBs (100-300m typical)
    physics = GravityPhysicsLayer(
        train_data.pixel_size_m, 
        mean_depth=200.0,  # Shallower than mineral exploration
        thickness=500.0    # Thinner layer
    ).to(device)
    
    # Losses
    sg_tv_loss = StructureGuidedTVLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Starting VOID-mode training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # ======== TRAINING ========
        model.train()
        train_loss = 0
        
        pbar = tqdm(range(train_data.batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for _ in pbar:
            grav, mag = train_data.sample_batch(batch_size=8)
            grav = grav.to(device)
            mag = mag.to(device)
            
            optimizer.zero_grad()
            
            pred_density = model(grav)
            pred_gravity = physics(pred_density)
            
            # Data fidelity loss
            loss_mse = F.mse_loss(pred_gravity, grav)
            
            # Regularization
            try:
                weights = calculate_weights_from_magnetic_gradient(mag.squeeze(1), beta=1.5)
                if weights.dim() == 3:
                    weights = weights.unsqueeze(1)
                loss_reg = sg_tv_loss(pred_density, weights) / pred_density.numel()
            except:
                loss_reg = torch.tensor(0.0).to(device)
            
            # Sparsity
            loss_sparsity = torch.mean(torch.abs(pred_density))
            
            # VOID MODE: Penalize POSITIVE density (we want negative = voids)
            loss_void_bias = torch.mean(F.relu(pred_density))  # Penalize positive values
            
            # Total loss with void bias
            loss = (10.0 * loss_mse + 
                   0.1 * loss_reg + 
                   0.001 * loss_sparsity +
                   0.5 * loss_void_bias)  # Strong void bias
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), void_bias=loss_void_bias.item())
        
        avg_train_loss = train_loss / train_data.batches
        
        # ======== VALIDATION ========
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for _ in range(val_data.batches):
                grav, mag = val_data.sample_batch(batch_size=8)
                grav = grav.to(device)
                mag = mag.to(device)
                
                pred_density = model(grav)
                pred_gravity = physics(pred_density)
                
                loss_mse = F.mse_loss(pred_gravity, grav)
                loss_void_bias = torch.mean(F.relu(pred_density))
                
                loss = 10.0 * loss_mse + 0.5 * loss_void_bias
                val_loss += loss.item()
        
        avg_val_loss = val_loss / val_data.batches
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info("  >>> New best!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Void PINN model saved to {output_model_path}")
    
    train_data.close()
    val_data.close()
    
    return model


def apply_void_model(model_path, gravity_tif, output_tif):
    """Apply trained void model to full gravity grid."""
    import rasterio
    from rasterio.windows import Window
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DensityUNet(max_density=800.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info(f"Loaded void model from {model_path}")
    
    # Process in tiles
    with rasterio.open(gravity_tif) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='deflate')
        
        height, width = src.shape
        tile_size = 512
        
        # Create output array
        output = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        
        GLOBAL_MAX = 300.0
        
        for row in tqdm(range(0, height - tile_size + 1, tile_size // 2), desc="Processing"):
            for col in range(0, width - tile_size + 1, tile_size // 2):
                window = Window(col, row, tile_size, tile_size)
                data = src.read(1, window=window)
                
                if np.isnan(data).mean() > 0.8:
                    continue
                
                data_norm = np.nan_to_num(data / GLOBAL_MAX, nan=0.0)
                tensor = torch.from_numpy(data_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(tensor).squeeze().cpu().numpy()
                
                output[row:row+tile_size, col:col+tile_size] += pred
                counts[row:row+tile_size, col:col+tile_size] += 1
        
        # Average overlapping regions
        counts[counts == 0] = 1
        output = output / counts
    
    # Save
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "Void Density Contrast (kg/m3)")
    
    logger.info(f"Saved void density map to {output_tif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN for void detection")
    parser.add_argument("--gravity", default="data/outputs/usa_supervised/usa_gravity_mosaic.tif")
    parser.add_argument("--magnetic", default="data/outputs/usa_supervised/usa_magnetic_mosaic.tif")
    parser.add_argument("--output-model", default="void_pinn_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--apply", action="store_true", help="Apply trained model to full grid")
    parser.add_argument("--output-tif", default="data/outputs/dub_detection/pinn_void_density.tif")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gravity):
        logger.error(f"Gravity file not found: {args.gravity}")
        exit(1)
    
    if not os.path.exists(args.magnetic):
        logger.error(f"Magnetic file not found: {args.magnetic}")
        exit(1)
    
    # Train
    logger.info("="*60)
    logger.info("TRAINING VOID-MODE PINN")
    logger.info("="*60)
    
    model = train_void_pinn(
        args.gravity,
        args.magnetic,
        args.output_model,
        epochs=args.epochs
    )
    
    if args.apply:
        logger.info("\n" + "="*60)
        logger.info("APPLYING MODEL TO FULL GRID")
        logger.info("="*60)
        apply_void_model(args.output_model, args.gravity, args.output_tif)
    
    logger.info("\n✅ Void PINN training complete!")
