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
from generate_synthetic_dubs import generate_synthetic_batch

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

                # Gravity Normalization (Global Physical Normalization)
                # Research-backed: Scale by fixed global constant to preserve physical magnitude
                # FIXED: Expanded range to handle full Bouguer gravity spectrum
                GLOBAL_MAX_GRAVITY = 300.0 # mGal (was 100, too narrow)
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
                
                # Data augmentation: randomly flip
                if np.random.random() > 0.5:
                    g_norm = np.flip(g_norm, axis=0).copy()
                    m_norm = np.flip(m_norm, axis=0).copy()
                if np.random.random() > 0.5:
                    g_norm = np.flip(g_norm, axis=1).copy()
                    m_norm = np.flip(m_norm, axis=1).copy()
                
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

def train_usa(grav_path, mag_path, output_model_path, epochs=100, patch_size=256, mode='mineral'):
    """
    Train PINN for gravity or magnetic inversion.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logger.info(f"Training on {device} | Mode: {mode.upper()}")
    
    # Increase batch size for GPU efficiency
    batch_size = 5 if torch.cuda.is_available() else 4
    
    # Datasets
    train_dataset = USADataset(grav_path, mag_path, patch_size=patch_size, batches_per_epoch=80)
    val_dataset = USADataset(grav_path, mag_path, patch_size=patch_size, batches_per_epoch=20)
    
    # Model - higher max value for susceptibility (SI can be high for iron)
    max_val = 1.0 if mode == 'magnetic' else 800.0
    model = DensityUNet(max_val=max_val).to(device)
    
    # Physics
    if mode == 'magnetic':
        from pinn_gravity_inversion import MagneticPhysicsLayer
        physics = MagneticPhysicsLayer(train_dataset.pixel_size_m, mean_depth=200.0).to(device)
    else:
        physics = GravityPhysicsLayer(train_dataset.pixel_size_m, mean_depth=200.0).to(device)
    
    # Loss
    sg_tv_loss = StructureGuidedTVLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    logger.info("Starting Training with VALIDATION-BASED early stopping...")
    
    for epoch in range(epochs):
        # ============ TRAINING PHASE ============
        model.train()
        train_loss = 0.0  # Explicit float
        batch_count = 0
        
        pbar = tqdm(range(train_dataset.batches), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ascii=True)
        
        scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        
        for _ in pbar:
            if mode == 'synthetic':
                # Generate synthetic DUB density maps
                gt_density, _ = generate_synthetic_batch(batch_size=batch_size, patch_size=patch_size)
                gt_density = gt_density.to(device)
                # Forward project to get 'observed' gravity
                target = physics(gt_density)
                grav = target
                mag = target
            else:
                grav, mag = train_dataset.sample_batch(batch_size=batch_size)
                grav = grav.to(device)
                mag = mag.to(device)
                target = mag if mode == 'magnetic' else grav
            
            optimizer.zero_grad()
            
            # AMP Forward Pass
            with autocast('cuda', enabled=torch.cuda.is_available()):
                pred_val = model(grav)
                
                # Physics FFT must use FP32 to avoid overflow
                with torch.amp.autocast('cuda', enabled=False):
                    pred_obs = physics(pred_val.float())
                
                loss_mse = F.mse_loss(pred_obs, target)
                weights = calculate_weights_from_magnetic_gradient(mag, beta=1.5)
                loss_reg = sg_tv_loss(pred_val, weights) / pred_val.numel()
                loss_sparsity = torch.mean(torch.abs(pred_val))
                
                if mode == 'void':
                    loss_bias = torch.mean(F.relu(pred_val))
                elif mode == 'magnetic' or mode == 'synthetic':
                    loss_bias = torch.tensor(0.0, device=device)
                else:
                    loss_bias = torch.mean(F.relu(-pred_val))
                
                loss = 10.0 * loss_mse + 0.1 * loss_reg + 0.001 * loss_sparsity + 0.5 * loss_bias
            
            if torch.isnan(loss):
                continue
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            loss_val = loss.item()
            train_loss += loss_val
            batch_count += 1
            pbar.set_postfix(loss=loss_val)
        
        avg_train_loss = train_loss / max(batch_count, 1)
        
        # ============ VALIDATION PHASE ============
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for _ in range(val_dataset.batches):
                grav, mag = val_dataset.sample_batch(batch_size=8)
                grav = grav.to(device)
                mag = mag.to(device)
                
                pred_density = model(grav)
                pred_gravity = physics(pred_density)
                
                loss_mse = F.mse_loss(pred_gravity, grav)
                weights = calculate_weights_from_magnetic_gradient(mag, beta=1.5)
                loss_reg = sg_tv_loss(pred_density, weights) / pred_density.numel()
                loss_sparsity = torch.mean(torch.abs(pred_density))
                
                loss = 10.0 * loss_mse + 0.1 * loss_reg + 0.001 * loss_sparsity
                val_loss += loss.item()
        
        avg_val_loss = val_loss / val_dataset.batches
        
        logger.info(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # ============ EARLY STOPPING CHECK ============
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  >>> New best validation loss!")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                logger.info(f"EARLY STOPPING at epoch {epoch+1} (val loss started to rise)")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model from validation checkpoint")
    
    # Save Model
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Model saved to {output_model_path}")
    
    train_dataset.close()
    val_dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", default="data/outputs/usa_supervised/usa_gravity_mosaic.tif")
    parser.add_argument("--magnetic", default="data/outputs/usa_supervised/usa_magnetic_mosaic.tif")
    parser.add_argument("--output", default="usa_pinn_model.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mode", choices=['mineral', 'void', 'magnetic', 'synthetic'], default='mineral',
                       help="'mineral' for gravity/ore, 'void' for DUBs/caves, 'magnetic' for metal/susceptibility, 'synthetic' for pre-training")
    args = parser.parse_args()
    
    # Set output filename based on mode if not specified
    if args.output == "usa_pinn_model.pth":
        if args.mode == 'void': args.output = "usa_pinn_model_void.pth"
        elif args.mode == 'magnetic': args.output = "usa_pinn_model_magnetic.pth"
    
    if os.path.exists(args.gravity) and os.path.exists(args.magnetic):
        train_usa(args.gravity, args.magnetic, args.output, args.epochs, mode=args.mode)
    else:
        print("Mosaics not found. Skipping training.")
