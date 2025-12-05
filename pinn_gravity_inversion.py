import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration (RTX 4060 Ti Optimized)
# ==========================================
CONFIG = {
    "GPU_ID": 0,
    "SEED": 42,
    "LR": 1e-3,
    "EPOCHS": 1000,          # Train longer for better detail
    "BATCH_SIZE": 1,         # Keep 1 for single-tile inversion
    "TILE_SIZE": 1024,       # Crop size to fit 8GB VRAM (if image is larger)
    "DEPTH_ESTIMATE": 200,   # Estimated depth of anomalies (meters)
    "MAX_DENSITY": 500.0,    # Cap density contrast (kg/m^3)
    "PHYSICS_WEIGHT": 0.1,   # Weight of physics loss vs sparsity
    "USE_AMP": True          # Automatic Mixed Precision for 4060 Ti
}

# Set device
device = torch.device(f"cuda:{CONFIG['GPU_ID']}" if torch.cuda.is_available() else "cpu")
print(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# 2. The Physics Layer (Differentiable)
# ==========================================
class GravityPhysicsLayer(nn.Module):
    """
    Simulates gravity from a density map using FFT (Parker's Formula).
    This allows the network to 'check' its answers against the laws of physics.
    """
    def __init__(self, pixel_size_meters, mean_depth):
        super().__init__()
        self.pixel_size = pixel_size_meters
        self.depth = mean_depth
        self.G = 6.674e-11  # Gravitational Constant
        self.SI_to_mGal = 1e5

    def forward(self, density_map):
        """
        Input: Density Map (Batch, 1, H, W)
        Output: Simulated Gravity (Batch, 1, H, W)
        """
        B, C, H, W = density_map.shape
        
        # 1. Create Frequency Grid (Wavenumbers)
        # Optimized for PyTorch FFT standard
        freq_y = torch.fft.fftfreq(H, d=self.pixel_size).to(density_map.device)
        freq_x = torch.fft.fftfreq(W, d=self.pixel_size).to(density_map.device)
        KY, KX = torch.meshgrid(freq_y, freq_x, indexing='ij')
        K = torch.sqrt(KX**2 + KY**2)
        
        # Avoid division by zero at DC component (K=0)
        K[0, 0] = 1e-10

        # 2. Earth Filter (Bouguer Slab + Upward Continuation)
        # Formula: G(k) = 2 * pi * G * exp(-|k| * z) * Density(k)
        earth_filter = 2 * np.pi * self.G * torch.exp(-K * self.depth)
        
        # 3. FFT Convolution
        d_fft = torch.fft.fft2(density_map)
        g_fft = d_fft * earth_filter.unsqueeze(0).unsqueeze(0)
        gravity_pred = torch.real(torch.fft.ifft2(g_fft))
        
        # Convert SI (m/s^2) to mGal
        return gravity_pred * self.SI_to_mGal

# ==========================================
# 3. The U-Net (The "Brain")
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DensityUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight U-Net to fit 8GB VRAM
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(128 + 64, 64)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(64 + 32, 32)
        
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        # Handle slight padding mismatches if dimensions are odd
        if x.shape != x3.shape: x = F.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x2.shape: x = F.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x1.shape: x = F.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        # Output scaled to range [-MAX_DENSITY, MAX_DENSITY]
        return torch.tanh(self.outc(x)) * CONFIG['MAX_DENSITY']

# ==========================================
# 4. Training Loop (Mixed Precision)
# ==========================================
def invert_gravity(tif_path, output_path, lithology_path=None):
    # --- Load Data ---
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        profile = src.profile
        height, width = data.shape
        # Calculate pixel size in meters (approx for lat/lon)
        # Using the center latitude for cosine correction
        center_lat = (src.bounds.top + src.bounds.bottom) / 2
        deg_to_meter = 111320 * np.cos(np.deg2rad(center_lat))
        pixel_size = src.transform[0] * deg_to_meter
        print(f"Loaded {tif_path}: {data.shape}, Pixel Size: {pixel_size:.1f}m")

    # --- Load Lithology Prior ---
    prior_contrast = None
    if lithology_path and os.path.exists(lithology_path):
        try:
            with rasterio.open(lithology_path) as src:
                # Read and resize to match gravity data dimensions
                prior_data = src.read(
                    1,
                    out_shape=(height, width),
                    resampling=Resampling.nearest
                )
                # Convert absolute density to contrast (relative to standard crust 2670 kg/m3)
                # Physics layer expects density contrast for gravity anomaly
                prior_contrast_np = prior_data - 2670.0
                prior_contrast = torch.from_numpy(prior_contrast_np).float().unsqueeze(0).unsqueeze(0).to(device)
                print(f"Loaded Lithology Prior from {lithology_path} (converted to contrast)")
        except Exception as e:
            print(f"Warning: Failed to load lithology prior: {e}")

    # --- Preprocessing ---
    # Normalize input gravity for stability (Z-score)
    grav_mean = np.nanmean(data)
    grav_std = np.nanstd(data)
    if grav_std == 0: grav_std = 1
    data_norm = (data - grav_mean) / grav_std
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    # Convert to Tensor
    inp_tensor = torch.from_numpy(data_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    target_gravity = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Handle NaNs in target (mask them out of loss)
    target_mask = ~torch.isnan(target_gravity)
    target_gravity = torch.nan_to_num(target_gravity, nan=0.0)

    # --- Initialize Model ---
    model = DensityUNet().to(device)
    physics = GravityPhysicsLayer(pixel_size, CONFIG['DEPTH_ESTIMATE']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    # AMP Scaler for 4060 Ti
    scaler = GradScaler(enabled=CONFIG['USE_AMP'])
    
    # --- Training ---
    print("Starting Inversion...")
    loop = tqdm(range(CONFIG['EPOCHS']))
    
    model.train()
    for epoch in loop:
        optimizer.zero_grad()
        
        with autocast(enabled=CONFIG['USE_AMP']):
            # 1. Forward Pass (Predict Residual Density)
            # We pass the gravity map itself as input features
            pred_residual = model(inp_tensor)
            
            # 2. Physics Pass (Simulate Gravity)
            if prior_contrast is not None:
                # Combine known lithology contrast with predicted residual
                total_contrast = prior_contrast + pred_residual
                sim_gravity = physics(total_contrast)
                
                # Regularization: We want the residual to be sparse (trust the prior)
                # We still penalize positive mass in the residual if we are looking for voids
                loss_sparsity = torch.mean(torch.abs(pred_residual))
                loss_void_bias = torch.mean(F.relu(pred_residual))
            else:
                # No prior: Model predicts total contrast directly
                sim_gravity = physics(pred_residual)
                loss_sparsity = torch.mean(torch.abs(pred_residual))
                loss_void_bias = torch.mean(F.relu(pred_residual))
            
            # 3. Loss Calculation
            # A. Data Fidelity: Does simulated gravity match observed gravity?
            loss_data = F.mse_loss(sim_gravity[target_mask], target_gravity[target_mask])
            
            loss = loss_data + (0.1 * loss_sparsity) + (0.1 * loss_void_bias)

        # 4. Backward (Scaled)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item(), data_loss=loss_data.item())

    # --- Save Result ---
    model.eval()
    with torch.no_grad():
        final_density = model(inp_tensor).squeeze().cpu().numpy()

    profile.update(dtype=rasterio.float32, count=1, compress='deflate')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_density.astype(np.float32), 1)
    
    print(f"Inversion Complete. Saved to {output_path}")

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # Example Usage
    # Ensure you point this to your actual file
    INPUT_FILE = "data/processed/gravity_residual.tif"
    OUTPUT_FILE = "data/outputs/density_contrast_map.tif"
    
    LITHOLOGY_FILE = "data/processed/lithology_density.tif"
    
    if os.path.exists(INPUT_FILE):
        invert_gravity(INPUT_FILE, OUTPUT_FILE, LITHOLOGY_FILE)
    else:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please process your gravity data first.")