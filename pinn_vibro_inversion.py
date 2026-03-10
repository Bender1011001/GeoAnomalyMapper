#!/usr/bin/env python3
"""
PINN Vibro-Elastic Inversion: Helmholtz-Based 3D Subsurface Imaging
====================================================================

This module implements the Physics-Informed Neural Network (PINN) for
inverting 2D surface vibration maps into 3D subsurface structure using
the Helmholtz wave equation.

The Biondi Inversion Principle
------------------------------
Given a 2D map of surface vibration amplitudes (from sar_vibrometry.py),
this PINN learns the 3D underground density/wave-speed distribution that
would produce exactly those surface vibrations when excited by ambient
seismic noise.

Physics: The Helmholtz Equation
-------------------------------
The steady-state wave equation (Helmholtz) is:

    ∇²U + (ω²/c²)·U = 0

where:
    U(x,y,z) = vibration displacement amplitude
    ω = angular frequency of excitation
    c(x,y,z) = local seismic wave speed (function of material density/elasticity)

The PINN simultaneously learns:
  - U(x,y,z): The 3D vibration field
  - c(x,y,z): The 3D wave-speed field (→ density/void structure)

The boundary condition at z=0 is the SAR-measured surface vibration map.
Deep underground, c reverts to the mean crustal wave speed.

Architecture: Dual-Head PINN
-----------------------------
Input: (x, y, z) coordinates in the 3D subsurface volume
Output Head 1: U(x,y,z) — Predicted vibration amplitude
Output Head 2: c(x,y,z) — Predicted wave speed (→ inversely related to density contrast)

Loss = L_physics (Helmholtz residual) + L_data (surface fit) + L_regularization

Usage:
    python pinn_vibro_inversion.py --vibration data/vibrometry/outputs/vibration_amplitude.npy
    python pinn_vibro_inversion.py --vibration vib.npy --depth 2000 --epochs 5000
"""

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from project_paths import DATA_DIR, OUTPUTS_DIR, ensure_directories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# 1. Configuration
# ============================================================
INVERSION_DIR = DATA_DIR / "inversion_3d"
INVERSION_OUTPUTS = INVERSION_DIR / "outputs"

DEFAULT_INVERSION_CONFIG = {
    # Domain parameters
    "max_depth_m": 5000.0,         # Maximum inversion depth (meters)
    "grid_nx": 64,                  # X grid resolution for collocation
    "grid_ny": 64,                  # Y grid resolution
    "grid_nz": 32,                  # Z (depth) grid resolution
    "domain_width_m": 5000.0,       # Horizontal domain extent (meters)

    # Physics parameters
    "background_wave_speed": 3500.0,  # Mean crustal P-wave speed (m/s)
    "min_wave_speed": 300.0,          # Air/void wave speed (m/s)
    "max_wave_speed": 6000.0,         # Dense rock wave speed (m/s)
    "excitation_frequency_hz": 5.0,   # Dominant ambient seismic frequency
    "density_background": 2670.0,     # Mean crustal density (kg/m³)

    # Training parameters
    "epochs": 3000,
    "lr": 1e-4,
    "batch_size_collocation": 512,    # Interior collocation points per micro-batch (small for 8GB GPU + 2nd-order autograd graph)
    "batch_size_boundary": 256,       # Surface boundary points per micro-batch
    "gradient_accumulation_steps": 4, # Effective batch = 512*4 = 2048 collocation per optimizer step
    "physics_weight": 1.0,            # Helmholtz residual weight
    "data_weight": 10.0,              # Surface data fit weight
    "regularization_weight": 0.01,    # Smoothness/sparsity weight
    "deep_prior_weight": 0.1,         # Force c→background at depth

    # Network architecture
    "hidden_layers": 8,
    "hidden_neurons": 512,
    "activation": "swish",

    # GPU
    "use_amp": True,
    "use_compile": True,              # torch.compile for fused kernels (PyTorch 2.0+)
    "num_workers": 4,                 # CPU workers for data prep
    "pin_memory": True,               # Pin CPU tensors for faster H2D transfer
    "seed": 42,
}


# ============================================================
# 2. Neural Network Architecture
# ============================================================
class FourierFeatures(nn.Module):
    """
    Positional encoding using random Fourier features.

    Maps low-dimensional (x,y,z) coordinates to a higher-dimensional
    representation that helps the network learn high-frequency spatial
    patterns (critical for resolving sharp void boundaries).

    This is equivalent to the "random Fourier features" trick from
    Tancik et al. (2020) "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains".
    """

    def __init__(self, in_dim: int = 3, num_frequencies: int = 128, sigma: float = 10.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Random projection matrix (frozen, not learned)
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) tensor of (x, y, z) coordinates

        Returns:
            (N, 2*num_frequencies) tensor of Fourier features [sin(Bx), cos(Bx)]
        """
        proj = x @ self.B  # (N, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class Sine(nn.Module):
    """SIREN activation function — prevents wave PDE gradients from vanishing.
    
    Standard activations (ReLU, Swish) have zero or near-zero 2nd derivatives,
    making it impossible for the network to learn ∇²U accurately. sin(w0*x)
    has non-zero derivatives of all orders, which is mathematically necessary
    for 2nd-order wave equations like Helmholtz.
    """
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class VibroInversionPINN(nn.Module):
    """
    Dual-head Physics-Informed Neural Network for vibro-elastic inversion.

    Architecture:
        Input: (x, y, z) → Fourier Encoding → Shared Trunk → Two Heads
        Head 1: U(x,y,z) — Vibration amplitude field
        Head 2: c(x,y,z) — Wave speed field (→ density structure)

    The shared trunk learns common spatial representations while
    the heads specialize in their respective physical quantities.
    """

    def __init__(
        self,
        hidden_layers: int = 6,
        hidden_neurons: int = 256,
        num_frequencies: int = 128,
        c_background: float = 3500.0,
        c_min: float = 300.0,
        c_max: float = 6000.0,
    ):
        super().__init__()
        self.c_background = c_background
        self.c_min = c_min
        self.c_max = c_max

        # Fourier positional encoding
        self.fourier = FourierFeatures(3, num_frequencies, sigma=10.0)
        input_dim = 2 * num_frequencies

        # Shared trunk with residual connections
        self.trunk = nn.ModuleList()
        self.trunk.append(nn.Linear(input_dim, hidden_neurons))

        for i in range(hidden_layers - 1):
            self.trunk.append(nn.Linear(hidden_neurons, hidden_neurons))

        self.activation = Sine()

        # Head 1: Complex wavefield — outputs both U_real and U_imag
        # A single real output forces unphysical perfect standing waves.
        # Complex output allows proper wave propagation modeling.
        self.head_U = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            Sine(),
            nn.Linear(hidden_neurons // 2, 2),  # 2 outputs: real + imaginary
        )

        # Head 2: Wave speed c(x,y,z)
        self.head_c = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            Sine(),
            nn.Linear(hidden_neurons // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: (N, 3) tensor of (x, y, z) coordinates.
                    x, y normalized to [-1, 1]
                    z normalized to [0, 1] (0=surface, 1=max_depth)

        Returns:
            U_real: (N, 1) real part of vibration wavefield
            U_imag: (N, 1) imaginary part of vibration wavefield
            c: (N, 1) wave speed in m/s, bounded to [c_min, c_max]
        """
        # Fourier features
        h = self.fourier(coords)

        # Shared trunk with skip connections (every 2 layers)
        for i, layer in enumerate(self.trunk):
            h_new = self.activation(layer(h))
            if i > 0 and i % 2 == 0 and h_new.shape == h.shape:
                h_new = h_new + h  # Residual connection
            h = h_new

        # Head 1: Complex wavefield components
        U_complex = self.head_U(h)
        U_real = U_complex[:, 0:1]
        U_imag = U_complex[:, 1:2]

        # Head 2: Wave speed (bounded via sigmoid scaling)
        c_raw = self.head_c(h)
        c = self.c_min + (self.c_max - self.c_min) * torch.sigmoid(c_raw)

        return U_real, U_imag, c


# ============================================================
# 3. Physics Loss: Helmholtz Equation
# ============================================================
def helmholtz_physics_loss(
    model: VibroInversionPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    omega: float = 31.416,  # 2π * 5 Hz
    domain_width_m: float = 5000.0,
    max_depth_m: float = 2000.0,
    c_background: float = 3500.0,
) -> torch.Tensor:
    """
    Compute the Helmholtz equation residual at collocation points with
    proper chain-rule scaling from normalized to physical coordinates.

    The Helmholtz equation for steady-state acoustic waves:
        ∇²U + (ω²/c²) * U = 0

    CRITICAL: The network operates on normalized coordinates [-1, 1] but
    the Helmholtz term ω²/c² is in physical SI units (meters). Without
    chain-rule scaling, the Laplacian is off by a factor of ~10⁷, causing
    the physics loss to be effectively ignored.

    Chain rule: ∂U/∂x_phys = ∂U/∂x_norm * (2 / domain_width)

    Parameters
    ----------
    model : VibroInversionPINN
        The PINN model.
    x, y, z : torch.Tensor
        (N, 1) coordinate tensors with requires_grad=True.
    omega : float
        Angular frequency (2π * f_Hz).
    domain_width_m : float
        Physical domain width in meters (for x,y scaling).
    max_depth_m : float
        Physical max depth in meters (for z scaling).
    c_background : float
        Background wave speed for loss normalization.

    Returns
    -------
    torch.Tensor
        Scalar: mean squared Helmholtz residual.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)

    coords = torch.cat([x, y, z], dim=-1)
    U_real, U_imag, c = model(coords)

    # Chain-rule scaling: maps normalized gradients to physical space
    # x,y ∈ [-1, 1] maps to [-domain_width/2, domain_width/2], so scale = 2/domain_width
    # z ∈ [0, 1] maps to [0, max_depth], so scale = 1/max_depth
    scale_xy = 2.0 / domain_width_m
    scale_z = 1.0 / max_depth_m

    def compute_laplacian(U):
        """Compute ∇²U with chain-rule corrections."""
        dU_dx = torch.autograd.grad(
            U, x, grad_outputs=torch.ones_like(U),
            create_graph=True, retain_graph=True
        )[0]
        dU_dy = torch.autograd.grad(
            U, y, grad_outputs=torch.ones_like(U),
            create_graph=True, retain_graph=True
        )[0]
        dU_dz = torch.autograd.grad(
            U, z, grad_outputs=torch.ones_like(U),
            create_graph=True, retain_graph=True
        )[0]

        d2U_dx2 = torch.autograd.grad(
            dU_dx, x, grad_outputs=torch.ones_like(dU_dx),
            create_graph=True, retain_graph=True
        )[0]
        d2U_dy2 = torch.autograd.grad(
            dU_dy, y, grad_outputs=torch.ones_like(dU_dy),
            create_graph=True, retain_graph=True
        )[0]
        d2U_dz2 = torch.autograd.grad(
            dU_dz, z, grad_outputs=torch.ones_like(dU_dz),
            create_graph=True, retain_graph=True
        )[0]

        # Apply chain-rule scaling to convert normalized → physical gradients
        return (d2U_dx2 + d2U_dy2) * (scale_xy ** 2) + d2U_dz2 * (scale_z ** 2)

    laplacian_Ur = compute_laplacian(U_real)
    laplacian_Ui = compute_laplacian(U_imag)

    # Helmholtz term in physical SI units
    helmholtz_term = (omega ** 2) / (c ** 2 + 1e-8)

    # Residuals for both real and imaginary components
    residual_real = laplacian_Ur + helmholtz_term * U_real
    residual_imag = laplacian_Ui + helmholtz_term * U_imag

    # Normalize loss for numerical stability in float32
    scaling_factor = (c_background / omega) ** 2
    return torch.mean((residual_real * scaling_factor) ** 2 + (residual_imag * scaling_factor) ** 2)


def surface_data_loss(
    model: VibroInversionPINN,
    surface_coords: torch.Tensor,
    observed_vibration: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce the boundary condition at z=0: the PINN's predicted
    vibration MAGNITUDE must match the SAR-observed surface vibration.

    The observed vibration is a scalar amplitude (always positive),
    so we fit |U| = sqrt(U_real² + U_imag²) to the observations.

    Parameters
    ----------
    model : VibroInversionPINN
        The PINN model.
    surface_coords : torch.Tensor
        (N, 3) surface coordinates with z=0.
    observed_vibration : torch.Tensor
        (N, 1) observed vibration amplitude from SAR.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    U_real, U_imag, _ = model(surface_coords)
    # Target is the MAGNITUDE of the complex wavefield
    U_mag = torch.sqrt(U_real**2 + U_imag**2 + 1e-8)
    return F.mse_loss(U_mag, observed_vibration)


def deep_boundary_loss(
    model: VibroInversionPINN,
    deep_coords: torch.Tensor,
    c_background: float = 3500.0,
) -> torch.Tensor:
    """
    At great depth, wave speed should revert to the background crustal value.
    This prevents the PINN from hallucinating structure where there is none.

    Parameters
    ----------
    model : VibroInversionPINN
        The PINN model.
    deep_coords : torch.Tensor
        (N, 3) coordinates near maximum depth (z ≈ 1).
    c_background : float
        Background wave speed in m/s.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss penalizing deviation from background.
    """
    _, _, c_pred = model(deep_coords)
    target = torch.full_like(c_pred, c_background)
    # Normalize by c_background² so loss measures fractional deviation (~0-1 scale)
    # Without this, MSE of (c_pred - 3500)² is ~millions, completely dominating
    # the total loss despite a small weight.
    return F.mse_loss(c_pred, target) / (c_background ** 2)


def tv_regularization(
    model: VibroInversionPINN,
    coords: torch.Tensor,
    c_background: float = 3500.0,
) -> torch.Tensor:
    """
    Total Variation (TV) regularization on wave speed.
    
    Unlike L1 sparsity which just penalizes deviation from background,
    TV encourages SHARP geological boundaries rather than blurry ones.
    This is physically correct — real geological layers have abrupt transitions.

    Parameters
    ----------
    model : VibroInversionPINN
        The PINN model.
    coords : torch.Tensor
        (N, 3) sample coordinates (must have requires_grad enabled).
    c_background : float
        Background wave speed (unused, kept for API compatibility).

    Returns
    -------
    torch.Tensor
        Scalar TV penalty.
    """
    coords = coords.requires_grad_(True)
    _, _, c_pred = model(coords)
    grad_c = torch.autograd.grad(
        c_pred, coords, grad_outputs=torch.ones_like(c_pred),
        create_graph=False  # No need for 2nd-order grads through regularizer — saves ~40% VRAM
    )[0]
    return torch.mean(torch.sqrt(torch.sum(grad_c**2, dim=-1) + 1e-8))


# ============================================================
# 4. Collocation Point Sampling
# ============================================================
def sample_collocation_points(
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample random collocation points in the normalized 3D domain.

    Domain: x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 1]
    z is biased toward the surface (more points near z=0) using
    a quadratic distribution, since structure near the surface has
    the strongest effect on surface vibrations.

    Returns
    -------
    x, y, z : torch.Tensor
        Each (batch_size, 1) with requires_grad enabled.
    """
    x = (2 * torch.rand(batch_size, 1, device=device) - 1)
    y = (2 * torch.rand(batch_size, 1, device=device) - 1)
    # Quadratic bias: more samples near surface (z=0)
    z_raw = torch.rand(batch_size, 1, device=device)
    z = z_raw ** 2  # Concentrates points near z=0

    return x, y, z


class SurfaceSamplerGPU:
    """
    Surface point sampler with pre-computed importance weights.
    
    For small maps (<=1M pixels): keeps CDF/data on GPU for fast sampling.
    For large maps (>1M pixels): keeps CDF/data on CPU, transfers only
    the sampled batch to GPU. This prevents VRAM exhaustion on real
    SAR data (e.g. 19669x25757 = 506M pixels would eat 4+ GB VRAM).
    """
    
    def __init__(self, vibration_map: np.ndarray, device: torch.device):
        h, w = vibration_map.shape
        self.h = h
        self.w = w
        self.device = device
        
        # Decide storage: GPU for small maps, CPU for large ones
        n_pixels = h * w
        self.on_gpu = (n_pixels <= 1_000_000)  # ~4 MB on GPU vs ~2 GB+
        storage_device = device if self.on_gpu else torch.device('cpu')
        
        # Pre-compute importance weights and CDF
        weights = np.abs(np.nan_to_num(vibration_map.flatten(), nan=0.0)).astype(np.float32)
        weights += 1e-6
        weights /= weights.sum()
        cdf = np.cumsum(weights).astype(np.float32)
        cdf[-1] = 1.0  # Ensure exact 1.0 at end
        self.cdf = torch.tensor(cdf, device=storage_device)
        
        # Pre-compute all coordinate grids
        cols_all = torch.arange(w, device=storage_device, dtype=torch.float32)
        rows_all = torch.arange(h, device=storage_device, dtype=torch.float32)
        self.x_norm_lut = 2.0 * cols_all / (w - 1) - 1.0  # (W,)
        self.y_norm_lut = 2.0 * rows_all / (h - 1) - 1.0  # (H,)
        
        # Pre-compute flat vibration values
        self.vib_flat = torch.tensor(
            np.nan_to_num(vibration_map.flatten(), nan=0.0).astype(np.float32),
            device=storage_device
        )
        
        storage_mb = n_pixels * 4 * 3 / (1024**2)  # CDF + vib_flat + LUTs
        logger.info(f"SurfaceSampler: {h}x{w} = {n_pixels:,} pixels, "
                    f"storage={'GPU' if self.on_gpu else 'CPU'} ({storage_mb:.0f} MB)")
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample surface points using pre-computed CDF."""
        # Importance sampling via inverse CDF (on whichever device holds the data)
        u = torch.rand(batch_size, device=self.cdf.device)
        indices = torch.searchsorted(self.cdf, u).clamp(0, self.h * self.w - 1)
        
        rows = indices // self.w
        cols = indices % self.w
        
        # Look up pre-computed normalized coordinates
        x_norm = self.x_norm_lut[cols]
        y_norm = self.y_norm_lut[rows]
        z_norm = torch.zeros(batch_size, device=self.cdf.device)
        
        coords = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (N, 3)
        values = self.vib_flat[indices].unsqueeze(-1)  # (N, 1)
        
        # Transfer to GPU if data was on CPU
        if not self.on_gpu:
            coords = coords.to(self.device)
            values = values.to(self.device)
        
        return coords, values


def sample_surface_points(
    vibration_map: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points from the surface vibration map for data fitting.
    Legacy wrapper — prefer SurfaceSamplerGPU for training loops.
    """
    h, w = vibration_map.shape
    weights = vibration_map.flatten().copy()
    weights = np.nan_to_num(weights, nan=0.0)
    weights = np.abs(weights) + 1e-6
    weights /= weights.sum()

    indices = np.random.choice(h * w, size=batch_size, replace=True, p=weights)
    rows = indices // w
    cols = indices % w

    x_norm = 2.0 * cols / (w - 1) - 1.0
    y_norm = 2.0 * rows / (h - 1) - 1.0
    z_norm = np.zeros(batch_size)

    coords = torch.tensor(
        np.stack([x_norm, y_norm, z_norm], axis=-1),
        dtype=torch.float32, device=device
    )
    values = torch.tensor(
        vibration_map[rows, cols].reshape(-1, 1),
        dtype=torch.float32, device=device
    )

    return coords, values


def sample_deep_points(
    batch_size: int,
    device: torch.device,
    z_min: float = 0.8,
) -> torch.Tensor:
    """
    Sample points near the maximum depth for the deep boundary condition.

    Parameters
    ----------
    batch_size : int
        Number of deep points.
    device : torch.device
        Target device.
    z_min : float
        Minimum normalized depth (0.8 = bottom 20% of domain).

    Returns
    -------
    coords : torch.Tensor
        (batch_size, 3) coordinates near max depth.
    """
    x = 2 * torch.rand(batch_size, 1, device=device) - 1
    y = 2 * torch.rand(batch_size, 1, device=device) - 1
    z = z_min + (1.0 - z_min) * torch.rand(batch_size, 1, device=device)
    return torch.cat([x, y, z], dim=-1)


# ============================================================
# 5. Training Loop
# ============================================================
def train_vibro_pinn(
    vibration_map: np.ndarray,
    output_dir: str,
    config: Optional[Dict] = None,
    frequency_map: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Train the Helmholtz PINN to invert surface vibrations into 3D structure.

    Parameters
    ----------
    vibration_map : np.ndarray
        2D array of surface vibration amplitudes (from sar_vibrometry.py).
    output_dir : str
        Directory to save model, outputs, and visualizations.
    config : dict, optional
        Override default configuration.
    frequency_map : np.ndarray, optional
        2D array of dominant vibration frequencies. If provided, uses
        spatially-varying omega instead of a single frequency.

    Returns
    -------
    dict
        Paths to outputs: 'model', 'wave_speed_volume', 'density_volume',
        'void_probability_surface'.
    """
    cfg = DEFAULT_INVERSION_CONFIG.copy()
    if config:
        cfg.update(config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Downsample vibration map if too large for GPU
    # The PINN grid is only 64x64 at the surface, so keeping millions of
    # observation pixels in VRAM is wasteful and causes OOM on 8GB GPUs.
    # Real Sentinel-1 SLC bursts are often 19669x25757 = 506M pixels.
    MAX_VIB_DIM = 1024  # Max dimension for the vibration map
    vib_map = vibration_map.astype(np.float32)
    orig_shape = vib_map.shape
    if max(vib_map.shape) > MAX_VIB_DIM:
        from scipy.ndimage import zoom
        scale = MAX_VIB_DIM / max(vib_map.shape)
        vib_map = zoom(vib_map, scale, order=1)  # Bilinear downsampling
        logger.info(f"Downsampled vibration map: {orig_shape} -> {vib_map.shape} "
                    f"(scale={scale:.4f}, saves {orig_shape[0]*orig_shape[1]*4/1024**2:.0f} MB VRAM)")

    # Normalize vibration map
    vib_mean = np.nanmean(vib_map)
    vib_std = np.nanstd(vib_map)
    if vib_std < 1e-8:
        vib_std = 1.0
    vib_normalized = (vib_map - vib_mean) / vib_std
    vib_normalized = np.nan_to_num(vib_normalized, nan=0.0)

    logger.info(f"Vibration map: {vib_map.shape}, mean={vib_mean:.6f}, std={vib_std:.6f}")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    use_amp = cfg["use_amp"] and device.type == "cuda"

    # Seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Angular frequency
    omega = 2 * np.pi * cfg["excitation_frequency_hz"]

    # Initialize model
    model = VibroInversionPINN(
        hidden_layers=cfg["hidden_layers"],
        hidden_neurons=cfg["hidden_neurons"],
        c_background=cfg["background_wave_speed"],
        c_min=cfg["min_wave_speed"],
        c_max=cfg["max_wave_speed"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"PINN parameters: {param_count:,}")

    # torch.compile for fused kernels (PyTorch 2.0+)
    use_compile = cfg.get("use_compile", True) and device.type == "cuda"
    if use_compile:
        import sys
        if sys.platform == "win32":
            logger.info("torch.compile disabled (Triton is not natively supported on Windows)")
        else:
            try:
                _dynamo = getattr(__import__('torch', fromlist=['_dynamo']), '_dynamo')
                _dynamo.config.suppress_errors = True
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile enabled (reduce-overhead mode) — first epochs will be slower")
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back to eager mode: {e}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # NaN recovery: track best state for rollback
    best_state = None
    nan_recoveries = 0
    max_nan_recoveries = 5

    # GPU-accelerated surface sampler (eliminates CPU bottleneck)
    surface_sampler = SurfaceSamplerGPU(vib_normalized, device)
    logger.info("GPU surface sampler initialized (pre-computed CDF + coordinate LUTs)")

    # Gradient accumulation config
    accum_steps = cfg.get("gradient_accumulation_steps", 1)
    effective_colloc = cfg["batch_size_collocation"] * accum_steps
    effective_boundary = cfg["batch_size_boundary"] * accum_steps

    # Training loop
    logger.info(f"\nStarting PINN training: {cfg['epochs']} epochs")
    logger.info(f"  Physics weight: {cfg['physics_weight']}")
    logger.info(f"  Data weight: {cfg['data_weight']}")
    logger.info(f"  Regularization: {cfg['regularization_weight']}")
    logger.info(f"  Collocation batch: {cfg['batch_size_collocation']} x {accum_steps} accum = {effective_colloc}")
    logger.info(f"  Surface batch: {cfg['batch_size_boundary']} x {accum_steps} accum = {effective_boundary}")
    logger.info(f"  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  GPU Memory: {gpu_mem:.1f} GB")
    sys.stdout.flush()

    training_start = time.time()
    loss_history = {"total": [], "physics": [], "data": [], "reg": [], "deep": []}
    best_loss = float('inf')

    model.train()
    loop = tqdm(range(cfg["epochs"]), desc="PINN Training")

    for epoch in loop:
      try:
        optimizer.zero_grad(set_to_none=True)

        device_type = 'cuda' if device.type == 'cuda' else 'cpu'

        # Verbose logging for first 3 epochs to diagnose hangs
        verbose = (epoch < 3)

        # Gradient accumulation: run multiple micro-batches per optimizer step
        epoch_loss_physics = 0.0
        epoch_loss_data = 0.0
        epoch_loss_deep = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_total = 0.0

        for accum_step in range(accum_steps):
            if verbose:
                logger.info(f"  Epoch {epoch} accum {accum_step+1}/{accum_steps}: sampling collocation...")
                sys.stdout.flush()

            # === Physics loss ALWAYS in float32 (2nd-order autograd overflows in float16) ===
            x_coll, y_coll, z_coll = sample_collocation_points(
                cfg["batch_size_collocation"], device
            )

            if verbose:
                logger.info(f"  Epoch {epoch} accum {accum_step+1}/{accum_steps}: computing physics loss...")
                if device.type == 'cuda':
                    logger.info(f"    VRAM before physics: {torch.cuda.memory_allocated(0)/(1024**2):.0f} MB")
                sys.stdout.flush()

            loss_physics = helmholtz_physics_loss(
                model, x_coll, y_coll, z_coll, omega,
                domain_width_m=cfg["domain_width_m"],
                max_depth_m=cfg["max_depth_m"],
                c_background=cfg["background_wave_speed"],
            )

            if verbose:
                logger.info(f"  Epoch {epoch} accum {accum_step+1}/{accum_steps}: physics loss = {loss_physics.item():.6f}")
                if device.type == 'cuda':
                    logger.info(f"    VRAM after physics: {torch.cuda.memory_allocated(0)/(1024**2):.0f} MB")
                sys.stdout.flush()

            # === Data + regularization losses can use AMP safely ===
            with torch.amp.autocast(device_type, enabled=use_amp):
                if verbose:
                    logger.info(f"  Epoch {epoch} accum {accum_step+1}/{accum_steps}: sampling surface points...")
                    sys.stdout.flush()

                # 2. Data loss: Surface vibration fit (GPU-accelerated sampler)
                surface_coords, surface_values = surface_sampler.sample(
                    cfg["batch_size_boundary"]
                )
                loss_data = surface_data_loss(model, surface_coords, surface_values)

                if verbose:
                    logger.info(f"  Epoch {epoch} accum {accum_step+1}/{accum_steps}: data loss = {loss_data.item():.6f}")
                    sys.stdout.flush()

                # 3. Deep boundary loss: c → background at depth
                deep_coords = sample_deep_points(
                    cfg["batch_size_boundary"] // 4, device
                )
                loss_deep = deep_boundary_loss(
                    model, deep_coords, cfg["background_wave_speed"]
                )

                # 4. TV regularization (uses detached coords, no create_graph)
                reg_coords = torch.cat([
                    x_coll.detach(),
                    y_coll.detach(),
                    z_coll.detach()
                ], dim=-1)
                loss_reg = tv_regularization(
                    model, reg_coords, cfg["background_wave_speed"]
                )

            # Total loss (scaled by accumulation steps for gradient averaging)
            phys_weighted = cfg["physics_weight"] * loss_physics
            data_weighted = cfg["data_weight"] * loss_data
            deep_weighted = cfg["deep_prior_weight"] * loss_deep
            reg_weighted = cfg["regularization_weight"] * loss_reg
            loss_unscaled = phys_weighted + data_weighted + deep_weighted + reg_weighted
            loss = loss_unscaled / accum_steps

            if verbose:
                logger.info(
                    f"  E{epoch} μ{accum_step+1}/{accum_steps}: "
                    f"phys={loss_physics.item():.4f}×{cfg['physics_weight']}={phys_weighted.item():.4f}, "
                    f"data={loss_data.item():.4f}×{cfg['data_weight']}={data_weighted.item():.4f}, "
                    f"deep={loss_deep.item():.4f}×{cfg['deep_prior_weight']}={deep_weighted.item():.4f}, "
                    f"reg={loss_reg.item():.4f}×{cfg['regularization_weight']}={reg_weighted.item():.4f}, "
                    f"total_unscaled={loss_unscaled.item():.4f}, "
                    f"loss_for_backward={loss.item():.6f}"
                )
                if device.type == 'cuda':
                    logger.info(f"    VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.0f} MB allocated")
                sys.stdout.flush()

            # Backprop (accumulate gradients)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Aggressively free autograd graph memory (2nd-order graph can hold ~2-4 GB)
            # Extract scalar values BEFORE deleting tensors
            _phys_val = loss_physics.item()
            _data_val = loss_data.item()
            _deep_val = loss_deep.item()
            _reg_val = loss_reg.item()
            _loss_val = loss.item()

            del loss, loss_unscaled, phys_weighted, data_weighted, deep_weighted, reg_weighted
            del loss_physics, loss_data, loss_deep, loss_reg
            del x_coll, y_coll, z_coll, surface_coords, surface_values, deep_coords, reg_coords
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Track per-step losses (average across accum steps)
            epoch_loss_physics += _phys_val / accum_steps
            epoch_loss_data += _data_val / accum_steps
            epoch_loss_deep += _deep_val / accum_steps
            epoch_loss_reg += _reg_val / accum_steps
            epoch_loss_total += _loss_val

        # === NaN detection and rollback ===
        if not np.isfinite(epoch_loss_total):
            nan_recoveries += 1
            if nan_recoveries > max_nan_recoveries:
                logger.error(f"NaN loss after {max_nan_recoveries} recoveries — aborting.")
                break
            logger.warning(f"NaN detected at epoch {epoch}! Recovery {nan_recoveries}/{max_nan_recoveries}")
            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info("  Rolled back to best checkpoint")
            # Halve LR for stability
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            logger.info(f"  LR halved to {optimizer.param_groups[0]['lr']:.2e}")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Optimizer step after all accumulation steps
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step(epoch_loss_total)

        # Record
        loss_history["total"].append(epoch_loss_total)
        loss_history["physics"].append(epoch_loss_physics)
        loss_history["data"].append(epoch_loss_data)
        loss_history["reg"].append(epoch_loss_reg)
        loss_history["deep"].append(epoch_loss_deep)

        if epoch_loss_total < best_loss:
            best_loss = epoch_loss_total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Gradient norm logging every 20 epochs
        if epoch % 20 == 0:
            grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
            if grad_norms:
                total_grad_norm = torch.norm(torch.stack(grad_norms), 2).item()
            else:
                total_grad_norm = 0.0

        # Exact objective decomposition every 50 epochs
        if epoch % 50 == 0:
            logger.info(
                f"[E{epoch:04d}] OBJECTIVE: "
                f"phys_raw={epoch_loss_physics:.6f} data_raw={epoch_loss_data:.6f} "
                f"deep_raw={epoch_loss_deep:.6f} reg_raw={epoch_loss_reg:.6f} | "
                f"weighted: phys={epoch_loss_physics*cfg['physics_weight']:.4f} "
                f"data={epoch_loss_data*cfg['data_weight']:.4f} "
                f"deep={epoch_loss_deep*cfg['deep_prior_weight']:.4f} "
                f"reg={epoch_loss_reg*cfg['regularization_weight']:.4f} | "
                f"total={epoch_loss_total:.6f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"grad_norm={total_grad_norm:.4f}"
            )
            sys.stdout.flush()

        if epoch % 10 == 0:
            mem_info = ""
            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                mem_info = f" GPU:{mem_used:.1f}/{mem_total:.1f}GB"
            loop.set_postfix(
                loss=f"{epoch_loss_total:.4f}",
                phys=f"{epoch_loss_physics:.4f}",
                data=f"{epoch_loss_data:.4f}",
                deep=f"{epoch_loss_deep:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                mem=mem_info,
            )

      except Exception as e:
          logger.error(f"Training failed at epoch {epoch}: {e}")
          logger.error(traceback.format_exc())
          sys.stdout.flush()
          raise

    training_time = time.time() - training_start
    logger.info(f"\nTraining complete in {training_time:.1f}s (best loss: {best_loss:.6f})")

    # ============================================================
    # 6. Extract 3D Volume
    # ============================================================
    logger.info("\nExtracting 3D subsurface volume...")
    model.eval()

    nx, ny, nz = cfg["grid_nx"], cfg["grid_ny"], cfg["grid_nz"]
    max_depth = cfg["max_depth_m"]
    domain_width = cfg["domain_width_m"]

    # Create evaluation grid
    x_grid = np.linspace(-1, 1, nx)
    y_grid = np.linspace(-1, 1, ny)
    z_grid = np.linspace(0, 1, nz)

    wave_speed_volume = np.zeros((nz, ny, nx), dtype=np.float32)
    vibration_volume = np.zeros((nz, ny, nx), dtype=np.float32)

    with torch.no_grad():
        for iz, z_val in enumerate(tqdm(z_grid, desc="Evaluating depth slices")):
            # Create grid for this depth
            xx, yy = np.meshgrid(x_grid, y_grid)
            coords = np.stack([
                xx.flatten(),
                yy.flatten(),
                np.full(nx * ny, z_val)
            ], axis=-1).astype(np.float32)

            coords_t = torch.tensor(coords, device=device)
            U_real, U_imag, c_pred = model(coords_t)

            wave_speed_volume[iz] = c_pred.cpu().numpy().reshape(ny, nx)
            # Store wavefield magnitude: |U| = sqrt(U_real² + U_imag²)
            U_mag = torch.sqrt(U_real**2 + U_imag**2).cpu().numpy().reshape(ny, nx)
            vibration_volume[iz] = U_mag

    # Convert wave speed anomaly to density contrast
    # Using empirical Birch's law: ρ ≈ 0.32 * Vp + 0.77 (g/cm³)
    # Density contrast = ρ_predicted - ρ_background
    density_volume = 0.32 * wave_speed_volume / 1000.0 + 0.77  # g/cm³
    density_background = 0.32 * cfg["background_wave_speed"] / 1000.0 + 0.77
    density_contrast = (density_volume - density_background) * 1000.0  # kg/m³

    # Void probability at each depth (low wave speed = high void probability)
    void_threshold = cfg["background_wave_speed"] * 0.5
    void_probability = np.clip(
        1.0 - (wave_speed_volume - cfg["min_wave_speed"]) /
        (cfg["background_wave_speed"] - cfg["min_wave_speed"]),
        0, 1
    )

    # Save outputs
    outputs = {}

    # Model checkpoint
    model_path = Path(output_dir) / "vibro_pinn_model.pth"
    torch.save(model.state_dict(), model_path)
    outputs["model"] = str(model_path)

    # 3D volumes as numpy
    ws_path = Path(output_dir) / "wave_speed_volume.npy"
    np.save(ws_path, wave_speed_volume)
    outputs["wave_speed_volume"] = str(ws_path)

    dc_path = Path(output_dir) / "density_contrast_volume.npy"
    np.save(dc_path, density_contrast)
    outputs["density_contrast_volume"] = str(dc_path)

    vp_path = Path(output_dir) / "void_probability_volume.npy"
    np.save(vp_path, void_probability)
    outputs["void_probability_volume"] = str(vp_path)

    vib_vol_path = Path(output_dir) / "vibration_field_volume.npy"
    np.save(vib_vol_path, vibration_volume)
    outputs["vibration_field_volume"] = str(vib_vol_path)

    # Surface void probability (max void probability along depth axis)
    surface_void_prob = np.max(void_probability, axis=0)
    svp_path = Path(output_dir) / "surface_void_probability.npy"
    np.save(svp_path, surface_void_prob)
    outputs["void_probability_surface"] = str(svp_path)

    # Save depth metadata
    metadata = {
        "grid_shape": f"({nz}, {ny}, {nx})",
        "max_depth_m": max_depth,
        "domain_width_m": domain_width,
        "depth_values_m": [f"{z * max_depth:.1f}" for z in z_grid],
        "x_values_m": [f"{x * domain_width/2:.1f}" for x in x_grid],
        "y_values_m": [f"{y * domain_width/2:.1f}" for y in y_grid],
        "background_wave_speed_ms": cfg["background_wave_speed"],
        "excitation_freq_hz": cfg["excitation_frequency_hz"],
        "training_time_s": f"{training_time:.1f}",
        "best_loss": f"{best_loss:.6f}",
        "epochs": cfg["epochs"],
    }
    meta_path = Path(output_dir) / "inversion_metadata.txt"
    with open(meta_path, 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")
    outputs["metadata"] = str(meta_path)

    # Save loss history
    loss_path = Path(output_dir) / "loss_history.npy"
    np.save(loss_path, {k: np.array(v) for k, v in loss_history.items()})
    outputs["loss_history"] = str(loss_path)

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("3D INVERSION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Volume shape: ({nz}, {ny}, {nx})")
    logger.info(f"  Depth range: 0 to {max_depth:.0f} m")
    logger.info(f"  Wave speed: [{wave_speed_volume.min():.0f}, {wave_speed_volume.max():.0f}] m/s")
    logger.info(f"  Density contrast: [{density_contrast.min():.0f}, {density_contrast.max():.0f}] kg/m³")
    logger.info(f"  Max void probability: {void_probability.max():.4f}")
    logger.info(f"  Voxels with void_prob > 0.5: {(void_probability > 0.5).sum()}")
    logger.info(f"  Voxels with void_prob > 0.8: {(void_probability > 0.8).sum()}")
    logger.info("=" * 60)

    return outputs


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PINN Vibro-Elastic 3D Inversion (Helmholtz)"
    )
    parser.add_argument(
        "--vibration", required=True,
        help="Path to vibration amplitude map (.npy or .tif)"
    )
    parser.add_argument("--frequency", default=None, help="Optional frequency map")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--depth", type=float, default=2000.0, help="Max depth (m)")
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--freq-hz", type=float, default=5.0, help="Excitation frequency (Hz)")
    parser.add_argument("--nx", type=int, default=64, help="X grid resolution")
    parser.add_argument("--ny", type=int, default=64, help="Y grid resolution")
    parser.add_argument("--nz", type=int, default=32, help="Z grid resolution")
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--hidden-neurons", type=int, default=256)
    parser.add_argument("--wave-speed-bg", type=float, default=3500.0, help="Background wave speed (m/s)")
    parser.add_argument("--batch-colloc", type=int, default=None, help="Collocation batch size (default: 65536)")
    parser.add_argument("--batch-boundary", type=int, default=None, help="Boundary batch size (default: 16384)")
    parser.add_argument("--accum-steps", type=int, default=None, help="Gradient accumulation steps (default: 4)")

    args = parser.parse_args()

    # Load vibration map
    vib_path = args.vibration
    if vib_path.endswith(".npy"):
        vibration_map = np.load(vib_path)
    elif vib_path.endswith(".tif") or vib_path.endswith(".tiff"):
        import rasterio
        with rasterio.open(vib_path) as src:
            vibration_map = src.read(1)
    else:
        raise ValueError(f"Unsupported format: {vib_path}")

    # Optional frequency map
    freq_map = None
    if args.frequency:
        if args.frequency.endswith(".npy"):
            freq_map = np.load(args.frequency)
        elif args.frequency.endswith(".tif"):
            import rasterio
            with rasterio.open(args.frequency) as src:
                freq_map = src.read(1)

    # Output directory
    if args.output_dir is None:
        output_dir = str(INVERSION_OUTPUTS)
    else:
        output_dir = args.output_dir

    # Config overrides
    config = {
        "max_depth_m": args.depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "excitation_frequency_hz": args.freq_hz,
        "grid_nx": args.nx,
        "grid_ny": args.ny,
        "grid_nz": args.nz,
        "hidden_layers": args.hidden_layers,
        "hidden_neurons": args.hidden_neurons,
        "background_wave_speed": args.wave_speed_bg,
    }
    if args.batch_colloc is not None:
        config["batch_size_collocation"] = args.batch_colloc
    if args.batch_boundary is not None:
        config["batch_size_boundary"] = args.batch_boundary
    if args.accum_steps is not None:
        config["gradient_accumulation_steps"] = args.accum_steps

    # Run inversion
    outputs = train_vibro_pinn(
        vibration_map,
        output_dir,
        config=config,
        frequency_map=freq_map,
    )

    print("\nOutputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
