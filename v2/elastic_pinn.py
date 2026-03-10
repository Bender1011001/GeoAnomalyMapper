#!/usr/bin/env python3
"""
Elastic Mixed-Variable PINN for Subsurface Shear Modulus Inversion
==================================================================

Key innovations over v1 (Helmholtz PINN):

1. ELASTIC WAVE EQUATION instead of acoustic Helmholtz:
   ∇·(μ∇U) + ρω²U = 0
   Where μ = shear modulus (Pa). Voids have μ=0, giving infinite contrast.

2. MIXED-VARIABLE FORMULATION (no 2nd-order autograd):
   Head 1: Predicts displacement U(x,y,z)
   Head 2: Predicts stress σ(x,y,z) = μ·∇U
   Physics: ∇·σ + ρω²U = 0  (only 1st derivatives!)
   
   This eliminates the ∇²U computation that caused NaN in v1.

3. PROPER SIREN INITIALIZATION (Sitzmann et al. 2020):
   Xavier uniform is fatal for Sine activations. We use the
   mathematically proven U(-sqrt(6/fan_in)/w0) bounds.

4. SHEAR MODULUS OUTPUT (not wave speed):
   Rock:  μ ~ 10-30 GPa
   Water: μ = 0 (exactly)  
   Air:   μ = 0 (exactly)
   Voids produce infinite contrast vs. ~3x contrast with wave speed.

References:
  - Sitzmann et al. 2020: "Implicit Neural Representations with Periodic Activations"
  - Medical MRE: Manduca et al. 2001: "Magnetic Resonance Elastography"
  - Mixed-Variable PINNs: Haghighat et al. 2021: "Physics-Informed Deep Learning for 
    Computational Elastodynamics"
"""

import math
import sys
import time
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# ACTIVATION: Sine (SIREN)
# =============================================================================
class Sine(nn.Module):
    """Sine activation for SIREN networks.
    
    w0 controls the frequency of the initial mapping. Higher w0 = the network
    can represent higher-frequency signals, but makes training harder.
    For subsurface imaging at 1-10 Hz, w0=1.0 is sufficient.
    """
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


# =============================================================================
# FOURIER FEATURE EMBEDDING
# =============================================================================
class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding.
    
    Maps (x, y, z) → [sin(Bx), cos(Bx)] where B ~ N(0, σ²).
    This helps the network learn spatial frequency content that matches
    the scale of subsurface structures (10m-1000m).
    """
    def __init__(self, in_dim: int = 3, num_features: int = 128, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_features) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# =============================================================================
# MIXED-VARIABLE ELASTIC PINN
# =============================================================================
class ElasticPINN(nn.Module):
    """Mixed-variable Physics-Informed Neural Network for elastic inversion.
    
    Architecture:
        Input: (x, y, z) normalized coordinates
        ↓ Fourier Features
        ↓ Shared SIREN trunk (N hidden layers)
        ↓ Three output heads:
           - Head U: displacement (real + imaginary) — 2 outputs
           - Head σ: stress tensor (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz) — 6 outputs  
           - Head μ: shear modulus — 1 output (sigmoid-scaled to [μ_min, μ_max])
    
    The mixed-variable trick:
        Instead of computing ∇²U (2nd derivative, NaN-prone), we predict
        stress σ directly. The physics loss enforces:
            1. Constitutive: σ = μ·∇U (stress-strain relation)
            2. Equilibrium: ∇·σ + ρω²U = 0 (equation of motion)
        Both only require 1st-order autograd derivatives!
    """
    
    def __init__(
        self,
        hidden_layers: int = 8,
        hidden_neurons: int = 512,
        mu_background: float = 15e9,   # ~15 GPa for average crustal rock
        mu_min: float = 0.0,           # Voids/air/water
        mu_max: float = 40e9,          # Hard crystalline rock
        fourier_features: int = 128,
        fourier_sigma: float = 10.0,
        w0: float = 1.0,
    ):
        super().__init__()
        
        self.mu_background = mu_background
        self.mu_min = mu_min
        self.mu_max = mu_max
        
        # Fourier embedding: (x,y,z) → 256-dim
        self.fourier = FourierFeatures(3, fourier_features, fourier_sigma)
        input_dim = self.fourier.out_dim  # 2 * fourier_features
        
        # Shared SIREN trunk
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(Sine(w0))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(Sine(w0))
        self.trunk = nn.Sequential(*layers)
        
        # Head U: displacement field (real + imaginary parts)
        # Output: (U_real, U_imag) — complex displacement
        self.head_u = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            Sine(w0),
            nn.Linear(hidden_neurons // 2, 2),  # U_real, U_imag
        )
        
        # Head σ: stress tensor (symmetric, so 6 independent components)
        # For 2.5D (quasi-3D from SAR), we use 3 diagonal + 3 off-diagonal
        # σ = [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
        self.head_sigma = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            Sine(w0),
            nn.Linear(hidden_neurons // 2, 6),  # 6 stress components
        )
        
        # Head μ: shear modulus (scalar field)
        # Output goes through sigmoid → scaled to [mu_min, mu_max]
        self.head_mu = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons // 4),
            Sine(w0),
            nn.Linear(hidden_neurons // 4, 1),
        )
        
        # Initialize with proper SIREN weights
        self._siren_init()
    
    def _siren_init(self):
        """Proper SIREN initialization (Sitzmann et al. 2020).
        
        Xavier uniform is FATAL for Sine activations. With 8 layers of sin(),
        xavier causes outputs to grow exponentially: layer N output ~ O(N!).
        
        SIREN paper specifies:
          - First layer: weights ~ U(-1/fan_in, 1/fan_in)
          - Hidden layers: weights ~ U(-sqrt(6/fan_in)/w0, sqrt(6/fan_in)/w0)
        This keeps sin() outputs bounded in [-1, 1] across all layers.
        """
        with torch.no_grad():
            is_first_linear = True
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    fan_in = m.weight.size(1)
                    if is_first_linear:
                        bound = 1.0 / fan_in
                        is_first_linear = False
                    else:
                        bound = math.sqrt(6.0 / fan_in) / 1.0  # w0=1.0
                    
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.uniform_(-bound, bound)
    
    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords: (N, 3) tensor of (x, y, z) coordinates.
                    x, y normalized to [-1, 1]
                    z normalized to [0, 1] (0=surface, 1=max_depth)

        Returns:
            dict with keys:
                'U_real': (N, 1) real part of displacement
                'U_imag': (N, 1) imaginary part of displacement
                'sigma': (N, 6) stress tensor components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
                'mu': (N, 1) shear modulus in Pa, bounded to [mu_min, mu_max]
        """
        # Fourier features
        h = self.fourier(coords)
        
        # Shared trunk
        h = self.trunk(h)
        
        # Displacement head
        u_out = self.head_u(h)
        U_real = u_out[:, 0:1]
        U_imag = u_out[:, 1:2]
        
        # Stress head
        sigma = self.head_sigma(h)
        
        # Shear modulus head — sigmoid scaled to [mu_min, mu_max]
        mu_raw = self.head_mu(h)
        mu = self.mu_min + (self.mu_max - self.mu_min) * torch.sigmoid(mu_raw)
        
        return {
            'U_real': U_real,
            'U_imag': U_imag,
            'sigma': sigma,
            'mu': mu,
        }


# =============================================================================
# COLLOCATION POINT SAMPLING
# =============================================================================
def sample_collocation_points(
    batch_size: int,
    device: torch.device,
    depth_bias: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample collocation points in the 3D volume with depth-biased distribution.
    
    Samples more points near the surface (where we have data) and fewer deep
    (where we rely on physics). Uses a beta distribution for depth sampling.
    
    Args:
        batch_size: Number of points to sample
        device: CUDA or CPU device
        depth_bias: Controls depth distribution. 0.5 = uniform, 0.7 = surface-biased
    
    Returns:
        x, y, z tensors each of shape (batch_size, 1), requires_grad=True
    """
    x = (torch.rand(batch_size, 1, device=device) * 2 - 1)  # [-1, 1]
    y = (torch.rand(batch_size, 1, device=device) * 2 - 1)  # [-1, 1]
    
    # Depth: beta distribution biased toward surface (z=0)
    alpha_param = 1.0  # shape parameter
    beta_param = 1.0 / (1.0 - depth_bias + 1e-8)  # higher = more surface bias
    z_raw = torch.distributions.Beta(alpha_param, beta_param).sample(
        (batch_size, 1)
    ).to(device)
    z = z_raw  # [0, 1] where 0=surface, 1=max_depth
    
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    return x, y, z


def sample_surface_points(
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample points on the surface (z=0) for boundary conditions."""
    x = (torch.rand(batch_size, 1, device=device) * 2 - 1)
    y = (torch.rand(batch_size, 1, device=device) * 2 - 1)
    z = torch.zeros(batch_size, 1, device=device)
    
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    return x, y, z


# =============================================================================
# PHYSICS LOSSES — MIXED-VARIABLE ELASTIC
# =============================================================================
def elastic_constitutive_loss(
    model: ElasticPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    domain_width_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    """Constitutive relation loss: σ = μ · ∇U (stress-strain relation).
    
    This enforces that the predicted stress is physically consistent with
    the predicted displacement gradient and shear modulus.
    
    Only requires 1st-order autograd derivatives of U — no ∇²U!
    
    The chain rule scaling converts from normalized coords to physical:
        ∂U/∂x_physical = ∂U/∂x_norm · (2/domain_width_m)
    """
    coords = torch.cat([x, y, z], dim=-1)
    
    # Cast to float64 for numerical stability in autograd
    orig_dtype = coords.dtype
    if coords.dtype != torch.float64:
        coords = coords.double()
        # Temporarily cast model to double for this computation
        model_was_float = next(model.parameters()).dtype == torch.float32
        if model_was_float:
            model.double()
    else:
        model_was_float = False
    
    coords.requires_grad_(True)
    
    outputs = model(coords)
    U_real = outputs['U_real']
    U_imag = outputs['U_imag']
    sigma_pred = outputs['sigma']  # (N, 6): σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz
    mu = outputs['mu']
    
    # Compute ∇U_real and ∇U_imag (1st derivatives only!)
    grad_U_real = torch.autograd.grad(
        U_real, coords,
        grad_outputs=torch.ones_like(U_real),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3): ∂U_real/∂x, ∂U_real/∂y, ∂U_real/∂z
    
    grad_U_imag = torch.autograd.grad(
        U_imag, coords,
        grad_outputs=torch.ones_like(U_imag),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3)
    
    # Chain-rule scaling: normalized coords → physical coords
    # x, y ∈ [-1, 1] map to [-W/2, W/2], so dx_phys = dx_norm * W/2
    # z ∈ [0, 1] maps to [0, D], so dz_phys = dz_norm * D
    scale_x = 2.0 / domain_width_m
    scale_y = 2.0 / domain_width_m
    scale_z = 1.0 / max_depth_m
    
    # Scale gradients to physical units
    dUr_dx = grad_U_real[:, 0:1] * scale_x
    dUr_dy = grad_U_real[:, 1:2] * scale_y
    dUr_dz = grad_U_real[:, 2:3] * scale_z
    
    dUi_dx = grad_U_imag[:, 0:1] * scale_x
    dUi_dy = grad_U_imag[:, 1:2] * scale_y
    dUi_dz = grad_U_imag[:, 2:3] * scale_z
    
    # For an isotropic elastic medium, the stress-strain relation for the
    # displacement gradient is:
    #   σ_ij = μ (∂U_i/∂x_j + ∂U_j/∂x_i)  (simplified, ignoring λ for shear)
    #
    # For scalar wavefield (our case), the stress components are:
    #   σ_xx = 2μ · ∂U/∂x (normal stress in x)
    #   σ_yy = 2μ · ∂U/∂y
    #   σ_zz = 2μ · ∂U/∂z
    #   σ_xy = μ · (∂U/∂y + ∂U/∂x) ≈ μ · 2·∂U/∂(avg) — simplified
    #   etc.
    #
    # For the scalar wavefield approximation (anti-plane shear):
    #   σ_xz = μ · ∂U/∂x
    #   σ_yz = μ · ∂U/∂y  
    #   σ_zz = μ · ∂U/∂z
    
    # Compute expected stress from constitutive relation for real part
    sigma_expected_real = torch.cat([
        mu * dUr_dx,    # σ_xx
        mu * dUr_dy,    # σ_yy
        mu * dUr_dz,    # σ_zz
        mu * (dUr_dx + dUr_dy) * 0.5,  # σ_xy (symmetric)
        mu * (dUr_dx + dUr_dz) * 0.5,  # σ_xz
        mu * (dUr_dy + dUr_dz) * 0.5,  # σ_yz
    ], dim=-1)
    
    # Constitutive loss: predicted stress must match μ·∇U
    constitutive_residual = sigma_pred - sigma_expected_real
    loss = torch.mean(constitutive_residual ** 2)
    
    # Restore model to float32 if needed
    if model_was_float:
        model.float()
    
    return loss.float() if loss.dtype == torch.float64 else loss


def elastic_equilibrium_loss(
    model: ElasticPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    omega: float,
    rho: float,
    domain_width_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    """Equilibrium equation loss: ∇·σ + ρω²U = 0 (equation of motion).
    
    This is the wave equation in stress form. Because σ is a direct network
    output, we only need 1st derivatives of σ — NOT 2nd derivatives of U.
    
    This is the critical advantage of the mixed-variable formulation:
    v1 computed ∇²U (2nd-order autograd → NaN at epoch 13).
    v2 computes ∇·σ (1st-order autograd → stable at 5000+ epochs).
    
    Args:
        model: The ElasticPINN
        x, y, z: Collocation points with requires_grad=True
        omega: Angular frequency (2π·f)
        rho: Rock density (kg/m³), typically ~2700
        domain_width_m: Physical domain width in meters
        max_depth_m: Physical max depth in meters
    
    Returns:
        Scalar loss (MSE of equilibrium residual)
    """
    coords = torch.cat([x, y, z], dim=-1)
    
    # Cast to float64 for stability
    orig_dtype = coords.dtype
    if coords.dtype != torch.float64:
        coords = coords.double()
        model_was_float = next(model.parameters()).dtype == torch.float32
        if model_was_float:
            model.double()
    else:
        model_was_float = False
    
    coords.requires_grad_(True)
    
    outputs = model(coords)
    U_real = outputs['U_real']
    sigma = outputs['sigma']  # (N, 6): σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz
    
    # Chain-rule scales
    scale_x = 2.0 / domain_width_m
    scale_y = 2.0 / domain_width_m
    scale_z = 1.0 / max_depth_m
    
    # Compute divergence of stress: ∇·σ
    # For each row of the stress tensor:
    #   (∇·σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z
    #   (∇·σ)_y = ∂σ_xy/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z
    #   (∇·σ)_z = ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z
    
    # For scalar wavefield (anti-plane shear SH wave), the equilibrium reduces to:
    #   ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z + ρω²U = 0
    #
    # σ_xz = sigma[:, 4:5], σ_yz = sigma[:, 5:6], σ_zz = sigma[:, 2:3]
    
    sigma_xz = sigma[:, 4:5]
    sigma_yz = sigma[:, 5:6]  
    sigma_zz = sigma[:, 2:3]
    
    # ∂σ_xz/∂x
    grad_sigma_xz = torch.autograd.grad(
        sigma_xz, coords,
        grad_outputs=torch.ones_like(sigma_xz),
        create_graph=True,
        retain_graph=True,
    )[0]
    dsigma_xz_dx = grad_sigma_xz[:, 0:1] * scale_x
    
    # ∂σ_yz/∂y
    grad_sigma_yz = torch.autograd.grad(
        sigma_yz, coords,
        grad_outputs=torch.ones_like(sigma_yz),
        create_graph=True,
        retain_graph=True,
    )[0]
    dsigma_yz_dy = grad_sigma_yz[:, 1:2] * scale_y
    
    # ∂σ_zz/∂z
    grad_sigma_zz = torch.autograd.grad(
        sigma_zz, coords,
        grad_outputs=torch.ones_like(sigma_zz),
        create_graph=True,
        retain_graph=True,
    )[0]
    dsigma_zz_dz = grad_sigma_zz[:, 2:3] * scale_z
    
    # Equilibrium residual: ∇·σ + ρω²U = 0
    div_sigma = dsigma_xz_dx + dsigma_yz_dy + dsigma_zz_dz
    body_force = rho * (omega ** 2) * U_real
    
    residual = div_sigma + body_force
    loss = torch.mean(residual ** 2)
    
    if model_was_float:
        model.float()
    
    return loss.float() if loss.dtype == torch.float64 else loss


# =============================================================================
# DATA LOSS — SURFACE VIBRATION MATCHING
# =============================================================================
class SurfaceSamplerGPU:
    """GPU-accelerated importance sampler for surface vibration data.
    
    Pre-computes CDF and coordinate LUTs on GPU for fast sampling.
    Samples more points from high-amplitude vibration regions.
    """
    
    def __init__(self, vibration_map: np.ndarray, device: torch.device):
        """
        Args:
            vibration_map: 2D array of vibration amplitudes (already normalized)
            device: CUDA or CPU device
        """
        h, w = vibration_map.shape
        
        # Flatten and compute importance weights
        flat = np.abs(vibration_map).ravel().astype(np.float64)
        flat_sum = flat.sum()
        if flat_sum > 0:
            probs = flat / flat_sum
        else:
            probs = np.ones_like(flat) / len(flat)
        
        # CDF for inverse transform sampling on GPU
        cdf = np.cumsum(probs)
        cdf = cdf / cdf[-1]  # Normalize
        self.cdf = torch.from_numpy(cdf).float().to(device)
        
        # Pre-compute coordinate lookup tables
        ys, xs = np.mgrid[0:h, 0:w]
        # Normalize to [-1, 1]
        x_norm = (xs.ravel().astype(np.float32) / (w - 1)) * 2 - 1
        y_norm = (ys.ravel().astype(np.float32) / (h - 1)) * 2 - 1
        
        self.x_lut = torch.from_numpy(x_norm).to(device)
        self.y_lut = torch.from_numpy(y_norm).to(device)
        
        # Store vibration values for target matching
        self.vib_values = torch.from_numpy(
            vibration_map.ravel().astype(np.float32)
        ).to(device)
        
        self.device = device
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample surface points with importance weighting.
        
        Returns:
            x: (N, 1) normalized x coordinates [-1, 1]
            y: (N, 1) normalized y coordinates [-1, 1]
            z: (N, 1) zeros (surface)
            vib: (N, 1) target vibration amplitudes
        """
        # Inverse transform sampling via CDF
        u = torch.rand(batch_size, device=self.device)
        indices = torch.searchsorted(self.cdf, u)
        indices = indices.clamp(0, len(self.x_lut) - 1)
        
        x = self.x_lut[indices].unsqueeze(1)
        y = self.y_lut[indices].unsqueeze(1)
        z = torch.zeros(batch_size, 1, device=self.device)
        vib = self.vib_values[indices].unsqueeze(1)
        
        return x, y, z, vib


def surface_data_loss(
    model: ElasticPINN,
    x_surf: torch.Tensor,
    y_surf: torch.Tensor,
    z_surf: torch.Tensor,
    vib_target: torch.Tensor,
) -> torch.Tensor:
    """Match predicted surface displacement to observed vibration.
    
    Loss = MSE(|U_pred|, vib_observed) at z=0 (surface)
    
    Where |U_pred| = sqrt(U_real² + U_imag²) is the displacement magnitude.
    """
    coords = torch.cat([x_surf, y_surf, z_surf], dim=-1)
    outputs = model(coords)
    
    U_real = outputs['U_real']
    U_imag = outputs['U_imag']
    
    # Displacement magnitude
    U_mag = torch.sqrt(U_real ** 2 + U_imag ** 2 + 1e-12)
    
    # Normalize both to [0, 1] for comparison
    U_mag_norm = U_mag / (U_mag.max() + 1e-12)
    vib_norm = vib_target / (vib_target.max() + 1e-12)
    
    return F.mse_loss(U_mag_norm, vib_norm)


# =============================================================================
# REGULARIZATION LOSSES
# =============================================================================
def mu_smoothness_loss(
    model: ElasticPINN,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    domain_width_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    """Total variation regularization on the shear modulus field.
    
    Encourages μ to be piecewise-smooth with sharp boundaries at voids.
    Uses L1 norm of gradients (TV) rather than L2 (which blurs edges).
    """
    coords = torch.cat([x, y, z], dim=-1)
    coords.requires_grad_(True)
    
    outputs = model(coords)
    mu = outputs['mu']
    
    grad_mu = torch.autograd.grad(
        mu, coords,
        grad_outputs=torch.ones_like(mu),
        create_graph=True,
    )[0]
    
    # Scale to physical coordinates
    scale_x = 2.0 / domain_width_m
    scale_y = 2.0 / domain_width_m
    scale_z = 1.0 / max_depth_m
    
    grad_mu_phys = grad_mu * torch.tensor(
        [scale_x, scale_y, scale_z], device=grad_mu.device
    )
    
    # TV: L1 norm of gradients
    return torch.mean(torch.abs(grad_mu_phys))


def deep_boundary_loss(
    model: ElasticPINN,
    batch_size: int,
    device: torch.device,
    mu_background: float,
) -> torch.Tensor:
    """Enforce μ → μ_background at depth boundaries.
    
    At the maximum depth and domain edges, we expect undisturbed rock.
    This prevents the network from hallucinating deep anomalies.
    """
    # Sample points at z=1 (max depth) and domain edges
    n_each = batch_size // 4
    
    # Bottom boundary (z=1)
    x_bot = (torch.rand(n_each, 1, device=device) * 2 - 1)
    y_bot = (torch.rand(n_each, 1, device=device) * 2 - 1)
    z_bot = torch.ones(n_each, 1, device=device)
    
    # Left edge (x=-1)
    x_left = -torch.ones(n_each, 1, device=device)
    y_left = (torch.rand(n_each, 1, device=device) * 2 - 1)
    z_left = torch.rand(n_each, 1, device=device)
    
    # Right edge (x=1)
    x_right = torch.ones(n_each, 1, device=device)
    y_right = (torch.rand(n_each, 1, device=device) * 2 - 1)
    z_right = torch.rand(n_each, 1, device=device)
    
    # Deep interior (z > 0.8)
    x_deep = (torch.rand(n_each, 1, device=device) * 2 - 1)
    y_deep = (torch.rand(n_each, 1, device=device) * 2 - 1)
    z_deep = 0.8 + 0.2 * torch.rand(n_each, 1, device=device)
    
    # Concatenate all boundary points
    x_all = torch.cat([x_bot, x_left, x_right, x_deep], dim=0)
    y_all = torch.cat([y_bot, y_left, y_right, y_deep], dim=0)
    z_all = torch.cat([z_bot, z_left, z_right, z_deep], dim=0)
    
    coords = torch.cat([x_all, y_all, z_all], dim=-1)
    outputs = model(coords)
    mu_pred = outputs['mu']
    
    # Should be close to background
    mu_target = torch.full_like(mu_pred, mu_background)
    
    return F.mse_loss(mu_pred, mu_target)


# =============================================================================
# GRAVITY CONSTRAINT LOSS (Joint Inversion)
# =============================================================================
def gravity_constraint_loss(
    model: ElasticPINN,
    x_grav: torch.Tensor,
    y_grav: torch.Tensor,
    gravity_observed: torch.Tensor,
    rho_rock: float = 2700.0,
    rho_void: float = 0.0,
    mu_background: float = 15e9,
    domain_width_m: float = 800.0,
    max_depth_m: float = 500.0,
    grid_nz: int = 32,
) -> torch.Tensor:
    """Joint gravity inversion: compare predicted gravity from μ field
    to observed USGS Bouguer anomaly.
    
    The key insight: low μ (void) implies low density → negative gravity anomaly.
    We compute the vertical integral of the density perturbation along z for
    each (x, y) position and compare to observed gravity.
    
    Gravity anomaly ≈ -2πG ∫ Δρ(x,y,z) dz  (Bouguer slab approximation)
    
    Where Δρ = ρ_rock · (μ/μ_background) for void-detection:
      - μ = μ_background → Δρ = 0 (no anomaly)
      - μ = 0 (void) → Δρ = -ρ_rock (maximum negative anomaly)
    
    Args:
        model: ElasticPINN
        x_grav, y_grav: (N, 1) gravity station coordinates (normalized)
        gravity_observed: (N, 1) observed Bouguer anomaly in mGal
        rho_rock: Rock density in kg/m³
        rho_void: Void density (0 for air, 1000 for water)
        mu_background: Background shear modulus
        domain_width_m: Physical domain width
        max_depth_m: Physical max depth
        grid_nz: Number of depth samples for vertical integration
    """
    G = 6.674e-11  # gravitational constant
    N = x_grav.shape[0]
    device = x_grav.device
    
    # Sample depth column for each gravity station
    z_samples = torch.linspace(0, 1, grid_nz, device=device)
    dz = max_depth_m / grid_nz  # Physical depth increment
    
    # Expand: (N*grid_nz, 1) for batch processing
    x_expanded = x_grav.repeat_interleave(grid_nz, dim=0)
    y_expanded = y_grav.repeat_interleave(grid_nz, dim=0)
    z_expanded = z_samples.repeat(N).unsqueeze(1)
    
    coords = torch.cat([x_expanded, y_expanded, z_expanded], dim=-1)
    
    with torch.no_grad():
        outputs = model(coords)
        mu_pred = outputs['mu']  # (N*grid_nz, 1)
    
    # Reshape to (N, grid_nz)
    mu_grid = mu_pred.view(N, grid_nz)
    
    # Density perturbation: Δρ = ρ_rock · (1 - μ/μ_background)
    # When μ = μ_background: Δρ = 0
    # When μ = 0 (void): Δρ = ρ_rock
    delta_rho = rho_rock * (1.0 - mu_grid / mu_background)
    
    # Vertical integral: gravity anomaly in m/s² 
    # g_z ≈ 2πG · Σ(Δρ · dz)
    gravity_predicted = 2 * math.pi * G * torch.sum(delta_rho * dz, dim=1, keepdim=True)
    
    # Convert to mGal (1 mGal = 1e-5 m/s²)
    gravity_predicted_mgal = gravity_predicted / 1e-5
    
    return F.mse_loss(gravity_predicted_mgal, gravity_observed)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
DEFAULT_CONFIG = {
    # Network architecture
    "hidden_layers": 8,
    "hidden_neurons": 512,
    "fourier_features": 128,
    "fourier_sigma": 10.0,
    
    # Physical parameters
    "mu_background": 15e9,      # 15 GPa — average crustal rock
    "mu_min": 0.0,              # Air/vacuum
    "mu_max": 40e9,             # Hard crystalline rock  
    "rho": 2700.0,              # Rock density kg/m³
    "excitation_frequency_hz": 2.0,  # Ambient seismic frequency
    "domain_width_m": 800.0,
    "max_depth_m": 500.0,
    
    # Training parameters
    "epochs": 5000,
    "lr": 1e-4,
    "batch_size_collocation": 8192,
    "batch_size_surface": 4096,
    "batch_size_boundary": 1024,
    "gradient_accumulation_steps": 1,
    
    # Loss weights (with warmup)
    "constitutive_weight": 0.001,   # σ = μ·∇U
    "equilibrium_weight": 0.0001,   # ∇·σ + ρω²U = 0
    "data_weight": 50.0,            # Surface vibration match
    "gravity_weight": 10.0,         # Joint gravity constraint
    "smoothness_weight": 0.01,      # TV on μ
    "boundary_weight": 0.1,         # μ → μ_bg at boundaries
    
    # Stability
    "physics_warmup_epochs": 500,
    "grad_clip_norm": 0.1,
    "use_amp": False,               # NEVER true for physics autograd
    "max_nan_recoveries": 20,
    "nan_lr_decay": 0.7,
    
    # Grid resolution for 3D output
    "grid_nx": 128,
    "grid_ny": 128,
    "grid_nz": 64,
}


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_elastic_pinn(
    vibration_map: np.ndarray,
    cfg: Dict[str, Any],
    device: torch.device,
    gravity_data: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = False,
) -> Tuple[ElasticPINN, Dict[str, list]]:
    """Train the mixed-variable elastic PINN.
    
    Args:
        vibration_map: 2D numpy array of surface vibration amplitudes
        cfg: Training configuration dict
        device: CUDA or CPU device
        gravity_data: Optional dict with 'x', 'y', 'anomaly' arrays for joint inversion
        verbose: Log detailed per-step info
    
    Returns:
        Trained model and loss history dict
    """
    omega = 2 * math.pi * cfg["excitation_frequency_hz"]
    
    # Build model
    model = ElasticPINN(
        hidden_layers=cfg["hidden_layers"],
        hidden_neurons=cfg["hidden_neurons"],
        mu_background=cfg["mu_background"],
        mu_min=cfg["mu_min"],
        mu_max=cfg["mu_max"],
        fourier_features=cfg.get("fourier_features", 128),
        fourier_sigma=cfg.get("fourier_sigma", 10.0),
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"ElasticPINN parameters: {param_count:,}")
    logger.info(f"  Architecture: {cfg['hidden_layers']} layers × {cfg['hidden_neurons']} neurons")
    logger.info(f"  Output heads: displacement (2), stress (6), shear modulus (1)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6)
    
    # NaN recovery state
    best_state = None
    nan_recoveries = 0
    max_nan_recoveries = cfg.get("max_nan_recoveries", 20)
    grad_clip_norm = cfg.get("grad_clip_norm", 0.1)
    
    # Physics warmup
    physics_warmup_epochs = cfg.get("physics_warmup_epochs", 500)
    target_constitutive_weight = cfg["constitutive_weight"]
    target_equilibrium_weight = cfg["equilibrium_weight"]
    
    # Surface data sampler
    vib_normalized = vibration_map.copy()
    vib_max = np.abs(vib_normalized).max()
    if vib_max > 0:
        vib_normalized = vib_normalized / vib_max
    
    surface_sampler = SurfaceSamplerGPU(vib_normalized, device)
    logger.info("GPU surface sampler initialized")
    
    # Gravity data (if available for joint inversion)
    has_gravity = gravity_data is not None
    if has_gravity:
        grav_x = torch.from_numpy(gravity_data['x']).float().to(device).unsqueeze(1)
        grav_y = torch.from_numpy(gravity_data['y']).float().to(device).unsqueeze(1)
        grav_anomaly = torch.from_numpy(gravity_data['anomaly']).float().to(device).unsqueeze(1)
        logger.info(f"Joint gravity inversion: {len(grav_x)} stations loaded")
    else:
        logger.info("No gravity data — running SAR-only inversion")
    
    # Logging
    logger.info(f"\nStarting ElasticPINN training: {cfg['epochs']} epochs")
    logger.info(f"  Constitutive weight: {target_constitutive_weight} (warmup over {physics_warmup_epochs} epochs)")
    logger.info(f"  Equilibrium weight: {target_equilibrium_weight} (warmup over {physics_warmup_epochs} epochs)")
    logger.info(f"  Data weight: {cfg['data_weight']}")
    logger.info(f"  Gravity weight: {cfg['gravity_weight'] if has_gravity else 'N/A'}")
    logger.info(f"  Grad clip: {grad_clip_norm}")
    logger.info(f"  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  GPU Memory: {gpu_mem:.1f} GB")
    sys.stdout.flush()
    
    # Training
    training_start = time.time()
    loss_history = {
        "total": [], "constitutive": [], "equilibrium": [],
        "data": [], "gravity": [], "smoothness": [], "boundary": [],
    }
    best_loss = float('inf')
    
    model.train()
    loop = tqdm(range(cfg["epochs"]), desc="ElasticPINN Training")
    
    for epoch in loop:
        optimizer.zero_grad()
        
        # === Physics warmup ===
        if physics_warmup_epochs > 0 and epoch < physics_warmup_epochs:
            warmup_factor = epoch / physics_warmup_epochs
        else:
            warmup_factor = 1.0
        
        current_constitutive_weight = target_constitutive_weight * warmup_factor
        current_equilibrium_weight = target_equilibrium_weight * warmup_factor
        
        # === Constitutive loss: σ = μ·∇U ===
        x_coll, y_coll, z_coll = sample_collocation_points(
            cfg["batch_size_collocation"], device
        )
        
        loss_constitutive = elastic_constitutive_loss(
            model, x_coll, y_coll, z_coll,
            domain_width_m=cfg["domain_width_m"],
            max_depth_m=cfg["max_depth_m"],
        )
        
        # === Equilibrium loss: ∇·σ + ρω²U = 0 ===
        x_eq, y_eq, z_eq = sample_collocation_points(
            cfg["batch_size_collocation"] // 2, device
        )
        
        loss_equilibrium = elastic_equilibrium_loss(
            model, x_eq, y_eq, z_eq, omega, cfg["rho"],
            domain_width_m=cfg["domain_width_m"],
            max_depth_m=cfg["max_depth_m"],
        )
        
        # === Surface data loss ===
        x_surf, y_surf, z_surf, vib_target = surface_sampler.sample(
            cfg["batch_size_surface"]
        )
        loss_data = surface_data_loss(model, x_surf, y_surf, z_surf, vib_target)
        
        # === Gravity loss (joint inversion) ===
        if has_gravity:
            loss_gravity = gravity_constraint_loss(
                model, grav_x, grav_y, grav_anomaly,
                rho_rock=cfg["rho"],
                mu_background=cfg["mu_background"],
                domain_width_m=cfg["domain_width_m"],
                max_depth_m=cfg["max_depth_m"],
                grid_nz=cfg["grid_nz"],
            )
        else:
            loss_gravity = torch.tensor(0.0, device=device)
        
        # === Regularization ===
        x_reg, y_reg, z_reg = sample_collocation_points(
            cfg["batch_size_boundary"], device
        )
        loss_smoothness = mu_smoothness_loss(
            model, x_reg, y_reg, z_reg,
            domain_width_m=cfg["domain_width_m"],
            max_depth_m=cfg["max_depth_m"],
        )
        
        loss_boundary = deep_boundary_loss(
            model, cfg["batch_size_boundary"], device,
            mu_background=cfg["mu_background"],
        )
        
        # === Total loss ===
        loss_total = (
            current_constitutive_weight * loss_constitutive +
            current_equilibrium_weight * loss_equilibrium +
            cfg["data_weight"] * loss_data +
            (cfg["gravity_weight"] * loss_gravity if has_gravity else 0.0) +
            cfg["smoothness_weight"] * loss_smoothness +
            cfg["boundary_weight"] * loss_boundary
        )
        
        # === NaN check ===
        if not torch.isfinite(loss_total):
            nan_recoveries += 1
            if nan_recoveries > max_nan_recoveries:
                logger.error(f"NaN recovery limit ({max_nan_recoveries}) exceeded. Stopping.")
                break
            
            logger.warning(f"NaN at epoch {epoch}! Recovery {nan_recoveries}/{max_nan_recoveries}")
            
            if best_state is not None:
                model.load_state_dict(best_state)
                for pg in optimizer.param_groups:
                    pg['lr'] *= cfg.get("nan_lr_decay", 0.7)
                logger.info(f"  Rolled back model, LR → {optimizer.param_groups[0]['lr']:.2e}")
            
            continue
        
        # === Backward + step ===
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        
        epoch_loss = loss_total.item()
        scheduler.step(epoch_loss)
        
        # Track losses
        loss_history["total"].append(epoch_loss)
        loss_history["constitutive"].append(loss_constitutive.item())
        loss_history["equilibrium"].append(loss_equilibrium.item())
        loss_history["data"].append(loss_data.item())
        loss_history["gravity"].append(loss_gravity.item() if has_gravity else 0.0)
        loss_history["smoothness"].append(loss_smoothness.item())
        loss_history["boundary"].append(loss_boundary.item())
        
        # Save best state
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Update tqdm
        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            warmup_pct = warmup_factor * 100
            loop.set_postfix({
                'loss': f'{epoch_loss:.4f}',
                'data': f'{loss_data.item():.4f}',
                'const': f'{loss_constitutive.item():.4f}',
                'equil': f'{loss_equilibrium.item():.4f}',
                'lr': f'{lr_now:.1e}',
                'warmup': f'{warmup_pct:.0f}%',
                'nan_rec': nan_recoveries,
            })
        
        # Periodic logging
        if epoch % 100 == 0:
            elapsed = time.time() - training_start
            lr_now = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:5d} | loss={epoch_loss:.6f} | "
                f"data={loss_data.item():.4f} | const={loss_constitutive.item():.4f} | "
                f"equil={loss_equilibrium.item():.6f} | smooth={loss_smoothness.item():.4f} | "
                f"warmup={warmup_factor*100:.0f}% | lr={lr_now:.2e} | "
                f"elapsed={elapsed:.0f}s"
            )
            sys.stdout.flush()
    
    elapsed_total = time.time() - training_start
    logger.info(f"\nTraining complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"  Best loss: {best_loss:.6f}")
    logger.info(f"  NaN recoveries: {nan_recoveries}")
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model state")
    
    return model, loss_history


# =============================================================================
# 3D VOLUME INFERENCE
# =============================================================================
def infer_shear_modulus_volume(
    model: ElasticPINN,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    """Run inference over a dense 3D grid to extract the shear modulus volume.
    
    Returns:
        3D numpy array of shape (nz, ny, nx) containing μ values in Pa.
        Voids will have μ ≈ 0, rock will have μ ≈ 15e9.
    """
    model.eval()
    
    nx = cfg["grid_nx"]
    ny = cfg["grid_ny"]
    nz = cfg["grid_nz"]
    
    # Generate grid coordinates
    x_lin = torch.linspace(-1, 1, nx)
    y_lin = torch.linspace(-1, 1, ny)
    z_lin = torch.linspace(0, 1, nz)
    
    zz, yy, xx = torch.meshgrid(z_lin, y_lin, x_lin, indexing='ij')
    coords_flat = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    
    total_points = coords_flat.shape[0]
    mu_flat = torch.zeros(total_points)
    
    logger.info(f"Inferring μ volume: {nx}×{ny}×{nz} = {total_points:,} points")
    
    with torch.no_grad():
        for i in range(0, total_points, batch_size):
            batch = coords_flat[i:i+batch_size].to(device)
            outputs = model(batch)
            mu_flat[i:i+batch_size] = outputs['mu'].squeeze().cpu()
    
    mu_volume = mu_flat.numpy().reshape(nz, ny, nx)
    
    logger.info(f"  μ range: {mu_volume.min():.2e} — {mu_volume.max():.2e} Pa")
    logger.info(f"  μ mean: {mu_volume.mean():.2e} Pa")
    logger.info(f"  Low-μ voxels (< 1 GPa): {(mu_volume < 1e9).sum():,}")
    
    return mu_volume
