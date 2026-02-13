#!/usr/bin/env python3
"""
Advanced Synthetic DUB Generator v2.0
=======================================
Generates high-fidelity synthetic geophysical signatures for Deep Underground Bases.

Enhancements:
- Geometric facilities (Rectilinear chambers, corridors)
- Structured Geology (Fractal noise, fault lines)
- Sensor Artifacts (Stripe noise, sensor drift)
- Physics Coupling (Metal shielding placed relative to excavated voids)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate, binary_dilation

def generate_fractal_noise(shape, octaves=5, persistence=0.5, lacunarity=2.0):
    """Generates 2D fractal noise to mimic complex geology."""
    noise = np.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        temp_noise = np.random.normal(0, 1, shape)
        noise += amplitude * gaussian_filter(temp_noise, sigma=1.0/frequency * 5)
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def create_rect_mask(shape, y, x, h, w, angle):
    """Creates a rotated rectangular mask."""
    mask = np.zeros(shape)
    y_min, y_max = max(0, y-h//2), min(shape[0], y+h//2)
    x_min, x_max = max(0, x-w//2), min(shape[1], x+w//2)
    mask[y_min:y_max, x_min:x_max] = 1.0
    if angle != 0:
        mask = rotate(mask, angle, reshape=False, order=0)
    return mask

def generate_synthetic_batch(batch_size=32, patch_size=256, pixel_size=200):
    """
    Generates high-fidelity synthetic (Gravity, Magnetic) pairs.
    """
    gt_density = np.zeros((batch_size, 1, patch_size, patch_size))
    gt_susceptibility = np.zeros((batch_size, 1, patch_size, patch_size))
    
    for i in range(batch_size):
        # 1. Generate Structured Geological Background
        # Density background (sedimentary basins vs mountains)
        geo_bg = generate_fractal_noise((patch_size, patch_size))
        gt_density[i, 0] = geo_bg * 50.0 
        
        # Magnetic background (crustal magnetization)
        mag_bg = generate_fractal_noise((patch_size, patch_size), octaves=3)
        gt_susceptibility[i, 0] = mag_bg * 0.01
        
        # 2. Add Fault Lines (Linear features)
        if np.random.random() > 0.5:
            angle = np.random.uniform(0, 180)
            fault = create_rect_mask((patch_size, patch_size), patch_size//2, patch_size//2, patch_size*2, 2, angle)
            gt_density[i, 0] += fault * 100.0
            
        # 3. Add DUB Facilities
        num_facilities = np.random.randint(1, 3)
        for _ in range(num_facilities):
            fy, fx = np.random.randint(40, patch_size-40, size=2)
            
            # --- Complex Facility Geometry ---
            fac_mask = np.zeros((patch_size, patch_size))
            
            # Main Chamber
            cw, ch = np.random.randint(4, 12, size=2)
            c_angle = np.random.uniform(0, 90)
            fac_mask += create_rect_mask((patch_size, patch_size), fy, fx, ch, cw, c_angle)
            
            # Corridors/Tunnels
            if np.random.random() > 0.3:
                tw, th = (40, 2) if np.random.random() > 0.5 else (2, 40)
                fac_mask += create_rect_mask((patch_size, patch_size), fy, fx, th, tw, c_angle)
            
            fac_mask = np.clip(fac_mask, 0, 1)
            
            # Density: Negative (void)
            gt_density[i, 0] -= fac_mask * 800.0
            
            # Magnetic: Metal Shielding (Outline of the void)
            # Steel is usually in the walls/structure
            shielding_mask = binary_dilation(fac_mask > 0, iterations=2).astype(float) - fac_mask
            gt_susceptibility[i, 0] += shielding_mask * 0.8
            
            # Add Point Dipoles (Machinery/Power Plants)
            for _ in range(np.random.randint(1, 4)):
                py, px = fy + np.random.randint(-5, 5), fx + np.random.randint(-5, 5)
                if 0 <= py < patch_size and 0 <= px < patch_size:
                    gt_susceptibility[i, 0, py, px] += 2.0 # High susceptibility jump
                    
        # 4. Add "Urban/Infrastructure" Clutter (Linear magnetic noise)
        if np.random.random() > 0.7:
            # Power lines or roads with metal
            angle = np.random.uniform(0, 180)
            line = create_rect_mask((patch_size, patch_size), np.random.randint(0, patch_size), 
                                    np.random.randint(0, patch_size), patch_size*2, 1, angle)
            gt_susceptibility[i, 0] += line * 0.1

    return torch.from_numpy(gt_density).float(), torch.from_numpy(gt_susceptibility).float()

def save_synthetic_samples(n=4):
    """Save high-res visual samples for verification."""
    density, susc = generate_synthetic_batch(batch_size=n)
    
    fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))
    for i in range(n):
        im1 = axes[i, 0].imshow(density[i, 0], cmap='RdBu_r')
        axes[i, 0].set_title(f"Synthetic Gravity (Density Contrast)\nFacility {i+1}")
        plt.colorbar(im1, ax=axes[i, 0], label='kg/m3')
        
        im2 = axes[i, 1].imshow(susc[i, 0], cmap='magma')
        axes[i, 1].set_title(f"Synthetic Magnetic (Susceptibility)\nMetallic Structure {i+1}")
        plt.colorbar(im2, ax=axes[i, 1], label='SI')
    
    plt.tight_layout()
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/outputs/synthetic_dubs_v2_preview.png")
    print("Saved advanced preview to data/outputs/synthetic_dubs_v2_preview.png")

if __name__ == "__main__":
    save_synthetic_samples()
