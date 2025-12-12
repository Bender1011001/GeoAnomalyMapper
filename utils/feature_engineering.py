"""
Feature Engineering Module for Geological Anomaly Detection (Expert Version)

This module provides functions to compute contextual features that describe
the shape and texture of geological anomalies.

Improvements over v1:
- Physics-Aware: Can handle physical units (mGal/m) via cell_size.
- Robust NaN Handling: Uses Normalized Convolution to prevent edge artifacts at land/ocean boundaries.
- Structure Detection: Uses Hessian Eigenvalues to distinguish ridges (faults) from blobs (deposits).
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict, Union

# Type alias
Grid2D = np.ndarray

def _nan_safe_window_stats(data: Grid2D, window_size: int) -> Tuple[Grid2D, Grid2D]:
    """
    Helper: Calculates local mean and variance ignoring NaNs (Normalized Convolution).
    Prevents "Zero-Halo" artifacts at survey boundaries.
    """
    mask = (~np.isnan(data)).astype(float)
    data_filled = np.nan_to_num(data, nan=0.0)
    
    # Calculate sum of values and weights (valid pixels)
    # uniform_filter calculates mean, so multiply by size^2 to get sum
    w_sq = window_size**2
    sum_data = ndimage.uniform_filter(data_filled, size=window_size, mode='reflect') * w_sq
    sum_mask = ndimage.uniform_filter(mask, size=window_size, mode='reflect') * w_sq
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = sum_data / sum_mask
        
        # Variance calculation: E[x^2] - (E[x])^2
        sum_sq = ndimage.uniform_filter(data_filled**2, size=window_size, mode='reflect') * w_sq
        local_mean_sq = sum_sq / sum_mask
        local_var = np.maximum(0, local_mean_sq - local_mean**2)

    # Restore NaNs where we had no valid data
    invalid = sum_mask < 1e-6
    local_mean[invalid] = np.nan
    local_var[invalid] = np.nan
    
    return local_mean, local_var

def calculate_gradient_magnitude(data: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """
    Calculate the Total Horizontal Gradient (THG).
    Uses np.gradient which handles boundaries better than simple convolution.
    """
    # np.gradient handles NaNs by propagating them, which is safer than zero-filling
    # for physical fields.
    grad_y, grad_x = np.gradient(data, cell_size)
    magnitude = np.hypot(grad_x, grad_y)
    return magnitude

def calculate_local_roughness(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate local roughness (standard deviation) robust to NaNs.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    _, local_var = _nan_safe_window_stats(data, window_size)
    return np.sqrt(local_var)

def calculate_local_mean(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Calculate local background trend (Robust Moving Average).
    """
    local_mean, _ = _nan_safe_window_stats(data, window_size)
    return local_mean

def calculate_curvature_shape(data: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Shape Index and Curvedness using Hessian Eigenvalues.
    Useful for distinguishing linear features (faults) from circular (pipes).
    """
    # Smooth first to remove noise (using nan-safe gaussian logic)
    mask = (~np.isnan(data)).astype(float)
    data_filled = np.nan_to_num(data, nan=0.0)
    
    smooth = ndimage.gaussian_filter(data_filled, sigma) / \
             (ndimage.gaussian_filter(mask, sigma) + 1e-6)
             
    # Derivatives
    gy, gx = np.gradient(smooth)
    gxx = np.gradient(gx, axis=1)
    gyy = np.gradient(gy, axis=0)
    gxy = np.gradient(gx, axis=0)
    
    # Eigenvalues of Hessian
    trace = gxx + gyy
    det = gxx*gyy - gxy**2
    term = np.sqrt(np.maximum(0, (trace/2)**2 - det))
    k1 = trace/2 + term
    k2 = trace/2 - term
    
    # Shape Index (2/pi * arctan((k1+k2)/(k1-k2)))
    # Ranges -1 (cup) to +1 (cap). 
    with np.errstate(divide='ignore', invalid='ignore'):
        shape_index = (2.0/np.pi) * np.arctan((k1+k2)/(k1-k2))
        
    return shape_index, k1 # Return shape index and max curvature

def calculate_all_features(data: np.ndarray, 
                          gradient_method: str = 'sobel', # Kept for API compatibility but ignored
                          roughness_window: int = 3,
                          mean_window: int = 5) -> dict:
    """
    Calculate all features.
    """
    # Note: We ignore 'gradient_method' argument to enforce the physics-based np.gradient
    
    shape_idx, max_curve = calculate_curvature_shape(data)
    
    features = {
        'gradient': calculate_gradient_magnitude(data),
        'roughness': calculate_local_roughness(data, window_size=roughness_window),
        'local_mean': calculate_local_mean(data, window_size=mean_window),
        'shape_index': shape_idx,
        'max_curvature': max_curve
    }
    
    return features

def stack_features(gravity_data: np.ndarray, 
                  magnetic_data: Optional[np.ndarray] = None,
                  **kwargs) -> Tuple[np.ndarray, list]:
    """
    Stack features for ML. Matches original API signature.
    """
    # Calculate gravity features
    g_feats = calculate_all_features(gravity_data, **kwargs)
    
    feature_stack = [
        gravity_data,
        g_feats['gradient'],
        g_feats['roughness'],
        g_feats['local_mean'],
        g_feats['shape_index'] # New Expert Feature
    ]
    feature_names = [
        'gravity',
        'gravity_gradient',
        'gravity_roughness',
        'gravity_local_mean',
        'gravity_shape'
    ]
    
    if magnetic_data is not None and magnetic_data.size > 0:
        m_feats = calculate_all_features(magnetic_data, **kwargs)
        feature_stack.extend([
            magnetic_data,
            m_feats['gradient'],
            m_feats['roughness'],
            m_feats['local_mean'],
            m_feats['shape_index']
        ])
        feature_names.extend([
            'magnetic',
            'magnetic_gradient',
            'magnetic_roughness',
            'magnetic_local_mean',
            'magnetic_shape'
        ])
        
    return np.stack(feature_stack, axis=0), feature_names