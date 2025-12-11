"""
Feature Engineering Module for Geological Anomaly Detection

This module provides functions to compute contextual features that describe
the shape and texture of geological anomalies, making the model more robust
to location-specific variations and better at learning generalizable patterns.

Key Features:
- Gradient magnitude: Captures edges and sharp transitions
- Local roughness: Measures texture and variability
- Local mean: Provides background trend information

These features are geologically significant and help the model focus on
anomaly characteristics rather than absolute values.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def calculate_gradient_magnitude(data: np.ndarray, 
                                method: str = 'sobel') -> np.ndarray:
    """
    Calculate the magnitude of the gradient (slope) for a 2D array.
    
    This feature captures edges and sharp transitions in the data,
    which are often associated with geological boundaries and anomalies.
    
    Args:
        data: 2D numpy array of raster data
        method: Method to use for gradient calculation ('sobel' or 'gradient')
        
    Returns:
        2D array of gradient magnitudes with same shape as input
    """
    # Handle NaN values by creating a mask
    mask = np.isnan(data)
    data_clean = np.where(mask, 0, data)
    
    if method == 'sobel':
        # Use Sobel filters for edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
        
        grad_x = ndimage.convolve(data_clean, sobel_x, mode='constant', cval=0.0)
        grad_y = ndimage.convolve(data_clean, sobel_y, mode='constant', cval=0.0)
        
    elif method == 'gradient':
        # Use numpy's gradient function
        grad_y, grad_x = np.gradient(data_clean)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sobel' or 'gradient'.")
    
    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Restore NaN values where original data had NaNs
    magnitude[mask] = np.nan
    
    return magnitude


def calculate_local_roughness(data: np.ndarray, 
                             window_size: int = 3) -> np.ndarray:
    """
    Calculate local roughness as the standard deviation within a sliding window.
    
    This feature measures the texture and local variability of the data,
    helping identify areas with complex geological structures.
    
    Args:
        data: 2D numpy array of raster data
        window_size: Size of the sliding window (must be odd)
        
    Returns:
        2D array of local roughness values with same shape as input
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    # Handle NaN values
    mask = np.isnan(data)
    data_clean = np.where(mask, 0, data)
    
    # Calculate local mean
    kernel = np.ones((window_size, window_size))
    local_mean = ndimage.convolve(data_clean, kernel, mode='constant', cval=0.0) / (window_size**2)
    
    # Calculate local variance
    squared_data = data_clean**2
    local_mean_squared = ndimage.convolve(squared_data, kernel, mode='constant', cval=0.0) / (window_size**2)
    
    # Calculate local standard deviation (roughness)
    roughness = np.sqrt(np.maximum(0, local_mean_squared - local_mean**2))
    
    # Restore NaN values where original data had NaNs
    roughness[mask] = np.nan
    
    return roughness


def calculate_local_mean(data: np.ndarray, 
                        window_size: int = 5) -> np.ndarray:
    """
    Calculate local background trend using a moving average filter.
    
    This feature provides context about the local background,
    helping distinguish anomalies from regional trends.
    
    Args:
        data: 2D numpy array of raster data
        window_size: Size of the moving average window (must be odd)
        
    Returns:
        2D array of local mean values with same shape as input
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    # Handle NaN values by using a masked array approach
    mask = np.isnan(data)
    data_clean = np.where(mask, 0, data)
    
    # Create kernel
    kernel = np.ones((window_size, window_size))
    
    # Calculate sum of values and count of non-NaN values in each window
    value_sum = ndimage.convolve(data_clean, kernel, mode='constant', cval=0.0)
    count_sum = ndimage.convolve(np.where(mask, 0, 1), kernel, mode='constant', cval=0.0)
    
    # Calculate local mean, avoiding division by zero
    local_mean = np.where(count_sum > 0, value_sum / count_sum, np.nan)
    
    # Restore NaN values where original data had NaNs or no valid neighbors
    local_mean[mask & (count_sum == 0)] = np.nan
    
    return local_mean


def calculate_all_features(data: np.ndarray, 
                          gradient_method: str = 'sobel',
                          roughness_window: int = 3,
                          mean_window: int = 5) -> dict:
    """
    Calculate all feature engineering transformations for a given data array.
    
    Args:
        data: 2D numpy array of raster data
        gradient_method: Method for gradient calculation ('sobel' or 'gradient')
        roughness_window: Window size for roughness calculation
        mean_window: Window size for local mean calculation
        
    Returns:
        Dictionary containing all computed features
    """
    features = {
        'gradient': calculate_gradient_magnitude(data, method=gradient_method),
        'roughness': calculate_local_roughness(data, window_size=roughness_window),
        'local_mean': calculate_local_mean(data, window_size=mean_window)
    }
    
    return features


def stack_features(gravity_data: np.ndarray, 
                  magnetic_data: Optional[np.ndarray] = None,
                  **kwargs) -> Tuple[np.ndarray, list]:
    """
    Stack all features into a multi-band array for machine learning.
    
    Args:
        gravity_data: 2D array of gravity data
        magnetic_data: Optional 2D array of magnetic data
        **kwargs: Additional arguments passed to calculate_all_features
        
    Returns:
        Tuple of (stacked_features_array, feature_names_list)
    """
    # Calculate gravity features
    gravity_features = calculate_all_features(gravity_data, **kwargs)
    
    # Stack features
    feature_stack = []
    feature_names = []
    
    # Add gravity features
    feature_stack.extend([
        gravity_data,
        gravity_features['gradient'],
        gravity_features['roughness'],
        gravity_features['local_mean']
    ])
    feature_names.extend([
        'gravity',
        'gravity_gradient',
        'gravity_roughness',
        'gravity_local_mean'
    ])
    
    # Add magnetic features if available
    if magnetic_data is not None and magnetic_data.size > 0:
        magnetic_features = calculate_all_features(magnetic_data, **kwargs)
        feature_stack.extend([
            magnetic_data,
            magnetic_features['gradient'],
            magnetic_features['roughness'],
            magnetic_features['local_mean']
        ])
        feature_names.extend([
            'magnetic',
            'magnetic_gradient',
            'magnetic_roughness',
            'magnetic_local_mean'
        ])
    
    # Stack into 3D array (bands, height, width)
    stacked_features = np.stack(feature_stack, axis=0)
    
    return stacked_features, feature_names


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Engineering Module")
    print("=" * 40)
    
    # Create sample data with some structure
    sample_data = np.random.randn(100, 100)
    sample_data[40:60, 40:60] += 5  # Add a "anomaly"
    
    # Calculate features
    features = calculate_all_features(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Gradient magnitude shape: {features['gradient'].shape}")
    print(f"Roughness shape: {features['roughness'].shape}")
    print(f"Local mean shape: {features['local_mean'].shape}")
    
    print("\nFeature statistics:")
    for name, feature in features.items():
        print(f"{name}: mean={np.nanmean(feature):.3f}, std={np.nanstd(feature):.3f}")