#!/usr/bin/env python3
"""
Phase 6: Interactive Visualization for GeoAnomalyMapper.

Generates a standalone HTML map with:
- Satellite imagery base layer.
- Anomaly probability overlay (color-coded).
- Layer controls to toggle transparency.
- Automatic reprojection to Web Mercator (EPSG:4326) for browser compatibility.
"""

import os
import argparse
import logging
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import folium
from folium import plugins
import matplotlib.pyplot as plt
from matplotlib import cm

from project_paths import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_array(arr):
    """Normalize array to 0-1 range, handling NaNs."""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def get_overlay_image(tiff_path, max_dim=2048, colormap='plasma'):
    """
    Reads a GeoTIFF, reprojects to Lat/Lon (EPSG:4326), applies a colormap,
    and returns the bounds and a base64 encoded PNG string.
    """
    logger.info(f"Processing overlay: {tiff_path}")
    
    with rasterio.open(tiff_path) as src:
        # 1. Calculate transform to EPSG:4326 (Required for Folium)
        dst_crs = 'EPSG:4326'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        # 2. Downsample if the image is too large for the browser
        # We limit the max dimension to ensure the HTML loads quickly
        scale_factor = min(max_dim / width, max_dim / height)
        if scale_factor < 1.0:
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, 
                dst_width=width, dst_height=height
            )
            logger.info(f" - Downsampling to {width}x{height} for web viewing")
        
        # 3. Create destination array
        destination = np.zeros((height, width), dtype=np.float32)
        
        # 4. Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        # 5. Get Bounds [min_lat, min_lon, max_lat, max_lon]
        # rasterio bounds are (left, bottom, right, top) -> (min_lon, min_lat, max_lon, max_lat)
        # Folium expects [[min_lat, min_lon], [max_lat, max_lon]]
        bounds_rio = rasterio.transform.array_bounds(height, width, transform)
        bounds_folium = [[bounds_rio[1], bounds_rio[0]], [bounds_rio[3], bounds_rio[2]]]
        
    # 6. Apply Colormap
    # Handle NaNs (make them transparent)
    data_norm = normalize_array(destination)
    
    # Apply matplotlib colormap
    # 'plasma' is great because high values (yellow) pop against satellite maps
    cmap = plt.get_cmap(colormap)
    rgba_img = cmap(data_norm) 
    
    # Set Alpha channel: 0 for NaNs, 0.7 for valid data (transparency)
    nan_mask = np.isnan(destination)
    rgba_img[..., 3] = np.where(nan_mask, 0, 0.7)
    
    # 7. Convert to Base64 PNG
    # We must flip the array vertically because image coordinates (0,0 is top-left)
    # vs map coordinates are often handled differently, but Reproject usually handles orientation.
    # However, to save as PNG, we convert 0-1 float to 0-255 uint8
    img_uint8 = (rgba_img * 255).astype(np.uint8)
    
    img_buffer = BytesIO()
    plt.imsave(img_buffer, img_uint8, format='png')
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return bounds_folium, f"data:image/png;base64,{img_b64}"

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Interactive Visualization")
    parser.add_argument("--input", type=str, default=str(OUTPUTS_DIR / "spatial_anomaly_v2.tif"), help="Input anomaly TIFF")
    parser.add_argument("--output", type=str, default=str(OUTPUTS_DIR / "anomaly_map.html"), help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("Generating map...")
    
    # Get image data
    bounds, image_b64 = get_overlay_image(input_path)
    
    # Calculate center
    center_lat = (bounds[0][0] + bounds[1][0]) / 2
    center_lon = (bounds[0][1] + bounds[1][1]) / 2
    
    # Initialize Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None, # We add custom tiles below
        control_scale=True
    )
    
    # Base Layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite (Esri)',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Streets',
        overlay=False,
        control=True
    ).add_to(m)

    # Anomaly Overlay
    folium.raster_layers.ImageOverlay(
        image=image_b64,
        bounds=bounds,
        name='Anomaly Probability (Target Void)',
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.MousePosition().add_to(m)
    
    # Save
    m.save(args.output)
    logger.info(f"Map saved to: {args.output}")
    logger.info("Open this file in your web browser to view the results.")

if __name__ == "__main__":
    main()