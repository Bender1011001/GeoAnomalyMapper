#!/usr/bin/env python3
"""
Phase 9: Visualization for GeoAnomalyMapper.

Generates visualization products for the spatial anomaly analysis:
1. Interactive HTML map (Folium)
2. Transparent PNG overlay
3. Google Earth KMZ file

Usage:
    python phase9_visualization.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import simplekml
from folium import plugins
from rasterio.warp import calculate_default_transform, reproject, Resampling

from utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to 0-1 range, handling NaNs.
    
    Args:
        arr: Input numpy array.
        
    Returns:
        Normalized array with values between 0 and 1.
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    
    if arr_max == arr_min:
        return np.zeros_like(arr)
        
    return (arr - arr_min) / (arr_max - arr_min)

def reproject_to_latlon(src_path: Path) -> Tuple[np.ndarray, dict, Tuple[float, float, float, float]]:
    """
    Read and reproject raster to EPSG:4326 (Lat/Lon).
    
    Args:
        src_path: Path to input GeoTIFF.
        
    Returns:
        Tuple containing:
        - Reprojected data array (band 1)
        - Updated profile/metadata
        - Bounds (min_lon, min_lat, max_lon, max_lat)
    """
    dst_crs = 'EPSG:4326'

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        destination = np.zeros((height, width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        # Calculate bounds: (left, bottom, right, top) -> (min_lon, min_lat, max_lon, max_lat)
        bounds = rasterio.transform.array_bounds(height, width, transform)
        
        return destination, kwargs, bounds

def create_png_overlay(
    data: np.ndarray,
    output_path: Path,
    colormap: str = 'plasma',
    opacity: float = 0.7
) -> None:
    """
    Generate a transparent PNG overlay from raster data.
    
    Args:
        data: Input 2D numpy array (normalized or raw).
        output_path: Path to save the PNG.
        colormap: Matplotlib colormap name.
        opacity: Opacity for valid data pixels (0-1).
    """
    # Normalize data
    data_norm = normalize_array(data)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    rgba_img = cmap(data_norm)
    
    # Set Alpha channel: 0 for NaNs, 'opacity' for valid data
    nan_mask = np.isnan(data)
    rgba_img[..., 3] = np.where(nan_mask, 0, opacity)
    
    # Convert to uint8 (0-255)
    img_uint8 = (rgba_img * 255).astype(np.uint8)
    
    # Save using matplotlib
    plt.imsave(output_path, img_uint8, format='png')
    logger.info(f"Saved PNG overlay: {output_path}")

def create_folium_map(
    png_path: Path,
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    layer_name: str = "Anomaly Map"
) -> None:
    """
    Create an interactive Folium map with the PNG overlay.
    
    Args:
        png_path: Path to the PNG overlay image.
        bounds: Bounding box (min_lon, min_lat, max_lon, max_lat).
        output_path: Path to save the HTML map.
        layer_name: Name of the overlay layer.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Calculate center
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Initialize Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
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
    
    # Image Overlay
    # Folium expects bounds as [[lat_min, lon_min], [lat_max, lon_max]]
    folium_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    
    # Pass the path as a string so Folium can find and read the file.
    # Folium will automatically base64 encode it into the HTML, making it standalone.
    image_url = str(png_path)
        
    folium.raster_layers.ImageOverlay(
        image=image_url,
        bounds=folium_bounds,
        name=layer_name,
        opacity=0.8,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.MousePosition().add_to(m)
    
    m.save(output_path)
    logger.info(f"Saved HTML map: {output_path}")

def create_kmz(
    png_path: Path,
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    name: str = "Anomaly Map"
) -> None:
    """
    Create a Google Earth KMZ file.
    
    Args:
        png_path: Path to the PNG overlay image.
        bounds: Bounding box (min_lon, min_lat, max_lon, max_lat).
        output_path: Path to save the KMZ file.
        name: Name of the KML document/overlay.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    kml = simplekml.Kml()
    kml.document.name = name
    
    overlay = kml.newgroundoverlay(name=name)
    # Point to the actual file path so simplekml can find and bundle it
    overlay.icon.href = str(png_path)
    
    # SimpleKML LatLonBox uses north, south, east, west
    overlay.latlonbox.north = max_lat
    overlay.latlonbox.south = min_lat
    overlay.latlonbox.east = max_lon
    overlay.latlonbox.west = min_lon
    
    # Save KMZ (simplekml automatically bundles local files referenced in href)
    kml.savekmz(str(output_path))
    logger.info(f"Saved KMZ file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Visualization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, help="Override input file path")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Get settings
    if args.input:
        input_path = Path(args.input)
    else:
        input_file = config.get('output', {}).get('anomaly_map', 'data/outputs/spatial_anomaly_v1.tif')
        
        # Handle relative paths in config
        input_path = Path(input_file)
        if not input_path.exists():
            # Try looking in data/outputs if not found directly
            alt_path = Path("data/outputs") / input_file
            if alt_path.exists():
                input_path = alt_path
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Output directory
    output_dir = input_path.parent
    base_name = input_path.stem
    
    # Visualization settings
    viz_config = config.get('visualization', {})
    colormap = viz_config.get('colormap', 'plasma')
    opacity = viz_config.get('opacity', 0.7)
    
    logger.info(f"Processing {input_path}...")
    
    # 1. Reproject to Lat/Lon
    try:
        data, meta, bounds = reproject_to_latlon(input_path)
    except Exception as e:
        logger.error(f"Error reprojecting raster: {e}")
        sys.exit(1)
        
    # 2. Generate PNG Overlay
    png_path = output_dir / f"{base_name}.png"
    create_png_overlay(data, png_path, colormap=colormap, opacity=opacity)
    
    # 3. Generate HTML Map
    html_path = output_dir / f"{base_name}.html"
    create_folium_map(png_path, bounds, html_path, layer_name=f"Anomaly: {base_name}")
    
    # 4. Generate KMZ
    kmz_path = output_dir / f"{base_name}.kmz"
    create_kmz(png_path, bounds, kmz_path, name=f"Anomaly: {base_name}")
    
    logger.info("Visualization complete.")
    logger.info(f"Outputs:\n - {html_path}\n - {png_path}\n - {kmz_path}")

if __name__ == "__main__":
    main()