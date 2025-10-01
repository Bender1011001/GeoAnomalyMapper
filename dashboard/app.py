"""
GeoAnomalyMapper Dashboard - Async API Version
================================================
Modernized Streamlit UI using FastAPI backend for asynchronous job handling.
Supports real-time progress tracking via polling.

Run with: streamlit run dashboard/app.py
Backend: uvicorn gam.api.main:app --reload
"""

import logging
import os
import sys
import time
import requests
from urllib.parse import urljoin
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import socket

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import folium.plugins
from folium.plugins import Draw, HeatMap
from branca.element import Figure, Html, MacroElement
import pandas as pd
import numpy as np
from scipy import interpolate
import pyvista as pv
try:
    import stpyvista.stpyvista as st_plotter
    render_plotter = st_plotter.stpyvista
except ImportError as e:
    try:
        # Try alternative import path
        from stpyvista import stpyvista
        render_plotter = stpyvista
    except ImportError:
        logger.warning(f"Failed to import stpyvista (error: {e}). 3D visualization features disabled. Install/upgrade stpyvista in your active environment for full support.")
        render_plotter = None
from matplotlib import cm

try:
    from .presets import get_all_presets, get_preset
except ImportError:
    # Handle case when running directly with Streamlit
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from presets import get_all_presets, get_preset

# Global render_plotter for 3D viz (set in import block above)

# Add GAM to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gam.core.exceptions import PipelineError, ConfigurationError
from gam.api.main import run_analysis

# API Configuration
API_TIMEOUT = 30  # seconds


@st.cache_data(ttl=300)  # Cache map for 5 min
def create_bbox_map(center_lat=0, center_lon=0, zoom=2):
    """Create interactive map for bbox drawing."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add Draw plugin for rectangle
    draw = Draw(
        position='topleft',
        shape_options={
            'rectangle': {
                'shapeOptions': {
                    'color': '#3388ff',
                    'weight': 2,
                    'fill': True,
                    'fillColor': '#3388ff',
                    'fillOpacity': 0.2
                }
            }
        },
        draw_options={
            'rectangle': {
                'show': True,
                'shapeOptions': {
                    'color': '#3388ff',
                    'weight': 2,
                    'fill': True,
                    'fillColor': '#3388ff',
                    'fillOpacity': 0.2
                }
            }
        }
    )
    draw.add_to(m)
    
    # Instructions popup
    folium.Marker(
        [center_lat, center_lon],
        popup=folium.Popup(
            """
            <div style="width: 300px;">
                <h4>Draw Analysis Region</h4>
                <p>1. Click and drag to draw a rectangle on the map</p>
                <p>2. The bounding box will be used for data ingestion</p>
                <p>3. Click 'Run Analysis' to start processing</p>
                <p><em>Tip: Start with small regions (e.g., 2x2 degrees) for faster results</em></p>
            </div>
            """,
            max_width=300
        ),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    return m


def extract_bbox_from_draw(data: dict) -> Optional[Tuple[float, float, float, float]]:
    """Extract bbox from drawn rectangle in st_folium data."""
    if not data or 'last_active_drawing' not in data or not data['last_active_drawing']:
        return None
    
    drawing = data['last_active_drawing']
    if drawing['geometry']['type'] != 'Rectangle':
        return None
    
    # Rectangle bounds: [[min_lat, min_lon], [max_lat, max_lon]]
    bounds = drawing['geometry']['coordinates'][0]  # Outer ring
    min_lat = min(coord[0] for coord in bounds)
    max_lat = max(coord[0] for coord in bounds)
    min_lon = min(coord[1] for coord in bounds)
    max_lon = max(coord[1] for coord in bounds)
    
    # Validate reasonable size (e.g., < 20 degrees)
    if (max_lat - min_lat > 20) or (max_lon - min_lon > 20):
        st.warning("Drawn region too large. Please select a smaller area (max 20° span).")
        return None
    
    return (min_lon, min_lat, max_lon, max_lat)


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def extract_anomalies(results_data: dict) -> list:
    """
    Extract and validate anomaly data from API results.
    
    Args:
        results_data: Dict from FastAPI response containing 'results' key with 'anomalies'
    
    Returns:
        List of validated anomaly dicts with 'lat', 'lon', 'confidence', 'type', 'intensity', 'id'
    """
    anomalies = []
    if not results_data or 'results' not in results_data or 'anomalies' not in results_data['results']:
        logger.warning("No anomalies found in results data")
        return anomalies
    
    raw_anomalies = results_data['results']['anomalies']
    if not isinstance(raw_anomalies, list):
        logger.warning("Anomalies data is not a list")
        return anomalies
    
    for idx, anomaly in enumerate(raw_anomalies):
        try:
            # Handle different formats: assume point data with lat/lon; skip polygons for now
            if isinstance(anomaly, dict):
                lat = anomaly.get('lat') or anomaly.get('latitude')
                lon = anomaly.get('lon') or anomaly.get('longitude')
                if lat is None or lon is None:
                    logger.debug(f"Skipping anomaly {idx}: missing coordinates")
                    continue
                
                # Validate coords (rough global bounds check)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    logger.debug(f"Skipping anomaly {idx}: invalid coordinates {lat}, {lon}")
                    continue
                
                validated = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'confidence': float(anomaly.get('confidence', 0.0)),
                    'type': anomaly.get('type', 'unknown'),
                    'intensity': float(anomaly.get('intensity', 0.0)),
                    'id': anomaly.get('id', f'anom_{idx:03d}'),
                    'modality': anomaly.get('modality', anomaly.get('detection_method', 'unknown'))
                }
                anomalies.append(validated)
            else:
                logger.debug(f"Skipping anomaly {idx}: not a dict")
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping anomaly {idx}: validation error {e}")
            continue
    
    logger.info(f"Extracted {len(anomalies)} valid anomalies from {len(raw_anomalies)} raw entries")
    return anomalies


def create_anomaly_map(results_data: dict, bbox: tuple) -> folium.Map:
    """
    Create Folium map with individual anomaly markers and popups.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Folium Map object
    """
    anomalies = extract_anomalies(results_data)
    if not anomalies:
        # Create empty map with message
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
        folium.Marker(
            [center_lat, center_lon],
            popup=folium.Popup("No anomalies detected in this region.", parse_html=True),
            icon=folium.Icon(color='lightgray', icon='info-sign')
        ).add_to(m)
        return m
    
    # Performance check: warn for large datasets
    if len(anomalies) > 100:
        logger.warning(f"Large dataset ({len(anomalies)} anomalies) - consider using clustering for better performance")
    
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    
    # Color mapping by type
    type_colors = {
        'gravity': 'blue',
        'magnetic': 'orange',
        'insar': 'green',
        'seismic': 'purple',
        'fusion': 'red',
        'unknown': 'gray'
    }
    
    for anomaly in anomalies:
        color = type_colors.get(anomaly['type'], 'gray')
        # Standardize marker size to reduce visual noise
        marker_radius = 10  # fixed radius for consistency
        
        # Create popup HTML
        popup_html = f"""
        <div style="width: 250px;">
            <h4 style="color: {color}; margin: 0 0 10px 0;">Anomaly {anomaly['id']}</h4>
            <p><strong>Type:</strong> {anomaly['type'].title()}</p>
            <p><strong>Coordinates:</strong> {anomaly['lat']:.4f}°N, {anomaly['lon']:.4f}°E</p>
            <p><strong>Confidence:</strong> {anomaly['confidence']:.2f}</p>
            <p><strong>Intensity:</strong> {anomaly['intensity']:.2f}</p>
            <p><strong>Modality:</strong> {anomaly['modality'].title()}</p>
        </div>
        """
        popup = folium.Popup(Html(popup_html, script=True), max_width=300)
        
        folium.CircleMarker(
            location=[anomaly['lat'], anomaly['lon']],
            radius=marker_radius,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Subtle single legend (reduced visual noise)
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 12px; left: 12px; z-index: 9999;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:600; margin-bottom:6px;">Anomaly Types</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:blue;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Gravity</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:orange;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Magnetic</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:green;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>InSAR</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:purple;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Seismic</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:red;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Fusion</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def create_anomaly_heatmap(results_data: dict, bbox: tuple) -> folium.Map:
    """
    Create Folium map with heatmap overlay using anomaly confidence as weight.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Folium Map object
    """
    anomalies = extract_anomalies(results_data)
    if not anomalies:
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
        folium.Marker(
            [center_lat, center_lon],
            popup=folium.Popup("No anomalies detected for heatmap.", parse_html=True),
            icon=folium.Icon(color='lightgray', icon='info-sign')
        ).add_to(m)
        return m
    
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    
    # Prepare heatmap data: [lat, lon, weight] where weight = confidence * intensity (normalized)
    max_intensity = max(a['intensity'] for a in anomalies) if anomalies else 1
    max_conf = max(a['confidence'] for a in anomalies) if anomalies else 1
    heat_data = []
    for anomaly in anomalies:
        weight = (anomaly['confidence'] * anomaly['intensity']) / (max_conf * max_intensity) if max_conf * max_intensity > 0 else 0
        heat_data.append([anomaly['lat'], anomaly['lon'], weight])
    
    # Add heatmap with perceptual Viridis gradient to improve contrast
    viridis_colors = [cm.viridis(x) for x in np.linspace(0, 1, 6)]
    def _rgba_to_hex(rgba):
        r, g, b = [int(255 * c) for c in rgba[:3]]
        return f'#{r:02x}{g:02x}{b:02x}'
    viridis_gradient = {i/5: _rgba_to_hex(c) for i, c in enumerate(viridis_colors)}
    HeatMap(heat_data, gradient=viridis_gradient, min_opacity=0.4, radius=15, blur=15).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Subtle single legend for heatmap (Viridis)
    heatmap_legend = '''
    <div style="
        position: fixed;
        bottom: 12px; right: 12px; z-index: 9999;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        min-width: 180px;
    ">
      <div style="font-weight:600; margin-bottom:6px;">Heat Intensity</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span>Low</span>
        <span style="flex:1;height:10px;display:block;background:linear-gradient(to right,#440154,#3b528b,#21918c,#5ec962,#fde725);border:1px solid rgba(0,0,0,0.1);"></span>
        <span>High</span>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(heatmap_legend))
    
    return m


def create_anomaly_clusters(results_data: dict, bbox: tuple) -> folium.Map:
    """
    Create Folium map with clustered anomaly markers, color-coded by type.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Folium Map object
    """
    anomalies = extract_anomalies(results_data)
    if not anomalies:
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
        folium.Marker(
            [center_lat, center_lon],
            popup=folium.Popup("No anomalies detected for clustering.", parse_html=True),
            icon=folium.Icon(color='lightgray', icon='info-sign')
        ).add_to(m)
        return m
    
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    
    # Color mapping
    type_colors = {
        'gravity': 'blue',
        'magnetic': 'orange',
        'insar': 'green',
        'seismic': 'purple',
        'fusion': 'red',
        'unknown': 'gray'
    }
    
    # Create marker cluster
    marker_cluster = folium.plugins.MarkerCluster().add_to(m)
    
    for anomaly in anomalies:
        color = type_colors.get(anomaly['type'], 'gray')
        
        # Popup HTML same as individual map
        popup_html = f"""
        <div style="width: 250px;">
            <h4 style="color: {color}; margin: 0 0 10px 0;">Anomaly {anomaly['id']}</h4>
            <p><strong>Type:</strong> {anomaly['type'].title()}</p>
            <p><strong>Coordinates:</strong> {anomaly['lat']:.4f}°N, {anomaly['lon']:.4f}°E</p>
            <p><strong>Confidence:</strong> {anomaly['confidence']:.2f}</p>
            <p><strong>Intensity:</strong> {anomaly['intensity']:.2f}</p>
            <p><strong>Modality:</strong> {anomaly['modality'].title()}</p>
        </div>
        """
        popup = folium.Popup(Html(popup_html, script=True), max_width=300)
        
        folium.CircleMarker(
            location=[anomaly['lat'], anomaly['lon']],
            radius=10,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Subtle single legend (same as individual map)
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 12px; left: 12px; z-index: 9999;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:600; margin-bottom:6px;">Anomaly Types</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:blue;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Gravity</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:orange;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Magnetic</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:green;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>InSAR</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:purple;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Seismic</div>
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:red;margin-right:6px;border:1px solid rgba(0,0,0,0.1);"></span>Fusion</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def extract_3d_model(results_data: dict, bbox: tuple) -> Optional[dict]:
    """
    Extract and process 3D fused model data from results.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Dict with 'grid' (UniformGrid), 'values' (np.array), or None if no 3D data
    """
    if not results_data or 'results' not in results_data:
        return None
    
    fused = results_data['results'].get('fused', {})
    if not fused:
        return None
    
    try:
        if 'mesh' in fused:
            # Assume VTK-compatible dict: points, cells, point_data with 'anomaly'
            points = np.array(fused['mesh']['points']).reshape(-1, 3)
            values = np.array(fused['mesh']['point_data']['anomaly'])
            mesh = pv.PolyData(points)
            mesh['anomaly'] = values
            # Decimate for performance if large
            if len(points) > 100000:
                mesh = mesh.decimate(0.9)
            return {'mesh': mesh, 'type': 'mesh'}
        
        elif 'volume' in fused:
            # Assume structured grid data
            dims = fused['volume']['dimensions']  # (nx, ny, nz)
            origin = fused['volume']['origin']
            spacing = fused['volume']['spacing']
            values = np.array(fused['volume']['data']).reshape(dims)
            grid = pv.UniformGrid(dimensions=dims, origin=origin, spacing=spacing)
            grid['anomaly'] = values
            return {'grid': grid, 'type': 'grid'}
        
        else:
            # Fallback: interpolate anomalies to 3D grid using bbox
            anomalies = extract_anomalies(results_data)
            if not anomalies:
                return None
            
            # Simple 3D grid: assume depth range 0-1000m, resolution based on bbox
            lon_min, lat_min, lon_max, lat_max = bbox
            dlon = (lon_max - lon_min) / 50  # 50x50 horizontal grid
            dlat = (lat_max - lat_min) / 50
            depth_levels = np.linspace(0, 1000, 20)  # 20 depth levels
            
            # Create coordinate grid (simplified, no real interpolation for demo but production-ready structure)
            lons = np.linspace(lon_min, lon_max, 50)
            lats = np.linspace(lat_min, lat_max, 50)
            depths = depth_levels
            Lon, Lat, Depth = np.meshgrid(lons, lats, depths, indexing='ij')
            
            # Dummy values: gaussian blobs around anomalies (replace with real scipy.interpolate.griddata)
            values = np.zeros(Lon.shape)
            for i, anomaly in enumerate(anomalies[:5]):  # Limit for perf
                dist = np.sqrt((Lon - anomaly['lon'])**2 + (Lat - anomaly['lat'])**2 + (Depth - 500)**2)
                values += anomaly['intensity'] * np.exp(-dist**2 / (2 * 100**2))
            
            grid = pv.UniformGrid(dimensions=(50, 50, 20), origin=(lon_min, lat_min, 0), spacing=(dlon, dlat, 50))
            grid['anomaly'] = values
            return {'grid': grid, 'type': 'interpolated'}
    
    except Exception as e:
        logger.error(f"Error extracting 3D model: {e}")
        return None


def create_3d_volume_viewer(results_data: dict, bbox: tuple) -> pv.Plotter:
    """
    Create PyVista plotter for volume rendering of subsurface anomalies.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Configured PyVista Plotter
    """
    model_data = extract_3d_model(results_data, bbox)
    if not model_data:
        plotter = pv.Plotter(off_screen=False)
        plotter.add_text("No 3D volume data available", font_size=12, color='white')
        return plotter
    
    plotter = pv.Plotter(off_screen=False, window_size=[800, 600])
    
    if model_data['type'] == 'mesh':
        # For mesh, use glyph or extract surface, but for volume fallback to grid
        grid = model_data['mesh'].delaunay_3d()  # Triangulate to volume
        grid['anomaly'] = model_data['mesh']['anomaly']  # Approximate
    else:
        grid = model_data['grid']
    
    # Volume rendering
    grid.set_active_scalars('anomaly')
    vol = plotter.add_volume(grid, cmap='viridis', opacity='linear', opacity_unit=0.5)
    
    # Axes and orientation
    plotter.add_axes()
    plotter.camera_position = 'iso'
    
    return plotter


def create_3d_slice_viewer(results_data: dict, bbox: tuple) -> pv.Plotter:
    """
    Create PyVista plotter with interactive slice planes through 3D model.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Configured PyVista Plotter with slices
    """
    model_data = extract_3d_model(results_data, bbox)
    if not model_data:
        plotter = pv.Plotter(off_screen=False)
        plotter.add_text("No 3D slice data available", font_size=12, color='white')
        return plotter
    
    plotter = pv.Plotter(off_screen=False, window_size=[800, 600])
    
    if model_data['type'] == 'mesh':
        grid = model_data['mesh'].delaunay_3d()
        grid['anomaly'] = model_data['mesh']['anomaly']
    else:
        grid = model_data['grid']
    
    grid.set_active_scalars('anomaly')
    
    # Add three orthogonal slices
    slice_x = grid.slice(normal=[1, 0, 0], origin=grid.center)
    slice_y = grid.slice(normal=[0, 1, 0], origin=grid.center)
    slice_z = grid.slice(normal=[0, 0, 1], origin=grid.center)
    
    plotter.add_mesh(slice_x, cmap='plasma', show_edges=True, opacity=0.8)
    plotter.add_mesh(slice_y, cmap='plasma', show_edges=True, opacity=0.8)
    plotter.add_mesh(slice_z, cmap='plasma', show_edges=True, opacity=0.8)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    
    return plotter


def create_3d_isosurface_viewer(results_data: dict, bbox: tuple, threshold: float = 0.5) -> pv.Plotter:
    """
    Create PyVista plotter with isosurfaces at specified thresholds.
    
    Args:
        results_data: Dict from API
        bbox: (min_lon, min_lat, max_lon, max_lat)
        threshold: Isosurface level (0-1 normalized)
    
    Returns:
        Configured PyVista Plotter with isosurfaces
    """
    model_data = extract_3d_model(results_data, bbox)
    if not model_data:
        plotter = pv.Plotter(off_screen=False)
        plotter.add_text("No 3D isosurface data available", font_size=12, color='white')
        return plotter
    
    plotter = pv.Plotter(off_screen=False, window_size=[800, 600])
    
    if model_data['type'] == 'mesh':
        grid = model_data['mesh'].delaunay_3d()
        grid['anomaly'] = model_data['mesh']['anomaly']
    else:
        grid = model_data['grid']
    
    grid.set_active_scalars('anomaly')
    
    # Normalize values for threshold
    values = grid['anomaly']
    min_val, max_val = values.min(), values.max()
    norm_threshold = min_val + threshold * (max_val - min_val)
    
    # Generate isosurface
    iso = grid.contour([norm_threshold])
    plotter.add_mesh(iso, cmap='coolwarm', opacity=0.7, show_edges=True)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    
    return plotter


def check_api_connection() -> bool:
    """Check if FastAPI backend is available."""
    api_url = st.session_state.api_base_url
    logger.info(f"Checking API connection to: {api_url}")
    full_url = urljoin(api_url, "/")
    logger.info(f"Full URL for check: {full_url}")
    try:
        response = requests.get(full_url, timeout=5)
        response.raise_for_status()
        logger.info("API connection successful")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"API connection failed: {e}")
        return False


def start_analysis_job(
    bbox: Tuple[float, float, float, float],
    modalities: List[str],
    resolution: float,
    output_dir: str,
    config_path: Optional[str],
    verbose: bool
) -> Optional[str]:
    """
    Start an analysis job via FastAPI backend.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        modalities: List of data types
        resolution: Grid resolution in meters (logged but not passed to API)
        output_dir: Base output directory
        config_path: Path to config file
        verbose: Verbose flag (logged but not passed to API)

    Returns:
        job_id if successful, None otherwise
    """
    api_url = st.session_state.api_base_url
    logger.info(f"Starting analysis job using API base URL: {api_url}")
    full_url = urljoin(api_url, "/analysis")
    logger.info(f"Full POST URL for analysis: {full_url}")
    try:
        # Prepare request matching AnalysisRequest schema
        request_data = {
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],  # min_lon, min_lat, max_lon, max_lat
            "modalities": modalities,
            "output_dir": output_dir,
            "config_path": config_path,
            "verbose": verbose
        }
        logger.info(f"Starting job with bbox={bbox}, modalities={modalities}")

        response = requests.post(
            full_url,
            json=request_data,
            timeout=API_TIMEOUT
        )
        logger.info(f"API response status: {response.status_code}")
        response.raise_for_status()
        job_id = response.json()["job_id"]
        logger.info(f"Job started successfully: {job_id}")
        return job_id
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to start analysis job: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None
    except KeyError as e:
        error_msg = f"Invalid response from API: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the current status of a job."""
    api_url = st.session_state.api_base_url
    try:
        response = requests.get(
            urljoin(api_url, f"/analysis/{job_id}/status"),
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        status_data = response.json()
        logger.debug(f"Job {job_id} status: {status_data['status']}, progress: {status_data['progress']}")
        return status_data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"Job {job_id} not found. It may have expired."
            logger.warning(error_msg)
            st.warning(error_msg)
        else:
            error_msg = f"HTTP error getting job status: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error getting job status: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None


def get_job_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the results of a completed job."""
    api_url = st.session_state.api_base_url
    try:
        response = requests.get(
            urljoin(api_url, f"/analysis/{job_id}/results"),
            timeout=API_TIMEOUT
        )
        if response.status_code == 425:
            logger.debug(f"Job {job_id} not ready yet")
            return None
        response.raise_for_status()
        results_data = response.json()
        logger.info(f"Job {job_id} results retrieved successfully")
        return results_data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 425:
            return None
        elif e.response.status_code == 404:
            error_msg = f"Job {job_id} not found."
            logger.error(error_msg)
            st.error(error_msg)
        else:
            error_msg = f"HTTP error getting job results: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error getting job results: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None


def initialize_session_state():
    """Initialize Streamlit session state for async job tracking."""
    defaults = {
        'current_job_id': None,
        'job_status': None,
        'job_progress': 0.0,
        'job_results': None,
        'api_available': False,  # Will be set after URL config
        'is_running': False,
        'job_history': [],
        'last_poll_time': 0,
        'job_start_time': None,  # For timeout enforcement
        'pipeline': None,  # For potential fallback
        'results': None,   # Legacy
        'status': "Ready",
        'bbox': None,
        'modalities': [],
        'selected_preset': 'Custom Configuration',
        'preset_applied': False,
        'preset_defaults': {},
        'api_base_url': os.getenv("GAM_API_URL", "http://localhost:8000"),
        'api_port': 8000,
        'dashboard_port': 8501
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Initial API check after URL set
    st.session_state.api_available = check_api_connection()


def poll_job_status():
    """Fetch job status once and update session state.

    Returns:
        bool: True if job is still running, False otherwise.
    """
    if not st.session_state.current_job_id or not st.session_state.is_running:
        return False

    job_id = st.session_state.current_job_id
    status_data = get_job_status(job_id)
    if not status_data:
        # Treat missing status as terminal; avoid looping forever
        st.session_state.is_running = False
        st.session_state.job_status = "UNKNOWN"
        return False

    st.session_state.job_status = status_data.get('status')
    st.session_state.job_progress = status_data.get('progress', 0.0)
    st.session_state.status = status_data.get('stage', st.session_state.status)

    if st.session_state.job_status == 'COMPLETED':
        results = get_job_results(job_id)
        if results:
            st.session_state.job_results = results
        st.session_state.is_running = False
        if job_id not in st.session_state.job_history:
            st.session_state.job_history.append(job_id)
        return False

    if st.session_state.job_status == 'FAILED':
        st.session_state.is_running = False
        st.error(f"Job failed: {status_data.get('message', 'Unknown error')}")
        return False

    return True


def retry_job_start():
    """Callback for retry button."""
    st.session_state.current_job_id = None
    st.session_state.job_status = None
    st.session_state.job_progress = 0.0
    st.session_state.job_results = None
    st.session_state.is_running = False
    st.rerun()


def main():
    """Main dashboard application with async API integration."""
    # Page configuration
    st.set_page_config(
        page_title="GeoAnomalyMapper Dashboard",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("🗺️ GeoAnomalyMapper Dashboard")
    st.markdown("""
    Modernized interface with asynchronous processing via FastAPI backend.
    Start the backend with: `uvicorn gam.api.main:app --reload`
    """)

    # Initialize session state
    initialize_session_state()

    # Inject modern CSS theme (non-intrusive styling only)
    st.markdown(
        """
        <style>
        :root {
            --bg: #0B0F14;
            --panel: #111827;
            --text: #E5E7EB;
            --muted: #9CA3AF;
            --primary: #60A5FA;
            --primary-600: #2563EB;
            --border: rgba(255,255,255,0.08);

            --spacing-1: 4px;
            --spacing-2: 8px;
            --spacing-3: 12px;
            --spacing-4: 16px;
            --spacing-5: 20px;
            --spacing-6: 24px;
            --spacing-8: 32px;

            --font-base: 16px;
            --line-height-base: 1.5;

            --radius-card: 16px;
            --radius-control: 8px;

            --shadow-subtle: 0 1px 0 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(0,0,0,0.6);
            --transition: all 150ms ease-out;
        }
        html, body, .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            font-size: var(--font-base);
            line-height: var(--line-height-base);
            letter-spacing: -0.01em;
        }
        .block-container {
            padding-top: var(--spacing-6);
            padding-bottom: var(--spacing-6);
        }
        .page-shell {
            max-width: 1280px;
        }
        /* Smooth transitions for dynamic UI elements */
        [data-testid="stMetric"] {
            transition: var(--transition);
        }
        [data-testid="stProgress"] {
            transition: var(--transition);
        }
        .stButton > button {
            transition: var(--transition);
        }
            margin: 0 auto;
            padding: var(--spacing-6);
        }
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: var(--spacing-5);
            box-shadow: var(--shadow-subtle);
        }
        /* Buttons */
        .stButton > button {
            height: 40px;
            padding: 0 1rem;
            border-radius: var(--radius-control);
            border: 1px solid transparent;
            background-color: var(--primary);
            color: rgba(0,0,0,0.9);
            font-size: 0.875rem;
            font-weight: 600;
            transition: var(--transition);
        }
        .stButton > button:hover:not(:disabled) {
            background-color: var(--primary-600);
            color: #FFF;
        }
        .stButton > button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
        }
        /* Inputs */
        input[type="text"], input[type="number"], input[type="search"], textarea, select {
            height: 40px;
            border-radius: var(--radius-control);
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            padding: 0 12px;
            color: var(--text);
            outline: none;
            transition: var(--transition);
        }
        input[type="text"]:focus, input[type="number"]:focus, input[type="search"]:focus, textarea:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(96,165,250,0.35);
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 10px 10px 0 0;
            border: 1px solid var(--border);
        }
        /* Metrics */
        [data-testid="stMetricDelta"] { font-weight: 600; }
        /* Map container */
        .map-container {
            padding: var(--spacing-4);
            border-radius: var(--radius-card);
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid var(--border);
            box-shadow: var(--shadow-subtle);
        }
        /* Headers */
        h1, h2, h3 { letter-spacing: 0.2px; color: var(--text); }
        /* Sidebar spacing */
        [data-testid="stSidebar"] > div { padding-top: 0.5rem; }
        /* Responsive tweaks */
        @media (max-width: 768px) {
            .page-shell { padding: var(--spacing-4); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # API Connectivity Check and Warning
    if not st.session_state.api_available:
        st.warning(f"""
        🚨 **Backend API not available at {st.session_state.api_base_url}.**
        Start the FastAPI server on the specified port or update the API URL above.
        Async features disabled. Contact admin for sync fallback.
        """)
        st.session_state.is_running = False

    # Sidebar: Essential controls only
    st.sidebar.header("⚙️ Controls")

    # API URL Configuration (keep in sidebar)
    api_url_input = st.sidebar.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        key="api_url_input",
        help="Override the API endpoint (e.g., http://localhost:8001 for custom ports)"
    )
    if api_url_input != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url_input
        st.session_state.api_available = check_api_connection()
        st.rerun()

    api_port = st.sidebar.number_input(
        "API Port",
        value=st.session_state.api_port,
        min_value=1,
        max_value=65535,
        key="api_port_input",
        help="Port for the FastAPI backend (default: 8000)"
    )
    dashboard_port = st.sidebar.number_input(
        "Dashboard Port",
        value=st.session_state.dashboard_port,
        min_value=1,
        max_value=65535,
        key="dashboard_port_input",
        help="Port for the Streamlit dashboard (default: 8501)"
    )

    if api_port != st.session_state.api_port:
        st.session_state.api_port = api_port
        st.session_state.api_base_url = f"http://localhost:{api_port}"
        st.session_state.api_available = check_api_connection()
        st.rerun()

    if st.sidebar.button("Check Ports"):
        api_in_use = is_port_in_use(api_port)
        dashboard_in_use = is_port_in_use(dashboard_port)
        if api_in_use:
            st.sidebar.error(f"API Port {api_port}: In use")
        else:
            st.sidebar.success(f"API Port {api_port}: Available")
        if dashboard_in_use:
            st.sidebar.error(f"Dashboard Port {dashboard_port}: In use")
        else:
            st.sidebar.success(f"Dashboard Port {dashboard_port}: Available")

    # PageShell wrapper
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    # Tabs for clean workflow separation
    tab_params, tab_analysis, tab_results = st.tabs([
        "📁 Upload & Parameters",
        "🚀 Analysis",
        "📊 Results & Visualization"
    ])

    # -------------------- Tab 1: Upload & Parameters --------------------
    with tab_params:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Two-column layout: left (presets & region), right (parameters)
        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("📦 Presets")
            preset_options = ['Custom Configuration'] + list(get_all_presets().keys())
            selected_preset = st.selectbox(
                "Analysis Preset",
                options=preset_options,
                index=0 if st.session_state.get('selected_preset') == 'Custom Configuration'
                    else preset_options.index(st.session_state['selected_preset']),
                help="Select a preset for common use cases or 'Custom Configuration' for manual setup."
            )
            st.session_state.selected_preset = selected_preset

            if selected_preset != 'Custom Configuration':
                preset_config = get_preset(selected_preset)
                if preset_config:
                    st.session_state.preset_applied = True
                    st.session_state.preset_defaults = preset_config
                    st.info(f"**{preset_config['description']}**")

                    with st.expander("Preset Details", expanded=True):
                        st.markdown("**Typical Use Cases:**")
                        for case in preset_config['typical_use_cases']:
                            st.markdown(f"• {case}")
                        st.info(f"**Recommended BBox Size:** {preset_config['typical_bbox_size']}")
                        st.info(f"**Analysis Focus:** {preset_config['analysis_focus']}")
                        st.markdown("**Note:** You can override any parameters on the right.")
                else:
                    st.session_state.preset_applied = False
                    st.warning("Preset not found. Using custom configuration.")
            else:
                st.session_state.preset_applied = False
                st.session_state.preset_defaults = {}

            if selected_preset != st.session_state.get('last_preset', ''):
                st.session_state.last_preset = selected_preset
                st.rerun()

            st.subheader("🗺️ Region Selection")
            with st.container():
                m = folium.Map(
                    location=[30.0, 31.2],  # Giza default
                    zoom_start=8,
                    tiles="OpenStreetMap"
                )
                draw = folium.plugins.Draw(
                    export=False,
                    position='topleft',
                    draw_options={
                        'rectangle': {'shapeOptions': {'color': '#ff7800', 'weight': 2, 'fillOpacity': 0.3}},
                        'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False
                    }
                )
                m.add_child(draw)
                st.markdown('<div class="map-container">', unsafe_allow_html=True)
                map_data = st_folium(m, height=600, key="bbox_map", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if map_data and 'last_active_drawing' in map_data:
                drawing = map_data['last_active_drawing']
                if drawing and drawing.get('geometry') and drawing['geometry'].get('type') == 'Rectangle':
                    bounds = drawing['geometry']['coordinates'][0]
                    min_lat, min_lon = min(bounds[0][0], bounds[1][0]), min(bounds[0][1], bounds[1][1])
                    max_lat, max_lon = max(bounds[0][0], bounds[1][0]), max(bounds[0][1], bounds[1][1])
                    bbox_str = f"{min_lon:.1f},{min_lat:.1f},{max_lon:.1f},{max_lat:.1f}"
                    st.session_state.bbox_str = bbox_str
                    st.success(f"Selected: {bbox_str}")

            # Fallback bbox input
            bbox_str = st.session_state.get('bbox_str', st.text_input(
                "Bounding Box",
                value="29.0,29.5,31.5,31.0",
                help="min_lon,min_lat,max_lon,max_lat"
            ))
            st.session_state.bbox_str = bbox_str

            # Parse bbox
            bbox = None
            try:
                bbox_parts = [float(p.strip()) for p in bbox_str.split(',')]
                if len(bbox_parts) != 4:
                    raise ValueError("Must have 4 values")
                min_lon, min_lat, max_lon, max_lat = bbox_parts
                bbox = (min_lon, min_lat, max_lon, max_lat)
                st.session_state.bbox = bbox
            except ValueError as e:
                st.error(f"Invalid bbox: {e}")

        with right:
            st.subheader("⚙️ Parameters")
            # Modalities
            preset_modalities_str = ','.join(
                st.session_state.preset_defaults.get('default_modalities', ['gravity', 'magnetic'])
            ) if st.session_state.preset_applied else "gravity,magnetic"

            default_modal_str = st.text_input(
                "Modalities",
                value=preset_modalities_str,
                help="Comma-separated list (e.g., gravity,magnetic,insar,seismic)"
            )

            # Override indicator
            if st.session_state.preset_applied:
                user_modalities = [m.strip() for m in default_modal_str.split(',') if m.strip()]
                preset_mods = st.session_state.preset_defaults.get('default_modalities', [])
                if set(user_modalities) != set(preset_mods):
                    st.warning("🔧 Manual override: Modalities differ from preset recommendation")

            try:
                selected_modalities = [m.strip() for m in default_modal_str.split(',') if m.strip()]
                if not selected_modalities:
                    selected_modalities = ['gravity', 'magnetic']
                st.session_state.modalities = selected_modalities
            except:
                selected_modalities = ['gravity', 'magnetic']

            # Resolution (logged, not passed to API per schema)
            resolution = st.slider(
                "Grid Resolution (meters)",
                min_value=100.0, max_value=5000.0, value=1000.0, step=100.0
            )

            # Output directory
            output_dir = st.text_input("Output Directory", value="results/dashboard")

            # Config file
            config_file_input = st.text_input("Config File", value="config.yaml")
            config_path = config_file_input if config_file_input != "config.yaml" else None

            # Verbose
            verbose = st.checkbox("Verbose Logging", value=False)

            run_synchronously = st.checkbox("Run Synchronously (CLI-like, no API needed)", key="run_synchronously")
            if run_synchronously:
                st.warning("Sync mode blocks the UI until the analysis is complete. It is best used for small regions.")

            with st.expander("Notes", expanded=False):
                st.info("Use the sidebar 'Run Analysis' to start processing. Tabs separate configuration, progress, and results for clarity.")

        st.markdown('</div>', unsafe_allow_html=True)
    # Compute disabled state after parameters are defined
    disabled = st.session_state.is_running or (not st.session_state.get('run_synchronously', False) and not st.session_state.api_available) or st.session_state.get('bbox') is None

    # Job History in Sidebar (keep)
    with st.sidebar.expander("📋 Job History", expanded=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.job_history:
            for job_id in st.session_state.job_history[-5:]:
                status_color = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "orange"}.get(
                    st.session_state.job_status if job_id == st.session_state.current_job_id else "unknown", "gray"
                )
                st.markdown(f"**{job_id[:8]}...** : {status_color}")
                if st.button(f"View Results {job_id[:8]}", key=f"view_{job_id}"):
                    results = get_job_results(job_id)
                    if results:
                        st.session_state.job_results = results
                        st.session_state.current_job_id = job_id
                        st.rerun()
        else:
            st.info("No completed jobs yet.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.info("**GeoAnomalyMapper v1.0.0**")
    # Run Button (kept in sidebar)
    if st.sidebar.button("🚀 Run Analysis", type="primary", disabled=disabled):
        if st.session_state.get('run_synchronously', False):
            # Synchronous execution mimicking CLI
            try:
                bbox_tuple = st.session_state.get('bbox')
                if not bbox_tuple:
                    st.error("No bounding box selected for analysis.")
                    st.stop()

                bbox_str = f"{bbox_tuple[0]:.6f},{bbox_tuple[1]:.6f},{bbox_tuple[2]:.6f},{bbox_tuple[3]:.6f}"
                output_path = Path(output_dir)
                config_p = config_file_input if config_file_input and config_file_input != "config.yaml" else None

                results = run_analysis(
                    bbox_str=bbox_str,
                    modalities=st.session_state.get('modalities', ['gravity', 'magnetic']),
                    output_dir=output_path,
                    config_path=config_p,
                    verbose=verbose
                )

                # Collect output files for download
                output_files = {}
                for file_path in output_path.rglob('*'):
                    if file_path.is_file():
                        output_files[file_path.name] = str(file_path)

                st.session_state.job_results = {'results': results, 'output_files': output_files}
                job_id = f"sync_{int(time.time())}"
                st.session_state.current_job_id = job_id
                st.session_state.is_running = False
                st.session_state.job_status = "COMPLETED"
                st.session_state.job_progress = 1.0
                # Sync jobs don't go to API history
                st.success(f"Synchronous analysis completed successfully!")
                st.rerun()

            except (PipelineError, ConfigurationError) as e:
                error_msg = f"Synchronous analysis failed: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                st.session_state.is_running = False
                st.session_state.job_status = "FAILED"

            except Exception as e:
                error_msg = f"Unexpected error in synchronous analysis: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
                st.session_state.is_running = False
                st.session_state.job_status = "FAILED"

        else:
            # Asynchronous execution via API
            job_id = start_analysis_job(
                bbox=st.session_state.get('bbox'),
                modalities=st.session_state.get('modalities', ['gravity', 'magnetic']),
                resolution=resolution,
                output_dir=output_dir,
                config_path=config_path,
                verbose=verbose
            )
            if job_id:
                st.session_state.current_job_id = job_id
                st.session_state.is_running = True
                st.session_state.job_status = "QUEUED"
                st.session_state.job_progress = 0.0
                st.session_state.job_start_time = time.time()  # Start timeout clock
                st.success(f"Analysis started! Job ID: {job_id}")
                st.rerun()

    # Optimized polling: timer-based refresh every 5s without blocking sleep
    if st.session_state.is_running and st.session_state.api_available:
        current_time = time.time()
        # Enforce 300s timeout
        if st.session_state.job_start_time and (current_time - st.session_state.job_start_time > 300):
            st.session_state.job_status = "FAILED"
            st.session_state.is_running = False
            st.session_state.job_progress = 0.0
            st.error("Job timed out after 5 minutes. Please retry.")
            st.session_state.job_start_time = None
        elif current_time - st.session_state.last_poll_time >= 5:
            if poll_job_status():  # Still running after poll
                st.session_state.last_poll_time = current_time
                st.rerun()  # Trigger refresh to update placeholders
            else:
                st.session_state.last_poll_time = current_time
                # No rerun needed; terminal state will show on next interaction

    # Retry Button if Failed (keep in sidebar)
    if st.session_state.job_status == "FAILED" and not st.session_state.is_running:
        if st.sidebar.button("🔄 Retry Analysis", disabled=disabled):
            retry_job_start()

    # -------------------- Tab 2: Analysis --------------------
    with tab_analysis:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.current_job_id and st.session_state.api_available:
            st.header("📊 Job Progress")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_placeholder = st.empty()
                status_placeholder.metric("Status", st.session_state.job_status, delta=None)
            with col2:
                progress_metric_placeholder = st.empty()
                progress_bar_placeholder = st.empty()
                progress_metric_placeholder.metric("Progress", f"{st.session_state.job_progress * 100:.1f}%")
                progress_bar_placeholder.progress(st.session_state.job_progress)
            with col3:
                stage_placeholder = st.empty()
                stage_placeholder.metric("Current Stage", st.session_state.status or "N/A")
            with col4:
                st.empty()

            if st.session_state.job_progress > 0:
                st.info(f"Job ID: {st.session_state.current_job_id} | Estimated time remaining: ~{(1 - st.session_state.job_progress) * 10:.1f} min (approx)")
        else:
            st.info("Use the sidebar to start an analysis. Progress will appear here.")

        st.markdown('</div>', unsafe_allow_html=True)
    # -------------------- Tab 3: Results & Visualization --------------------
    with tab_results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.job_results:
            st.header("✅ Analysis Results")
            results = st.session_state.job_results

            # Key Metrics
            col1, col2 = st.columns(2)
            with col1:
                anomaly_count = len(results.get('results', {}).get('anomalies', []))
                st.metric("Anomaly Count", anomaly_count)
            with col2:
                st.metric("Status", "Completed")

            # Downloads
            st.subheader("📦 Output Files")
            output_files = results.get('output_files', {})
            for file_name, file_path in output_files.items():
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=file.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                else:
                    st.warning(f"File {file_name} not found at {file_path}")

            # Interactive Map Viewer (UI only; uses existing functions)
            bbox = st.session_state.get('bbox')
            if bbox:
                st.subheader("🗺️ Interactive Map")
                map_mode = st.radio("Map mode", options=["Markers", "Heatmap", "Clusters"], horizontal=True, index=0)
                try:
                    if map_mode == "Markers":
                        fmap = create_anomaly_map(results, bbox)
                    elif map_mode == "Heatmap":
                        fmap = create_anomaly_heatmap(results, bbox)
                    else:
                        fmap = create_anomaly_clusters(results, bbox)
                    st.markdown('<div class="map-container">', unsafe_allow_html=True)
                    st_folium(fmap, height=600, use_container_width=True, key=f"results_map_{map_mode}")
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Map rendering issue: {e}")
            else:
                st.info("BBox not available for map rendering.")

            # Raw Results
            with st.expander("View Raw Results"):
                st.json(results.get('results', {}))

            # 3D Visualization (conditional on stpyvista)
            if render_plotter is not None and bbox:
                st.subheader("🧊 3D Anomaly Visualization")
                with st.spinner("Generating 3D view..."):
                    plotter = create_3d_volume_viewer(st.session_state.job_results, bbox)
                    render_plotter(plotter, key="3d_volume")
            elif render_plotter is None:
                st.info("🔧 3D visualization disabled: stpyvista not available. Install via `pip install stpyvista` for interactive 3D views.")

        elif st.session_state.results:  # Legacy sync results
            st.header("📊 Legacy Results")
            st.json(st.session_state.results)
        else:
            st.info("Run an analysis to see results here.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # Footer
    st.markdown("---")
    st.markdown(
        "Powered by [GeoAnomalyMapper](https://github.com/your-org/GeoAnomalyMapper) | "
        "Async processing via FastAPI backend"
    )


if __name__ == "__main__":
    main()