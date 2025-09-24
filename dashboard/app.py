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
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

import streamlit as st
import folium
from streamlit_folium import st_folium
import folium.plugins
from branca.element import Figure, Html, MacroElement
import pandas as pd
import numpy as np
from scipy import interpolate
import pyvista as pv
from stpyvista import st_plotter as render_plotter
from matplotlib import cm

from .presets import get_all_presets, get_preset

# Add GAM to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


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
        size = max(8, int(anomaly['confidence'] * 20))  # Scale size by confidence
        
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
            radius=size,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add simple legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 150px; height: 120px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4>Anomaly Types</h4>
    <p><i class="fa fa-circle" style="color:blue"></i> Gravity</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Magnetic</p>
    <p><i class="fa fa-circle" style="color:green"></i> InSAR</p>
    <p><i class="fa fa-circle" style="color:purple"></i> Seismic</p>
    <p><i class="fa fa-circle" style="color:red"></i> Fusion</p>
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
    
    # Add heatmap
    HeatMap(heat_data, gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'},
            min_opacity=0.4, radius=15, blur=15).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Legend for heatmap
    heatmap_legend = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 120px; height: 100px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px">
    <h4>Heatmap Intensity</h4>
    <p>Low <span style="background:linear-gradient(to right, blue, red); display:block; height:10px;"></span> High</p>
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
        
        folium.Marker(
            location=[anomaly['lat'], anomaly['lon']],
            popup=popup,
            icon=folium.Icon(color=color, icon='cloud')
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Legend same as individual map
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 150px; height: 120px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4>Anomaly Types</h4>
    <p><i class="fa fa-circle" style="color:blue"></i> Gravity</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Magnetic</p>
    <p><i class="fa fa-circle" style="color:green"></i> InSAR</p>
    <p><i class="fa fa-circle" style="color:purple"></i> Seismic</p>
    <p><i class="fa fa-circle" style="color:red"></i> Fusion</p>
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
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
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
            f"{API_BASE_URL}/analysis",
            json=request_data,
            timeout=API_TIMEOUT
        )
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
    try:
        response = requests.get(
            f"{API_BASE_URL}/analysis/{job_id}/status",
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
    try:
        response = requests.get(
            f"{API_BASE_URL}/analysis/{job_id}/results",
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
        'api_available': check_api_connection(),
        'is_running': False,
        'job_history': [],
        'last_poll_time': 0,
        'pipeline': None,  # For potential fallback
        'results': None,   # Legacy
        'status': "Ready",
        'bbox': None,
        'modalities': [],
        'selected_preset': 'Custom Configuration',
        'preset_applied': False,
        'preset_defaults': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def poll_job_status():
    """Poll job status and update session state. Returns True if still running."""
    if not st.session_state.current_job_id or not st.session_state.is_running:
        return False

    job_id = st.session_state.current_job_id
    status_data = get_job_status(job_id)
    if not status_data:
        st.session_state.is_running = False
        return False

    st.session_state.job_status = status_data['status']
    st.session_state.job_progress = status_data['progress']
    st.session_state.status = status_data['stage']  # For display

    if status_data['status'] == 'COMPLETED':
        results = get_job_results(job_id)
        if results:
            st.session_state.job_results = results
            st.session_state.is_running = False
            # Add to history
            if job_id not in st.session_state.job_history:
                st.session_state.job_history.append(job_id)
            st.rerun()
        return False
    elif status_data['status'] == 'FAILED':
        st.session_state.is_running = False
        st.error(f"Job failed: {status_data.get('message', 'Unknown error')}")
        return False

    return True  # Still running


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

    # API Connectivity Check and Warning
    if not st.session_state.api_available:
        st.warning("""
        🚨 **Backend API not available.** 
        Start the FastAPI server: `uvicorn gam.api.main:app --reload` (port 8000)
        Async features disabled. Contact admin for sync fallback.
        """)
        st.session_state.is_running = False
        # Optionally implement sync fallback here, but per task, just warn

    # Sidebar: Parameter selection
    st.sidebar.header("📍 Analysis Parameters")

    # Preset Selection
    preset_options = ['Custom Configuration'] + list(get_all_presets().keys())
    selected_preset = st.sidebar.selectbox(
        "Analysis Preset",
        options=preset_options,
        index=0 if st.session_state.get('selected_preset') == 'Custom Configuration' else preset_options.index(st.session_state['selected_preset']),
        help="Select a preset for common use cases or 'Custom Configuration' for manual setup."
    )
    st.session_state.selected_preset = selected_preset

    if selected_preset != 'Custom Configuration':
        preset_config = get_preset(selected_preset)
        if preset_config:
            st.session_state.preset_applied = True
            st.session_state.preset_defaults = preset_config
            st.sidebar.info(f"**{preset_config['description']}**")
            
            with st.sidebar.expander("Preset Details", expanded=True):
                st.markdown("**Typical Use Cases:**")
                for case in preset_config['typical_use_cases']:
                    st.markdown(f"• {case}")
                st.info(f"**Recommended BBox Size:** {preset_config['typical_bbox_size']}")
                st.info(f"**Analysis Focus:** {preset_config['analysis_focus']}")
                st.markdown("**Note:** You can override any parameters below for customization.")
        else:
            st.session_state.preset_applied = False
            st.sidebar.warning("Preset not found. Using custom configuration.")
    else:
        st.session_state.preset_applied = False
        st.session_state.preset_defaults = {}

    if selected_preset != st.session_state.get('last_preset', ''):
        st.session_state.last_preset = selected_preset
        st.rerun()

    # Interactive map for bbox selection (unchanged)
    st.sidebar.subheader("🗺️ Region Selection")
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
    map_data = st_folium(m, width=300, height=250, key="bbox_map")

    if map_data and 'last_active_drawing' in map_data:
        drawing = map_data['last_active_drawing']
        if drawing['geometry']['type'] == 'Rectangle':
            bounds = drawing['geometry']['coordinates'][0]
            min_lat, min_lon = min(bounds[0][0], bounds[1][0]), min(bounds[0][1], bounds[1][1])
            max_lat, max_lon = max(bounds[0][0], bounds[1][0]), max(bounds[0][1], bounds[1][1])
            bbox_str = f"{min_lon:.1f},{min_lat:.1f},{max_lon:.1f},{max_lat:.1f}"
            st.session_state.bbox_str = bbox_str
            st.sidebar.success(f"Selected: {bbox_str}")

    # Fallback bbox input
    bbox_str = st.session_state.get('bbox_str', st.sidebar.text_input(
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
        st.sidebar.error(f"Invalid bbox: {e}")

    # Modalities
    preset_modalities_str = ','.join(st.session_state.preset_defaults.get('default_modalities', ['gravity', 'magnetic'])) if st.session_state.preset_applied else "gravity,magnetic"
    default_modal_str = st.sidebar.text_input(
        "Modalities",
        value=preset_modalities_str,
        help="Comma-separated list of modalities (e.g., gravity,magnetic,insar,seismic)"
    )
    
    # Override indicator
    if st.session_state.preset_applied:
        user_modalities = [m.strip() for m in default_modal_str.split(',') if m.strip()]
        preset_mods = st.session_state.preset_defaults.get('default_modalities', [])
        if set(user_modalities) != set(preset_mods):
            st.sidebar.warning("🔧 Manual override: Modalities differ from preset recommendation")
    
    try:
        selected_modalities = [m.strip() for m in default_modal_str.split(',') if m.strip()]
        if not selected_modalities:
            selected_modalities = ['gravity', 'magnetic']
        st.session_state.modalities = selected_modalities
    except:
        selected_modalities = ['gravity', 'magnetic']

    # Resolution (logged, not passed to API per schema)
    resolution = st.sidebar.slider(
        "Grid Resolution (meters)",
        min_value=100.0, max_value=5000.0, value=1000.0, step=100.0
    )

    # Output directory
    output_dir = st.sidebar.text_input(
        "Output Directory", value="results/dashboard"
    )

    # Config file
    config_path = st.sidebar.text_input(
        "Config File", value="config.yaml"
    ) if config_path != "config.yaml" else None

    # Verbose
    verbose = st.sidebar.checkbox("Verbose Logging", value=False)

    # Job History in Sidebar
    with st.sidebar.expander("📋 Job History", expanded=False):
        if st.session_state.job_history:
            for job_id in st.session_state.job_history[-5:]:  # Last 5
                status_color = {"COMPLETED": "green", "FAILED": "red", "RUNNING": "orange"}.get(
                    st.session_state.job_status if job_id == st.session_state.current_job_id else "unknown", "gray"
                )
                st.markdown(f"**{job_id[:8]}...** : {status_color}")
                if st.button(f"View Results {job_id[:8]}", key=f"view_{job_id}"):
                    # Fetch and display results for historical job
                    results = get_job_results(job_id)
                    if results:
                        st.session_state.job_results = results
                        st.session_state.current_job_id = job_id
                        st.rerun()
        else:
            st.info("No completed jobs yet.")

    # Run Button with Disable
    disabled = st.session_state.is_running or not st.session_state.api_available or bbox is None
    if st.sidebar.button("🚀 Run Analysis", type="primary", disabled=disabled):
        if not st.session_state.api_available:
            st.error("Cannot start job: API not available.")
            st.stop()

        job_id = start_analysis_job(
            bbox=bbox,
            modalities=selected_modalities,
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
            st.success(f"Analysis started! Job ID: {job_id}")
            st.rerun()

    # Polling Loop for Progress
    if st.session_state.is_running and st.session_state.api_available:
        current_time = time.time()
        if current_time - st.session_state.last_poll_time >= 2:
            if poll_job_status():
                st.session_state.last_poll_time = current_time
                time.sleep(2)  # Wait before next poll
                st.rerun()
            else:
                st.session_state.last_poll_time = current_time

    # Retry Button if Failed
    if st.session_state.job_status == "FAILED" and not st.session_state.is_running:
        if st.sidebar.button("🔄 Retry Analysis", disabled=disabled):
            retry_job_start()

    # Main Content Area
    if st.session_state.current_job_id and st.session_state.api_available:
        st.header("📊 Job Progress")
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = {
                "QUEUED": "orange", "RUNNING": "blue", "COMPLETED": "green", "FAILED": "red"
            }.get(st.session_state.job_status, "gray")
            st.metric("Status", st.session_state.job_status, delta=None, delta_color=status_color)
        with col2:
            st.metric("Progress", f"{st.session_state.job_progress * 100:.1f}%")
            progress_bar = st.progress(st.session_state.job_progress)
        with col3:
            st.metric("Current Stage", st.session_state.status or "N/A")

        if st.session_state.job_progress > 0:
            st.info(f"Job ID: {st.session_state.current_job_id} | Estimated time remaining: ~{(1 - st.session_state.job_progress) * 10} min (approx)")

    # Results Display
    if st.session_state.job_results:
        st.header("✅ Analysis Results")
        results = st.session_state.job_results

        # Key Metrics
        col1, col2 = st.columns(2)
        with col1:
            anomaly_count = len(results.get('results', {}).get('anomalies', []))
            st.metric("Anomaly Count", anomaly_count)
        with col2:
            # Processing time not directly available; approximate from history if needed
            st.metric("Status", "Completed")

        st.subheader("Output Files")
        output_files = results.get('output_files', {})
        for file_name, file_path in output_files.items():
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file.read(),
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
            else:
                st.warning(f"File {file_name} not found at {file_path}")

        # Raw Results
        with st.expander("View Raw Results"):
            st.json(results['results'])

    elif st.session_state.results:  # Legacy sync results
        st.header("📊 Legacy Results")
        st.json(st.session_state.results)
    else:
        st.info("Run an analysis to see results here.")

    # Footer
    st.markdown("---")
    st.markdown(
        "Powered by [GeoAnomalyMapper](https://github.com/your-org/GeoAnomalyMapper) | "
        "Async processing via FastAPI backend"
    )


if __name__ == "__main__":
    main()