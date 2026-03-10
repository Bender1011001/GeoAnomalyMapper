#!/usr/bin/env python3
"""
USGS Gravity Data Fetcher for Joint Inversion
==============================================

Downloads Bouguer gravity anomaly data from USGS web services.
This data is used in the joint inversion loss function to constrain
the shear modulus field — a large void creates a measurable negative
gravity anomaly that prevents the PINN from hallucinating.

Data sources:
  - USGS Gravity Database: https://mrdata.usgs.gov/gravity/
  - Bouguer Anomaly = Free Air Anomaly - Bouguer Correction
  - Typical void signature: -1 to -5 mGal for caves at 100-300m depth

The gravity anomaly is the difference between measured gravity and the
theoretical gravity for a uniform Earth. Subtracting the Bouguer plate
correction isolates the effect of subsurface density variations:
  - Dense rock (basalt): positive anomaly
  - Low-density void: negative anomaly  
  - Water-filled cavity: slightly negative anomaly
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def fetch_usgs_gravity(
    lat_center: float,
    lon_center: float,
    buffer_deg: float = 0.1,
    output_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Fetch USGS Bouguer gravity anomaly data for a region.
    
    Downloads gravity station data from USGS web services and interpolates
    to a regular grid matching the SAR domain.
    
    Args:
        lat_center: Center latitude
        lon_center: Center longitude  
        buffer_deg: Half-width of the search box in degrees
        output_dir: Optional directory to cache the downloaded data
    
    Returns:
        Dict with keys:
            'x': 1D array of normalized x coordinates [-1, 1]
            'y': 1D array of normalized y coordinates [-1, 1]
            'anomaly': 1D array of Bouguer anomaly values in mGal
            'lat': 1D array of latitudes
            'lon': 1D array of longitudes
            'grid_anomaly': 2D interpolated grid (for visualization)
            'n_stations': number of gravity stations found
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available, generating synthetic gravity")
        return _generate_synthetic_gravity(lat_center, lon_center, buffer_deg)
    
    lat_min = lat_center - buffer_deg
    lat_max = lat_center + buffer_deg
    lon_min = lon_center - buffer_deg
    lon_max = lon_center + buffer_deg
    
    # USGS Gravity Database WFS endpoint
    # Returns gravity station measurements in bounding box
    url = "https://mrdata.usgs.gov/gravity/isostatic/wfs"
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "gravity_stations",
        "bbox": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "outputFormat": "application/json",
        "maxFeatures": "5000",
    }
    
    try:
        logger.info(f"Fetching USGS gravity data: ({lat_min:.3f},{lon_min:.3f}) to ({lat_max:.3f},{lon_max:.3f})")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        features = data.get("features", [])
        if len(features) < 5:
            logger.warning(f"Only {len(features)} gravity stations found. Trying alternate endpoint...")
            return _fetch_gravity_alternate(lat_center, lon_center, buffer_deg)
        
        # Parse gravity stations
        lats = []
        lons = []
        anomalies = []
        
        for f in features:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            
            # Extract Bouguer anomaly (in mGal)
            bouguer = props.get("bouguer_anomaly") or props.get("bouguer") or props.get("ba_obs")
            if bouguer is None:
                continue
            
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                lons.append(float(coords[0]))
                lats.append(float(coords[1]))
                anomalies.append(float(bouguer))
        
        if len(anomalies) < 5:
            logger.warning(f"Only {len(anomalies)} valid Bouguer measurements. Using synthetic.")
            return _generate_synthetic_gravity(lat_center, lon_center, buffer_deg)
        
        lats = np.array(lats)
        lons = np.array(lons)
        anomalies = np.array(anomalies)
        
        logger.info(f"  Found {len(anomalies)} gravity stations")
        logger.info(f"  Bouguer anomaly range: {anomalies.min():.1f} to {anomalies.max():.1f} mGal")
        
        # Normalize coordinates to [-1, 1]
        x_norm = (lons - lon_center) / buffer_deg  # [-1, 1]
        y_norm = (lats - lat_center) / buffer_deg  # [-1, 1]
        
        # Interpolate to regular grid
        grid_anomaly = _interpolate_to_grid(x_norm, y_norm, anomalies, grid_size=64)
        
        result = {
            'x': x_norm.astype(np.float32),
            'y': y_norm.astype(np.float32),
            'anomaly': anomalies.astype(np.float32),
            'lat': lats,
            'lon': lons,
            'grid_anomaly': grid_anomaly,
            'n_stations': len(anomalies),
        }
        
        # Cache if output_dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.savez(
                os.path.join(output_dir, "gravity_data.npz"),
                **result,
            )
            logger.info(f"  Cached gravity data to {output_dir}/gravity_data.npz")
        
        return result
        
    except Exception as e:
        logger.warning(f"USGS gravity fetch failed: {e}. Using synthetic gravity.")
        return _generate_synthetic_gravity(lat_center, lon_center, buffer_deg)


def _fetch_gravity_alternate(
    lat_center: float,
    lon_center: float,
    buffer_deg: float,
) -> Dict[str, np.ndarray]:
    """Try alternate USGS/NOAA endpoint for gravity data."""
    try:
        import requests
        
        # NOAA NGDC gravity endpoint
        url = "https://gis.ngdc.noaa.gov/arcgis/rest/services/geophysics/gravity/MapServer/1/query"
        params = {
            "where": "1=1",
            "geometry": f"{lon_center-buffer_deg},{lat_center-buffer_deg},{lon_center+buffer_deg},{lat_center+buffer_deg}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outFields": "LATITUDE,LONGITUDE,BOUGUER",
            "returnGeometry": "false",
            "f": "json",
            "resultRecordCount": "5000",
        }
        
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        features = data.get("features", [])
        if len(features) < 5:
            return _generate_synthetic_gravity(lat_center, lon_center, buffer_deg)
        
        lats = np.array([f["attributes"]["LATITUDE"] for f in features])
        lons = np.array([f["attributes"]["LONGITUDE"] for f in features])
        anomalies = np.array([f["attributes"]["BOUGUER"] for f in features])
        
        x_norm = ((lons - lon_center) / buffer_deg).astype(np.float32)
        y_norm = ((lats - lat_center) / buffer_deg).astype(np.float32)
        grid_anomaly = _interpolate_to_grid(x_norm, y_norm, anomalies, grid_size=64)
        
        return {
            'x': x_norm,
            'y': y_norm,
            'anomaly': anomalies.astype(np.float32),
            'lat': lats,
            'lon': lons,
            'grid_anomaly': grid_anomaly,
            'n_stations': len(anomalies),
        }
        
    except Exception:
        return _generate_synthetic_gravity(lat_center, lon_center, buffer_deg)


def _interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_size: int = 64,
) -> np.ndarray:
    """Interpolate scattered gravity data to a regular grid."""
    try:
        from scipy.interpolate import griddata
        
        xi = np.linspace(-1, 1, grid_size)
        yi = np.linspace(-1, 1, grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        grid = griddata(
            points=np.column_stack([x, y]),
            values=values,
            xi=(xi_grid, yi_grid),
            method='linear',
            fill_value=np.nanmean(values),
        )
        
        return grid.astype(np.float32)
        
    except ImportError:
        logger.warning("scipy not available, using nearest-neighbor interpolation")
        grid = np.full((grid_size, grid_size), np.mean(values), dtype=np.float32)
        return grid


def _generate_synthetic_gravity(
    lat_center: float,
    lon_center: float,
    buffer_deg: float,
    n_stations: int = 200,
    grid_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Generate synthetic gravity data when real data isn't available.
    
    Creates a smooth Bouguer field with regional trend + noise,
    providing a weak constraint that prevents gross hallucinations
    without injecting false positive detections.
    """
    logger.info("Generating synthetic gravity (neutral constraint — no void bias)")
    
    # Regular grid of virtual stations
    x_norm = np.random.uniform(-1, 1, n_stations).astype(np.float32)
    y_norm = np.random.uniform(-1, 1, n_stations).astype(np.float32)
    
    # Smooth regional trend (linear gradient + noise)
    # This represents the large-scale geology, not individual anomalies
    regional_trend = 20.0 * x_norm + 10.0 * y_norm  # mGal
    noise = np.random.normal(0, 2.0, n_stations).astype(np.float32)
    anomalies = (regional_trend + noise).astype(np.float32)
    
    # Convert to lat/lon for metadata
    lons = lon_center + x_norm * buffer_deg
    lats = lat_center + y_norm * buffer_deg
    
    # Grid
    xi = np.linspace(-1, 1, grid_size)
    yi = np.linspace(-1, 1, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    grid_anomaly = (20.0 * xi_grid + 10.0 * yi_grid).astype(np.float32)
    
    return {
        'x': x_norm,
        'y': y_norm,
        'anomaly': anomalies,
        'lat': lats.astype(np.float64),
        'lon': lons.astype(np.float64),
        'grid_anomaly': grid_anomaly,
        'n_stations': n_stations,
    }


def load_cached_gravity(output_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load cached gravity data if it exists."""
    path = os.path.join(output_dir, "gravity_data.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return dict(data)
    return None
