import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import requests
from shapely.geometry import shape, box

# ==========================================
# 1. Expert Configuration
# ==========================================
# Mapping geological classes to standard crustal densities (kg/m^3).
# Based on Telford et al. Applied Geophysics.
LITHOLOGY_DENSITY_MAP = {
    # Sedimentary
    "sedimentary": 2400,
    "sandstone": 2350,
    "limestone": 2550,
    "shale": 2400,
    "dolomite": 2700,
    "conglomerate": 2400,
    "alluvium": 1900,  # Loose soil, often covers voids
    
    # Igneous
    "igneous": 2800,
    "basalt": 2900,
    "granite": 2650,
    "andesite": 2600,
    "rhyolite": 2500,
    "gabbro": 3000,
    
    # Metamorphic
    "metamorphic": 2700,
    "gneiss": 2700,
    "schist": 2650,
    "slate": 2750,
    "quartzite": 2650,
    
    # Defaults
    "water": 1000,
    "ice": 917,
    "unknown": 2670  # Standard crustal average
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2. The Fetcher Engine
# ==========================================
def fetch_macrostrat_geojson(bounds: Tuple[float, float, float, float]) -> Dict:
    """
    Query Macrostrat API v2 for geologic units within a bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # API Endpoint
    url = "https://macrostrat.org/api/v2/geologic_units/map"
    
    # Buffer slightly to avoid edge artifacts
    params = {
        "envelope": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "format": "geojson"
    }
    
    logger.info(f"Querying Macrostrat for region: {params['envelope']}...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        count = data.get("success", {}).get("count", 0)
        features = data.get("success", {}).get("data", {}).get("features", [])
        
        # Fallback if structure differs (API v2 vs v1 quirks)
        if not features and "features" in data:
            features = data["features"]
            count = len(features)
            
        logger.info(f"Retrieved {count} geological features.")
        return features
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)

def parse_density_from_lithology(props: Dict) -> int:
    """
    Extract lithology string and map to density value.
    Prioritizes specific rock types over general classes.
    """
    # Macrostrat provides 'lith', 'lith_type', 'best_int_name'
    # We join them to search for keywords
    text_blob = " ".join([
        str(props.get("lith", "")),
        str(props.get("lith_type", "")),
        str(props.get("name", ""))
    ]).lower()
    
    # 1. Search for specific matches first (e.g., "basalt" before "igneous")
    for key, density in LITHOLOGY_DENSITY_MAP.items():
        if key in text_blob and key != "unknown":
            return density
            
    # 2. Fallback to broad classes if specific not found
    if "sedimentary" in text_blob: return LITHOLOGY_DENSITY_MAP["sedimentary"]
    if "igneous" in text_blob: return LITHOLOGY_DENSITY_MAP["igneous"]
    if "metamorphic" in text_blob: return LITHOLOGY_DENSITY_MAP["metamorphic"]
    
    return LITHOLOGY_DENSITY_MAP["unknown"]

# ==========================================
# 3. The Rasterizer (Vector to Grid)
# ==========================================
def rasterize_lithology(
    features: List[Dict], 
    reference_profile: Dict, 
    fill_value: int = 2670
) -> np.ndarray:
    """
    Burn vector shapes into a raster grid matching the reference GeoTIFF.
    """
    height = reference_profile['height']
    width = reference_profile['width']
    transform = reference_profile['transform']
    
    # Prepare (geometry, value) pairs
    shapes = []
    for feat in features:
        geom = shape(feat['geometry'])
        density = parse_density_from_lithology(feat['properties'])
        shapes.append((geom, density))
        
    if not shapes:
        logger.warning("No shapes to rasterize. Returning constant density map.")
        return np.full((height, width), fill_value, dtype=np.float32)
    
    logger.info(f"Rasterizing {len(shapes)} shapes into {height}x{width} grid...")
    
    # Rasterize
    # all_touched=True ensures small features aren't missed
    density_grid = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill_value,
        all_touched=True,
        dtype=np.float32
    )
    
    return density_grid

# ==========================================
# 4. Main Execution
# ==========================================
def fetch_and_rasterize(reference_tif_path: str, output_path: str) -> bool:
    """
    Fetch lithology data from Macrostrat and rasterize it to match the reference GeoTIFF.
    
    Args:
        reference_tif_path: Path to the reference GeoTIFF (e.g., gravity residual).
        output_path: Path where the output density map will be saved.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    ref_path = Path(reference_tif_path)
    out_path = Path(output_path)
    
    if not ref_path.exists():
        logger.error(f"Reference file not found: {ref_path}")
        return False
        
    try:
        # 1. Read Reference Grid Info
        with rasterio.open(ref_path) as src:
            bounds = src.bounds # (left, bottom, right, top)
            profile = src.profile
            # Macrostrat expects (min_lon, min_lat, max_lon, max_lat)
            # Rasterio bounds are (left, bottom, right, top) -> Same order
            query_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
        # 2. Fetch Vector Data
        features = fetch_macrostrat_geojson(query_bounds)
        
        # 3. Rasterize to Density Map
        density_map = rasterize_lithology(features, profile)
        
        # 4. Save Output
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='deflate',
            nodata=np.nan
        )
        
        # Optional: Smooth boundaries slightly to avoid hard edges in gravity inversion
        # density_map = gaussian_filter(density_map, sigma=1)
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(density_map, 1)
            dst.set_band_description(1, "Estimated Crustal Density (kg/m3)")
            
        logger.info(f"Success! Density map saved to: {out_path}")
        logger.info(f"  - Resolution: {profile['width']}x{profile['height']}")
        logger.info(f"  - Bounds: {query_bounds}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fetch and rasterize lithology: {e}")
        return False

def main(region=None, resolution=None, output_dir=None):
    if region is not None and resolution is not None and output_dir is not None:
        # Workflow mode
        output_path = Path(output_dir) / "processed/lithology_density.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy profile from region/resolution
        min_lon, min_lat, max_lon, max_lat = region
        width = int((max_lon - min_lon) / resolution)
        height = int((max_lat - min_lat) / resolution)
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
            'compress': 'deflate'
        }
        
        # Fetch Vector Data
        features = fetch_macrostrat_geojson(region)
        
        # Rasterize to Density Map
        density_map = rasterize_lithology(features, profile)
        
        # Save Output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(density_map, 1)
            dst.set_band_description(1, "Estimated Crustal Density (kg/m3)")
            
        logger.info(f"Success! Density map saved to: {output_path}")
        return

    parser = argparse.ArgumentParser(description="Macrostrat Lithology to Density Map Fetcher")
    parser.add_argument("--reference", required=True, help="Path to reference GeoTIFF (gravity/dem) to match grid")
    parser.add_argument("--output", required=True, help="Output path for Density GeoTIFF")
    args = parser.parse_args()
    
    success = fetch_and_rasterize(args.reference, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()