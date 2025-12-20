"""
Prepare High-Resolution Data for Residual PINN Training.

Steps:
1. Mosaic GGMplus tiles for CONUS (USA)
2. Rasterize SGMC lithology to density grid
3. Save outputs ready for training
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import struct
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# USA bounds (CONUS + buffer)
USA_BOUNDS = {
    'min_lat': 24,
    'max_lat': 50,
    'min_lon': -125,
    'max_lon': -66
}

# Rock type to density mapping (kg/m³)
# Based on standard geophysical references
LITHOLOGY_DENSITY = {
    # Sedimentary
    'sandstone': 2350,
    'shale': 2400,
    'limestone': 2600,
    'dolomite': 2800,
    'conglomerate': 2500,
    'siltstone': 2450,
    'mudstone': 2350,
    'chalk': 2200,
    'coal': 1500,
    'evaporite': 2200,
    'salt': 2170,
    'gypsum': 2300,
    # Igneous
    'granite': 2670,
    'granodiorite': 2720,
    'diorite': 2800,
    'gabbro': 2900,
    'basalt': 2900,
    'rhyolite': 2500,
    'andesite': 2650,
    'dacite': 2550,
    'obsidian': 2400,
    'pumice': 800,
    'tuff': 2000,
    'volcanic': 2600,
    'intrusive': 2700,
    # Metamorphic
    'gneiss': 2700,
    'schist': 2750,
    'slate': 2750,
    'marble': 2700,
    'quartzite': 2650,
    'phyllite': 2700,
    'amphibolite': 2900,
    'greenstone': 2850,
    'serpentinite': 2600,
    # Other
    'ultramafic': 3200,
    'peridotite': 3300,
    'dunite': 3300,
    'alluvium': 2000,
    'unconsolidated': 1800,
    'sediment': 2200,
    'fill': 1900,
    'water': 1000,
    'ice': 920,
}

def read_ggmplus_tile(filepath):
    """
    Read a GGMplus binary tile.
    Format: 2500x2500 float32 grid, big-endian
    Covers 5x5 degrees at ~200m resolution
    """
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='>f4')  # Big-endian float32
    
    # Reshape to 2500x2500
    data = data.reshape((2500, 2500))
    
    # Parse filename for bounds
    name = Path(filepath).stem.replace('.ga', '').replace('.dg', '')
    
    # Parse lat/lon from filename (e.g., N35W120)
    lat_dir = name[0]
    lon_idx = name.find('E') if 'E' in name else name.find('W')
    lat = int(name[1:lon_idx])
    lon_dir = name[lon_idx]
    lon = int(name[lon_idx+1:])
    
    if lat_dir == 'S':
        lat = -lat
    if lon_dir == 'W':
        lon = -lon
    
    # Tile covers 5 degrees
    bounds = (lon, lat, lon + 5, lat + 5)
    
    return data, bounds

def mosaic_ggmplus_tiles(tile_dir, output_path, usa_only=True):
    """
    Mosaic all GGMplus tiles for the USA region.
    """
    tile_files = list(Path(tile_dir).glob('*.ga'))
    if not tile_files:
        tile_files = list(Path(tile_dir).glob('*.dg'))
    
    logger.info(f"Found {len(tile_files)} tiles")
    
    # Filter for USA region
    usa_tiles = []
    for tf in tile_files:
        name = tf.stem.replace('.ga', '').replace('.dg', '')
        
        # Parse coordinates
        lat_dir = name[0]
        lon_idx = name.find('E') if 'E' in name else name.find('W')
        lat = int(name[1:lon_idx])
        lon_dir = name[lon_idx]
        lon = int(name[lon_idx+1:])
        
        if lat_dir == 'S':
            lat = -lat
        if lon_dir == 'W':
            lon = -lon
        
        # Check if tile overlaps USA
        tile_max_lat = lat + 5
        tile_max_lon = lon + 5
        
        if usa_only:
            if (lat < USA_BOUNDS['max_lat'] and tile_max_lat > USA_BOUNDS['min_lat'] and
                lon < USA_BOUNDS['max_lon'] and tile_max_lon > USA_BOUNDS['min_lon']):
                usa_tiles.append((tf, (lon, lat, lon + 5, lat + 5)))
        else:
            usa_tiles.append((tf, (lon, lat, lon + 5, lat + 5)))
    
    logger.info(f"Processing {len(usa_tiles)} tiles covering USA region")
    
    if not usa_tiles:
        logger.error("No tiles found for USA region!")
        return
    
    # Calculate output dimensions
    # GGMplus is ~200m resolution = 0.002 degrees per pixel (approx)
    pixel_size = 5.0 / 2500  # 0.002 degrees
    
    out_width = int((USA_BOUNDS['max_lon'] - USA_BOUNDS['min_lon']) / pixel_size)
    out_height = int((USA_BOUNDS['max_lat'] - USA_BOUNDS['min_lat']) / pixel_size)
    
    logger.info(f"Output dimensions: {out_width} x {out_height}")
    
    # Create output array
    output = np.full((out_height, out_width), np.nan, dtype=np.float32)
    
    # Read and place tiles
    for tile_path, bounds in tqdm(usa_tiles, desc="Mosaicking tiles"):
        try:
            data, _ = read_ggmplus_tile(tile_path)
            
            # Calculate position in output
            col_start = int((bounds[0] - USA_BOUNDS['min_lon']) / pixel_size)
            row_start = int((USA_BOUNDS['max_lat'] - bounds[3]) / pixel_size)  # Flip Y
            
            col_end = col_start + data.shape[1]
            row_end = row_start + data.shape[0]
            
            # Handle boundary clipping
            data_col_start = max(0, -col_start)
            data_row_start = max(0, -row_start)
            data_col_end = data.shape[1] - max(0, col_end - out_width)
            data_row_end = data.shape[0] - max(0, row_end - out_height)
            
            col_start = max(0, col_start)
            row_start = max(0, row_start)
            col_end = min(out_width, col_end)
            row_end = min(out_height, row_end)
            
            output[row_start:row_end, col_start:col_end] = data[data_row_start:data_row_end, data_col_start:data_col_end]
            
        except Exception as e:
            logger.warning(f"Error processing {tile_path}: {e}")
    
    # Save output
    transform = from_bounds(
        USA_BOUNDS['min_lon'], USA_BOUNDS['min_lat'],
        USA_BOUNDS['max_lon'], USA_BOUNDS['max_lat'],
        out_width, out_height
    )
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': out_width,
        'height': out_height,
        'count': 1,
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'compress': 'deflate',
        'nodata': np.nan
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output, 1)
    
    logger.info(f"Saved mosaic to {output_path}")
    logger.info(f"Valid pixels: {np.count_nonzero(~np.isnan(output))} / {output.size}")

def assign_density(row):
    """
    Assign density based on rock type description.
    """
    lith = str(row.get('ROCKTYPE1', '')).lower() if row.get('ROCKTYPE1') else ''
    lith2 = str(row.get('ROCKTYPE2', '')).lower() if row.get('ROCKTYPE2') else ''
    unit_name = str(row.get('UNIT_NAME', '')).lower() if row.get('UNIT_NAME') else ''
    
    combined = f"{lith} {lith2} {unit_name}"
    
    # Search for keywords
    for keyword, density in LITHOLOGY_DENSITY.items():
        if keyword in combined:
            return density
    
    # Default crustal density
    return 2670.0

def rasterize_sgmc_lithology(shapefile_path, output_path, resolution=0.002):
    """
    Convert SGMC polygon shapefile to density raster.
    """
    logger.info(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    logger.info(f"Loaded {len(gdf)} polygons")
    logger.info(f"Columns: {list(gdf.columns)}")
    
    # Assign density values
    logger.info("Assigning density values based on rock types...")
    gdf['density'] = gdf.apply(assign_density, axis=1)
    
    # Filter to USA bounds
    gdf = gdf.cx[USA_BOUNDS['min_lon']:USA_BOUNDS['max_lon'], 
                  USA_BOUNDS['min_lat']:USA_BOUNDS['max_lat']]
    
    logger.info(f"Filtered to {len(gdf)} polygons in USA region")
    
    # Calculate raster dimensions
    width = int((USA_BOUNDS['max_lon'] - USA_BOUNDS['min_lon']) / resolution)
    height = int((USA_BOUNDS['max_lat'] - USA_BOUNDS['min_lat']) / resolution)
    
    transform = from_bounds(
        USA_BOUNDS['min_lon'], USA_BOUNDS['min_lat'],
        USA_BOUNDS['max_lon'], USA_BOUNDS['max_lat'],
        width, height
    )
    
    logger.info(f"Rasterizing to {width} x {height}...")
    
    # Create shapes for rasterization
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf['density']))
    
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=2670.0,  # Default crustal density
        dtype='float32'
    )
    
    # Save
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'compress': 'deflate',
        'nodata': -9999
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)
    
    logger.info(f"Saved lithology density raster to {output_path}")
    
    # Stats
    logger.info(f"Density range: {np.min(raster):.0f} - {np.max(raster):.0f} kg/m³")
    logger.info(f"Mean density: {np.mean(raster):.0f} kg/m³")

def main():
    parser = argparse.ArgumentParser(description="Prepare high-resolution data for PINN training")
    parser.add_argument("--ggmplus-dir", default="D:/Geo_data/ga/ga", help="Path to GGMplus tiles")
    parser.add_argument("--sgmc-shapefile", default="D:/Geo_data/USGS_SGMC_Shapefiles/USGS_SGMC_Shapefiles/SGMC_Geology.shp", 
                        help="Path to SGMC shapefile")
    parser.add_argument("--output-dir", default="data/outputs/highres", help="Output directory")
    parser.add_argument("--gravity-only", action="store_true", help="Only process gravity data")
    parser.add_argument("--lithology-only", action="store_true", help="Only process lithology data")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.lithology_only:
        # Mosaic GGMplus gravity
        gravity_output = os.path.join(args.output_dir, "usa_ggmplus_gravity.tif")
        mosaic_ggmplus_tiles(args.ggmplus_dir, gravity_output)
    
    if not args.gravity_only:
        # Rasterize SGMC lithology
        lithology_output = os.path.join(args.output_dir, "usa_sgmc_density.tif")
        rasterize_sgmc_lithology(args.sgmc_shapefile, lithology_output)
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main()
