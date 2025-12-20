import os
import zipfile
import numpy as np
import rasterio
from rasterio.transform import from_origin
import logging
import requests
from tqdm import tqdm
from rasterio.features import rasterize
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unzip_all(target_dir):
    for item in os.listdir(target_dir):
        if item.endswith(".zip"):
            file_path = os.path.join(target_dir, item)
            extract_path = os.path.join(target_dir, item.replace(".zip", ""))
            if not os.path.exists(extract_path):
                logger.info(f"Extracting {item}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            else:
                logger.info(f"Skipping extraction for {item}, already extracted.")

def convert_ggmplus_to_tif(dg_folder, output_tif):
    """
    GGMplus tiles are 5x5 degrees, 2500x2500 points.
    dg folder contains .dg files (int16, big-endian, 0.1 mGal units).
    """
    import glob
    import re
    
    bin_files = glob.glob(os.path.join(dg_folder, "*.dg"))
    if not bin_files:
        logger.warning(f"No .dg files found in {dg_folder}")
        return

    logger.info(f"Mosaicking {len(bin_files)} GGMplus tiles...")
    
    # 7.2 arc-seconds in degrees
    res_deg = 7.2 / 3600.0
    
    temp_tifs = []
    temp_dir = os.path.join(os.path.dirname(dg_folder), "temp_tifs")
    os.makedirs(temp_dir, exist_ok=True)

    for bin_file in bin_files:
        # Expected filename: N00E005.dg
        fname = os.path.basename(bin_file)
        # Regex for N00E005 or S10W080 etc
        match = re.search(r'([NS])(\d+)([EW])(\d+)', fname)
        if not match:
            continue
            
        lat_dir, lat_val, lon_dir, lon_val = match.groups()
        lat = int(lat_val) * (1 if lat_dir == 'N' else -1)
        lon = int(lon_val) * (1 if lon_dir == 'E' else -1)
        
        # Read binary data (int16, big-endian)
        with open(bin_file, 'rb') as f:
            data = np.fromfile(f, dtype='>i2').reshape((2500, 2500))
        
        # Convert to float and scale (0.1 mGal units)
        data = data.astype(np.float32) * 0.1
        
        # Define transform: GGMplus is center-of-pixel
        # Tile covers [lat, lat+5] and [lon, lon+5]
        # Binary data is typically North-to-South (top-row at max latitude)
        # For South latitudes (e.g. S10), the top edge is -10 + 5 = -5 ? 
        # Actually GGMplus tiles are named by their SOUTHERNWESTERN corner.
        # N00 means 0 to 5. S05 means -5 to 0. S10 means -10 to -5.
        # E005 means 5 to 10. W005 means -5 to 0.
        north_edge = lat + 5
        transform = from_origin(lon, north_edge, res_deg, res_deg)
        
        tif_path = os.path.join(temp_dir, fname.replace(".dg", ".tif"))
        with rasterio.open(
            tif_path, 'w', driver='GTiff',
            height=2500, width=2500, count=1, dtype='float32',
            crs='EPSG:4326', transform=transform, nodata=-9999
        ) as dst:
            dst.write(data, 1)
        temp_tifs.append(tif_path)

    # Build VRT instead of merging in memory (more efficient for 881 tiles)
    from rasterio.vrt import WarpedVRT
    import subprocess
    
    logger.info("Building VRT mosaic...")
    vrt_path = output_tif.replace(".tif", ".vrt")
    
    # Use gdalbuildvrt command line tool for efficiency
    cmd = ["gdalbuildvrt", vrt_path] + temp_tifs
    subprocess.run(cmd, check=True)
    
    logger.info(f"Translating VRT to compressed GeoTIFF...")
    # Use gdal_translate to convert VRT to final TIF with compression
    cmd = ["gdal_translate", "-co", "COMPRESS=LZW", "-co", "TILED=YES", 
           "-co", "BIGTIFF=YES", vrt_path, output_tif]
    subprocess.run(cmd, check=True)
        
    logger.info(f"Global mosaic saved to {output_tif}")
    
    # Cleanup temp and VRT
    for f in temp_tifs:
        os.remove(f)
    os.remove(vrt_path)
    os.rmdir(temp_dir)

def download_file(url, target_path):
    if os.path.exists(target_path):
        logger.info(f"Skipping download, {target_path} already exists.")
        return True
    logger.info(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    return True

def rasterize_glim(gdb_path, ref_tif, output_tif):
    """
    Rasterize GLiM lithology to match gravity grid.
    Uses Geopandas to process attributes, then gdal_rasterize for stable rasterization.
    """
    if os.path.exists(output_tif):
        logger.info(f"Skipping GLiM rasterization, {output_tif} already exists.")
        return
    
    temp_gpkg = output_tif.replace(".tif", "_temp.gpkg")
    
    try:
        import fiona
        import geopandas as gpd
        import subprocess
        
        # 1. Load and Process Vector Data
        logger.info(f"Loading GLiM vectors (utilizing 64GB RAM)...")
        layers = fiona.listlayers(gdb_path)
        layer_name = layers[0]
        
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        
        # Map Lithology to Density
        dens_map = {
            'su': 2.0, 'ss': 2.4, 'sc': 2.7, 'sm': 2.5, 'py': 2.2,
            'va': 2.5, 'vi': 2.7, 'vb': 3.0, 'pa': 2.6, 'pi': 2.8,
            'pb': 3.1, 'mt': 2.8, 'ev': 2.3, 'ig': 0.9, 'wb': 1.0, 'nd': 2.6
        }
        gdf['density'] = gdf['xx'].map(dens_map).fillna(2.6)
        
        # 2. Export to temporary GeoPackage (gdal_rasterize needs a file source)
        logger.info(f"Exporting processed vectors to {temp_gpkg}...")
        gdf[['geometry', 'density']].to_file(temp_gpkg, driver="GPKG")
        
        # Free memory
        del gdf
        
        # 3. Get reference raster bounds and resolution
        with rasterio.open(ref_tif) as src:
            bounds = src.bounds
            width = src.width
            height = src.height
            
        logger.info(f"Rasterizing with gdal_rasterize (Size: {width}x{height})...")
        
        # 4. Run gdal_rasterize
        # -a density: use 'density' attribute
        # -te xmin ymin xmax ymax: target extent
        # -ts width height: target size
        # -ot Float32: output type
        # -a_nodata 2.67: nodata/background
        cmd = [
            "gdal_rasterize",
            "-a", "density",
            "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
            "-ts", str(width), str(height),
            "-ot", "Float32",
            "-a_nodata", "2.67",
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=YES",
            temp_gpkg,
            output_tif
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info(f"GLiM lithology rasterized successfully to {output_tif}")

    except Exception as e:
        logger.error(f"Failed to rasterize GLiM: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_gpkg):
            try:
                os.remove(temp_gpkg)
            except:
                pass

def main():
    target_dir = r"D:\Geo_data"
    
    # No need to unzip_all if user did it
    
    # 1. Process GGMplus (Gravity)
    ggmplus_path = os.path.join(target_dir, "dg")
    output_grav = os.path.join(target_dir, "global_gravity_200m.tif")
    if os.path.exists(ggmplus_path) and not os.path.exists(output_grav):
        convert_ggmplus_to_tif(ggmplus_path, output_grav)
        
    # 2. Download Magnetics (EMAG2v3)
    mag_url = "https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530/EMAG2_V3_20170530_UpCont.tif"
    output_mag = os.path.join(target_dir, "global_magnetics_2arcmin.tif")
    download_file(mag_url, output_mag)

    # 3. Process Lithology (GLiM)
    gdb_path = os.path.join(target_dir, "LiMW_GIS 2015.gdb")
    output_lith = os.path.join(target_dir, "global_lithology_density.tif")
    if os.path.exists(gdb_path) and os.path.exists(output_grav):
        rasterize_glim(gdb_path, output_grav, output_lith)
        
    # 4. Results checking
    if os.path.exists(output_grav):
        logger.info(f"SUCCESS: Gravity mosaic ready at {output_grav}")

    logger.info("Data preparation completed.")

if __name__ == "__main__":
    main()
