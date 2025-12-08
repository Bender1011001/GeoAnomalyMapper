import os
import zipfile
import rasterio
from pathlib import Path

def convert_asc_to_tif(input_path, output_path):
    """
    Converts an ArcInfo ASCII Grid file (.asc) to GeoTIFF (.tif).
    Enforces EPSG:4326 CRS.
    """
    print(f"Converting {input_path} to {output_path}...")
    try:
        with rasterio.open(input_path) as src:
            # Read the data
            data = src.read(1)
            profile = src.profile
            
            # Update profile for GeoTIFF
            profile.update(
                driver='GTiff',
                count=1,
                compress='lzw'
            )
            
            # Ensure CRS is WGS84 if missing or incorrect (based on filename hint)
            if not src.crs:
                print("CRS not found in source, setting to EPSG:4326")
                profile['crs'] = 'EPSG:4326'
            
            # Write to GeoTIFF
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
                
        print(f"Successfully created {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def setup_lithology():
    """
    Checks for Lithology .asc file and converts to .tif if needed.
    """
    lithology_dir = Path('data/raw/lithology')
    lithology_dir.mkdir(parents=True, exist_ok=True)
    
    asc_file = lithology_dir / 'glim_wgs84_0point5deg.txt.asc'
    tif_file = lithology_dir / 'glim_wgs84_0.5deg.tif'
    
    if tif_file.exists():
        print(f"Lithology TIF already exists: {tif_file}")
        return

    if asc_file.exists():
        convert_asc_to_tif(asc_file, tif_file)
    else:
        print(f"MISSING: {asc_file}")
        print(f"Please place 'glim_wgs84_0point5deg.txt.asc' in {lithology_dir}")

def setup_gravity():
    """
    Checks for Gravity .zip file and extracts it if needed.
    """
    gravity_dir = Path('data/raw/gravity')
    gravity_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = gravity_dir / 'GeophysicsGravity_USCanada.zip'
    
    # We can't easily know the extracted filename without opening the zip or knowing the dataset.
    # But we can check if the zip exists and extract it.
    
    if zip_file.exists():
        print(f"Found Gravity ZIP: {zip_file}")
        print("Extracting...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(gravity_dir)
            print("Gravity data extracted successfully.")
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")
    else:
        # Check if we might already have data (heuristic)
        files = list(gravity_dir.glob('*.tif')) + list(gravity_dir.glob('*.asc'))
        if files:
            print(f"Gravity data seems to exist ({len(files)} files found in {gravity_dir}). Skipping ZIP check.")
        else:
            print(f"MISSING: {zip_file}")
            print(f"Please place 'GeophysicsGravity_USCanada.zip' in {gravity_dir}")

if __name__ == "__main__":
    print("Starting Data Setup...")
    setup_lithology()
    setup_gravity()
    print("Data Setup Complete.")