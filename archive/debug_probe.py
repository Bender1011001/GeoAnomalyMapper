import rasterio
import numpy as np

def probe(path, lat, lon, name):
    try:
        with rasterio.open(path) as src:
            # Check bounds
            print(f"--- Probing {name} ---")
            print(f"Path: {path}")
            print(f"Bounds: {src.bounds}")
            print(f"CRS: {src.crs}")
            
            # Sample
            try:
                vals = list(src.sample([(lon, lat)]))
                val = vals[0][0]
                print(f"Value at ({lat}, {lon}): {val}")
                
                # Check pixel index
                row, col = src.index(lon, lat)
                print(f"Pixel Index (row, col): {row}, {col}")
                
                # Check if inside array
                if 0 <= row < src.height and 0 <= col < src.width:
                    print("Index is VALID (inside raster dimensions)")
                else:
                    print("Index is OUT OF BOUNDS")
                    
            except Exception as e:
                print(f"Sampling failed: {e}")
            
    except Exception as e:
        print(f"Could not open {path}: {e}")

# Bingham Canyon (approx)
lat = 40.52
lon = -112.15

probe("data/outputs/usa_supervised/usa_gravity_mosaic.tif", lat, lon, "Gravity Mosaic")
probe("data/outputs/usa_production_probability_map.tif", lat, lon, "Probability Map")
