import geopandas as gpd
import fiona
import os

gdb_path = r"D:\Geo_data\LiMW_GIS 2015.gdb"

print(f"Checking GDB at: {gdb_path}")

try:
    layers = fiona.listlayers(gdb_path)
    print(f"Layers found: {layers}")
    
    if layers:
        target_layer = layers[0]
        print(f"Attempting to read first 5 rows of layer: {target_layer}")
        gdf = gpd.read_file(gdb_path, layer=target_layer, rows=5)
        print("Success! Head of dataframe:")
        print(gdf.head())
        print(f"Columns: {gdf.columns}")
    else:
        print("No layers found!")
        
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
