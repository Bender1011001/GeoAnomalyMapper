import os
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_data():
    """Create dummy gravity and magnetic GeoTIFFs for testing."""
    os.makedirs("data/test", exist_ok=True)
    
    # Create 64x64 grid
    data = np.random.rand(64, 64).astype(np.float32)
    # Add a "structure"
    data[20:40, 20:40] += 1.0
    
    transform = from_origin(-115.0, 35.0, 0.01, 0.01) # Approx 1km pixel
    
    profile = {
        'driver': 'GTiff',
        'height': 64,
        'width': 64,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    grav_path = "data/test/gravity_dummy.tif"
    mag_path = "data/test/magnetic_dummy.tif"
    
    with rasterio.open(grav_path, 'w', **profile) as dst:
        dst.write(data, 1)
        
    with rasterio.open(mag_path, 'w', **profile) as dst:
        # Magnetic guide has sharp edges
        mag_data = np.zeros((64, 64), dtype=np.float32)
        mag_data[20:40, 20:40] = 100.0 # Block
        dst.write(mag_data, 1)
        
    print("Dummy data created at data/test/")
    return grav_path, mag_path

def run_pipeline():
    print("1. Creating Dummy Data...")
    grav_path, mag_path = create_dummy_data()
    
    output_path = "data/test/density_output.tif"
    
    print("\n2. Running Inversion with Structure Guidance...")
    # Import here to test standard import path
    from pinn_gravity_inversion import invert_gravity
    
    # Run with small epochs for speed
    # We need to mock config or load it, but the function handles defaults.
    # We can patch the DEFAULT_CONFIG in the module or just trust it runs.
    # To speed it up, we should hack the module's default config if possible 
    # OR just let it run for 1000 epochs on 64x64 (should be fast on CPU even).
    # Let's override the EPOCHS by monkeypatching for the test
    import pinn_gravity_inversion
    pinn_gravity_inversion.DEFAULT_CONFIG['EPOCHS'] = 10
    pinn_gravity_inversion.DEFAULT_CONFIG['USE_AMP'] = False # Disable AMP to avoid NaNs in small synthetic test
    pinn_gravity_inversion.DEFAULT_CONFIG['LR'] = 1e-4 # Lower LR for safety
    
    invert_gravity(
        tif_path=grav_path,
        output_path=output_path,
        magnetic_guide_path=mag_path,
        target_mode='mineral'
    )
    
    print("\n3. Verifying Output...")
    if os.path.exists(output_path):
        print(" Output file exists!")
        with rasterio.open(output_path) as src:
            d = src.read(1)
            print(f" Output shape: {d.shape}")
            print(f" Output range: {d.min()} to {d.max()}")
    else:
        print(" Output file MISSING!")

    print("\n4. Testing P-A Plot...")
    from validation_pa import PredictionAreaPlotter
    # Dummy deposits
    deposits = [(30, 30), (25, 25)] # Inside the block
    
    with rasterio.open(output_path) as src:
        pmap = src.read(1)
        
    plotter = PredictionAreaPlotter(pmap, deposits)
    oa, pr = plotter.calculate_curve()
    print(f" P-A Calc successful. AUC Points: {len(oa)}")
    
    plotter.plot("data/test/pa_valid.png")
    if os.path.exists("data/test/pa_valid.png"):
        print(" P-A Plot saved!")

if __name__ == "__main__":
    run_pipeline()
