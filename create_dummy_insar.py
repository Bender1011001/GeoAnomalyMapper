import numpy as np
import rasterio
from pathlib import Path

def create_dummy_insar():
    # Use gravity processed as template for dimensions
    ref_path = Path("data/processed/gravity/gravity_processed.tif")
    output_dir = Path("data/processed/insar")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        shape = src.read(1).shape
        
    # Create 2 dummy coherence files
    for i in range(2):
        # Random coherence between 0 and 1
        data = np.random.rand(*shape).astype(np.float32)
        
        output_path = output_dir / f"coh_dummy_{i}.tif"
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"Created dummy InSAR: {output_path}")

if __name__ == "__main__":
    create_dummy_insar()