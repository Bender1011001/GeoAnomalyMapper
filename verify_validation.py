
import os
import rasterio
from validation_pa import PredictionAreaPlotter

def test_validation():
    print("Testing P-A Plot...")
    output_path = "data/test/density_output.tif"
    
    if not os.path.exists(output_path):
        print("Output file missing!")
        return

    # Dummy deposits
    deposits = [(30, 30), (25, 25)] # Inside the block
    
    with rasterio.open(output_path) as src:
        pmap = src.read(1)
        
    print(f"Map shape: {pmap.shape}")
    
    plotter = PredictionAreaPlotter(pmap, deposits)
    oa, pr = plotter.calculate_curve()
    print(f"P-A Calc successful. AUC Points: {len(oa)}")
    
    plotter.plot("data/test/pa_check.png")
    if os.path.exists("data/test/pa_check.png"):
        print("P-A Plot saved!")

if __name__ == "__main__":
    test_validation()
