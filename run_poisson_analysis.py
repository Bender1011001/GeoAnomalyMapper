import sys
import os
from pathlib import Path

# Ensure current directory is in path
sys.path.append(os.getcwd())

try:
    from poisson_analysis import analyze_poisson_correlation
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    gravity_residual_path = "data/processed/gravity_residual.tif"
    magnetic_path = "data/processed/magnetic/magnetic_processed.tif"
    output_path = "data/processed/poisson_correlation.tif"
    
    if not Path(gravity_residual_path).exists():
        print(f"Error: Gravity residual file {gravity_residual_path} not found.")
        sys.exit(1)
        
    if not Path(magnetic_path).exists():
        print(f"Error: Magnetic file {magnetic_path} not found.")
        sys.exit(1)
        
    print(f"Running Poisson analysis...")
    try:
        analyze_poisson_correlation(gravity_residual_path, magnetic_path, output_path)
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error running Poisson analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()