import sys
import os
from pathlib import Path

# Ensure current directory is in path
sys.path.append(os.getcwd())

try:
    from detect_voids import dempster_shafer_fusion
    from process_data import compute_tilt_derivative
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    # Inputs
    gravity_residual_path = Path("data/processed/gravity_residual.tif")
    gravity_tdr_path = Path("data/processed/gravity_tdr.tif")
    
    # We need TDR for fusion. If it doesn't exist, create it from residual.
    if not gravity_tdr_path.exists():
        if gravity_residual_path.exists():
            print("Computing TDR from residual...")
            compute_tilt_derivative(gravity_residual_path, gravity_tdr_path)
        else:
            print(f"Error: Gravity residual {gravity_residual_path} missing, cannot compute TDR.")
            sys.exit(1)
            
    artif_path = Path("data/processed/insar/structural_artificiality.tif")
    poisson_path = Path("data/processed/poisson_correlation.tif")
    output_path = Path("data/processed/fused_belief_reinforced.tif")
    
    # Check inputs
    if not gravity_tdr_path.exists():
        print(f"Error: TDR file {gravity_tdr_path} missing.")
        sys.exit(1)
        
    # Note: artif_path might be missing if Phase 2 failed. 
    # The fusion script handles missing secondary inputs by treating them as neutral.
    if not artif_path.exists():
        print(f"Warning: Artificiality file {artif_path} missing. Fusion will proceed without it.")
        
    if not poisson_path.exists():
        print(f"Warning: Poisson file {poisson_path} missing. Fusion will proceed without it.")

    print("Running Dempster-Shafer Fusion...")
    try:
        dempster_shafer_fusion(
            tdr_path=gravity_tdr_path,
            artificiality_path=artif_path,
            poisson_path=poisson_path,
            output_belief_path=output_path
        )
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error running fusion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()