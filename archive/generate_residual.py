import sys
import os
from pathlib import Path

# Ensure current directory is in path
sys.path.append(os.getcwd())

try:
    from process_data import wavelet_decompose_gravity
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    input_path = Path("data/processed/gravity/gravity_processed.tif")
    output_path = Path("data/processed/gravity_residual.tif")

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    print(f"Generating gravity residual from {input_path}...")
    try:
        wavelet_decompose_gravity(input_path, output_path)
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error generating residual: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()