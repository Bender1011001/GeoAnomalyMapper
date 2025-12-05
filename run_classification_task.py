import sys
import os
from pathlib import Path

# Ensure current directory is in path
sys.path.append(os.getcwd())

try:
    from classify_anomalies import classify_dumb_candidates
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    # Define inputs
    feature_paths = [
        "data/processed/gravity_residual.tif",
        "data/processed/fused_belief_reinforced.tif",
        "data/processed/poisson_correlation.tif",
        # "data/processed/insar/structural_artificiality.tif" # Missing
    ]
    
    # Filter existing files
    existing_features = [p for p in feature_paths if Path(p).exists()]
    
    if not existing_features:
        print("Error: No feature files found for classification.")
        sys.exit(1)
        
    output_path = "data/processed/dumb_probability_v2.tif"
    
    print(f"Running classification with features: {existing_features}")
    try:
        classify_dumb_candidates(existing_features, output_path)
        print(f"Successfully created {output_path}")
    except Exception as e:
        print(f"Error running classification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()