from classify_anomalies import classify_dumb_candidates
import os

# Define the paths to your processed layers
# Note: structural_artificiality.tif, poisson_correlation.tif, fused_belief_reinforced.tif not found in searches.
# Proceeding with available files, but script will fail on missing ones.
feature_paths = [
    "data/processed/gravity_residual.tif",
    "data/processed/structural_artificiality.tif",
    "data/processed/poisson_correlation.tif",
    "data/processed/fused_belief_reinforced.tif",
    "final_density_map.tif"
]

# Filter out paths that don't exist to avoid errors, but warn about them
existing_paths = []
for path in feature_paths:
    if os.path.exists(path):
        existing_paths.append(path)
    else:
        print(f"Warning: Input file not found: {path}")

if not existing_paths:
    print("Error: No input files found. Cannot proceed with classification.")
    exit(1)

output_path = "data/outputs/final_dumb_probability.tif"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Generate the final probability map
print("Starting classification...")
classify_dumb_candidates(existing_paths, output_path)
print(f"Classification complete. Output saved to {output_path}")