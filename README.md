# GeoAnomalyMapper

**Automated AI Mineral Vectoring System**

This repository contains the "Truth Machine" — a physics-informed deep learning system that identifies hidden mineral deposits across North America using gravity, magnetic, and seismic data, coupled with an automated verification agent to perform due diligence.

## Key Features
*   **Physics-Informed Inversion**: Generates 3D density models using structure-guided PINNs.
*   **4,000+ Targets**: Identified across the Continental US, Alaska, and Hawaii.
*   **Auto-Verification Agent**: Determining land status, claims, and geology automatically.
*   **Interactive Mapping**: Visualization of clusters and mineral districts.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
python verification/setup_environment.py
```

### 2. Run the Verification Pipeline
The verification system (located in `verification/`) filters raw targets.

**Quick Check (2 mins):**
```bash
python verification/quick_verify.py
```

**Full Analysis (~4 hours):**
```bash
python verification/run_verification.py
```
*Note: This requires Google Earth Engine authentication (`earthengine authenticate`) and reference data in `data/reference/`.*

## Outputs
*   **`data/outputs/target_map.html`**: Interactive map of all high-value targets.
*   **`data/outputs/FINAL_high_confidence_targets.csv`**: The "Gold List" (unclaimed, verified).

## Documentation
*   **[Monetization Strategy](docs/MONETIZATION.md)**: How to turn these targets into revenue.
*   **[Research Paper](docs/RESEARCH_PAPER.md)**: Methodology and scientific basis.

## Project Structure
*   `verification/`: Scripts for verifying and analyzing targets.
*   `data/`: Inputs (targets) and Outputs (maps, verified lists).
*   `train_*.py`: Model training scripts (PINN).
*   `predict_*.py`: Inference scripts.
