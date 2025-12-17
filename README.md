# GeoAnomalyMapper

**AI-Powered Mineral Exploration System**

GeoAnomalyMapper is an advanced geophysics analysis pipeline that combines Physics-Informed Neural Networks (PINNs) with Supervised Machine Learning to identify high-probability mineral deposits across the United States.

## Key Features
-   **National Scale**: Processes Gravity and Magnetic mosaics for the entire contiguous USA.
-   **Physics-Driven**: Uses PINN inversion to model subsurface density.
-   **Data-Verified**: Validated against 1,500+ USGS "Goldilocks" deposits with **91% Sensitivity**.
-   **New Discoveries**: Identifies high-grade anomalies that are geologically similar to known mines but located in unexplored "greenfield" areas.

## Usage

The entire pipeline is orchestrated by a single master script:

```bash
python process_everything.py
```

This will:
1.  **Prepare Data**: Process raw Gravity and Magnetic GeoTIFFs.
2.  **Train & Invert**: Run the PINN model to generate a 3D density map.
3.  **Classify**: Apply the Random Forest classifier to generate a Probability Map.
4.  **Extract**: Identify and grade specific targets.

## Outputs

All results are saved to `data/outputs/`:

-   **`usa_targets.csv`**: The master list of ~1,600 high-confidence exploration targets.
-   **`usa_supervised_probability.tif`**: The probability heatmap (GeoTIFF).
-   **`high_value_targets_map.png`**: Static visualization of top prospects.

## Validation

To verify the model's integrity yourself, run:

```bash
python verify_skeptic.py
```

This performs a "Triple Check":
1.  **Spatial Holdout**: Trains on West US, predicts East US (Tests generalization).
2.  **Null Hypothesis**: Trains on random labels (Tests for lucky guessing).
3.  **Feature Audit**: confirms physical inputs drive the model.

## Legacy Code
Older experimental scripts and phased workflows have been moved to the `archive/` directory to maintain a clean workspace.
