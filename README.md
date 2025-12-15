# GeoAnomalyMapper: Continental-Scale Mineral Exploration System (v2.0)

GeoAnomalyMapper is a robust, validation-driven geophysical pipeline designed for identifying mineral deposit signatures across the contiguous United States. It integrates rigorous data processing, physics-guided feature engineering, and high-performance machine learning to generate exploration targets defined by high statistical confidence.

## 🚀 Key Achievements
- **Sensitivity**: **90.7%** (Validated on 1,590 held-out deposits via Spatial 10-Fold CV).
- **Specificity**: **2.4%** Flagged Area (Precision targeting).
- **Scale**: Full Continuous US (CONUS) coverage.

## 🏗️ Architecture
The pipeline operates in three distinct phases:

### 1. Robust Data Processing
- **Source**: USGS Isostatic Gravity, USGS Aeromagnetics, and USGS MRDS (Mineral Resources Data System).
- **Preprocessing**: Automated mosacing, reprojection, and outlier removal.
- **Feature Engineering**: Physics-based derivatives including Tilt Angle, Total Horizontal Gradient, and Analytic Signal.

### 2. Physics-Guided Inversion (Experimental Layer)
- **Methodology**: Uses a custom **Physics-Guided Neural Network (PGNN)** (`pinn_gravity_inversion.py`) that incorporates a spectral forward gravity operator (Parker's Formula).
- **Role**: Generates a subsurface density contrast map to serve as a high-value feature for the downstream classifier.
- *Note*: This component replaces traditional "Collocation PINN" approaches with a direct forward-modeling constraint for better stability on real-world data.

### 3. Supervised Classification ("The Truth Machine")
- **Core Engine**: Balanced Random Forest Ensemble (`classify_supervised.py`).
- **Training Strategy**: Trained on **1,590 verified "Goldilocks" deposits** (USGS Producers/Past-Producers of key commodities).
- **Negative Sampling**: Spatial exclusion zones ensure the model learns from "true background" geology.
- **Fusion**: Integrates Gravity, Magnetics, and InSAR-derived stability features using Random Forest-based fusion (`multi_resolution_fusion.py`).

## 📊 Validation & Verification
Rigorous validation is the cornerstone of this project.

| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **Sensitivity** | **90.7%** | >50% | ✅ Exceeded |
| **Flagged Area** | **2.38%** | <5% | ✅ Passed |
| **Validation Method** | Spatial 10-Fold CV | - | ✅ Robust |

*Results verified via `validate_robustness.py` on 2025-12-14.*

## 📂 Key Artifacts
- **Production Model**: `data/outputs/usa_models/usa_production_model.joblib`
- **Probability Map**: `data/outputs/usa_production_probability_map.tif`
- **Validation Log**: `full_usa_validation.log`

## 🛠️ Usage

### Quick Start
```bash
# Run the full pipeline (End-to-End)
python run_robust_pipeline.py
```

### Reproduce Validation
```bash
# Run the "Truth Machine" (Validation Suite)
python validate_robustness.py
```

## 📜 License
MIT License.
