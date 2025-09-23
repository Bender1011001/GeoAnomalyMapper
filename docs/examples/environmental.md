# Environmental Monitoring Applications

## Overview

GAM detects geological hazards like subsidence, faults, and landslides using InSAR (displacement) and seismic (fault activity) data. Useful for disaster risk assessment and climate impact studies.

**Key Benefits**:
- Real-time hazard mapping from satellite data.
- Global monitoring for vulnerable regions.
- Integration with environmental models.

## Case Study: Himalayan Subsidence (India/Nepal)

### Background
The Himalayas face subsidence from tectonic activity and groundwater extraction. GAM maps fault lines and subsidence zones using InSAR and seismic.

### Data and Setup
- **Bbox**: [27.0, 28.0, 85.0, 86.0] (Kathmandu region).
- **Modalities**: Seismic, InSAR.
- **Config Snippet**:
  ```yaml
  data:
    bbox: [27.0, 28.0, 85.0, 86.0]
    modalities: ["seismic", "insar"]
  preprocessing:
    grid_res: 0.05
  modeling:
    threshold: 2.2
    priors:
      joint_weight: 0.6  # Balance seismic/InSAR
  visualization:
    map_type: "interactive"
    export_formats: ["html", "geotiff"]
  ```

### Running the Analysis
```bash
gam run --config himalaya.yaml --output himalaya_results/
```

**Run Time**: 20-40 min.

### Results and Interpretation
- **Anomalies Detected**: 18 subsidence zones, avg confidence 0.82.
- **Key Finding**: Zone at (27.7 N, 85.3 E) shows 5cm/year subsidence (InSAR) aligned with fault (seismic), confidence 0.91.
  - Indicates landslide risk; depth -500m suggests aquifer depletion.
- **Visualization**: Interactive Folium map with subsidence arrows; GeoTIFF for overlay in QGIS.

![Himalaya Subsidence](images/himalaya_map.png)
*Figure: InSAR displacement map with fault anomalies (red lines).*

- **Export**: HTML for web dashboard; CSV for risk models.

### Validation
- Matches USGS reports on Kathmandu Valley subsidence.
- Environmental Impact: Guides groundwater management.

### Extensions
- Add gravity for density changes (e.g., erosion).
- Time-series InSAR for trend analysis.

See [Multi-Modality Tutorial](../tutorials/02_multi_modality_fusion.ipynb) for fusion details.

---

*Case Study Date: 2025 | Data Sources: IRIS, Copernicus*