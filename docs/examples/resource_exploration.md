# Resource Exploration Use Cases

## Overview

GAM supports mineral, oil, and gas exploration by mapping subsurface density/magnetic anomalies indicative of deposits. Fusion of gravity and magnetic data highlights potential reservoirs; seismic adds structural detail.

**Key Benefits**:
- Global coverage from public data, reducing exploration costs.
- Probabilistic scoring for risk assessment.
- Scalable to basins (1-5° bbox) or global prospecting.

## Case Study: North Sea Oil Basin (Europe)

### Background
The North Sea is a mature oil province. GAM identifies undrilled prospects by fusing gravity (basin thickness) and magnetic (basement faults).

### Data and Setup
- **Bbox**: [55.0, 62.0, 0.0, 10.0] (UK/Norway sector).
- **Modalities**: Gravity, magnetic, seismic.
- **Config Snippet**:
  ```yaml
  data:
    bbox: [55.0, 62.0, 0.0, 10.0]
    modalities: ["gravity", "magnetic", "seismic"]
  preprocessing:
    grid_res: 0.1
  modeling:
    threshold: 2.0
    priors:
      regularization: "l1"  # Sparse for deposit detection
  visualization:
    map_type: "2d"
    export_formats: ["geotiff", "vtk"]
  ```

### Running the Analysis
```bash
gam run --config north_sea.yaml --parallel-workers 8 --output north_sea_results/
```

**Run Time**: 30-60 min.

### Results and Interpretation
- **Anomalies Detected**: 25 prospects, avg confidence 0.75.
- **Key Finding**: Cluster at (57.5 N, 2.5 E, -2000m) with low gravity (porous reservoir), high magnetic contrast (fault trap), confidence 0.88.
  - Potential oil trap; depth aligns with Jurassic reservoirs.
- **Visualization**: Contour map shows basin edges; 3D VTK for fault visualization.

![North Sea Prospects](images/north_sea_map.png)
*Figure: Gravity anomaly map with prospect hotspots (yellow).*

- **Export**: CSV for seismic survey planning; GeoTIFF for seismic interpretation software.

### Validation
- Correlated with known fields (e.g., Brent field anomalies match).
- Economic: Identified 3 high-potential sites for drilling.

### Extensions
- Integrate well logs (custom ingestion).
- Global scan for greenfield exploration.

See [Global Tutorial](../tutorials/03_global_processing.ipynb) for basin-scale.

---

*Case Study Date: 2025 | Data Sources: USGS, IRIS*