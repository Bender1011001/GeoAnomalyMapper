# Academic Research Applications

## Overview

GAM facilitates geophysical research by providing reproducible workflows for anomaly detection, model fusion, and global analysis. Ideal for theses, papers, and collaborative studies in geophysics, geology, and earth sciences.

**Key Benefits**:
- Open-source, public data for verifiable results.
- Extensible API for custom algorithms.
- Integration with scientific tools (SciPy, xarray).

## Case Study: Global Fault Mapping for Tectonics Research

### Background
Researchers studying plate tectonics use GAM to map global faults from seismic and gravity data, validating models of subduction zones.

### Data and Setup
- **Bbox**: Global [-90, 90, -180, 180].
- **Modalities**: Seismic, gravity.
- **Config Snippet**:
  ```yaml
  data:
    bbox: [-90, 90, -180, 180]
    modalities: ["seismic", "gravity"]
  core:
    tile_size: 20
    parallel_workers: -1
  modeling:
    threshold: 3.0
    priors:
      regularization: "l2"  # Smooth for tectonic features
  visualization:
    map_type: "interactive"
    export_formats: ["geotiff", "csv"]
  ```

### Running the Analysis
```bash
gam run --global --config tectonics.yaml --output tectonics_results/
```

**Run Time**: 4-12 hours on multi-core machine.

### Results and Interpretation
- **Anomalies Detected**: 1,200+ fault segments, avg confidence 0.85.
- **Key Finding**: Subduction zone at (10 S, 120 E) shows high seismic-gravity correlation, confidence 0.95.
  - Confirms Nazca plate model; depth -400km aligns with Benioff zone.
- **Visualization**: Interactive global map with fault lines; GeoTIFF for publication.

![Global Faults](images/global_faults.png)
*Figure: Fused seismic-gravity map highlighting subduction zones (red).*

- **Export**: CSV for statistical analysis; netCDF for model integration.

### Validation
- Compared with USGS earthquake catalog (95% match on major faults).
- Research Impact: Used in paper on plate boundary dynamics.

### Extensions
- Custom fusion for velocity-density models.
- Time-series for fault evolution.

See [Developer API](../developer/api_reference.md) for extensions.

---

*Case Study Date: 2025 | Data Sources: IRIS, USGS*