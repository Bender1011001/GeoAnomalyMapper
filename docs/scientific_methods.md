# Scientific Methods in GeoAnomalyMapper (GAM)

This document provides a high-level overview of the geophysical methods implemented in GAM for anomaly detection and modeling. Each section describes the core algorithms, key configuration options, and relevant literature references. These methods integrate public data sources with advanced inversion techniques to map subsurface anomalies across scales.

GAM emphasizes scientific rigor: methods are grounded in established geophysical theory, with configurable parameters for reproducibility and customization. For implementation details, see the [API Reference](../developer/api_reference.md) and source code.

## Gravity Method

### Description
GAM's gravity modeling estimates 3D density contrasts from observed gravity anomalies using SimPEG's potential fields framework. The pipeline includes terrain corrections via prism-based integration (accounting for topographic effects on Bouguer anomalies) and joint inversion with L1 (sparse) or L2 (smooth) regularization to recover compact or diffuse density models. Data is gridded to user-specified resolution, with forward modeling on adaptive TreeMesh for efficiency. Anomalies are flagged via z-score thresholding on residuals post-inversion.

This approach is ideal for detecting density voids (e.g., archaeological cavities) or intrusions (e.g., mineral deposits), with uncertainty propagation from model covariance.

### Configuration Options
- `modeling.gravity.dem_path`: Path to DEM GeoTIFF for terrain correction (optional; default: None, skips correction).
- `modeling.gravity.base_density`: Reference crustal density (kg/m³; default: 2000).
- `modeling.gravity.regularization`: 'l1' for sparse models or 'l2' for smooth (default: 'l2').
- `modeling.gravity.alpha_s`: Smallness regularization weight (default: 1e-4).
- `modeling.gravity.alpha_x/y/z`: Directional smoothness weights (default: 1.0).
- `modeling.gravity.max_iterations`: Inversion convergence limit (default: 20).
- `preprocessing.gravity.grid_res`: Output grid resolution (degrees; default: 0.1).
- `modeling.threshold`: Z-score for anomaly detection (default: 2.0).

Example in config.yaml:
```yaml
modeling:
  gravity:
    dem_path: "data/dem_giza.tif"
    regularization: "l1"
    alpha_s: 1e-5
```

### Literature References
- SimPEG documentation: Cockett et al. (2015), "SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications."
- Terrain correction: Nagy et al. (2000), "The gravitational potential and its derivatives for the prism."
- Regularization: Oldenburg (1996), "The choice of parameterization in non-linear inverse problems," Geophysical Journal International.

**Assumptions and Limits**: L2 misfit assumes Gaussian errors; non-Gaussian noise may bias. Uncertainty from diagonal Hessian underestimates correlations. Valid for regional scales; high-frequency noise requires preprocessing.

## Seismic Method

### Description
Seismic processing in GAM focuses on travel-time tomography and event detection. Waveforms are bandpass filtered (configurable frequency range) before applying the STA/LTA (Short-Term Average / Long-Term Average) algorithm for P-wave picking, which identifies onset times for first-arrival tomography. Inversion uses PyGIMLi for 2D/3D velocity models via finite-element eikonal solvers, enabling anomaly detection in velocity perturbations (e.g., low-velocity zones indicating fluids or faults). Fusion with other modalities weights seismic contributions by uncertainty.

This method excels in hazard monitoring (e.g., fault mapping) and resource exploration, with picking robust to noise via threshold tuning.

### Configuration Options
- `preprocessing.seismic.lowcut/highcut`: Bandpass filter bounds (Hz; default: 0.1/1.0).
- `preprocessing.seismic.sta_len/lta_len`: STA/LTA window lengths (s; default: 1.0/10.0).
- `preprocessing.seismic.threshold`: STA/LTA trigger threshold (default: 3.5).
- `modeling.seismic.mesh_type`: '2d' or '3d' for PyGIMLi (default: '2d').
- `modeling.seismic.regularization`: 'tikhonov' or 'sparsity' (default: 'tikhonov').
- `modeling.seismic.max_iterations`: Tomography solver limit (default: 50).
- `preprocessing.seismic.grid_res`: Spatial grid resolution (degrees; default: 0.1).
- `modeling.threshold`: Anomaly z-score (default: 2.0).

Example in config.yaml:
```yaml
preprocessing:
  seismic:
    lowcut: 0.05
    highcut: 2.0
    sta_len: 0.5
    threshold: 4.0
modeling:
  seismic:
    mesh_type: "3d"
```

### Literature References
- STA/LTA picking: Trnkoczy (2010), "Understanding and removing effects of instrumental irregularities in seismic data," in New Manual of Seismological Observatory Practice.
- Tomography: Günther et al. (2022), "PyGIMLi: An open-source library for modelling and inversion in geophysics," Computers & Geosciences.

## InSAR Method

### Description
GAM processes InSAR interferograms for surface deformation mapping, using SNAPHU for robust phase unwrapping (statistical-cost network-flow algorithm) to resolve ambiguities in wrapped phase. Atmospheric delays are corrected via spatio-temporal filtering (temporal median + spatial Gaussian), followed by conversion to line-of-sight (LOS) displacements. Inversion employs elastic half-space models: Mogi for point sources (volcanic inflation) or Okada (via PyCoulomb) for rectangular dislocations (fault slip). Anomalies are detected as deformation hotspots exceeding thresholds.

Suitable for monitoring subsidence, tectonics, or anthropogenic changes, with LOS projections accounting for incidence/heading angles.

### Configuration Options
- `preprocessing.insar.wavelength`: Radar wavelength (m; default: 0.056 for Sentinel-1).
- `preprocessing.insar.snaphu_region_size`: Unwrapping tile size (pixels; default: 512).
- `preprocessing.insar.apply_atm_correction`: Enable/disable atmospheric filtering (default: True).
- `preprocessing.insar.time_window`: Temporal median window (acquisitions; default: 5).
- `preprocessing.insar.spatial_sigma`: Gaussian spatial filter sigma (default: 1.0).
- `modeling.insar.source_type`: 'mogi' or 'okada' (default: 'mogi').
- `modeling.insar.poisson`: Poisson ratio (default: 0.25).
- `modeling.insar.modulus`: Young's modulus (Pa; default: 80e9).
- `modeling.insar.max_iterations`: Least-squares limit (default: 100).
- `preprocessing.insar.grid_res`: Resolution (degrees; default: 0.01 for high-res InSAR).
- `modeling.threshold`: Deformation threshold (mm; default: 5.0).

Example in config.yaml:
```yaml
preprocessing:
  insar:
    apply_atm_correction: true
    time_window: 3
    spatial_sigma: 0.5
modeling:
  insar:
    source_type: "okada"
    poisson: 0.3
```

### Literature References
- Phase unwrapping: Chen & Zebker (2001), "Network approaches to two-dimensional phase unwrapping: InSAR and MRI," IEEE Transactions on Geoscience and Remote Sensing.
- Atmospheric correction: Yun et al. (2011), "Spatio-temporal atmospheric phase screen estimation for InSAR time series," IEEE Transactions on Geoscience and Remote Sensing.
- Okada modeling: Okada (1985), "Surface deformation due to shear and tensile faults in a half-space," Bulletin of the Seismological Society of America. Implemented via PyCoulomb (Barnhart & Lohman, 2010).

---

*Last Updated: 2025-09-26 | For updates, see GAM changelog.*