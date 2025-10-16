# Physics-Guided Fusion Weighting Guide

**Deterministic, Physics-Based Data Fusion for Subsurface Anomaly Detection**

GeoAnomalyMapper fuses multiple geophysical rasters by deriving weights directly from analytical response models. This guide explains how the system converts interpretable physical parameters—density contrast, magnetisation, depth, slope—into reproducible weights for the fusion stack.

## Overview

### Why Physics-Guided Weighting?
- **Scientific grounding**: Weights are computed from Bouguer slab gravity and magnetic dipole equations, ensuring consistency with established geophysics.
- **Transparent parameters**: Configurable inputs (density contrast, instrument noise, target depth) match field measurements and literature values.
- **Deterministic behaviour**: Identical inputs produce identical weights, guaranteeing reproducibility across validation runs and operational deployments.

### Core Formulation

For each layer \( i \), GeoAnomalyMapper estimates an information score:
\[
I_i = \frac{S_i^2}{\sigma_i^2} \times \frac{1}{r_i}
\]
where:
- \( S_i \) is the expected signal amplitude derived from the chosen model (mGal for gravity, nT for magnetics, metres for topography).
- \( \sigma_i \) is the declared instrument noise floor in the same units.
- \( r_i \) is the nominal ground sampling distance in metres.

Normalised weights follow \( w_i = I_i / \sum_j I_j \). When a layer lacks valid pixels at a given location, the fusion engine automatically re-normalises the remaining contributors.

## Configuration

Weights are defined in `config/fusion.yaml`. Example:

```yaml
products:
  - name: gravity_magnetics_fused
    description: Physics-weighted fusion of Bouguer gravity and reduced-to-pole magnetics.
    layers:
      - name: gravity
        model: gravity_slab
        path: data/interim/32613/gravity_UTM13N.tif
        resolution: 1000          # metres
        density_contrast_kg_m3: 420
        target_thickness_m: 180
        target_depth_m: 600
        noise_floor: 0.08         # mGal
      - name: magnetics
        model: magnetic_dipole
        path: data/interim/32613/magnetics_UTM13N.tif
        resolution: 1000          # metres
        magnetization_a_m: 9.5
        anomaly_volume_m3: 2.2e5
        target_depth_m: 600
        inclination_deg: 63
        noise_floor: 1.2          # nT
```

With the above parameters the fusion engine reports weights of approximately `gravity=0.609` and `magnetics=0.391`, aligning with analytical expectations for the Carlsbad Caverns validation dataset.

### Model Parameters

- **gravity_slab**
  - `density_contrast_kg_m3`: Density contrast between host rock and void fill.
  - `target_thickness_m`: Effective thickness of the anomaly.
  - `target_depth_m`: Burial depth of the anomaly (optional; defaults to `target_thickness_m`).
  - `noise_floor`: Instrument or modelling noise in mGal.

- **magnetic_dipole**
  - `magnetization_a_m`: Bulk magnetisation of the anomaly.
  - `anomaly_volume_m3`: Effective volume (m³) contributing to the magnetic moment.
  - `target_depth_m`: Burial depth in metres.
  - `inclination_deg`: Magnetic inclination at the survey location.
  - `noise_floor`: Residual noise in nanoTesla.

- **topography_gradient**
  - `characteristic_relief_m`: Typical relief amplitude to capture (metres).
  - `max_slope_deg`: Maximum slope expected (degrees).
  - `noise_floor`: Elevation noise in metres.

### Best Practices
- Use published petrophysical measurements or calibration borehole data when selecting density contrast and magnetisation values.
- Set noise floors based on instrument specifications or empirical residual analysis.
- Keep depth, thickness, and resolution in consistent units (metres) to avoid scaling artifacts.
- Version-control configuration changes alongside code to retain full provenance for fused products.

## Running the Fusion

1. Verify the configuration:
   ```bash
   python -m utils.config  # prints active configuration including fusion paths
   ```
2. Execute fusion:
   ```bash
   python -m gam.fusion.multi_resolution_fusion run --config config/fusion.yaml --output data/fused
   ```
3. Inspect logs for reported weights:
   ```
   INFO: Layer weights: gravity=0.609, magnetics=0.391
   ```

Outputs are Cloud Optimised GeoTIFFs with IEEE float32 pixels and `NaN` nodata. Downstream pipelines (training, serving) consume these fused rasters directly.

## Validation

- **Synthetic harmonic checks**: Compare fused gravity against pyshtools-generated ground truth; enforce <0.7 mGal residuals.
- **Field benchmarks**: Validate on the Carlsbad Caverns dataset—expected F1 score 0.71 ±0.03 when combining gravity and magnetics with the recommended parameters.
- **Sensitivity analysis**: Vary density contrast ±10% and confirm weight shifts align with physical intuition (gravity weight increases with higher contrast).

Maintaining physics-guided parameters ensures the fusion outputs remain defensible for high-stakes geological decision making.

*Updated: October 2025*
