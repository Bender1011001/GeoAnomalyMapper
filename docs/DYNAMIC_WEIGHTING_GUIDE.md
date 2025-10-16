# Dynamic Weighting System User Guide

GeoAnomalyMapper fuses multi-resolution rasters using an adaptive weighting
scheme implemented in `gam.fusion.weight_calculator`. This guide explains the
mathematics, configuration controls, and diagnostic workflows needed to tune the
system for high-confidence anomaly detection.

## Algorithm overview

Each fusion product defines the contributing layers and their nominal ground
sample distance in metres. The weighting engine performs three steps:

1. **Resolution inversion** – Convert each resolution value \(r_i\) into an
   inverse-resolution score \(s_i = 1 / r_i\). Finer grids produce larger
   scores.
2. **Temperature scaling** – Apply a softmax temperature \(T\) to control how
   sharply weights concentrate on high-resolution inputs:
   \[
   s'_i = s_i^{1/T}
   \]
3. **Normalisation** – Produce final weights via a softmax:
   \[
   w_i = \frac{s'_i}{\sum_j s'_j}
   \]

Setting \(T < 1\) emphasises the highest-resolution layers while
\(T > 1\) yields a more balanced blend. The default \(T = 1\) provides a
straightforward inverse-resolution weighting.

The fused pixel is then computed as the weighted average of the aligned inputs:
\[
F(x, y) = \frac{\sum_i w_i \cdot L_i(x, y)}{\sum_i w_i}
\]

## Configuration

Fusion behaviour is described in two places:

- `config/fusion.yaml` – Defines fusion products, listing each source raster,
  its path, and its resolution in metres.
- `config/config.json` – Holds global fusion settings, including whether dynamic
  weighting is enabled and the default temperature.

### Example configuration

```yaml
# config/fusion.yaml
products:
  - name: carlsbad_priority
    layers:
      - name: insar_velocity
        path: data/features/insar_velocity.tif
        resolution: 10.0
      - name: xgm2019e_gravity
        path: data/features/xgm2019e_gravity.tif
        resolution: 250.0
```

```json
// config/config.json
{
  "fusion": {
    "dynamic_weighting": true,
    "temperature": 1.0
  }
}
```

Set `dynamic_weighting` to `false` to fall back to equal weighting (useful for
sensitivity analyses). The CLI also exposes a `--static` flag for ad-hoc runs.

### Environment overrides

Every configuration value can be overridden without editing JSON files. For
example:

```bash
export GAM__FUSION__TEMPERATURE=0.75
export GAM__FUSION__DYNAMIC_WEIGHTING=true
```

## Running the fusion pipeline

Invoke the module via the provided CLI wrapper:

```bash
python -m gam.fusion.multi_resolution_fusion run \
  --config config/fusion.yaml \
  --output data/products/fusion \
  --temperature 0.9
```

For each product entry, the pipeline:

1. Loads raster metadata from the YAML definition.
2. Computes weights using `resolution_weighting` and the configured temperature.
3. Applies the weighted average to produce a fused Cloud Optimised GeoTIFF using
   `gam.io.cogs.write_cog`.
4. Logs the computed weight vector for traceability.

Include `--static` to disable dynamic weighting for specific runs without
modifying configuration files.

## Inspecting weights programmatically

Use the weighting helper directly to audit the contribution of each layer:

```python
from gam.fusion.weight_calculator import resolution_weighting

resolutions = {"insar": 10.0, "gravity": 250.0, "magnetic": 500.0}
weights = resolution_weighting(resolutions, temperature=0.8)
print(weights.weights)
# {'insar': 0.88, 'gravity': 0.10, 'magnetic': 0.02}
```

For custom diagnostics, persist weights alongside fusion outputs:

```python
from pathlib import Path
from gam.fusion.multi_resolution_fusion import fuse_layers

layers = {
    "insar": {"path": Path("data/features/insar_velocity.tif"), "resolution": 10.0},
    "gravity": {"path": Path("data/features/xgm2019e_gravity.tif"), "resolution": 250.0},
}
output = fuse_layers(layers, Path("data/products/custom_fusion.tif"))
```

## Best practices

- **Validate resolution metadata** – Ensure the metre values in
  `config/fusion.yaml` reflect the actual sampling distance of each raster. The
  weighting calculation trusts these numbers.
- **Monitor weight distributions** – Log the resulting weights for each product
  and verify they align with expectations (e.g., InSAR dominates in urban areas,
  gravity has a larger share in sparsely observed regions).
- **Experiment with temperature** – Slight adjustments (0.7–1.3) often improve
  results when blending very heterogeneous datasets.
- **Combine with uncertainty masks** – If additional confidence measures exist
  (e.g., coherence rasters), multiply the final fused product by those masks to
  suppress unreliable areas.

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| All layers receive similar weights | Temperature too high | Lower `fusion.temperature` or supply an explicit value when calling `resolution_weighting`. |
| A coarse layer dominates | Resolution metadata incorrect | Update the `resolution` value in `config/fusion.yaml` to the true sampling distance. |
| Output appears blurred | One layer has significantly lower resolution | Consider resampling higher-resolution layers to a coarser grid before fusion or run separate products per scale. |
| Pipeline writes empty output | Input rasters missing or misaligned | Validate paths in `config/fusion.yaml` and ensure rasters share a common grid (see `make harmonize`). |

The dynamic weighting system is deterministic, lightweight, and designed for
reproducibility. Combined with rigorous validation, it delivers traceable fusion
products suitable for high-stakes geological assessments.
