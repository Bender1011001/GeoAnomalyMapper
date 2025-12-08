# GeoAnomalyMapper Configuration Presets Comparison

This table compares the key configurable parameters across the default `config.yaml` and the three proposed presets. Changes are designed to meet the specific goals:

- **Fast Mode**: Prioritizes speed (<2 min runtime) by reducing epochs, increasing stride (fewer chips), and optimizing batch size/LR for quick convergence.
- **Production Mode**: Maximizes quality/smoothness with more epochs, higher overlap (smaller stride), and conservative LR.
- **Mining Mode**: Targets large-scale geological features (e.g., deposits/voids) with larger chip size for broader context, moderate overlap, more epochs, lower LR for precision, and z-score normalization (requires minor code update in [`phase8_spatial_analysis.py`](phase8_spatial_analysis.py:99) to use StandardScaler per channel instead of minmax).

| Parameter          | Default (`config.yaml`) | Fast (`config_fast.yaml`) | Production (`config_production.yaml`) | Mining (`config_mining.yaml`) |
|--------------------|--------------------------|---------------------------|--------------------------------------|-------------------------------|
| **chip_size**     | 64                      | 64                       | 64                                  | 128                          |
| **stride**        | 32                      | 64                       | 16                                  | 32                           |
| **batch_size**    | 128                     | 512                      | 64                                  | 64                           |
| **epochs**        | 15                      | 3                        | 50                                  | 30                           |
| **learning_rate** | 0.001                   | 0.005                    | 0.0005                              | 0.0002                       |
| **normalization** | minmax                  | minmax                   | minmax                              | zscore                       |
| **optimizer**     | adam                    | adam                     | adam                                | adam                         |
| **loss**          | mse                     | mse                      | mse                                 | mse                          |
| **Estimated Runtime** | ~10-15 min            | <2 min                | ~45-60 min                          | ~20-30 min                   |

**Notes**:
- Model architecture (`conv_autoencoder_v1`) and other sections (data.inputs, output, visualization) remain identical across presets.
- `validation_split` (0.1) unused in current code.
- Larger `chip_size=128` in Mining works with existing model (encoder/decoder strides scale).
- Runtime estimates assume ~5000x5000 USA mosaic on RTX 4060 Ti; actual varies by hardware/data size.
- For Mining `zscore`: Update [`phase8_spatial_analysis.py`](phase8_spatial_analysis.py:136) to use `StandardScaler().fit_transform(arr.reshape(-1,1)).reshape(arr.shape)` per layer.