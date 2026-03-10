# GeoAnomalyMapper

## Status
- **Working**: Data acquisition (Sentinel-1 SLC via ASF), SAR vibrometry (GPU-accelerated), PINN training loop, 3D visualization, anomaly extraction
- **Fixed (2026-03-10)**: Trivial collapse, domain mismatch, loss history serialization, sparsity weight, deep prior
- **Needs Validation**: Synthetic data test (Sentinel-1 is noisy — pipeline math needs proof via synthetic)
- **Blocked**: Real-world accuracy requires Spotlight SAR (Umbra/Capella) — Sentinel-1 TOPSAR has insufficient dwell time for Doppler vibrometry

## Tech Stack
- Python 3.10+, PyTorch (CUDA), NumPy, SciPy, Matplotlib, tqdm
- GPU: Tested on H100 (80GB) and RTX 4090 (24GB)
- Data: Sentinel-1 SLC via ASF DAAC / EarthData

## Key Files
- `pinn_vibro_inversion.py` — PINN model, training loop, loss functions, volume extraction
- `run_biondi_exploration.py` — Orchestrator: profiles, target configs, pipeline execution
- `sar_vibrometry.py` — SAR Doppler vibrometry signal extraction (GPU-accelerated)
- `visualize_3d_subsurface.py` — 3D rendering, anomaly body extraction, cross-sections
- `slc_data_fetcher.py` — Sentinel-1 SLC data acquisition via ASF
- `elastic_pinn.py` — v2 elastic PINN (outputs shear modulus, avoids 2nd-order derivatives)

## Architecture Quirks
- PINN uses SIREN activation (sin(w0*x)) instead of ReLU/Swish — periodic basis functions match wave physics
- Fourier Features (Random Fourier encoding) handle coordinates — NOT positional encoding
- Physics loss (Helmholtz) uses chain-rule scaling for normalized→physical domain conversion
- Training loop does NaN rollback with LR halving — critical for stability
- Surface sampler uses CDF-based importance sampling on GPU for large SAR images

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| Trivial collapse (0 anomalies) | Hardcoded sparsity weight 5.0 + L1 penalty crushed all voids | Replaced with Cauchy prior (gamma=0.05), weight 0.01 |
| Uniform 3500 m/s output | Physics weight 0.0001 vs data 50.0 — physics had zero influence | Rebalanced: physics=1.0, data=20.0 |
| Visualization domain mismatch | Visualizer defaults to 5000m when not passed domain from PINN | Added metadata failsafe + explicit passing from orchestrator |
| loss_history.npy scalar | np.save(dict) saves whole dict as 0-d array, unanalyzable | Switched to JSON serialization |
| deep_prior_weight unused | Config key existed but was never added to the loss graph | Implemented deep_boundary_loss() for z>0.8 |
| CUDA OOM on H100 with float64 | float64 physics doubles VRAM for all batch tensors | Set use_float64_physics=False in 'high' profile for H100 |
| torch.compile crash with float64 | compile doesn't support dynamic dtype casting | Conditional: only compile when not use_float64_physics |
| Sentinel-1 can't do vibrometry | TOPSAR dwell time ~0.1s, need >1s for sub-Hz vibration pickup | Fundamental limitation — need Spotlight SAR (Umbra/Capella) |

## Anti-Patterns (DO NOT)
- NEVER hardcode loss weights in the training loop — always read from cfg dict
- NEVER use L1 sparsity for void detection — it linearly penalizes large deviations, preventing real anomalies
- NEVER save Python dicts with np.save() — use json.dump()
- NEVER assume visualizer gets domain from orchestrator — always have metadata failsafe
- NEVER use AMP (float16) for 2nd-order autograd — causes NaN in physics loss
- NEVER skip physics warmup — raw Helmholtz loss ~20M causes NaN within 15 epochs

## Loss Weight Reference (2026-03-10 Expert-Validated)
```
physics_weight:        1.0    # Helmholtz PDE — primary driver
data_weight:          20.0    # Surface vibration fit — lowered from 50 (noisy data)
sparsity_weight:       0.01   # Cauchy prior — was 5.0 L1 (caused collapse)
regularization_weight: 0.1    # TV reg — promotes sharp boundaries
deep_prior_weight:     0.1    # Background at z>0.8 — prevents downward hallucination
```

## Build / Verify
```bash
# Synthetic validation (mandatory before real runs):
python run_biondi_exploration.py --synthetic-fallback

# Real target (standard profile):
python run_biondi_exploration.py --target "Carlsbad Caverns" --profile standard
```

## Saved Results
- Carlsbad (failed): `C:\tmp\carlsbad_results\` — trivial collapse, 0 anomalies
- Debug package: `debug_carlsbad_2026_03_10/` — expert prompt + 9 core files
