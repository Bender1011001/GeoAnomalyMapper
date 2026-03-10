# GeoAnomalyMapper

## Status
- **Working**: SLC download, SAR vibrometry processing, PINN network forward pass, anomaly extraction/visualization
- **Broken**: PINN training crashes to NaN after ~13 epochs due to physics loss magnitude. Currently testing physics warmup fix.
- **Known False Results**: All previous runs (Egypt+California) produced identical anomaly patterns regardless of location — these are PINN initialization artifacts, NOT real detections

## Tech Stack
- Python 3.11+, PyTorch 2.x (CUDA)
- Sentinel-1 C-band IW SLC data via NASA Earthdata API
- PINN with Helmholtz PDE constraint for wavefield inversion
- Remote execution on Vast.ai RTX 4090 instances

## Key Files
- `pinn_vibro_inversion.py` — Core PINN training & physics loss (Helmholtz equation, chain-rule scaling)
- `run_biondi_exploration.py` — Orchestrator: targets, resolution profiles, pipeline execution
- `sar_vibrometry.py` — SAR Doppler vibrometry processing (GPU-accelerated)
- `project_paths.py` — Path configuration
- `.env` — NASA Earthdata credentials (on remote only)

## Architecture Quirks
- The PINN outputs real+imaginary wavefield components (U_real, U_imag) and wave speed (c)
- Physics loss uses 2nd-order autograd derivatives (∇²U) which are inherently unstable in float32
- `torch.compile` is INCOMPATIBLE with this pipeline — the autograd graph is dynamic
- AMP (float16) causes NaN in the Helmholtz second derivatives
- Vibration maps are downsampled to 1024×1024 max for GPU memory management

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| "Trivial Collapse" — PINN predicts uniform 3500 m/s | deep_prior_weight too high (0.1), data_weight too low (10) | deep_prior=0.001, data_weight=50 |
| NaN at epoch ~13 | Physics loss ~20M overwhelms data loss ~2.3 | Physics warmup: ramp from 0→target over 500 epochs |
| Same anomalies at all locations | Only 24/5000 epochs run; output = initialization bias | Need stable training for 1000+ epochs |
| CUDA Graphs error | torch.compile uses CUDA Graphs, incompatible with autograd.grad | use_compile=False |
| Windows line ending errors | SCP'd .sh scripts have \\r\\n | Send commands inline via SSH, not .sh files |
| SCP glob failures | PowerShell doesn't expand glob patterns over SCP | Use tar on remote, then SCP the tarball |
| CUDA OOM in vibrometry | Full SLC burst too large for GPU | Chunk processing in sar_vibrometry.py |

## Anti-Patterns (DO NOT)
- Do NOT enable torch.compile — breaks autograd.grad physics loss
- Do NOT enable AMP (use_amp=True) — float16 causes NaN in 2nd-order gradients
- Do NOT set physics_weight > 0.001 without warmup — instant NaN divergence
- Do NOT trust anomaly results from runs with < 200 epochs — initialization artifacts
- Do NOT SCP shell scripts from Windows — line ending corruption

## Build / Verify
```bash
# Remote (Vast.ai):
cd /workspace/geo && python3 run_biondi_exploration.py --phase 1 --resolution high
# Monitor:
tail -f /tmp/stability_test.log
nvidia-smi -l 5
```

## Current Experiment (2026-03-10)
Testing physics warmup fix: physics_weight ramps from 0→0.0001 over 500 epochs.
Goal: Achieve 500+ epochs before NaN (was: 13 epochs) so the network actually learns from SAR data.
