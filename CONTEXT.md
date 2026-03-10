# GeoAnomalyMapper

## Status
- **Working**: SLC download, SAR vibrometry processing, PINN training (STABLE past epoch 150+), anomaly extraction/visualization
- **Training**: First successful training run in progress on Vast.ai RTX 4090. Loss dropping steadily: phys 180K→33K, data 6.1→1.5, total 393→79 over 150 epochs.
- **v2 Pipeline**: Elastic PINN with mixed-variable formulation + gravity joint inversion. Code complete, not yet tested on GPU.

## Tech Stack
- Python 3.11+, PyTorch 2.x (CUDA)
- Sentinel-1 C-band IW SLC data via NASA Earthdata API
- PINN with Helmholtz PDE constraint for wavefield inversion (v1)
- Elastic wave equation with shear modulus output (v2)
- Remote execution on Vast.ai RTX 4090 instances

## Key Files
- `pinn_vibro_inversion.py` — Core PINN training & physics loss (Helmholtz equation, SIREN init, chain-rule scaling)
- `run_biondi_exploration.py` — Orchestrator: targets, resolution profiles, pipeline execution
- `sar_vibrometry.py` — SAR Doppler vibrometry processing (GPU-accelerated)
- `project_paths.py` — Path configuration
- `.env` — NASA Earthdata credentials (on remote only)
- `v2/elastic_pinn.py` — Mixed-variable elastic PINN (no 2nd-order autograd)
- `v2/gravity_fetcher.py` — USGS Bouguer gravity anomaly data fetcher
- `v2/run_v2.py` — v2 orchestrator

## Architecture Quirks
- The PINN outputs real+imaginary wavefield components (U_real, U_imag) and wave speed (c)
- Physics loss uses 2nd-order autograd derivatives (∇²U) which are inherently unstable in float32
- `torch.compile` is INCOMPATIBLE with this pipeline — the autograd graph is dynamic
- AMP (float16) causes NaN in the Helmholtz second derivatives
- Vibration maps are downsampled to 1024×1024 max for GPU memory management
- float64 physics casts model to .double() temporarily — collocation coords stay float64 after!

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
| **xavier_uniform_ on SIREN** | xavier init causes exponential activation growth in sin() | **SIREN-specific init: W~U(-√(6/n), √(6/n)), first layer ω₀/n** |
| **CUDA OOM with float64 physics** | model.double() doubles VRAM (12→24GB) | Reduced high profile: neurons 768→512, layers 10→8, batch 8192→4096 with 2x accum |
| **dtype mismatch in TV reg** | After physics loss, coords remain float64 but model is back to float32 | .float() on detached coords before TV regularization |

## Anti-Patterns (DO NOT)
- Do NOT enable torch.compile — breaks autograd.grad physics loss
- Do NOT enable AMP (use_amp=True) — float16 causes NaN in 2nd-order gradients
- Do NOT set physics_weight > 0.001 without warmup — instant NaN divergence
- Do NOT trust anomaly results from runs with < 200 epochs — initialization artifacts
- Do NOT SCP shell scripts from Windows — line ending corruption
- Do NOT use xavier_uniform_ with SIREN (sin) activations — use SIREN-specific init
- Do NOT use hidden_neurons > 512 or hidden_layers > 8 with float64 physics on 24GB GPU

## Build / Verify
```bash
# Remote (Vast.ai):
cd /workspace/geo && python3 run_biondi_exploration.py --phase 1 --resolution high
# Monitor:
tail -f /tmp/run3.log
nvidia-smi -l 5
# Full pipeline (Phase 1 + Phase 2):
bash /workspace/geo/run_all_phases.sh
```

## Current Experiment (2026-03-10 ~03:30 PT)
**STABLE TRAINING CONFIRMED** — First successful run past epoch 150 (previously crashed at 13).
- Physics: 180,000 → 33,000 (↓5.5x, still dropping)
- Data: 6.13 → 1.55 (↓4x)
- Total: 393 → 79 (↓5x)
- VRAM: 0.1/23.5 GB (plenty of headroom)
- Speed: 2.19 s/epoch → ~3 hr/target at 5000 epochs
- Running Phase 1 (Carlsbad → Mammoth → Khufu → Khafre → Menkaure)
- Phase 2 (California) queued after Phase 1 via run_all_phases.sh
