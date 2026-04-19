---
project: GeoAnomalyMapper-1
status: working
updated: 2026-03-14
---

# GeoAnomalyMapper - SAR Doppler Tomography Pipeline

## Resume
- **Pick up at**: [Review and update]
- **Last session**: [Auto-migrated to CONTEXT v2]
- **Blocked on**: Nothing

## Status
- **Working**: SAR data acquisition (Sentinel-1 SLC), Doppler vibrometry, PINN 3D inversion, 3D visualization, batch exploration
- **Running**: 24-target batch (Egypt + California + Vacaville) on Vast.ai H100

## Tech Stack
- Python 3.10+, PyTorch (CUDA), NumPy, SciPy, Matplotlib
- GPU: Vast.ai H100 (34.44.101.231:6993)
- Data: ESA Sentinel-1 SLC via Earthdata

## Key Files
- `slc_data_fetcher.py` — Sentinel-1 SLC data acquisition from ASF/Earthdata
- `sar_vibrometry.py` — SAR Doppler vibrometry processing (GPU-accelerated)
- `pinn_vibro_inversion.py` — Physics-Informed Neural Network for 3D wave speed inversion
- `visualize_3d_subsurface.py` — 3D visualization + anomaly extraction
- `run_biondi_exploration.py` — End-to-end pipeline orchestrator + target definitions
- `project_paths.py` — Path configuration
- `deploy_vastai.sh` — GPU instance deployment script

## Architecture
```
SLC Fetch → Doppler Vibrometry → PINN Inversion → 3D Visualization
                                      ↓
                              wave_speed_volume.npy
                              void_probability.npy
```

## Architecture Quirks
- Void probability uses sigmoid mapping (not linear): `1/(1+exp(-(speed-thresh)/temp))`
- Temperature = `bg_speed * 0.05` for sharp detection
- Threshold lowered to 0.35 (from 0.5) to catch weaker anomalies from Sentinel-1
- Sentinel-1 C-band penetration is limited; deep anomalies (>800m) are geological, not structural
- Khafre and Khufu Phase 1 runs produced identical data (same Sentinel-1 scene, overlapping domain)

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| NaN loss during training | AMP mixed precision in physics loss | Disabled AMP for physics branch |
| CUDA OOM on H100 | Batch size too large | Reduced to 4096 + gradient accumulation |
| Void probability always near 0 | Linear mapping with partial volume effect | Sigmoid mapping with tight temperature |
| f-string syntax error in SSH heredoc | PowerShell escaping `=` in f-strings | Rewrote with string concatenation |
| Khafre = Khufu in Phase 1 | Same SAR scene, wide overlapping domain | Phase 2 uses tight per-pyramid domains |

## Anti-Patterns (DO NOT)
- Do NOT use f-strings in scripts uploaded via SSH heredoc
- Do NOT use linear void probability mapping — always sigmoid
- Do NOT set void_threshold > 0.4 with Sentinel-1 data (max vp ≈ 0.40)
- Do NOT run batch sizes > 8192 on H100 80GB

## Results
- `results/phase1/` — Phase 1 visualizations + reports (Carlsbad, Khufu, Mammoth)
