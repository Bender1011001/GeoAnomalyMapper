# GeoAnomalyMapper

## Status
- **Working**: Gravity/Magnetic PINN inversion (pinn_gravity_inversion.py), supervised classifier (classify_supervised.py), target extraction, validation pipeline, USGS MRDS data fetching, InSAR coherence features, 2D plotting (folium/matplotlib)
- **Working — SAR Doppler Tomography Pipeline**: SLC data fetching (slc_data_fetcher.py — Umbra + Sentinel-1 + Capella), Doppler sub-aperture vibrometry (sar_vibrometry.py), Helmholtz PINN 3D inversion (pinn_vibro_inversion.py), 3D visualization (visualize_3d_subsurface.py), Exploration orchestrator (run_biondi_exploration.py)
- **Broken**: None critical. Archive contains legacy/experimental code.

## Tech Stack
- Python 3.10+
- PyTorch >= 2.0 (CUDA preferred for PINN training)
- rasterio for GeoTIFF I/O
- scipy for FFT-based sub-aperture decomposition
- scikit-image for marching cubes / GLCM
- asf_search for Sentinel-1 SLC data from ASF DAAC
- requests for Umbra Space / Capella Space STAC API access
- pyvista for 3D volumetric visualization
- NASA Earthdata credentials required for Sentinel-1 SLC download
- Umbra Space open data requires NO credentials (free AWS S3)

## Key Files
- `run_biondi_exploration.py` — **Orchestrator**: Phased scanning (calibration → CA → USA) with resolution profiles, auto-advance, synthetic fallback, consolidated reports. Data priority: Umbra → Sentinel-1 → Synthetic.
- `pinn_gravity_inversion.py` — Parker-Oldenburg gravity → density inversion (2D, spectral)
- `pinn_vibro_inversion.py` — Helmholtz wave equation → 3D wave-speed/density inversion (domain=800m, grid=128³, data_weight=50, deep_prior=0.001)
- `sar_vibrometry.py` — Doppler sub-aperture decomposition for surface vibration mapping
- `slc_data_fetcher.py` — SLC data acquisition: Umbra Space X-band (16cm, free), Capella X-band, Sentinel-1 C-band
- `visualize_3d_subsurface.py` — 3D isosurface extraction, anomaly classification, PyVista rendering
- `project_paths.py` — Centralised path management (DATA_DIR, RAW_DIR, etc.)
- `utils/data_fetcher.py` — USGS MRDS mineral deposit data for training/validation
- `loss_functions.py` — Structure-guided TV loss, magnetic gradient weights
- `train_void_pinn.py` — Void-mode PINN training (DUB detection)
- `classify_supervised.py` — Supervised mineral exploration classifier

## Architecture Quirks
- **Dual PINN Paradigm**: The project now has TWO PINN approaches:
  1. `pinn_gravity_inversion.py` — U-Net based, spectral forward operator, 2D-only (gravity → density)
  2. `pinn_vibro_inversion.py` — Coordinate-based MLP with Fourier features, 3D (vibration → wave speed + density)
  These serve different purposes: (1) for surface mineral detection, (2) for deep 3D structure imaging.
- **SAR vibrometry operates on single-pass SLC data**, NOT multi-temporal InSAR. It measures vibrations during one satellite flyover by splitting the Doppler spectrum. This is fundamentally different from the existing InSAR coherence pipeline.
- **Fourier Features in PINN**: Random Fourier encoding (Tancik et al. 2020) is critical for the Helmholtz PINN to learn high-frequency spatial patterns. Without it, the MLP will smooth out sharp void boundaries.
- The `project_paths.py` module uses `GEOANOMALYMAPPER_DATA_DIR` env var for custom data roots.
- **Resolution Profiles**: The orchestrator has three profiles — `quick` (64³, 500 epochs), `standard` (128³, 3000 epochs), `high` (128³, 5000 epochs). Use `standard` for calibration and `high` for final scans.
- **Orchestrator outputs JSON + TXT reports** per phase under `data/biondi_exploration/phase_N_report.txt`.
- **Data Source Priority**: Umbra X-band (Spotlight, 16cm-1m, free) → Sentinel-1 C-band (TOPSAR, 5x20m, free) → Synthetic fallback.
- **Domain Sizing**: Domain shrunk from 5000m→800m, grid increased to 128³ for ~6.2m voxel resolution. Old 156m voxels completely erased sub-20m structures.
- **Depth-adaptive frequency**: excitation_frequency_hz adapts to target depth: shallow (5-15Hz), medium (1-5Hz), deep (0.5-1Hz).

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| SLC data stored as 2-band (I,Q) not complex | rasterio reads bands separately | `slc_data_fetcher.py` handles both formats |
| Helmholtz PINN diverges early | Gradient explosion from second derivatives | Gradient clipping (max_norm=1.0) + cosine warm restart LR |
| Sub-aperture coherence too low | Insufficient overlap between bands | Default 30% overlap; increase if coherence < 0.3 |
| Marching cubes fails on empty volumes | Threshold outside data range | Pre-check with data range validation |
| PyVista not available on headless servers | Needs GPU/display for interactive mode | Falls back to matplotlib cross-sections |
| **PINN Trivial Collapse** — wave speed = uniform 3500 m/s, 0 anomalies detected | **deep_prior_weight=0.1 penalized deep voids so heavily the PINN chose solid rock everywhere. data_weight=10 let physics smoothing dominate over satellite observations. 5000m domain / 32 grid = 156m voxels erased sub-20m structures. excitation_frequency_hz=5.0 too high for deep (500m+) structures.** | **deep_prior_weight→0.001, data_weight→50.0, domain→800m, grid→128³ (6.2m voxels), freq→0.5-1.0Hz. Also added Umbra X-band data source (Spotlight mode, 16cm res) to replace Sentinel-1 TOPSAR (5x20m).** |
| Phase 1 detected 0 anomalies | Grid 32×32×16 too coarse, 500 epochs insufficient, synthetic fallback used random noise | Fixed: use `standard` profile (128³, 3000 epochs), proper synthetic via generate_synthetic_vibration_test |
| Orchestrator called download_slc_products (plural) | Function doesn't exist, only download_slc_product (singular) | Fixed: call download_slc_product per product |
| .env pointed to D: drive | D: drive not mounted | Fixed: removed GEOANOMALYMAPPER_DATA_DIR, falls back to local data/ |
| slc_data_fetcher.np.save() used in orchestrator | Incorrect module reference | Fixed: use np.save() directly, replaced with proper synthetic generator |
| GPU only at 5% utilization | CPU-bound `np.random.choice` surface sampling every epoch + small batch sizes | Fixed: `SurfaceSamplerGPU` pre-computes CDF on GPU, `torch.searchsorted` replaces numpy. Batch 2K colloc + 1K boundary + 4x grad accum |
| Low effective batch per optim step | Single forward pass per step underutilizes GPU | Fixed: gradient accumulation (4 micro-batches per optimizer step = 8K effective collocation) |
| PINN training hangs at "Starting PINN training" | `standard` profile had `gradient_accumulation_steps: 64` — 64 second-order autograd passes before any log output. On 8GB GPU, either OOMs silently or takes hours per epoch with zero visible progress | Fixed: reduced to `accum=4, colloc=2048, boundary=1024`. Also `grid_nz` reduced from 64 to 32. Added per-substep verbose logging for first 3 epochs |
| TV regularization wasting VRAM | `tv_regularization()` used `create_graph=True` in its autograd.grad call — needlessly building a 2nd-order computation graph through the regularizer, ~40% extra VRAM | Fixed: changed to `create_graph=False` since we never differentiate through the TV loss |

## Anti-Patterns (DO NOT)
- Do NOT use pre-processed InSAR coherence for vibrometry — it averages out high-frequency Doppler content
- Do NOT use the gravity PINN loss (Poisson equation) for vibration inversion — use Helmholtz
- Do NOT train the Helmholtz PINN with uniform collocation sampling — use quadratic bias toward surface
- Do NOT skip Fourier feature encoding for the coordinate MLP — it will fail to learn sharp boundaries
- Do NOT use `quick` resolution for calibration targets — results will be empty at 64³ grid
- Do NOT set deep_prior_weight > 0.01 — causes trivial collapse where the PINN erases deep anomalies
- Do NOT set data_weight < 20 — allows physics smoothing to dominate over satellite observations
- Do NOT use domain_width > 1000m with grid < 128 — voxel resolution will be too coarse to resolve tunnels/chambers
- Do NOT use Sentinel-1 TOPSAR data when Umbra Spotlight data is available — Sentinel-1's 0.1s dwell time cannot track sub-second micro-vibrations

## Build / Verify
```bash
# Quick synthetic end-to-end test (no satellite data needed)
python sar_vibrometry.py synthetic --size 256 --anomalies 3 --run-pipeline

# Full pipeline (requires SLC data)
python slc_data_fetcher.py search --lat 29.9792 --lon 31.1342 --buffer 0.5
python sar_vibrometry.py process --input data/slc/bursts/burst_00.npy
python pinn_vibro_inversion.py --vibration data/vibrometry/outputs/vibration_amplitude.npy --epochs 5000
python visualize_3d_subsurface.py --volume data/inversion_3d/outputs/wave_speed_volume.npy

# Orchestrated multi-target scanning
python run_biondi_exploration.py --phase 1 --synthetic-fallback --resolution standard --clean-rerun
python run_biondi_exploration.py --phase 1 --synthetic-fallback --resolution standard --auto-advance
```

## Data Flow (SAR Doppler Tomography Pipeline)
```
Umbra X-band SLC (.tif)       Sentinel-1 SLC (.SAFE)        Capella X-band SLC
  [PREFERRED: 16cm res]          [FALLBACK: 5x20m]              [COMMERCIAL]
       │                              │                              │
       └──────────────┬───────────────┴──────────────────────────────┘
                      │
     slc_data_fetcher.py (burst extraction)
                      │
                      ▼
     Complex SLC burst (.npy)
                      │
     sar_vibrometry.py (sub-aperture decomposition)
                      │
                      ├─ vibration_amplitude.npy (2D surface mm/s)
                      ├─ vibration_frequency.npy (2D surface Hz)
                      └─ vibration_quality.npy   (2D coherence)
                      │
     pinn_vibro_inversion.py (Helmholtz PINN)
        Domain: 800m × 800m × 1000m
        Grid: 128 × 128 × 64 (~6.2m voxels)
        data_weight=50, deep_prior=0.001
        freq: 0.5-5 Hz (depth-adaptive)
                      │
                      ├─ wave_speed_volume.npy     (3D, m/s)
                      ├─ density_contrast_volume.npy (3D, kg/m³)
                      └─ void_probability_volume.npy (3D, 0-1)
                      │
     visualize_3d_subsurface.py (extraction + rendering)
                      │
                      ├─ isosurface_3d.png          (3D render)
                      ├─ depth_cross_sections.png   (2D slices)
                      ├─ subsurface_volume.vtk      (3D mesh)
                      ├─ void_surface.stl           (printable mesh)
                      ├─ detected_anomalies.csv     (anomaly catalog)
                      └─ anomaly_report.txt         (detailed report)
```
