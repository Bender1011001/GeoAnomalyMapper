# Task: Rebuild the PINN Physics Core for SAR-Doppler Full Waveform Inversion

## Context

This codebase implements a Physics-Informed Neural Network (PINN) that inverts 2D SAR surface vibration maps into 3D underground wave-speed volumes using the Helmholtz wave equation. A deep algorithmic audit has identified **5 fatal mathematical bugs** and **4 physics-correctness issues** that collectively prevent the PINN from learning meaningful physics. The pipeline architecture and orchestration code are solid — only the physics engine needs rebuilding.

## What to Rebuild

**Rebuild these functions in `pinn_vibro_inversion.py`:**

1. `helmholtz_physics_loss()` — the Helmholtz PDE residual computation
2. `tv_regularization()` — Total Variation regularization on wave speed
3. `surface_prior_loss()` — embedding-guided sparsity prior
4. The data normalization block in `train_vibro_pinn()` (lines 922-929)
5. `_init_weights()` in `VibroInversionPINN` — SIREN initialization

**Fix this function in `sar_vibrometry.py`:**

6. `phase_to_vibration_velocity()` — SAR phase-to-velocity conversion

**Fix these in `visualize_3d_subsurface.py`:**

7. Edge sharpness metric in `extract_anomaly_bodies()`
8. PyVista array flattening order in `render_3d_subsurface()`
9. Aspect ratio floor calculation

**Fix this in `satellite_embeddings.py`:**

10. K-Means spatial sampling bias in `cluster_embedding_anomalies()`

**Keep all existing function signatures and return types unchanged** so the orchestration code (`run_biondi_exploration.py`, `embedding_target_discovery.py`) continues to work.

---

## The 5 Fatal Math Bugs (Must Fix)

### Bug 1.1 — Model Casting Disconnects Optimizer (CRITICAL)

**Location:** `helmholtz_physics_loss()` lines 455-461, 533-536

**What happens:** When `use_float64_physics=True` (default), the code calls `model.double()` to cast the entire model to float64 before computing the PDE, then `model.float()` afterward. In PyTorch, `.double()` creates **new parameter tensors**. The `Adam` optimizer was initialized with references to the original float32 tensors. After casting, the physics gradients flow into the new float64 parameters, but `optimizer.step()` only updates the old detached float32 parameters → **the PDE constraint contributes zero learning**.

**The fix:** Never cast the model. Keep it in float32. Cast only the **output tensors** (U_real, U_imag, c) to float64 before computing autograd derivatives:
```python
coords_f32 = torch.cat([x.float(), y.float(), z.float()], dim=-1)
U_real_f32, U_imag_f32, c_f32 = model(coords_f32)
U_real = U_real_f32.to(torch.float64)  # autograd graph stays connected
U_imag = U_imag_f32.to(torch.float64)
c = c_f32.to(torch.float64)
```

### Bug 1.2 — Z-Score Creates Negative Targets for Non-Negative Output (CRITICAL)

**Location:** `train_vibro_pinn()` lines 922-929

**What happens:** The vibration map is Z-scored: `(vib - mean) / std`. This produces ~50% negative values. But `surface_data_loss()` fits `|U| = sqrt(U_real² + U_imag²)` which is strictly ≥ 0. The network minimizes MSE by predicting U=0 wherever targets are negative → hallucinated "dead zones" of infinite impedance.

**The fix:** Scale to positive range: `vib / max(abs(vib))`, then clip to ≥ 0.

### Bug 1.3 — TV Regularization is Dead (CRITICAL)

**Location:** `tv_regularization()` line 722

**What happens:** `create_graph=False` returns a detached gradient tensor. When `loss.backward()` is called, it cannot propagate through the TV term → TV contributes **zero gradients** to the model parameters. The regularization does absolutely nothing.

**The fix:** `create_graph=True`. Yes, this increases VRAM usage, but the alternative is a regularizer that doesn't regularize.

### Bug 1.4 — SIREN Init Wrong for Fourier Feature Input (MEDIUM)

**Location:** `_init_weights()` lines 357-359

**What happens:** The first linear layer receives output from `FourierFeatures` (values in [-1, 1]), not raw coordinates. Using `bound = 1.0 / fan_in` (SIREN first-layer init) causes vanishing gradients because the input magnitude is much larger than expected. Should use hidden-layer init: `bound = sqrt(6.0 / fan_in)`.

### Bug 1.5 — VRAM Graph Retention

**Location:** `compute_laplacian()` lines 495-506

The 2nd-order derivatives use `create_graph=True`. This is **correct and necessary** — without it, the Helmholtz loss would be dead (same issue as the TV bug). However, the training loop should aggressively `del` intermediate tensors after `.backward()` to prevent graph accumulation across gradient accumulation steps. The current cleanup at line 1199 already does this, but verify it's sufficient.

---

## 4 Physics-Correctness Issues

### Issue 2.1 — No Absorbing Boundary Conditions (HIGH)

**What's missing:** The domain has no boundary conditions on 5 of 6 faces (everything except z=0 surface). The box acts as a perfect acoustic mirror → standing wave artifacts → PINN hallucinates voids at resonance nodes.

**The fix:** Add a Sommerfeld radiation condition loss on the 5 non-surface boundaries:
$$\nabla U \cdot \hat{n} - i\frac{\omega}{c} U = 0$$

This should be a new function `sommerfeld_boundary_loss()` added as a loss term in the training loop.

### Issue 2.2 — Anisotropic TV from Unscaled Coordinates (HIGH)

**Location:** `tv_regularization()`

**What happens:** The TV gradient is computed in normalized coordinates. Z spans [0,1] mapping to 1000m depth, while X spans [-1,1] mapping to 800m width. The vertical gradients are penalized ~2.5x more per physical meter. This biases the PINN toward vertically-smooth, horizontally-sharp structure — the opposite of real geology.

**The fix:** Apply chain-rule physical scales inside the TV norm:
```python
scale_xy = 2.0 / domain_width_m
scale_z = 1.0 / max_depth_m
phys_scale = torch.tensor([scale_xy, scale_xy, scale_z], ...)
grad_c_phys = grad_c * phys_scale
```

### Issue 2.3 — Surface Prior Creates Columnar Hallucinations (HIGH)

**Location:** `surface_prior_loss()` line 639

**What happens:** `anomaly_weight = 1.0 - surface_anomaly_lut` removes the sparsity penalty for the **entire vertical column** under an anomalous surface pixel — from z=0 to z=1. The PINN dumps residual error into these unpenalized columns → vertical shaft artifacts.

**The fix:** Multiply the surface anomaly influence by an exponential depth decay:
```python
z_depth = coords[:, 2].clamp(0.0, 1.0)
depth_decay = torch.exp(-3.0 * z_depth)
anomaly_weight = 1.0 - (surface_anomaly_lut * depth_decay)
```

### Issue 2.4 — SAR Phase-to-Velocity Scaling is Wrong (HIGH)

**Location:** `sar_vibrometry.py` → `phase_to_vibration_velocity()` line 243

**What happens:** The conversion factor uses PRF to compute sub-aperture time separation:
```python
conv_factor = (wavelength * PRF) / (4π * N_sub)
```
But sub-apertures are separated in **Doppler frequency**, not time. The time separation between sub-aperture centers is `Δf_D / |K_a|` where K_a is the azimuth FM rate (~2020 Hz/s for Sentinel-1 IW mode).

**The fix:**
```python
K_a = 2020.0  # Sentinel-1 IW mode azimuth FM rate (Hz/s)
delta_fd = prf_hz / num_sub_apertures
delta_t = delta_fd / K_a
conv_factor = radar_wavelength_m / (4 * np.pi * delta_t)
```
Consider making K_a a parameter since it varies by sub-swath.

---

## Minor Algorithmic Fixes

### Fix 3.1 — K-Means Spatial Bias (`satellite_embeddings.py` line 534-544)

The sampling loop fills a sequential buffer: `if sample_count < max_pixels_for_fit: sample_rows.append(...)`. For large GeoTIFFs processed block-by-block (top→bottom), this only samples Northern rows. Fix with reservoir sampling.

### Fix 3.2 — Edge Sharpness Dilution (`visualize_3d_subsurface.py` line 269-270)

`edge_sharpness = mean(abs(gradient))` over the entire bounding box volume. For larger voids, the interior gradient is ~0, diluting the mean as 1/R. Fix: compute gradient magnitude only on boundary voxels where `grad_mag > threshold`.

### Fix 3.3 — PyVista F-Order Scrambling (`visualize_3d_subsurface.py` lines 453-458)

`flatten(order="F")` with numpy array shape `(nz, ny, nx)` and RectilinearGrid axes `(x, y, z)` means Z varies fastest — but PyVista expects X fastest for `(x, y, z)` grid definition. This renders the subsurface rotated/transposed. Fix: use `flatten(order="C")` or transpose the array.

### Fix 3.4 — Aspect Ratio Floor (`visualize_3d_subsurface.py` lines 240-241)

`max(extents[0], 1)` uses 1 meter as floor. When voxel resolution is 6.25m, the minimum extent is already > 6m, but if resolution changes this could produce aspect_ratio = inf. Use `max(extents[0], 1e-6)`.

---

## Architecture Reference

```
pinn_vibro_inversion.py
├── FourierFeatures          — Positional encoding (sin/cos)
├── Sine                     — sin() activation for SIREN
├── VibroInversionPINN       — Dual-head PINN (wavefield U + wave speed c)
│   ├── fourier              — FourierFeatures layer
│   ├── trunk                — 8-layer SIREN with skip connections
│   ├── head_U               — Complex wavefield output (U_real, U_imag)
│   └── head_c               — Wave speed output (sigmoid-bounded)
├── helmholtz_physics_loss() — Helmholtz PDE residual ∇²U + (ω²/c²)U = 0
├── surface_data_loss()      — Boundary condition: |U(z=0)| ≈ observed vibration
├── global_sparsity_loss()   — Cauchy prior on wave-speed deviation
├── deep_boundary_loss()     — Forces c→background at z>0.8
├── surface_prior_loss()     — Embedding-guided sparsity modulation
├── tv_regularization()      — Total Variation on wave-speed field
├── sample_collocation_points() — Random 3D domain sampling
├── SurfaceSamplerGPU        — Importance-weighted surface point sampler
└── train_vibro_pinn()       — Main training loop (3000 epochs, Adam, grad accum)
```

## Key Constraints

- **PyTorch autograd:** All physics loss functions use `torch.autograd.grad` for derivatives. The computational graph must stay connected from model parameters through the loss for gradients to flow.
- **`create_graph=True`** is mandatory for any loss term whose gradient needs to reach model parameters through the autograd graph.
- **Domain convention:** x,y ∈ [-1,1] (horizontal), z ∈ [0,1] (depth, 0=surface)
- **Chain-rule scaling:** Normalized-to-physical coordinate mapping requires multiplying each gradient by `2/domain_width` (x,y) or `1/max_depth` (z)
- **Default config:** 800m × 800m × 1000m domain, 128×128×64 grid, 1.0 Hz excitation, 3500 m/s background wave speed
- All outputs must remain float32 tensors for the optimizer

## Deliverables

1. Rebuilt `helmholtz_physics_loss()` with output-tensor casting (no model casting)
2. New `sommerfeld_boundary_loss()` function
3. Fixed `tv_regularization()` with `create_graph=True` and physical scaling
4. Fixed `surface_prior_loss()` with depth decay
5. Fixed `_init_weights()` for Fourier feature input
6. Fixed data normalization in `train_vibro_pinn()` (positive scaling)
7. Updated training loop to call new/fixed loss functions with correct arguments
8. Fixed `phase_to_vibration_velocity()` in `sar_vibrometry.py`
9. Fixed edge sharpness, F-order, and aspect ratio in `visualize_3d_subsurface.py`
10. Fixed K-Means sampling in `satellite_embeddings.py`

Return complete, production-ready implementations of each changed function. Do not use placeholders, TODOs, or pseudo-code.
