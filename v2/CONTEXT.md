# GeoAnomalyMapper v2 — Elastic Joint Inversion

## Status
- **In Development**: Building from scratch based on lessons from v1 failures
- **v1 Post-Mortem**: Acoustic Helmholtz + xavier init + AMP = NaN at epoch 13

## Architecture Overview
v2 replaces the entire physics engine:

| Component | v1 (broken) | v2 (new) |
|-----------|------------|----------|
| PDE | Acoustic Helmholtz (∇²U + ω²/c² U = 0) | Elastic (∇·(μ∇U) + ρω²U = 0) |
| Target variable | Wave speed c (m/s) | Shear modulus μ (Pa) — voids = 0 |
| Network arch | Single-head PINN | Mixed-variable: displacement + stress |
| 2nd derivatives | Yes (∇²U via autograd) | NO — stress σ=μ∇U, only 1st derivs |
| Initialization | Xavier (fatal for SIREN) | Proper SIREN (Sitzmann 2020) |
| Extra data | None | USGS Bouguer gravity anomaly |
| AMP | float16 (NaN) | Disabled, float64 physics |

## Key Files
- `elastic_pinn.py` — Mixed-variable elastic PINN network + training loop
- `gravity_fetcher.py` — Download USGS Bouguer gravity anomaly grids
- `joint_loss.py` — Joint inversion loss: L_sar + L_elastic + L_gravity + L_reg
- `run_v2.py` — Orchestrator for v2 pipeline

## Why Mixed-Variable?
The killer insight from medical MRE research:
Instead of computing ∇²U (2nd derivative → NaN), split into two heads:
- Head 1: displacement U(x,y,z) 
- Head 2: stress σ(x,y,z) = μ · ∇U

Physics loss becomes: ∇·σ + ρω²U = 0 (only 1st derivatives of σ!)
This completely eliminates the 2nd-order autograd instability.

## Why Shear Modulus?
- Rock: μ ~ 10-30 GPa
- Water: μ = 0 (exactly)
- Air/void: μ = 0 (exactly)
Voids pop out with infinite contrast. Wave speed c gave ~3x contrast at best.

## Anti-Patterns (DO NOT)
- Do NOT use xavier_uniform_ with Sine activations — exponential blowup
- Do NOT compute ∇²U via autograd — use mixed-variable formulation
- Do NOT use AMP (float16) for physics computations
- Do NOT use torch.compile with autograd.grad
- Do NOT start physics loss at full weight — always warmup
