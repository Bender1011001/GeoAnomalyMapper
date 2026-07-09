# InSAR Deformation Prototype (validated 2026-07-09)

Ground-truth-validated detection path for large underground features via
surface deformation, built after the single-pass Doppler vibrometry approach
failed its Carlsbad-vs-barren discrimination controls.

## Method

1. `submit_wink_insar.py` / `submit_wink_hires.py` — select a Sentinel-1 stack
   (asf_search) over a target and submit InSAR pairs to ASF HyP3
   (free on-demand INSAR_GAMMA processing; Earthdata credentials from `.env`).
2. `analyze_wink_v2.py` — frame-wide: coherence-masked, per-pair
   median-referenced stacking; ~5 km high-pass; long-pair sign-agreement vote;
   connected-component bowl detection with chance-rate accounting.
3. `analyze_wink_hires.py` — AOI-local: 10x2-look (40 m) short-pair stack for
   fast/small bowls (100–250 m, up to tens of cm/yr).
4. `final_insar_report.py` — consolidated findings + map.

## Validation result

Blindly detected the published actively-subsiding area of the Wink TX
sinkhole complex: −8.6 cm/yr bowl within 0.69 km of the literature-derived
location (robust noise σ = 1 cm/yr; 5 bowls in the 295 km² box; ~5% chance
probability). Outputs archived in `data/insar_wink/`
(`FINAL_deformation_findings.json`, `wink_deformation_map.png`).

## Hard-won pitfalls

- 80 m (20x4 looks) products CANNOT resolve small fast bowls — the phase
  gradient aliases and GAMMA's filtering silently flattens the signal.
  Use 10x2 looks with 6–12 day pairs for fast deformation.
- Long-pair product footprints (frame overlap) may not cover the AOI; check
  coverage before requiring long-pair agreement.
- HyP3 unwrapped products carry an arbitrary constant offset per pair —
  always reference locally (median over coherent pixels in the AOI).
- LOS, single geometry, short observation windows: candidates need SBAS
  time-series confirmation and cross-checking against oilfield operations
  before any "unknown void" claim.

Scripts are prototype quality (Wink-hardcoded paths/coordinates); generalize
into a proper module before pointing at new targets.
