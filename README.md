# GeoAnomalyMapper

**AI-Driven Continental-Scale Mineral Prospectivity Engine**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](./LICENSE)

> **Commercial licensing available.** Research use permitted. See [LICENSE](./LICENSE) for terms.

---

## What It Does

GeoAnomalyMapper uses a **Physics-Informed Neural Network (PINN)** to solve the gravity inverse problem at continental scale — converting Bouguer gravity anomaly data into subsurface density contrast models, then extracting ranked prospectivity targets.

```
Bouguer Gravity → Residual Separation → PINN Inversion → Dual-Pipeline Extraction → Scored Targets
```

The system runs a **dual pipeline** — finding both mass-excess anomalies (VMS, IOCG, skarns, Ni-Cu) and mass-deficit anomalies (epithermal gold, alteration halos, kimberlites) — validated against independent geochemical and MRDS datasets.

---

## Validated Performance

| Metric | Result |
|--------|--------|
| Targets generated (continental US) | 1,634 |
| Tier 1 high-confidence targets | 31 |
| Geochemical enrichment vs. baseline | **8.5x** (32.4% hit rate vs. 3.8% random) |
| Negative control specificity | **100%** (zero false positives in barren regions) |
| Statistical significance | >7 sigma |

Independent validation used the NURE geochemical database (397,000+ sediment samples) — completely separate from training data. See [docs/SCIENTIFIC_VALIDATION_REPORT.md](docs/SCIENTIFIC_VALIDATION_REPORT.md) for full methodology.

---


## Technical Summary

**Core model:** DensityUNet with physics layer implementing Parker-Oldenburg forward gravity.
Loss function: data fidelity + structural coupling (EMAG2 magnetic regularization) + sparsity.

**Resolution:** ~2km continental grid. Optimized for district-scale targets (1–20km). Not suitable
for resolving individual deposits <500m. Resolution limit is a documented design constraint,
not a bug — the filter is calibrated to district scale by design.

**Depth ambiguity:** All gravity inversions are non-unique. Outputs are prospectivity indicators,
not structural models. Ground truthing is required.

**Deposit types detected:**
- Mass-excess: IOCG, VMS, skarns, magmatic Ni-Cu, dense intrusives
- Mass-deficit: Epithermal gold systems, alteration halos, sediment-hosted Au, kimberlite pipes

---

## Project Structure

```
GeoAnomalyMapper/
├── pinn_gravity_inversion.py  # PINN architecture (DensityUNet + physics layer)
├── loss_functions.py          # Custom loss (structure-guided TV + magnetic coupling)
├── train_usa_pinn.py          # Training pipeline
├── predict_usa.py             # Continental-scale sliding window inference
├── extract_dual_targets.py    # Dual-pipeline target extraction
├── phase2_validation.py       # MRDS cross-reference + confidence scoring
├── verify_skeptic_v2.py       # Forensic validation + negative controls
├── data/outputs/              # Target CSVs, scored lists
└── docs/                      # Scientific validation, forensic audit, methodology
```

---

## Quick Start (Research Use)

```bash
pip install -r requirements.txt

python train_usa_pinn.py
python predict_usa.py
python extract_dual_targets.py data/outputs/usa_density_model.tif
python phase2_validation.py
```

Gravity input data: USGS Bouguer anomaly grid. Magnetic data: EMAG2.
Both are publicly available. See `setup_usgs_data.py` for download helpers.

---

## Known Limitations

- Continental public grids (~2km) cannot resolve deposits smaller than ~500m.
- Basin & Range regional gravity lows (Nevada) mask discrete epithermal anomalies at this resolution.
- Gravity inversion is inherently non-unique (depth vs. density ambiguity).
- Outputs are statistical prospectivity indicators — not drill targets. Field verification required.

The validation suite (negative controls, geochemical enrichment, forensic audit) documents
exactly where the model works and where it doesn't. See `docs/` for the full record.

---

## License

Proprietary. Commercial use requires a license agreement. Research use permitted.
See [LICENSE](./LICENSE) for terms.

---

## Acknowledgments

- USGS for public gravity and MRDS data
- NURE program for geochemical validation data
- The geophysics and mineral exploration research community
