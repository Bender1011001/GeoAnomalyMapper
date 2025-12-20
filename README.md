# GeoAnomalyMapper 🌍

**AI-Driven Continental-Scale Mineral Exploration**

[![Built with AI](https://img.shields.io/badge/Built%20With-AI%20Assistance-blueviolet)](https://github.com/bender1011001/geoanomalymapper)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> ⚠️ **Transparency Notice**: This entire project—from concept to code to validation—was built using AI assistants by a non-expert. This code is provided for research and educational purposes only. **Commercial use of this code, the derived model, or the resulting target data is strictly prohibited under the CC BY-NC 4.0 license.**

---

## What This Project Does

GeoAnomalyMapper uses **Physics-Informed Neural Networks (PINNs)** to analyze continental-scale gravity data and identify potential mineral exploration targets across the United States.

### The Pipeline

```
Bouguer Gravity Data → Residual Separation → PINN Inversion → Dual-Pipeline Extraction → Validation
```

1. **Ingests** USGS gravity anomaly data (Bouguer corrected)
2. **Preprocessing**: Computes Residual Gravity to remove regional crustal trends (crucial for finding local anomalies)
3. **Trains** a physics-informed neural network (PINN) to invert gravity to density
4. **Generates** a subsurface density contrast map
5. **Extracts** targets using a **Dual-Pipeline**:
   - **Mass-Excess (High Density)**: VMS, IOCG, Magmatic Ni-Cu
   - **Mass-Deficit (Low Density)**: Epithermal Gold, Carlin-style, Kimberlites
6. **Validates** against USGS Mineral Resources Data System (MRDS)

---

## Key Results

| Metric | Value |
|--------|-------|
| **Targets Identified** | 1,634 |
| **MRDS Correlation (Top 50)** | 46% within 50km of known deposits |
| **Tier 1 (High Confidence) Targets** | 31 |
| **Coverage** | Continental US |

The 46% hit rate means nearly half of the model's top predictions are near known mineral deposits—providing geological plausibility. The remaining 54% could be undiscovered deposits, false positives, or deposits not in the database.

---

## Honest Limitations

After extensive research, I learned that this approach has fundamental constraints:

### ✅ What It CAN Detect
- **Mass-Excess Targets**: IOCG, VMS, Skarns, Magmatic Ni-Cu
- **Mass-Deficit Targets**: Epithermal gold, alteration halos, geological contacts
- **Structural Traps**: Faults and fold hinges defined by density contrast

### ❌ Limitations & Lessons
- **Regional Masking**: In areas like Nevada (Basin & Range), the entire region is a gravity low. Without high-resolution local data, discrete Carlin-type anomalies are masked by the regional signal.
- **Resolution**: Continental grids (~2km) cannot resolve small (<500m) deposits.
- **Depth Ambiguity**: Gravity inversion is non-unique; a small shallow body can look like a large deep one.

> **Pivot Note**: We successfully pivoted to a **Dual-Pipeline** approach (finding lows and highs). While we didn't find specific targets in the Nevada Carlin Trend due to data resolution, the system successfully identified gold-associated anomalies in the California Coast Ranges.

---

## Project Structure

```
GeoAnomalyMapper/
├── train_usa_pinn.py      # Train the physics-informed neural network
├── predict_usa.py         # Generate continental density model
├── extract_targets.py     # Extract anomalous regions as targets
├── verify_skeptic_v2.py   # Forensic validation of results
├── phase2_validation.py   # MRDS cross-reference and scoring
├── data/
│   ├── inputs/           # Source gravity data
│   └── outputs/          # Generated models and targets
└── docs/
    └── SCIENTIFIC_VALIDATION_REPORT.md
```

---

## The AI-Assisted Development Story

### Who Made This
I'm a high school dropout who works in construction. I have no formal training in:
- Geophysics
- Machine learning
- Python programming
- Mineral exploration

### How It Was Built
Every component of this project was created through conversation with AI assistants:
- **Architecture design**: Asked AI to explain gravity inversion methods
- **Code generation**: AI wrote all Python scripts
- **Debugging**: AI diagnosed and fixed errors
- **Validation strategy**: AI suggested forensic checks
- **Research**: AI explained the limitations I didn't know existed

### Why I'm Sharing This
This isn't the project that "wins" for me—it's proof of concept that someone without traditional credentials can:
1. Understand complex scientific domains through AI dialogue
2. Build functional systems that would normally require specialized teams
3. Validate their own work by asking the right questions

The limiting factor isn't access to knowledge anymore. It's knowing what questions to ask.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bender1011001/geoanomalymapper.git
cd geoanomalymapper

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (requires gravity data)
python train_usa_pinn.py
python predict_usa.py
python extract_targets.py data/outputs/usa_density_model.tif
python phase2_validation.py
```

---

## What I Learned

1. **AI is a force multiplier, not a replacement for thinking.** The AI could write code, but I had to understand what to ask for.

2. **Validation is everything.** The initial results looked great until I asked "how do we know this isn't just finding tile boundaries?"—and it was. The skeptical questions matter more than the exciting results.

3. **Domain expertise still matters.** The deep research on gravity signatures revealed that my fundamental assumption (positive density = ore) was only half-right. No amount of AI could have saved me from that without explicitly asking "what are we missing?"

4. **Scope discipline is hard.** This project could expand forever. Knowing when to stop and ship is a skill.

---

## Future Directions (For Someone Else)

If someone wanted to take this further:
- [x] Implement dual-pipeline (mass-excess AND mass-deficit targets) **(Done!)**
- [x] Compute residual gravity derivatives **(Done!)**
- [ ] Add magnetic data integration (Started, but could go deeper with Joint Inversion)
- [ ] Validate against geochemical survey data
- [ ] Test on a single mining district with high-resolution drone gravity

---

## Acknowledgments

- **USGS** for public gravity and MRDS data
- **The AI assistants** that made this possible
- **The mining/geophysics community** whose published research guided the validation

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

**Summary of Terms:**
- **Attribution**: You must give appropriate credit.
- **NonCommercial**: You may not use the material for commercial purposes.
- **NoAdditionalRestrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For the full legal code, visit: [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

---

*Built by a construction worker with AI. December 2024.*
