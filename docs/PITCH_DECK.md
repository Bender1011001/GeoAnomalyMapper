# GeoAnomalyMapper — One-Page Pitch
## For: Junior Exploration Companies, Prospect Generators, Deal-Flow Teams

---

## The Problem You Have

Your team spends weeks doing desk work before field season:
- Pulling gravity and magnetic grids
- Cross-referencing MRDS deposit records
- Manually eyeballing anomalies in GIS

That's slow, expensive, and inconsistent between analysts. Half the anomalies you flag are
already next to known mines. The real value is finding the ones nobody has looked at yet.

---

## What We Built

**GeoAnomalyMapper** is a Physics-Informed Neural Network (PINN) that inverts continental-scale
gravity data into subsurface density models, then automatically ranks and scores anomalies as
mineral prospectivity targets.

It's not a "black box AI." The physics are explicit — we implement Newton's law of gravitation
as a differentiable layer, so every output is traceable to the input gravity signal.

**It runs a dual pipeline:**
- Mass-excess anomalies → VMS, IOCG, skarns, magmatic Ni-Cu
- Mass-deficit anomalies → Epithermal gold systems, alteration halos, kimberlites

---

## Validated Numbers

| Test | Result |
|------|--------|
| Geochemical enrichment (NURE, 397k samples) | **8.5x above random baseline** |
| Hit rate on sampled targets | **32.4%** (vs. 3.8% random) |
| Specificity (negative control — barren regions) | **100%** zero false positives |
| Statistical significance | >7 sigma |
| Coverage | Continental US, 1,634 targets generated |

The geochemical validation used a completely independent dataset — not the same data used to
build the model. That's the number that matters.

---

## What We're Selling

### Option 1: Regional Screening Pack — from $1,500
You tell us the commodity and the state/province.
We deliver:
- Top N ranked targets (CSV + GeoJSON + interactive HTML map)
- Density contrast raster (GeoTIFF)
- Short methodology PDF with evidence per target

Turnaround: 3–5 business days for any US region already processed.
Non-exclusive (cheaper) or exclusive (priced on region size).

### Option 2: "Bring Your Own Data" Inversion — from $5,000/project
You supply higher-resolution gravity and/or magnetic data.
We deliver targets localized to your data resolution — suitable for drill planning.

### Option 3: Due Diligence Reports — from $250/coordinate
Submit a list of coordinates from your existing target pipeline.
We cross-check each against our model layers, MRDS, and geochemical database
and return a standardized go/no-go desk report.

---

## Who This Is For

- Junior explorers who have geophysics data but no internal AI/ML processing capability
- Prospect generators who want to screen more ground faster
- Exploration managers evaluating incoming deal flow on new districts
- Anyone paying consultants to do manual gravity interpretation

---

## What This Is Not

- Drill targets (continental grid resolution is ~2km — too coarse for that)
- A substitute for ground-truth field work
- Investment advice

We are explicit about this. The value is in **cutting desk-work time** and **surfacing white-space
anomalies that manual screening misses**.

---

## How to Get Started

1. Tell us the commodity and region you're targeting
2. We'll quote a screening pack or scope a data inversion engagement
3. You receive deliverables within the agreed turnaround
4. You decide what to do with your field season

**Contact:** Open a GitHub issue or reach out directly.
**Repository:** github.com/bender1011001/geoanomalymapper (research version — commercial data available separately)

---

*GeoAnomalyMapper | Proprietary — © 2024-2026. Commercial use requires license agreement.*
