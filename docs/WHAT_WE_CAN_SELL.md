# What We Can Sell (Even If Public Data Is Too Low-Res For Drill Targets)

## Positioning (say this out loud)
GeoAnomalyMapper is a **regional screening and ranking engine**. With public continental gravity/mag, it is usually not defensible to promise “drill targets”.

What *is* defensible:
- Shortlisting areas and structures for follow-up
- Prioritizing field time and budget
- Turning proprietary higher-resolution data into localized targets

## Core sellable deliverables (SKUs)

### 1) “Bring Your Own Data” Inversion + Target Pack (Service)
Customer provides: higher-res gravity and/or mag (and optional geochem).
We deliver:
- Ranked anomalies/targets with uncertainty notes
- Map outputs (HTML map + GeoJSON/KML/CSV)
- Evidence bundle (what layers drove the score, nearby known deposits, sampling voids)

Good for: juniors with a geophysics budget but no internal inversion/ML pipeline.
Pricing model: per project area (fixed) + optional retainer for iterations.

### 2) Automated Due Diligence Reports (Product / Subscription)
Input: a list of candidate coordinates (pins) from a team’s own targeting.
Output: a standardized “go/no-go” report for each pin:
- Cross-checks against the model layers you already compute
- Distance to known occurrences (MRDS)
- “Coverage” flags (is this in a sampling void?)
- Exportable formats for internal GIS and presentations

Good for: exploration managers who need fast, repeatable desk screening.
Pricing model: pay-per-report or monthly seats.

### 3) Regional Screening Packs (Data Product)
Not “drill targets”; “regional leads”.
Deliver:
- Heatmap / prospectivity raster(s)
- Top N ranked leads per commodity/play type
- A short PDF methodology + limitations + suggested next steps (ground truth, data to acquire)

Good for: early-stage teams selecting which districts to enter.
Pricing model: non-exclusive (cheaper) vs exclusive (more expensive).

### 4) Workflow Software: Pin-First Exploration Notebook (SaaS)
Make the “pin works” part the product:
- Collaborative pins, tags, comments, attachments (photos, assays)
- Import/export (KML/GeoJSON/CSV), versioned target lists
- Auto-generated map packs for investors/partners
- Optional compute add-ons (run inversion/scoring jobs in the cloud)

Good for: small teams who live in spreadsheets + Google Earth today.
Pricing model: per-seat + add-on compute.

## What we should NOT sell (or should reword)
- “Drill targets” from continental public grids
- “Guaranteed deposits”
- Raw coordinates with no evidence/explanation (easy to copy, hard to defend)

## Licensing note (repo reality check)
This repo currently states non-commercial terms (CC BY-NC 4.0 + README language).
If monetization is the goal, we need to explicitly decide:
1) re-license, or
2) build a separate commercial codebase and keep this one as research.

