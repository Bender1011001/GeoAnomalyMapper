"""
generate_report.py — GeoAnomalyMapper Due Diligence Report Generator

Generates a per-target or per-list due diligence report from the scored target database.

Usage:
    # Single target by ID:
    python generate_report.py --target-id 1525

    # List of coordinates (CSV with lat,lon columns):
    python generate_report.py --coords-csv my_targets.csv

    # Top N targets from scored database:
    python generate_report.py --top-n 10 --tier "Tier 1"

    # Full regional pack (all targets in bounding box):
    python generate_report.py --bbox 35 42 -125 -114 --output-dir reports/california

Output:
    Markdown report(s) + summary CSV per target
"""

import argparse
import csv
import math
import os
import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
TARGETS_CSV = PROJECT_ROOT / "data" / "outputs" / "usa_targets_scored.csv"
TOP50_CSV = PROJECT_ROOT / "data" / "outputs" / "usa_top50_targets.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_targets(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def tier_label(score: float) -> str:
    if score >= 0.70:
        return "Tier 1 — High Confidence"
    elif score >= 0.55:
        return "Tier 2 — Moderate Confidence"
    else:
        return "Tier 3 — Speculative"


def deposit_type_hint(density_contrast: float, is_low: bool = False) -> str:
    if is_low:
        return "Mass-deficit anomaly — consistent with epithermal Au systems, silica/clay alteration, kimberlite"
    if density_contrast > 150:
        return "Strong mass-excess — consistent with dense intrusive, IOCG, or massive sulfide body"
    elif density_contrast > 50:
        return "Moderate mass-excess — consistent with skarn, VMS, or mafic intrusive"
    else:
        return "Weak mass-excess — consistent with mineralized vein zone or shallow alteration"


def generate_report(target: dict, output_path: Path) -> None:
    try:
        region_id = target.get("Region_ID", "?")
        lat = float(target["Latitude"])
        lon = float(target["Longitude"])
        density = float(target.get("Density_Contrast", 0))
        score = float(target.get("Confidence_Score", 0))
        tier = target.get("Tier", tier_label(score))
        dist_mrds = float(target.get("Dist_to_MRDS_km", target.get("Dist_to_Known_km", 0)))
        dist_road = float(target.get("Dist_to_Road_km", 0))
        elevation = float(target.get("Elevation", 0))
        is_undiscovered = str(target.get("Is_Undiscovered", "True")).lower() == "true"
        score_density = float(target.get("Score_Density", 0))
        score_novelty = float(target.get("Score_Novelty", 0))
        score_edge = float(target.get("Score_EdgeDist", 0))
        score_mrds = float(target.get("Score_MRDS", 0))
        area_px = int(float(target.get("Area_Pixels", 1)))
        area_km2 = area_px * 4  # ~2km x 2km per pixel
    except (KeyError, ValueError) as e:
        print(f"  Warning: could not parse field — {e}")
        return

    # --- Verdict logic ---
    go_nogo = "GO (Pending Field Verification)"
    risk_level = "Moderate"
    if score < 0.50:
        go_nogo = "NO-GO — Below confidence threshold"
        risk_level = "High"
    elif score >= 0.70 and is_undiscovered and dist_road < 50:
        go_nogo = "GO — Priority target"
        risk_level = "Low–Moderate"
    elif not is_undiscovered:
        go_nogo = "HOLD — Near known MRDS deposit; verify claim status before staking"
        risk_level = "Moderate"

    deposit_hint = deposit_type_hint(density)

    report = f"""# Due Diligence Report — Target {region_id}
## GeoAnomalyMapper | {date.today().isoformat()} | CONFIDENTIAL

---

## Verdict: {go_nogo}

| Field | Value |
|-------|-------|
| Target ID | {region_id} |
| Coordinates | {lat:.4f}, {lon:.4f} |
| Confidence Score | **{score:.3f}** |
| Classification | **{tier}** |
| Risk Level | {risk_level} |

---

## Signal Evidence

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Density Contrast | {density:.2f} kg/m³ | {deposit_hint} |
| Anomaly Area | {area_km2} km² | Footprint of anomalous zone |
| Score: Density Strength | {score_density:.3f} | 0=weak, 1=strong |
| Score: Novelty (vs. known deposits) | {score_novelty:.3f} | 1.0 = far from all MRDS records |
| Score: Not-an-artifact | {score_edge:.1f} | 1.0 = passes tile-boundary check |
| Score: MRDS Proximity | {score_mrds:.3f} | 1.0 = near proven deposit, 0.2 = distant |

---

## Location Assessment

| Factor | Value |
|--------|-------|
| Distance to nearest MRDS deposit | {dist_mrds:.1f} km |
| Distance to nearest road | {dist_road:.1f} km |
| Elevation | {elevation:.0f} m |
| Is Undiscovered (no MRDS within 50km) | {"YES — white space" if is_undiscovered else "NO — near known deposit"} |

**Accessibility:** {"Accessible (<20km to road)" if dist_road < 20 else "Remote (>20km to road) — helicopter or trail access likely required"}

---

## Recommended Next Steps

### Step 1 — Land Status (free, 1 hour)
- Check BLM MLRS: mlrs.blm.gov
- Confirm open to mineral entry, no existing claims, not in wilderness or special management area
- For Alaska targets: check ADFG and DNR records

### Step 2 — Satellite Review (free, 2–4 hours)
- Open coordinates in Google Earth Pro or Sentinel Hub
- Look for: outcrop exposure, alteration color (iron oxide = red/orange), access routes
- Check for existing exploration infrastructure (drill roads, pads, pits)

### Step 3 — Literature & State Survey Search (free, 2–8 hours)
- State geological survey mineral occurrence database
- USGS OFR and MDS bulletins for district
- SEDAR/EDGAR filings if any junior companies have reported on adjacent ground

### Step 4 — Geochemical Gap Check
- Cross-reference with NURE database for this coordinate
- If no samples within 5km: surface stream sediment or soil sampling is the lowest-cost next step
- If samples exist and are anomalous: escalate to Tier 1 field visit

### Step 5 — Field Verification (if Steps 1–4 positive)
- Contract geologist: $500–$800/day
- Collect 5–10 rock chip samples from outcrop
- Minimum assay package: Au, Ag, Cu, Pb, Zn, Mo (ICP-MS)
- Cost: ~$1,500–$3,000 total

### Step 6 — High-Resolution Geophysics (if field confirms anomaly)
- Drone gravity or ground magnetics: ~$5k–$20k depending on area
- This upgrades the 2km-resolution continental model to a drillable target

---

## Score Breakdown Chart

```
Density Strength  [{density_bar(score_density)}] {score_density:.2f}
Novelty           [{density_bar(score_novelty)}] {score_novelty:.2f}
Not-Artifact      [{density_bar(score_edge)}] {score_edge:.2f}
MRDS Proximity    [{density_bar(score_mrds)}] {score_mrds:.2f}
                  ─────────────────────────────
Composite Score   [{density_bar(score)}] {score:.3f} — {tier}
```

---

## Limitations

- This report is based on continental-scale gravity data (~2km resolution).
- It is a prospectivity indicator, NOT a drill target.
- Depth and density estimates carry inherent non-uniqueness (gravity inversion ambiguity).
- Land status, existing claims, and environmental constraints must be independently verified.
- This report does not constitute geological, legal, or investment advice.

---

*GeoAnomalyMapper | Proprietary — Commercial use requires license agreement.*
*Contact via github.com/bender1011001/geoanomalymapper*
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"  Report written: {output_path}")


def density_bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate due diligence reports from GeoAnomalyMapper targets")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--target-id", type=str, help="Single target by Region_ID")
    g.add_argument("--top-n", type=int, help="Top N targets from scored database")
    g.add_argument("--coords-csv", type=str, help="CSV file with lat,lon columns to match against database")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                   help="Bounding box (all targets within region)")

    p.add_argument("--tier", choices=["Tier 1", "Tier 2", "Tier 3", "all"], default="all",
                   help="Filter by tier (default: all)")
    p.add_argument("--targets-csv", type=str, default=str(TARGETS_CSV),
                   help="Path to scored targets CSV")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                   help="Output directory for reports")
    return p.parse_args()


def main():
    args = parse_args()
    targets_path = Path(args.targets_csv)
    out_dir = Path(args.output_dir)

    if not targets_path.exists():
        print(f"ERROR: targets CSV not found: {targets_path}")
        print("Run phase2_validation.py first to generate scored targets.")
        sys.exit(1)

    all_targets = load_targets(targets_path)

    # Apply tier filter
    if args.tier != "all":
        all_targets = [t for t in all_targets if t.get("Tier", "") == args.tier]

    selected = []

    if args.target_id:
        selected = [t for t in all_targets if str(t.get("Region_ID", "")) == args.target_id]
        if not selected:
            print(f"ERROR: Target ID '{args.target_id}' not found in {targets_path}")
            sys.exit(1)

    elif args.top_n:
        # Sort by confidence score descending
        sorted_targets = sorted(all_targets, key=lambda t: float(t.get("Confidence_Score", 0)), reverse=True)
        selected = sorted_targets[:args.top_n]

    elif args.coords_csv:
        coords_path = Path(args.coords_csv)
        if not coords_path.exists():
            print(f"ERROR: coords CSV not found: {coords_path}")
            sys.exit(1)
        with open(coords_path, newline="", encoding="utf-8") as f:
            coords = list(csv.DictReader(f))

        for row in coords:
            try:
                clat = float(row.get("lat", row.get("Lat", row.get("latitude", row.get("Latitude", "")))))
                clon = float(row.get("lon", row.get("Lon", row.get("longitude", row.get("Longitude", "")))))
            except ValueError:
                print(f"  Skipping row — cannot parse lat/lon: {row}")
                continue

            # Find nearest target within 10km
            best = None
            best_dist = 999999
            for t in all_targets:
                try:
                    tlat = float(t["Latitude"])
                    tlon = float(t["Longitude"])
                    d = haversine_km(clat, clon, tlat, tlon)
                    if d < best_dist:
                        best_dist = d
                        best = t
                except (KeyError, ValueError):
                    continue

            if best and best_dist <= 50:
                selected.append(best)
                print(f"  Matched ({clat}, {clon}) → Target {best['Region_ID']} ({best_dist:.1f} km away)")
            else:
                print(f"  No match within 50km for ({clat}, {clon})")

    elif args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
        for t in all_targets:
            try:
                tlat = float(t["Latitude"])
                tlon = float(t["Longitude"])
                if lat_min <= tlat <= lat_max and lon_min <= tlon <= lon_max:
                    selected.append(t)
            except (KeyError, ValueError):
                continue
        print(f"  Found {len(selected)} targets in bounding box")

    if not selected:
        print("No targets selected. Exiting.")
        sys.exit(0)

    print(f"\nGenerating {len(selected)} report(s) to {out_dir}/\n")
    for t in selected:
        rid = t.get("Region_ID", "unknown")
        score = float(t.get("Confidence_Score", 0))
        fname = out_dir / f"target_{rid}_score{score:.3f}.md"
        generate_report(t, fname)

    # Write summary CSV
    summary_path = out_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Region_ID", "Latitude", "Longitude", "Confidence_Score", "Tier",
                  "Density_Contrast", "Dist_to_MRDS_km", "Dist_to_Road_km", "Is_Undiscovered"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"\nSummary CSV written: {summary_path}")
    print(f"Done. {len(selected)} report(s) generated.")


if __name__ == "__main__":
    main()
