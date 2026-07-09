"""Final consolidated InSAR deformation report: validation + candidates + map."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))
WORK = REPO / "data" / "insar_wink"

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as warp_transform

from json_utils import write_strict_json

SINKS = {
    "Wink Sink 1 (1980)": (31.7775, -103.1129),
    "Wink Sink 2 (2002)": (31.7632, -103.1129),
    "documented fast area (2016)": (31.7632, -103.1020),
}
VALIDATION_HIT = {"lat": 31.76936, "lon": -103.10177, "rate_cm_yr": -8.6}
CANDIDATES = [
    {"lat": 31.75019, "lon": -103.02743, "rate_cm_yr": -21.8, "label": "candidate A (twin west)"},
    {"lat": 31.75158, "lon": -103.02360, "rate_cm_yr": -18.5, "label": "candidate B (twin east)"},
    {"lat": 31.83583, "lon": -103.03575, "rate_cm_yr": -8.0, "label": "candidate C (S of Kermit)"},
    {"lat": 31.75455, "lon": -103.02902, "rate_cm_yr": -7.4, "label": "candidate D (near twins)"},
]

with rasterio.open(WORK / "wink_hires_stack_rate_m_yr.tif") as src:
    sm = src.read(1).astype(np.float64) * 100.0  # cm/yr
    tr = src.transform
    crs = src.crs
    H, W = src.height, src.width

# corner coordinates for extent (approximate lat/lon axis)
xs = [tr.c, tr.c + tr.a * W]
ys = [tr.f + tr.e * H, tr.f]
lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
extent = [lons[0], lons[1], lats[0], lats[1]]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(np.clip(sm, -15, 15), extent=extent, cmap="RdBu", vmin=-15, vmax=15, origin="upper")
plt.colorbar(im, ax=ax, label="LOS velocity (cm/yr), negative = subsidence", shrink=0.75)
for name, (la, lo) in SINKS.items():
    ax.plot(lo, la, "k^", markersize=11, markerfacecolor="yellow")
    ax.annotate(name, (lo, la), textcoords="offset points", xytext=(8, 6), fontsize=9, weight="bold")
ax.plot(VALIDATION_HIT["lon"], VALIDATION_HIT["lat"], "o", color="lime", markersize=13, fillstyle="none", markeredgewidth=2.5)
ax.annotate(f"VALIDATION HIT {VALIDATION_HIT['rate_cm_yr']} cm/yr",
            (VALIDATION_HIT["lon"], VALIDATION_HIT["lat"]),
            textcoords="offset points", xytext=(10, -14), fontsize=9, color="darkgreen", weight="bold")
for c in CANDIDATES:
    ax.plot(c["lon"], c["lat"], "s", color="red", markersize=10, fillstyle="none", markeredgewidth=2)
    ax.annotate(f"{c['label']} {c['rate_cm_yr']}", (c["lon"], c["lat"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color="darkred")
ax.set_title("Wink-Kermit corridor: Sentinel-1 InSAR LOS velocity (12 pairs, Mar-Jun 2026, 40 m)\n"
             "Ground truth validated: documented active subsidence area detected at 0.69 km")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
png = WORK / "wink_deformation_map.png"
plt.tight_layout()
plt.savefig(png, dpi=170)
print(f"MAP: {png}")

v2 = json.loads((WORK / "wink_insar_report_v2.json").read_text(encoding="utf-8"))
frame_bowls = [b for b in v2.get("top_bowls", []) if b["type"] == "subsidence"][:15]

report = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "pipeline": "Sentinel-1 SLC -> ASF HyP3 INSAR_GAMMA -> coherence-masked locally-referenced stacking",
    "validation": {
        "verdict": "PASS",
        "ground_truth": "Wink Sinks, Winkler County TX; published active subsidence area ~1 km east of Wink Sink 2",
        "detection": VALIDATION_HIT,
        "offset_from_published_location_km": 0.69,
        "noise_floor_cm_yr_robust_sigma": 1.0,
        "bowls_in_295km2_box": 5,
        "chance_probability_within_1km": "~5%",
        "notes": [
            "Wink Sink rims themselves are currently quiescent (+0.7 to +4 cm/yr), consistent with published post-2016 deceleration.",
            "80 m products and 384-day pairs cannot resolve these small fast bowls; 40 m short pairs were required.",
        ],
    },
    "unknown_candidates_hires_box": CANDIDATES,
    "unknown_candidates_frame_wide_80m": frame_bowls,
    "caveats": [
        "LOS (not vertical) rates; single ascending geometry.",
        "3.4-month observation window for the hi-res stack; rates are annualized.",
        "Candidates A/B/D lie ~7 km east of the documented sink complex, beyond the TerraSAR-X footprints of the published studies; novelty not yet established against oilfield operations records (TX RRC).",
        "Frame-wide 80 m candidates are dominated by Delaware Basin oil/gas injection-extraction deformation; not all deformation implies an underground void.",
    ],
    "artifacts": {
        "map_png": str(png),
        "hires_stack_tif": str(WORK / "wink_hires_stack_rate_m_yr.tif"),
        "frame_velocity_tif": str(WORK / "wink_los_velocity_highpass_m_yr.tif"),
        "v2_report": str(WORK / "wink_insar_report_v2.json"),
        "hires_validation": str(WORK / "wink_hires_validation.json"),
    },
}
out = WORK / "FINAL_deformation_findings.json"
write_strict_json(out, report)
print(f"FINAL_REPORT: {out}")
