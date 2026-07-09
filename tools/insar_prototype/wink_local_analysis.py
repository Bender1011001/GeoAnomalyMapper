"""Wink-local deformation analysis with per-pair box-local referencing.

Detector: the 384-day pair (only long pair covering the box; coherence 0.95)
gives ~1 cm/yr sensitivity. Short pairs (6-12 d), each locally referenced to
the box median, provide an independent sign/consistency check at any bowl.
Ground truth: Wink Sinks ~6 mi south of Kermit, centered near (31.77, -103.12);
published rates 8-18 cm/yr at sink rims, 40-60+ cm/yr ~1 km east of Sink 2.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))
PRODUCTS = REPO / "data" / "insar_wink" / "products"
WORK = REPO / "data" / "insar_wink"

import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import gaussian_filter, label

from json_utils import write_strict_json

BOX = {"lat_min": 31.70, "lat_max": 31.84, "lon_min": -103.22, "lon_max": -103.02}
SINK_CENTER = (31.77, -103.12)
DATE_RE = re.compile(r"S1[ABC]{2}_(\d{8})T\d{6}_(\d{8})T\d{6}")
COH_FLOOR = 0.30

los_files = sorted(PRODUCTS.glob("*_los_disp.tif"))

# Choose the long pair covering the box as reference grid (the 384-day one).
long_los = None
for los in los_files:
    m = DATE_RE.search(los.name)
    days = abs((datetime.strptime(m.group(2), "%Y%m%d") - datetime.strptime(m.group(1), "%Y%m%d")).days)
    if days >= 300:
        long_los = (los, days)
print("LONG_PAIR:", long_los[0].name, long_los[1], "days")

with rasterio.open(long_los[0]) as src:
    crs = src.crs
    transform = src.transform
    xs, ys = warp_transform("EPSG:4326", crs,
                            [BOX["lon_min"], BOX["lon_max"]],
                            [BOX["lat_min"], BOX["lat_max"]])
    inv = ~transform
    (c0, r0) = inv * (min(xs), max(ys))
    (c1, r1) = inv * (max(xs), min(ys))
    r0, r1 = int(max(0, min(r0, r1))), int(min(src.height, max(r0, r1)))
    c0, c1 = int(max(0, min(c0, c1))), int(min(src.width, max(c0, c1)))
BH, BW = r1 - r0, c1 - c0
print(f"BOX_WINDOW {BH}x{BW} px")


def load_box(path, grid_src):
    """Read a raster window aligned to the reference grid box."""
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as src:
        if src.crs == crs and src.transform == transform:
            data = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float64)
            return data
        out = np.full((BH, BW), np.nan)
        box_transform = transform * transform.translation(c0, r0)
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=box_transform, dst_crs=crs,
            resampling=Resampling.bilinear, dst_nodata=np.nan,
        )
        return out


def local_rate(los_path):
    m = DATE_RE.search(los_path.name)
    days = abs((datetime.strptime(m.group(2), "%Y%m%d") - datetime.strptime(m.group(1), "%Y%m%d")).days)
    disp = load_box(los_path, None)
    valid = np.isfinite(disp) & (disp != 0.0)
    corr_path = los_path.with_name(los_path.name.replace("_los_disp.tif", "_corr.tif"))
    if corr_path.exists():
        corr = load_box(corr_path, None)
        valid &= np.isfinite(corr) & (corr >= COH_FLOOR)
    if valid.sum() < 100:
        return days, None
    disp = disp - float(np.median(disp[valid]))  # box-local reference
    rate = np.full_like(disp, np.nan)
    rate[valid] = disp[valid] / days * 365.25
    return days, rate


# Long-pair local rate (primary detector)
_, long_rate = local_rate(long_los[0])

# Short-pair median local rate (consistency)
short_stack = []
for los in los_files:
    m = DATE_RE.search(los.name)
    days = abs((datetime.strptime(m.group(2), "%Y%m%d") - datetime.strptime(m.group(1), "%Y%m%d")).days)
    if days > 30:
        continue
    _, rate = local_rate(los)
    if rate is not None:
        short_stack.append(rate)
short = np.stack(short_stack)
short_med = np.nanmedian(short, axis=0)
short_n = np.isfinite(short).sum(axis=0)
short_med = np.where(short_n >= 6, short_med, np.nan)
print(f"SHORT_PAIRS_IN_BOX: {len(short_stack)}")

# Light smoothing to suppress single-pixel noise (bowls are 100-250 m wide -> 2-3 px)
long_sm = gaussian_filter(np.nan_to_num(long_rate, nan=0.0), 1.0)
weight = gaussian_filter(np.isfinite(long_rate).astype(float), 1.0)
long_sm = np.where(weight > 0.3, long_sm / np.maximum(weight, 1e-6), np.nan)

finite = long_sm[np.isfinite(long_sm)]
print(f"LONG_LOCAL_RATE cm/yr: p01={np.percentile(finite,1)*100:.1f} p50={np.percentile(finite,50)*100:.1f} "
      f"p99={np.percentile(finite,99)*100:.1f} min={finite.min()*100:.1f} max={finite.max()*100:.1f}")

# Detect subsidence bowls in the box on the long-pair map
mask = np.isfinite(long_sm) & (long_sm <= -0.02)
labeled, n = label(mask)
bowls = []
for k in range(1, n + 1):
    px = labeled == k
    if px.sum() < 3:
        continue
    rows, cols = np.where(px)
    vals = long_sm[rows, cols]
    pi = np.argmax(-vals)
    pr, pc = int(rows[pi]), int(cols[pi])
    x, y = transform * (c0 + pc + 0.5, r0 + pr + 0.5)
    lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
    lat, lon = float(lats[0]), float(lons[0])
    dist_km = float(np.hypot((lat - SINK_CENTER[0]) * 111.32,
                             (lon - SINK_CENTER[1]) * 111.32 * np.cos(np.radians(lat))))
    sm_val = short_med[pr, pc]
    bowls.append({
        "peak_cm_yr": round(float(vals[pi]) * 100, 2),
        "mean_cm_yr": round(float(vals.mean()) * 100, 2),
        "pixels": int(px.sum()),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "km_from_sink_center": round(dist_km, 2),
        "short_pair_median_cm_yr_at_peak": round(float(sm_val) * 100, 2) if np.isfinite(sm_val) else None,
        "short_pairs_negative_at_peak": int(np.nansum(short[:, pr, pc] < 0)) if np.isfinite(sm_val) else None,
        "short_pairs_total_at_peak": int(np.isfinite(short[:, pr, pc]).sum()),
    })
bowls.sort(key=lambda b: b["peak_cm_yr"])
print(f"LOCAL_SUBSIDENCE_BOWLS: {len(bowls)}")
for b in bowls[:10]:
    print(f"  peak={b['peak_cm_yr']} cm/yr px={b['pixels']} at ({b['lat']}, {b['lon']}) "
          f"dist_from_sinks={b['km_from_sink_center']} km "
          f"short_med={b['short_pair_median_cm_yr_at_peak']} "
          f"neg={b['short_pairs_negative_at_peak']}/{b['short_pairs_total_at_peak']}")

near = [b for b in bowls if b["km_from_sink_center"] <= 4.0 and b["peak_cm_yr"] <= -3.0]
consistent = [b for b in near
              if b["short_pairs_negative_at_peak"] is not None
              and b["short_pairs_total_at_peak"] >= 8
              and b["short_pairs_negative_at_peak"] >= 0.65 * b["short_pairs_total_at_peak"]]
verdict = "PASS" if consistent else ("WEAK_PASS" if near else "FAIL")
print(f"WINK_LOCAL_VALIDATION_{verdict}")

write_strict_json(WORK / "wink_local_validation.json", {
    "schema_version": 1,
    "long_pair": long_los[0].name,
    "box": BOX,
    "sink_center_assumed": SINK_CENTER,
    "bowls": bowls,
    "near_sink_bowls": near,
    "consistent_near_sink_bowls": consistent,
    "verdict": verdict,
})
print("SAVED:", WORK / "wink_local_validation.json")
