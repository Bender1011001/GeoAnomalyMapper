"""Wink InSAR analysis v2 — robust bowl detection with artifact suppression.

Fixes over v1:
- High-pass the stacked velocity (normalized-convolution Gaussian, ~5 km) to
  remove atmospheric ramps and orbital trends before detection.
- Temporal consistency vote: a bowl must appear with the same sign in >=3 of
  the 4 independent long-interval pairs (median long-pair rate as robust rate).
- Physically bounded rates and a tight ground-truth box at the published Wink
  Sink location (sinks ~2-4 km NE of Wink town; documented 8-60+ cm/yr).
- Chance-rate accounting: reports detection density so the validation cannot
  pass by luck.
"""
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))
WORK = REPO / "data" / "insar_wink"
PRODUCTS = WORK / "products"

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy as transform_xy
from scipy.ndimage import gaussian_filter, label
import pyproj

from json_utils import dump_strict_json

COHERENCE_FLOOR = 0.30
HP_SIGMA_PX = 60                  # ~4.8 km at 80 m pixels
STACK_THRESH_M_YR = 0.03          # |high-passed stacked velocity| >= 3 cm/yr
LONGPAIR_THRESH_M_YR = 0.02       # |median long-pair rate| >= 2 cm/yr
MIN_SIGN_AGREE = 3                # of 4 long pairs
MIN_BOWL_PIXELS = 6
MAX_PLAUSIBLE_RATE = 1.0          # m/yr; beyond this = unwrapping artifact
LONG_PAIR_MIN_DAYS = 150
WINK_BOX = {"lat_min": 31.72, "lat_max": 31.82, "lon_min": -103.20, "lon_max": -103.05}

DATE_RE = re.compile(r"S1[ABC]{2}_(\d{8})T\d{6}_(\d{8})T\d{6}")


def pair_days(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    d0 = datetime.strptime(m.group(1), "%Y%m%d")
    d1 = datetime.strptime(m.group(2), "%Y%m%d")
    return abs((d1 - d0).days)


# ---- Collect extracted rasters (already unzipped by v1) ----
los_files = sorted(PRODUCTS.glob("*_los_disp.tif"))
pairs = []
for los in los_files:
    days = pair_days(los.name)
    corr = los.with_name(los.name.replace("_los_disp.tif", "_corr.tif"))
    if days:
        pairs.append({"los": los, "corr": corr if corr.exists() else None, "days": days})
print(f"PAIRS: {len(pairs)} (long: {sum(1 for p in pairs if p['days'] >= LONG_PAIR_MIN_DAYS)})", flush=True)

with rasterio.open(pairs[0]["los"]) as ref:
    ref_profile = ref.profile.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width


def load_on_grid(path):
    with rasterio.open(path) as src:
        if src.crs == ref_crs and src.transform == ref_transform and (src.height, src.width) == (H, W):
            data = src.read(1).astype(np.float64)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data
        out = np.full((H, W), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.bilinear, dst_nodata=np.nan,
        )
        return out


# ---- Stacked velocity (all pairs, referenced, coherence-masked) ----
sum_disp = np.zeros((H, W)); sum_days = np.zeros((H, W)); n_px = np.zeros((H, W), dtype=np.int32)
long_rates = []
long_days = []
for p in pairs:
    disp = load_on_grid(p["los"])
    valid = np.isfinite(disp) & (disp != 0.0)
    if p["corr"]:
        corr = load_on_grid(p["corr"])
        valid &= np.isfinite(corr) & (corr >= COHERENCE_FLOOR)
    if not valid.any():
        continue
    disp = disp - float(np.median(disp[valid]))
    disp_masked = np.where(valid, disp, np.nan)
    sum_disp[valid] += disp[valid]
    sum_days[valid] += p["days"]
    n_px[valid] += 1
    if p["days"] >= LONG_PAIR_MIN_DAYS:
        long_rates.append(disp_masked / p["days"] * 365.25)
        long_days.append(p["days"])

good = (n_px >= 4) & (sum_days > 0)
velocity = np.full((H, W), np.nan)
velocity[good] = sum_disp[good] / sum_days[good] * 365.25
print(f"LONG_PAIRS_USED: {len(long_rates)} days={long_days}", flush=True)

# ---- High-pass (normalized convolution to respect NaN) ----
w = np.isfinite(velocity).astype(np.float64)
v0 = np.where(np.isfinite(velocity), velocity, 0.0)
low = gaussian_filter(v0, HP_SIGMA_PX) / np.maximum(gaussian_filter(w, HP_SIGMA_PX), 1e-6)
vel_hp = np.where(w > 0, velocity - low, np.nan)

# ---- Long-pair robust rate + sign agreement ----
lr = np.stack(long_rates)                       # (L, H, W) with NaN
rate_med = np.nanmedian(lr, axis=0)
n_finite = np.isfinite(lr).sum(axis=0)
sign_pos = np.nansum(lr > 0, axis=0)
sign_neg = np.nansum(lr < 0, axis=0)
sign_agree = np.maximum(sign_pos, sign_neg)

# high-pass the long-pair median too (same artifact classes)
w2 = np.isfinite(rate_med).astype(np.float64)
r0 = np.where(np.isfinite(rate_med), rate_med, 0.0)
low2 = gaussian_filter(r0, HP_SIGMA_PX) / np.maximum(gaussian_filter(w2, HP_SIGMA_PX), 1e-6)
rate_med_hp = np.where(w2 > 0, rate_med - low2, np.nan)

# ---- Confirmed anomaly mask ----
plausible = (np.abs(vel_hp) <= MAX_PLAUSIBLE_RATE) & (np.abs(rate_med_hp) <= MAX_PLAUSIBLE_RATE)
same_sign = np.sign(vel_hp) == np.sign(rate_med_hp)
confirmed = (
    np.isfinite(vel_hp) & np.isfinite(rate_med_hp) & plausible & same_sign
    & (np.abs(vel_hp) >= STACK_THRESH_M_YR)
    & (np.abs(rate_med_hp) >= LONGPAIR_THRESH_M_YR)
    & (sign_agree >= MIN_SIGN_AGREE) & (n_finite >= MIN_SIGN_AGREE)
)
finite_hp = vel_hp[np.isfinite(vel_hp)]
print(f"HP_VEL_STATS m/yr: p01={np.percentile(finite_hp,1):.4f} p50={np.percentile(finite_hp,50):.4f} "
      f"p99={np.percentile(finite_hp,99):.4f} std={finite_hp.std():.4f}", flush=True)
print(f"CONFIRMED_PIXELS: {int(confirmed.sum())} of {int(np.isfinite(vel_hp).sum())}", flush=True)

# ---- Bowls ----
to_wgs84 = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
px_km2 = abs(ref_transform.a * ref_transform.e) / 1e6
labeled, n = label(confirmed)
bowls = []
for k in range(1, n + 1):
    px = labeled == k
    cnt = int(px.sum())
    if cnt < MIN_BOWL_PIXELS:
        continue
    rows, cols = np.where(px)
    vals = vel_hp[rows, cols]
    peak_i = np.argmax(np.abs(vals))
    pr, pc = int(rows[peak_i]), int(cols[peak_i])
    x, y = transform_xy(ref_transform, pr, pc)
    lon, lat = to_wgs84.transform(x, y)
    bowls.append({
        "type": "subsidence" if vals[peak_i] < 0 else "uplift",
        "pixels": cnt,
        "area_km2": round(cnt * px_km2, 3),
        "peak_velocity_cm_yr": round(float(vals[peak_i]) * 100, 2),
        "mean_velocity_cm_yr": round(float(vals.mean()) * 100, 2),
        "longpair_median_cm_yr": round(float(rate_med_hp[pr, pc]) * 100, 2),
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
    })
bowls.sort(key=lambda b: abs(b["peak_velocity_cm_yr"]), reverse=True)

coherent_km2 = float(np.isfinite(vel_hp).sum()) * px_km2
box_km2 = 111.32 * (WINK_BOX["lat_max"] - WINK_BOX["lat_min"]) * \
          111.32 * np.cos(np.radians(31.77)) * (WINK_BOX["lon_max"] - WINK_BOX["lon_min"])
density = len(bowls) / max(coherent_km2, 1)
expected_in_box = density * box_km2
in_box = [b for b in bowls
          if WINK_BOX["lat_min"] <= b["lat"] <= WINK_BOX["lat_max"]
          and WINK_BOX["lon_min"] <= b["lon"] <= WINK_BOX["lon_max"]]
subsidence_in_box = [b for b in in_box if b["type"] == "subsidence"]

print(f"CONFIRMED_BOWLS: {len(bowls)} over {coherent_km2:.0f} km2 "
      f"(density {density*1000:.2f}/1000km2, chance expectation in Wink box: {expected_in_box:.2f})", flush=True)
print(f"WINK_BOX_BOWLS: {len(in_box)} ({len(subsidence_in_box)} subsidence)", flush=True)
for b in in_box[:8]:
    print(f"  [WINK] {b['type']} peak={b['peak_velocity_cm_yr']} cm/yr area={b['area_km2']} km2 "
          f"long={b['longpair_median_cm_yr']} cm/yr at ({b['lat']}, {b['lon']})", flush=True)
print("TOP_FRAME_BOWLS:", flush=True)
for b in bowls[:20]:
    tag = "WINK" if b in in_box else "OTHER"
    print(f"  [{tag}] {b['type']} peak={b['peak_velocity_cm_yr']} cm/yr mean={b['mean_velocity_cm_yr']} "
          f"area={b['area_km2']} km2 long={b['longpair_median_cm_yr']} at ({b['lat']}, {b['lon']})", flush=True)

validation_pass = bool(
    subsidence_in_box
    and abs(subsidence_in_box[0]["peak_velocity_cm_yr"]) >= 5.0
    and len(in_box) >= max(3.0 * expected_in_box, 1.0)
)

report = {
    "schema_version": 2,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "method": "HyP3 INSAR_GAMMA; referenced coherence-masked stacking; ~5km high-pass; "
              "long-pair median rate with >=3/4 sign agreement",
    "thresholds": {
        "stack_cm_yr": STACK_THRESH_M_YR * 100,
        "longpair_cm_yr": LONGPAIR_THRESH_M_YR * 100,
        "min_sign_agree": MIN_SIGN_AGREE,
        "min_bowl_pixels": MIN_BOWL_PIXELS,
        "max_plausible_rate_cm_yr": MAX_PLAUSIBLE_RATE * 100,
    },
    "chance_accounting": {
        "confirmed_bowls": len(bowls),
        "coherent_km2": round(coherent_km2),
        "bowl_density_per_1000km2": round(density * 1000, 3),
        "wink_box_km2": round(box_km2),
        "expected_bowls_in_box_by_chance": round(expected_in_box, 2),
        "observed_bowls_in_box": len(in_box),
    },
    "wink_ground_truth": {
        "box": WINK_BOX,
        "published_rates_cm_yr": "8-18 near sinks (2011-2016); 40-60+ ~1km east of Sink 2",
        "detections": in_box,
    },
    "validation_pass": validation_pass,
    "top_bowls": bowls[:50],
}
report_path = WORK / "wink_insar_report_v2.json"
with open(report_path, "w", encoding="utf-8") as f:
    dump_strict_json(report, f, indent=2)

# save high-passed velocity raster
prof = ref_profile.copy()
prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
hp_path = WORK / "wink_los_velocity_highpass_m_yr.tif"
with rasterio.open(hp_path, "w", **prof) as dst:
    dst.write(vel_hp.astype(np.float32), 1)
    dst.set_band_description(1, "highpass_stacked_los_velocity_m_per_yr")

print(f"REPORT_V2: {report_path}", flush=True)
print("VALIDATION_" + ("PASS" if validation_pass else "FAIL"), flush=True)
