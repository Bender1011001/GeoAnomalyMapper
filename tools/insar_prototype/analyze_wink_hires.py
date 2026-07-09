"""Analyze 10x2-look short-pair stack over the Wink sinks.

Sensitivity: ~10-13 cm/yr stack noise -> detects only fast bowls (>=25 cm/yr),
which matches the published 40-60+ cm/yr feature east of Wink Sink 2.
"""
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))
SCRATCH = Path(r"C:\Users\admin\AppData\Local\Temp\claude\E--code-projects-GeoAnomalyMapper-1\a2a0531d-4552-488d-8ff4-e85ea3157475\scratchpad")
WORK = REPO / "data" / "insar_wink"
HIRES = WORK / "products_hires"
HIRES.mkdir(parents=True, exist_ok=True)

import rasterio
from rasterio.warp import transform as warp_transform, reproject, Resampling
from scipy.ndimage import gaussian_filter, label

import slc_data_fetcher
from json_utils import write_strict_json
from hyp3_sdk import HyP3

BOX = {"lat_min": 31.70, "lat_max": 31.84, "lon_min": -103.22, "lon_max": -103.02}
SINK_CENTER = (31.77, -103.12)
DATE_RE = re.compile(r"S1[ABC]{2}_(\d{8})T\d{6}_(\d{8})T\d{6}")
COH_FLOOR = 0.30
DETECT_CM_YR = 25.0
MIN_NEG_FRACTION = 0.65

slc_data_fetcher.load_env_file(REPO / ".env")
auth = slc_data_fetcher.resolve_earthdata_auth()
hyp3 = HyP3(username=auth["username"], password=auth["password"])
tracked = {j["job_id"] for j in json.loads((SCRATCH / "wink_hires_jobs.json").read_text())}
succeeded = [j for j in hyp3.find_jobs(name="gam_wink_hires") if j.job_id in tracked and j.succeeded()]
print(f"SUCCEEDED: {len(succeeded)}", flush=True)
for job in succeeded:
    try:
        job.download_files(HIRES)
    except Exception as exc:
        print(f"DOWNLOAD_FAILED {job.job_id}: {exc}", flush=True)

for zpath in sorted(HIRES.glob("*.zip")):
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if member.endswith(("_los_disp.tif", "_corr.tif")):
                target = HIRES / Path(member).name
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

los_files = sorted(HIRES.glob("*_los_disp.tif"))
print(f"LOS_RASTERS: {len(los_files)}", flush=True)

with rasterio.open(los_files[0]) as src:
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
box_transform = transform * transform.translation(c0, r0)
print(f"BOX {BH}x{BW} px at {abs(transform.a):.0f} m", flush=True)


def load_box(path):
    with rasterio.open(path) as src:
        if src.crs == crs and src.transform == transform:
            return src.read(1, window=((r0, r1), (c0, c1))).astype(np.float64)
        out = np.full((BH, BW), np.nan)
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=box_transform, dst_crs=crs,
                  resampling=Resampling.bilinear, dst_nodata=np.nan)
        return out


sum_disp = np.zeros((BH, BW)); sum_days = np.zeros((BH, BW)); n_px = np.zeros((BH, BW), dtype=int)
rates = []
used = []
for los in los_files:
    m = DATE_RE.search(los.name)
    days = abs((datetime.strptime(m.group(2), "%Y%m%d") - datetime.strptime(m.group(1), "%Y%m%d")).days)
    disp = load_box(los)
    valid = np.isfinite(disp) & (disp != 0.0)
    corr_path = los.with_name(los.name.replace("_los_disp.tif", "_corr.tif"))
    if corr_path.exists():
        corr = load_box(corr_path)
        valid &= np.isfinite(corr) & (corr >= COH_FLOOR)
    if valid.sum() < 500:
        print(f"SKIP {los.name[5:36]} valid={int(valid.sum())}", flush=True)
        continue
    disp = disp - float(np.median(disp[valid]))
    rate = np.where(valid, disp / days * 365.25, np.nan)
    rates.append(rate)
    sum_disp[valid] += disp[valid]
    sum_days[valid] += days
    n_px[valid] += 1
    used.append({"pair": los.name[5:36], "days": days, "valid_px": int(valid.sum())})
    print(f"USED {los.name[5:36]} days={days} valid={int(valid.sum())}", flush=True)

stack_rate = np.full((BH, BW), np.nan)
ok = (n_px >= max(6, len(rates) // 2)) & (sum_days > 0)
stack_rate[ok] = sum_disp[ok] / sum_days[ok] * 365.25

rate_cube = np.stack(rates)
neg_count = np.nansum(rate_cube < 0, axis=0)
fin_count = np.isfinite(rate_cube).sum(axis=0)

sm = gaussian_filter(np.nan_to_num(stack_rate, nan=0.0), 1.5)
wt = gaussian_filter(np.isfinite(stack_rate).astype(float), 1.5)
sm = np.where(wt > 0.3, sm / np.maximum(wt, 1e-6), np.nan)

finite = sm[np.isfinite(sm)]
print(f"HIRES_STACK cm/yr: p01={np.percentile(finite,1)*100:.1f} p50={np.percentile(finite,50)*100:.1f} "
      f"p99={np.percentile(finite,99)*100:.1f} min={finite.min()*100:.1f}", flush=True)

mask = np.isfinite(sm) & (sm <= -DETECT_CM_YR / 100.0)
labeled, n = label(mask)
bowls = []
for k in range(1, n + 1):
    px = labeled == k
    if px.sum() < 4:
        continue
    rows, cols = np.where(px)
    vals = sm[rows, cols]
    pi = int(np.argmax(-vals))
    pr, pc = int(rows[pi]), int(cols[pi])
    x, y = box_transform * (pc + 0.5, pr + 0.5)
    lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
    lat, lon = float(lats[0]), float(lons[0])
    dist_km = float(np.hypot((lat - SINK_CENTER[0]) * 111.32,
                             (lon - SINK_CENTER[1]) * 111.32 * np.cos(np.radians(lat))))
    nf = int(fin_count[pr, pc])
    bowls.append({
        "peak_cm_yr": round(float(vals[pi]) * 100, 1),
        "mean_cm_yr": round(float(np.mean(vals)) * 100, 1),
        "pixels": int(px.sum()),
        "area_km2": round(px.sum() * abs(box_transform.a * box_transform.e) / 1e6, 3),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "km_from_sink_center": round(dist_km, 2),
        "pairs_negative_at_peak": int(neg_count[pr, pc]),
        "pairs_finite_at_peak": nf,
        "neg_fraction": round(float(neg_count[pr, pc]) / max(nf, 1), 2),
    })
bowls.sort(key=lambda b: b["peak_cm_yr"])
print(f"FAST_BOWLS: {len(bowls)}", flush=True)
for b in bowls[:10]:
    print(f"  peak={b['peak_cm_yr']} cm/yr area={b['area_km2']} km2 at ({b['lat']}, {b['lon']}) "
          f"dist={b['km_from_sink_center']} km neg={b['pairs_negative_at_peak']}/{b['pairs_finite_at_peak']}", flush=True)

near = [b for b in bowls if b["km_from_sink_center"] <= 4.0 and b["neg_fraction"] >= MIN_NEG_FRACTION]
verdict = "PASS" if near else "FAIL"
print(f"HIRES_VALIDATION_{verdict}", flush=True)

# save stack raster
prof = {"driver": "GTiff", "height": BH, "width": BW, "count": 1, "dtype": "float32",
        "crs": crs, "transform": box_transform, "nodata": np.nan, "compress": "deflate"}
with rasterio.open(WORK / "wink_hires_stack_rate_m_yr.tif", "w", **prof) as dst:
    dst.write(sm.astype(np.float32), 1)

write_strict_json(WORK / "wink_hires_validation.json", {
    "schema_version": 1,
    "pairs_used": used,
    "detect_threshold_cm_yr": DETECT_CM_YR,
    "min_neg_fraction": MIN_NEG_FRACTION,
    "sink_center_assumed": SINK_CENTER,
    "bowls": bowls,
    "near_sink_fast_bowls": near,
    "verdict": verdict,
})
print("SAVED wink_hires_validation.json", flush=True)
