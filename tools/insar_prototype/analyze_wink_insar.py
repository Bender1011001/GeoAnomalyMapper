"""Download completed HyP3 InSAR products, stack LOS displacement into a
velocity map, validate against the Wink sinkholes, and scan for anomalies.

Method (classic interferogram stacking):
  velocity ~ sum(los_disp_i) / sum(dt_i)  over coherent pixels,
weighted implicitly by pair duration; pixels below a coherence floor in a
pair are excluded from that pair's sums. Deformation bowls are detected on
the velocity map with connected components, exactly mirroring the project's
audit-first conventions.
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
PRODUCTS = WORK / "products"
PRODUCTS.mkdir(parents=True, exist_ok=True)

import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import label

import slc_data_fetcher
from json_utils import dump_strict_json
from hyp3_sdk import HyP3

COHERENCE_FLOOR = 0.30
MIN_PAIRS_PER_PIXEL = 4
VELOCITY_THRESHOLD_M_YR = 0.02   # |LOS velocity| >= 2 cm/yr = anomaly
MIN_BOWL_PIXELS = 6              # at ~80 m pixels -> >= ~0.04 km^2
WINK_BOX = {"lat_min": 31.72, "lat_max": 31.90, "lon_min": -103.25, "lon_max": -103.02}

slc_data_fetcher.load_env_file(REPO / ".env")
auth = slc_data_fetcher.resolve_earthdata_auth()
hyp3 = HyP3(username=auth["username"], password=auth["password"])

tracked = {j["job_id"] for j in json.loads((SCRATCH / "wink_jobs.json").read_text(encoding="utf-8"))}
batch = hyp3.find_jobs(name="gam_wink_v1")
succeeded = [j for j in batch if j.job_id in tracked and j.succeeded()]
print(f"SUCCEEDED_JOBS: {len(succeeded)} of {len(tracked)}", flush=True)

# ---- Download ----
zips = []
for job in succeeded:
    existing = list(PRODUCTS.glob(f"*{job.job_id[:8]}*.zip"))
    try:
        files = job.download_files(PRODUCTS)
        zips.extend(Path(f) for f in files)
    except Exception as exc:
        print(f"DOWNLOAD_FAILED {job.job_id}: {exc}", flush=True)
zips = sorted(set(p for p in PRODUCTS.glob("*.zip")))
print(f"PRODUCT_ZIPS: {len(zips)}", flush=True)

DATE_RE = re.compile(r"S1[ABC]{2}_(\d{8})T\d{6}_(\d{8})T\d{6}")

def pair_days(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    d0 = datetime.strptime(m.group(1), "%Y%m%d")
    d1 = datetime.strptime(m.group(2), "%Y%m%d")
    return abs((d1 - d0).days)

# ---- Extract los_disp + corr rasters ----
pairs = []
for zpath in zips:
    days = pair_days(zpath.name)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        los = [n for n in names if n.endswith("_los_disp.tif")]
        corr = [n for n in names if n.endswith("_corr.tif")]
        if not los or days is None:
            print(f"SKIP {zpath.name}: los={bool(los)} days={days}", flush=True)
            continue
        for member in los + corr:
            target = PRODUCTS / Path(member).name
            if not target.exists():
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        pairs.append({
            "zip": zpath.name,
            "days": days,
            "los": PRODUCTS / Path(los[0]).name,
            "corr": (PRODUCTS / Path(corr[0]).name) if corr else None,
        })
print(f"PAIRS_WITH_LOS: {len(pairs)}", flush=True)
if len(pairs) < 4:
    print("FATAL: not enough pairs to stack")
    raise SystemExit(1)

# ---- Common grid: use first raster's grid; reproject others onto it ----
with rasterio.open(pairs[0]["los"]) as ref:
    ref_profile = ref.profile.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width

sum_disp = np.zeros((H, W), dtype=np.float64)
sum_days = np.zeros((H, W), dtype=np.float64)
n_pairs_px = np.zeros((H, W), dtype=np.int32)

def load_on_grid(path):
    with rasterio.open(path) as src:
        if src.crs == ref_crs and src.transform == ref_transform and (src.height, src.width) == (H, W):
            return src.read(1).astype(np.float64)
        out = np.full((H, W), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.bilinear, dst_nodata=np.nan,
        )
        return out

for p in pairs:
    disp = load_on_grid(p["los"])  # meters, LOS
    valid = np.isfinite(disp) & (disp != 0.0)
    if p["corr"] and p["corr"].exists():
        corr = load_on_grid(p["corr"])
        valid &= np.isfinite(corr) & (corr >= COHERENCE_FLOOR)
    if not valid.any():
        print(f"STACKED {p['zip'][:60]} days={p['days']} valid_px=0 (skipped)", flush=True)
        continue
    # HyP3 unwrapped products carry an arbitrary constant offset per pair;
    # reference each pair to its own coherent-area median so the stack
    # measures motion relative to the (stable) regional background.
    disp = disp - float(np.median(disp[valid]))
    sum_disp[valid] += disp[valid]
    sum_days[valid] += p["days"]
    n_pairs_px[valid] += 1
    print(f"STACKED {p['zip'][:60]} days={p['days']} valid_px={int(valid.sum())}", flush=True)

good = (n_pairs_px >= MIN_PAIRS_PER_PIXEL) & (sum_days > 0)
velocity = np.full((H, W), np.nan, dtype=np.float32)
velocity[good] = (sum_disp[good] / sum_days[good] * 365.25).astype(np.float32)  # m/yr LOS

vel_path = WORK / "wink_los_velocity_m_yr.tif"
prof = ref_profile.copy()
prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
with rasterio.open(vel_path, "w", **prof) as dst:
    dst.write(velocity, 1)
    dst.set_band_description(1, "stacked_los_velocity_m_per_yr")
print(f"VELOCITY_MAP: {vel_path}", flush=True)

finite = velocity[np.isfinite(velocity)]
print(f"VEL_STATS m/yr: n={finite.size} p01={np.percentile(finite,1):.4f} "
      f"p50={np.percentile(finite,50):.4f} p99={np.percentile(finite,99):.4f} "
      f"std={finite.std():.4f}", flush=True)

# ---- Anomaly detection: connected bowls beyond +/- threshold ----
from rasterio.transform import xy as transform_xy
import pyproj
to_wgs84 = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)

anomalies = []
for sign, mask in (("subsidence", velocity <= -VELOCITY_THRESHOLD_M_YR),
                   ("uplift", velocity >= VELOCITY_THRESHOLD_M_YR)):
    mask = mask & np.isfinite(velocity)
    labeled, n = label(mask)
    for k in range(1, n + 1):
        px = labeled == k
        cnt = int(px.sum())
        if cnt < MIN_BOWL_PIXELS:
            continue
        rows, cols = np.where(px)
        peak_idx = np.argmax(np.abs(velocity[rows, cols]))
        pr, pc = rows[peak_idx], cols[peak_idx]
        x, y = transform_xy(ref_transform, int(pr), int(pc))
        lon, lat = to_wgs84.transform(x, y)
        anomalies.append({
            "type": sign,
            "pixels": cnt,
            "area_km2": round(cnt * abs(ref_transform.a * ref_transform.e) / 1e6, 3),
            "peak_velocity_m_yr": round(float(velocity[pr, pc]), 4),
            "mean_velocity_m_yr": round(float(velocity[px].mean()), 4),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
        })

anomalies.sort(key=lambda a: abs(a["peak_velocity_m_yr"]), reverse=True)
print(f"ANOMALY_BOWLS: {len(anomalies)}", flush=True)

in_wink = [a for a in anomalies
           if WINK_BOX["lat_min"] <= a["lat"] <= WINK_BOX["lat_max"]
           and WINK_BOX["lon_min"] <= a["lon"] <= WINK_BOX["lon_max"]]
print(f"WINK_BOX_DETECTIONS: {len(in_wink)}", flush=True)
for a in anomalies[:15]:
    tag = "WINK_GT" if a in in_wink else "OTHER"
    print(f"  [{tag}] {a['type']} peak={a['peak_velocity_m_yr']*100:.1f} cm/yr "
          f"mean={a['mean_velocity_m_yr']*100:.1f} cm/yr area={a['area_km2']} km2 "
          f"at ({a['lat']}, {a['lon']})", flush=True)

report = {
    "schema_version": 1,
    "created_at_utc": datetime.utcnow().isoformat() + "Z",
    "method": "HyP3 INSAR_GAMMA pairs, coherence-masked LOS displacement stacking",
    "pairs_used": [{"zip": p["zip"], "days": p["days"]} for p in pairs],
    "coherence_floor": COHERENCE_FLOOR,
    "min_pairs_per_pixel": MIN_PAIRS_PER_PIXEL,
    "velocity_threshold_m_yr": VELOCITY_THRESHOLD_M_YR,
    "min_bowl_pixels": MIN_BOWL_PIXELS,
    "wink_ground_truth_box": WINK_BOX,
    "wink_box_detections": in_wink,
    "all_anomalies": anomalies,
    "velocity_map": str(vel_path),
}
report_path = WORK / "wink_insar_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    dump_strict_json(report, f, indent=2)
print(f"REPORT: {report_path}", flush=True)
print("VALIDATION_" + ("PASS" if in_wink else "FAIL"), flush=True)
