"""Select a Sentinel-1 stack over the Wink TX sinkholes and submit HyP3 InSAR jobs.

Ground truth: the Wink Sinks (collapse sinkholes over dissolved Permian salt,
between Wink and Kermit, TX) have documented ongoing cm/yr subsidence measured
by published InSAR studies. If our stack reproduces localized subsidence there,
the deformation pipeline is validated against a known underground feature.
"""
import inspect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))

import slc_data_fetcher
import asf_search as asf
from hyp3_sdk import HyP3

SCRATCH = Path(r"C:\Users\admin\AppData\Local\Temp\claude\E--code-projects-GeoAnomalyMapper-1\a2a0531d-4552-488d-8ff4-e85ea3157475\scratchpad")
POINT_LAT, POINT_LON = 31.80, -103.13  # Wink-Kermit corridor
MONTHS_BACK = 14
MAX_PAIRS = 12
JOB_NAME = "gam_wink_v1"

slc_data_fetcher.load_env_file(REPO / ".env")
auth = slc_data_fetcher.resolve_earthdata_auth()
if auth["mode"] != "credentials":
    print("NOTE: auth mode is", auth["mode"])

hyp3 = HyP3(username=auth["username"], password=auth["password"])
try:
    credits = hyp3.check_credits()
except Exception:
    credits = hyp3.my_info().get("remaining_credits")
print(f"CREDITS_REMAINING: {credits}")

end = datetime.now(timezone.utc)
start = end - timedelta(days=MONTHS_BACK * 30)
results = asf.geo_search(
    intersectsWith=f"POINT({POINT_LON} {POINT_LAT})",
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel=asf.PRODUCT_TYPE.SLC,
    beamMode=asf.BEAMMODE.IW,
    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    maxResults=250,
)
print(f"SCENES_FOUND: {len(results)}")

# Group by (path, direction); pick the group with the most acquisitions.
groups = {}
for r in results:
    p = r.properties
    key = (p.get("pathNumber"), p.get("flightDirection"))
    groups.setdefault(key, []).append(p)
best_key = max(groups, key=lambda k: len(groups[k]))
scenes = sorted(groups[best_key], key=lambda p: p["startTime"])
print(f"BEST_STACK: path={best_key[0]} dir={best_key[1]} scenes={len(scenes)}")

# Deduplicate by date (multiple frames same day covering point -> keep first)
by_date = {}
for p in scenes:
    by_date.setdefault(p["startTime"][:10], p)
dates = sorted(by_date)
scenes = [by_date[d] for d in dates]
print(f"UNIQUE_DATES: {len(scenes)} from {dates[0]} to {dates[-1]}")

# Consecutive pairs, most recent first, capped.
pairs = []
for i in range(len(scenes) - 1, 0, -1):
    ref = scenes[i - 1]["sceneName"]
    sec = scenes[i]["sceneName"]
    pairs.append((ref, sec))
    if len(pairs) >= MAX_PAIRS:
        break
print(f"PAIRS_PLANNED: {len(pairs)}")
for ref, sec in pairs:
    print(f"  {ref[17:25]} -> {sec[17:25]}")

# Build kwargs compatible with the installed SDK version.
sig = inspect.signature(hyp3.submit_insar_job)
kwargs = {"looks": "20x4"}
if "include_displacement_maps" in sig.parameters:
    kwargs["include_displacement_maps"] = True
elif "include_los_displacement" in sig.parameters:
    kwargs["include_los_displacement"] = True
if "include_look_vectors" in sig.parameters:
    kwargs["include_look_vectors"] = True
print("SUBMIT_KWARGS:", kwargs)

jobs = []
for ref, sec in pairs:
    job = hyp3.submit_insar_job(ref, sec, name=JOB_NAME, **kwargs)
    jobs.extend(job.jobs if hasattr(job, "jobs") else [job])
print(f"JOBS_SUBMITTED: {len(jobs)}")

out = SCRATCH / "wink_jobs.json"
out.write_text(json.dumps([j.to_dict() for j in jobs], indent=2, default=str), encoding="utf-8")
print(f"JOBS_SAVED: {out}")
