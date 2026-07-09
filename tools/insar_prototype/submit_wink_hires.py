"""Resubmit short-interval Wink pairs at 10x2 looks (40 m posting) so the
110-250 m sinkhole bowls are resolvable and fast deformation stays <1 fringe."""
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
JOB_NAME = "gam_wink_hires"
MAX_PAIRS = 12

slc_data_fetcher.load_env_file(REPO / ".env")
auth = slc_data_fetcher.resolve_earthdata_auth()
hyp3 = HyP3(username=auth["username"], password=auth["password"])
try:
    print("CREDITS:", hyp3.check_credits())
except Exception:
    pass

end = datetime.now(timezone.utc)
start = end - timedelta(days=200)
results = asf.geo_search(
    intersectsWith="POINT(-103.12 31.77)",   # directly over the sinks
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel=asf.PRODUCT_TYPE.SLC,
    beamMode=asf.BEAMMODE.IW,
    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    maxResults=200,
)
scenes = [r.properties for r in results
          if r.properties.get("pathNumber") == 78 and r.properties.get("flightDirection") == "ASCENDING"]
by_date = {}
for p in sorted(scenes, key=lambda p: p["startTime"]):
    by_date.setdefault(p["startTime"][:10], p)
dates = sorted(by_date)
print(f"DATES: {len(dates)} from {dates[0]} to {dates[-1]}")

pairs = []
for i in range(len(dates) - 1, 0, -1):
    pairs.append((by_date[dates[i - 1]]["sceneName"], by_date[dates[i]]["sceneName"]))
    if len(pairs) >= MAX_PAIRS:
        break

sig = inspect.signature(hyp3.submit_insar_job)
kwargs = {"looks": "10x2"}
if "include_displacement_maps" in sig.parameters:
    kwargs["include_displacement_maps"] = True
print("PAIRS:", len(pairs), "KWARGS:", kwargs)

jobs = []
for ref, sec in pairs:
    batch = hyp3.submit_insar_job(ref, sec, name=JOB_NAME, **kwargs)
    jobs.extend(batch.jobs if hasattr(batch, "jobs") else [batch])
print(f"JOBS_SUBMITTED: {len(jobs)}")
(SCRATCH / "wink_hires_jobs.json").write_text(
    json.dumps([j.to_dict() for j in jobs], indent=2, default=str), encoding="utf-8")
print("SAVED")
