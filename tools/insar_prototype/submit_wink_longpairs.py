"""Submit additional long-interval HyP3 pairs over Wink for robust rate estimation."""
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
JOB_NAME = "gam_wink_v1"

slc_data_fetcher.load_env_file(REPO / ".env")
auth = slc_data_fetcher.resolve_earthdata_auth()
hyp3 = HyP3(username=auth["username"], password=auth["password"])

end = datetime.now(timezone.utc)
start = end - timedelta(days=430)
results = asf.geo_search(
    intersectsWith="POINT(-103.13 31.80)",
    platform=asf.PLATFORM.SENTINEL1,
    processingLevel=asf.PRODUCT_TYPE.SLC,
    beamMode=asf.BEAMMODE.IW,
    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    maxResults=250,
)
scenes = [r.properties for r in results
          if r.properties.get("pathNumber") == 78 and r.properties.get("flightDirection") == "ASCENDING"]
by_date = {}
for p in sorted(scenes, key=lambda p: p["startTime"]):
    by_date.setdefault(p["startTime"][:10], p)
dates = sorted(by_date)
print("STACK_DATES:", dates[0], "...", dates[-1], f"({len(dates)} dates)")


def nearest(target: str) -> str:
    return min(dates, key=lambda d: abs((datetime.fromisoformat(d) - datetime.fromisoformat(target)).days))


latest = dates[-1]
pairs_wanted = [
    (nearest("2025-06-01"), latest),   # ~13 months
    (nearest("2025-09-01"), latest),   # ~10 months
    (nearest("2025-12-15"), latest),   # ~6 months
    (nearest("2025-06-01"), nearest("2025-12-15")),  # ~6 months, earlier window
]
sig = inspect.signature(hyp3.submit_insar_job)
kwargs = {"looks": "20x4"}
if "include_displacement_maps" in sig.parameters:
    kwargs["include_displacement_maps"] = True
if "include_look_vectors" in sig.parameters:
    kwargs["include_look_vectors"] = True

jobs = []
seen = set()
for d_ref, d_sec in pairs_wanted:
    if (d_ref, d_sec) in seen or d_ref == d_sec:
        continue
    seen.add((d_ref, d_sec))
    ref = by_date[d_ref]["sceneName"]
    sec = by_date[d_sec]["sceneName"]
    print(f"LONG_PAIR: {d_ref} -> {d_sec}")
    batch = hyp3.submit_insar_job(ref, sec, name=JOB_NAME, **kwargs)
    jobs.extend(batch.jobs if hasattr(batch, "jobs") else [batch])

print(f"LONG_JOBS_SUBMITTED: {len(jobs)}")
existing = json.loads((SCRATCH / "wink_jobs.json").read_text(encoding="utf-8"))
existing.extend(j.to_dict() for j in jobs)
(SCRATCH / "wink_jobs.json").write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
print(f"TOTAL_JOBS_TRACKED: {len(existing)}")
