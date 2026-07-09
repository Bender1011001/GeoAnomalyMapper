#!/usr/bin/env python3
"""
Run the pipeline on targets that have REAL downloaded SLC data.
No synthetic fallback - this is the real deal.
"""
import sys
import os

# Bootstrap credentials before any other imports
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

from run_biondi_exploration import (
    execute_biondi_pipeline_for_target,
    RESOLUTION_PROFILES,
    EXPLORE_DIR,
)
import logging
from datetime import datetime

from json_utils import dump_strict_json
import slc_data_fetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


ANOMALY_COUNT_KEYS = ("anomaly_count", "anomalies_detected")


def _coerce_anomaly_count(value):
    """Return a non-negative anomaly count, or None if the value is unusable."""
    if value is None:
        return None
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return None


def _get_anomaly_count(result):
    """Accept both current and legacy anomaly-count keys."""
    if not isinstance(result, dict):
        return 0
    for key in ANOMALY_COUNT_KEYS:
        count = _coerce_anomaly_count(result.get(key))
        if count is not None:
            return count
    return 0

# Targets with real SLC data on disk
REAL_DATA_TARGETS = [
    {
        "name": "Carlsbad Caverns (NM)",
        "lat": 32.1479,
        "lon": -104.5567,
        "buffer_deg": 0.1,
        "description": "Carlsbad Caverns - 120+ known caves, massive Big Room chamber. "
                       "Ground truth for natural cavern detection.",
        "expected_depth_m": 230,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Mammoth Cave (KY)",
        "lat": 37.1862,
        "lon": -86.1005,
        "buffer_deg": 0.1,
        "description": "World's longest known cave system - 680+ km mapped passages.",
        "expected_depth_m": 120,
        "expected_void_type": "natural_cave",
    },
]


def main():
    resolution = sys.argv[1] if len(sys.argv) > 1 else "standard"
    if resolution not in RESOLUTION_PROFILES:
        print(f"Unknown resolution: {resolution}. Options: {list(RESOLUTION_PROFILES.keys())}")
        sys.exit(1)

    credentials = {
        "EARTHDATA_USERNAME": os.environ.get("EARTHDATA_USERNAME", ""),
        "EARTHDATA_PASSWORD": os.environ.get("EARTHDATA_PASSWORD", ""),
        "EARTHDATA_TOKEN": os.environ.get("EARTHDATA_TOKEN", ""),
        "EARTHDATA_BEARER_TOKEN": os.environ.get("EARTHDATA_BEARER_TOKEN", ""),
    }
    auth_info = slc_data_fetcher.resolve_earthdata_auth(
        earthdata_username=credentials["EARTHDATA_USERNAME"],
        earthdata_password=credentials["EARTHDATA_PASSWORD"],
        earthdata_token=(credentials["EARTHDATA_TOKEN"] or credentials["EARTHDATA_BEARER_TOKEN"]),
    )
    logger.info("Earthdata auth configured: %s", slc_data_fetcher.describe_earthdata_auth(auth_info))

    results = []
    total_start = datetime.now()

    print("=" * 70)
    print(f"  REAL SLC DATA PIPELINE - {len(REAL_DATA_TARGETS)} targets @ {resolution}")
    print("=" * 70)
    print()

    for i, target in enumerate(REAL_DATA_TARGETS, 1):
        print(f"\n[Target {i}/{len(REAL_DATA_TARGETS)}] {target['name']}")
        print("-" * 50)

        try:
            result = execute_biondi_pipeline_for_target(
                target=target,
                credentials=credentials,
                use_synthetic_fallback=False,  # REAL DATA ONLY
                resolution=resolution,
            )
            results.append(result)
            anomalies = _get_anomaly_count(result)
            result["anomaly_count"] = anomalies
            result["anomalies_detected"] = anomalies
            source = result.get("data_source", "unknown")
            elapsed = result.get("elapsed_seconds", 0)
            logger.info(f"  -> {target['name']}: {anomalies} anomalies, "
                        f"source={source}, time={elapsed:.1f}s")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            results.append({
                "target": target["name"],
                "status": "error",
                "error": str(e),
                "anomaly_count": 0,
                "anomalies_detected": 0,
            })

    total_elapsed = (datetime.now() - total_start).total_seconds()

    # Report
    print("\n" + "=" * 70)
    print("  REAL DATA RESULTS")
    print("=" * 70)
    for r in results:
        name = r.get("target", r.get("target_name", "?"))
        status = r.get("status", "?")
        anomalies = _get_anomaly_count(r)
        source = r.get("data_source", "?")
        print(f"  {status.upper():8s} {name}: {anomalies} anomalies (source: {source})")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    # Save JSON report
    report_path = EXPLORE_DIR / "real_data_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        dump_strict_json({
            "timestamp": datetime.now().isoformat(),
            "resolution": resolution,
            "total_seconds": total_elapsed,
            "targets": results,
        }, f, indent=2)
        f.write("\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
