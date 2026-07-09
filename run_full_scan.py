#!/usr/bin/env python3
"""
Full Real-Data Subsurface Scan
================================
Downloads real Sentinel-1 SLC data from NASA Earthdata and runs the
full PINN inversion pipeline on each target. NO synthetic data.

Targets include:
  - California deep scan (military + geological)
  - Alleged Deep Underground Military Bases (DUMBs)
  - Confirmed underground facilities (ground truth)
  - Major cave systems (validation targets)
"""
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

from json_utils import dump_strict_json

# Bootstrap credentials
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


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_top_target_summary(target):
    """Format top target fields emitted by the 3D visualizer, with legacy fallbacks."""
    rank = target.get("deep_target_rank", target.get("rank", 0))
    depth = _as_float(target.get("depth_m", 0.0))
    score = _as_float(target.get("deep_target_score", target.get("fused_confidence_score", target.get("score", 0.0))))
    shape = target.get("shape_classification", target.get("shape", "?"))
    return f"Rank {rank}: depth={depth:.0f}m, score={score:.3f}, shape={shape}"


# ============================================================
# TARGET DATABASE - All Real Data, No Synthetics
# ============================================================

# --- CALIFORNIA DEEP SCAN ---
CALIFORNIA_TARGETS = [
    {
        "name": "China Lake NAWS (CA)",
        "lat": 35.6855,
        "lon": -117.6920,
        "buffer_deg": 0.15,
        "description": "Naval Air Weapons Station China Lake. 1.1M acres, "
                       "alleged massive underground research complex beneath "
                       "the Coso Range. Multiple reports of deep tunnel networks.",
        "expected_depth_m": 1500,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Edwards AFB (CA)",
        "lat": 34.9054,
        "lon": -117.8839,
        "buffer_deg": 0.15,
        "description": "Edwards Air Force Base. Alleged deep underground testing "
                       "facilities beneath Rogers Dry Lake and the surrounding "
                       "Antelope Valley complex.",
        "expected_depth_m": 800,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Tehachapi Ranch (CA)",
        "lat": 35.0575,
        "lon": -118.5600,
        "buffer_deg": 0.1,
        "description": "Northrop Grumman Tehachapi facility (Tejon Ranch area). "
                       "Alleged deep underground aerospace R&D complex built "
                       "into the Tehachapi Mountains.",
        "expected_depth_m": 1200,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Twentynine Palms MCAGCC (CA)",
        "lat": 34.2367,
        "lon": -116.0560,
        "buffer_deg": 0.15,
        "description": "Marine Corps Air Ground Combat Center. Largest USMC base. "
                       "Persistent reports of underground tunnel systems connecting "
                       "to China Lake.",
        "expected_depth_m": 600,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Mount Shasta (CA)",
        "lat": 41.4092,
        "lon": -122.1949,
        "buffer_deg": 0.1,
        "description": "Mount Shasta volcanic complex. Reports of anomalous "
                       "cavities and lava tubes. Geological interest for deep "
                       "volcanic plumbing detection.",
        "expected_depth_m": 2000,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Vandenberg SFB (CA)",
        "lat": 34.7420,
        "lon": -120.5724,
        "buffer_deg": 0.1,
        "description": "Vandenberg Space Force Base. Strategic missile launch "
                       "facility with confirmed underground launch silos and "
                       "alleged deeper command infrastructure.",
        "expected_depth_m": 500,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Catalina Island (CA)",
        "lat": 33.3870,
        "lon": -118.4160,
        "buffer_deg": 0.1,
        "description": "Santa Catalina Island. Reports of underground naval "
                       "facility. Geological interest for sea cave detection.",
        "expected_depth_m": 300,
        "expected_void_type": "unknown",
    },
]

# --- ALLEGED DEEP UNDERGROUND MILITARY BASES ---
DUMB_TARGETS = [
    {
        "name": "Dulce Base Area (NM)",
        "lat": 36.9336,
        "lon": -106.9989,
        "buffer_deg": 0.1,
        "description": "Archuleta Mesa near Dulce, NM. Most widely reported "
                       "alleged DUMB. Jicarilla Apache land. Multiple whistleblower "
                       "claims of multi-level underground facility.",
        "expected_depth_m": 2000,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Groom Lake Area 51 (NV)",
        "lat": 37.2350,
        "lon": -115.8111,
        "buffer_deg": 0.15,
        "description": "Area 51 / Groom Lake. Confirmed classified facility. "
                       "Reports of extensive underground hangar and testing "
                       "infrastructure beneath Papoose Lake and S-4.",
        "expected_depth_m": 1000,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Cheyenne Mountain (CO)",
        "lat": 38.7446,
        "lon": -104.8464,
        "buffer_deg": 0.05,
        "description": "NORAD / Cheyenne Mountain Complex. CONFIRMED underground "
                       "military facility. 2,000ft inside granite mountain. "
                       "Ground truth for artificial cavity detection.",
        "expected_depth_m": 600,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Denver Intl Airport (CO)",
        "lat": 39.8561,
        "lon": -104.6737,
        "buffer_deg": 0.1,
        "description": "Denver International Airport. Persistent reports of "
                       "underground structures beyond the confirmed baggage "
                       "tunnel system. Multi-level basement anomalies.",
        "expected_depth_m": 200,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Raven Rock Site R (PA)",
        "lat": 39.7264,
        "lon": -77.4227,
        "buffer_deg": 0.05,
        "description": "Raven Rock Mountain Complex (Site R). CONFIRMED "
                       "alternate Pentagon. Underground city inside granite "
                       "mountain near Blue Ridge Summit, PA.",
        "expected_depth_m": 200,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Mount Weather (VA)",
        "lat": 39.0809,
        "lon": -77.8894,
        "buffer_deg": 0.05,
        "description": "Mount Weather Emergency Operations Center. CONFIRMED "
                       "FEMA continuity-of-government facility. Underground "
                       "bunker complex in the Blue Ridge Mountains.",
        "expected_depth_m": 300,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Greenbrier Bunker (WV)",
        "lat": 37.7804,
        "lon": -80.3131,
        "buffer_deg": 0.05,
        "description": "The Greenbrier Congressional Bunker. CONFIRMED "
                       "decommissioned (1992) underground facility beneath "
                       "the Greenbrier Resort. Ground truth validation.",
        "expected_depth_m": 100,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Wright Patterson AFB (OH)",
        "lat": 39.8261,
        "lon": -84.0483,
        "buffer_deg": 0.1,
        "description": "Wright-Patterson Air Force Base. Alleged underground "
                       "research complex including Hangar 18. Foreign Technology "
                       "Division headquarters.",
        "expected_depth_m": 500,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "White Sands Missile Range (NM)",
        "lat": 32.3895,
        "lon": -106.4792,
        "buffer_deg": 0.15,
        "description": "White Sands Missile Range. Largest military installation "
                       "in US. Reports of underground testing facilities and "
                       "tunnel connections to Holloman AFB.",
        "expected_depth_m": 800,
        "expected_void_type": "artificial_cavity",
    },
    {
        "name": "Dugway Proving Ground (UT)",
        "lat": 40.1786,
        "lon": -112.9358,
        "buffer_deg": 0.1,
        "description": "Dugway Proving Ground. Chemical and biological defense "
                       "testing facility. Reports of underground containment "
                       "labs and tunnel networks beneath the Great Salt Lake Desert.",
        "expected_depth_m": 600,
        "expected_void_type": "artificial_cavity",
    },
]

# --- KNOWN CAVE SYSTEMS (Ground Truth Validation) ---
CAVE_TARGETS = [
    {
        "name": "Carlsbad Caverns (NM)",
        "lat": 32.1479,
        "lon": -104.5567,
        "buffer_deg": 0.1,
        "description": "Carlsbad Caverns National Park. 120+ known caves. "
                       "Big Room: 8.2 acres, 255ft high. Known depths to 1,027ft. "
                       "Premier ground truth for natural void detection.",
        "expected_depth_m": 300,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Mammoth Cave (KY)",
        "lat": 37.1862,
        "lon": -86.1005,
        "buffer_deg": 0.1,
        "description": "World's longest known cave system, 680+ km of surveyed "
                       "passages. Multi-level system in Mississippian limestone. "
                       "Ultimate validation target.",
        "expected_depth_m": 120,
        "expected_void_type": "natural_cave",
    },
    {
        "name": "Lechuguilla Cave (NM)",
        "lat": 32.1558,
        "lon": -104.5058,
        "buffer_deg": 0.05,
        "description": "Lechuguilla Cave, 5th longest in US (150+ miles). "
                       "Deepest limestone cave in US at 1,604 ft. Pristine "
                       "geological formations. Near Carlsbad.",
        "expected_depth_m": 490,
        "expected_void_type": "natural_cave",
    },
]


def main():
    resolution = sys.argv[1] if len(sys.argv) > 1 else "standard"

    # Allow selecting target groups
    group = sys.argv[2] if len(sys.argv) > 2 else "all"
    target_groups = {
        "california": ("CALIFORNIA DEEP SCAN", CALIFORNIA_TARGETS),
        "dumbs": ("DEEP UNDERGROUND MILITARY BASES", DUMB_TARGETS),
        "caves": ("CAVE SYSTEM VALIDATION", CAVE_TARGETS),
        "all": ("FULL USA SUBSURFACE SCAN", CALIFORNIA_TARGETS + DUMB_TARGETS + CAVE_TARGETS),
    }

    if group not in target_groups:
        print(f"Unknown group: {group}. Options: {list(target_groups.keys())}")
        sys.exit(1)

    if resolution not in RESOLUTION_PROFILES:
        print(f"Unknown resolution: {resolution}. Options: {list(RESOLUTION_PROFILES.keys())}")
        sys.exit(1)

    group_name, targets = target_groups[group]

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

    if auth_info["mode"] not in {"token", "credentials"}:
        print(
            "ERROR: NASA Earthdata auth not configured. Set EARTHDATA_TOKEN "
            "or EARTHDATA_BEARER_TOKEN, or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )
        sys.exit(1)
    logger.info("Earthdata auth configured: %s", slc_data_fetcher.describe_earthdata_auth(auth_info))

    results = []
    total_start = datetime.now()

    print("=" * 70)
    print(f"  {group_name}")
    print(f"  {len(targets)} targets @ {resolution} resolution")
    print("  Real Sentinel-1 SLC data - NO synthetic fallback")
    print("=" * 70)

    for i, target in enumerate(targets, 1):
        target_start = datetime.now()
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(targets)}] {target['name']}")
        print(f"  {target['description'][:80]}...")
        print(f"  Lat: {target['lat']}, Lon: {target['lon']}")
        print(f"  Expected depth: {target['expected_depth_m']}m, Type: {target['expected_void_type']}")
        print(f"{'='*70}")

        try:
            result = execute_biondi_pipeline_for_target(
                target=target,
                credentials=credentials,
                use_synthetic_fallback=False,
                resolution=resolution,
            )
            elapsed = (datetime.now() - target_start).total_seconds()
            result["elapsed_seconds"] = elapsed
            results.append(result)

            anomalies = _get_anomaly_count(result)
            result["anomaly_count"] = anomalies
            result["anomalies_detected"] = anomalies
            source = result.get("data_source", "unknown")
            top_targets = result.get("top_deep_targets", [])

            status_icon = "ANOMALY" if anomalies > 0 else "CLEAN"
            logger.info(f"\n  >> [{status_icon}] {target['name']}: "
                        f"{anomalies} anomalies (source={source}, {elapsed:.0f}s)")
            if top_targets:
                for t in top_targets[:3]:
                    logger.info(f"     {_format_top_target_summary(t)}")

        except Exception as e:
            elapsed = (datetime.now() - target_start).total_seconds()
            logger.error(f"  >> FAILED: {target['name']}: {e}")
            results.append({
                "target": target["name"],
                "status": "error",
                "error": str(e),
                "anomaly_count": 0,
                "anomalies_detected": 0,
                "elapsed_seconds": elapsed,
            })

        # Save intermediate results after each target
        _save_report(results, group, resolution, total_start)

    total_elapsed = (datetime.now() - total_start).total_seconds()

    # Final report
    print("\n" + "=" * 70)
    print(f"  {group_name} - FINAL RESULTS")
    print("=" * 70)

    anomaly_targets = []
    clean_targets = []
    failed_targets = []

    for r in results:
        name = r.get("target", r.get("target_name", "?"))
        status = r.get("status", "?")
        anomalies = _get_anomaly_count(r)
        source = r.get("data_source", "?")
        elapsed = r.get("elapsed_seconds", 0)

        if status == "error":
            failed_targets.append(name)
            print(f"  FAIL  {name}: {r.get('error', '?')[:60]}")
        elif anomalies > 0:
            anomaly_targets.append(name)
            print(f"  ** ANOMALY **  {name}: {anomalies} voids (source: {source}, {elapsed:.0f}s)")
        else:
            clean_targets.append(name)
            print(f"  CLEAN  {name}: no anomalies (source: {source}, {elapsed:.0f}s)")

    print("\n  SUMMARY:")
    print(f"    Targets scanned:   {len(results)}")
    print(f"    Anomalies found:   {len(anomaly_targets)}")
    print(f"    Clean:             {len(clean_targets)}")
    print(f"    Failed:            {len(failed_targets)}")
    print(f"    Total time:        {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)

    _save_report(results, group, resolution, total_start)


def _save_report(results, group, resolution, start_time):
    """Save JSON report (called after each target for crash resilience)."""
    total_elapsed = (datetime.now() - start_time).total_seconds()
    report_path = EXPLORE_DIR / f"scan_{group}_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scan_type": group,
        "resolution": resolution,
        "started": start_time.isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_seconds": total_elapsed,
        "targets_completed": len(results),
        "anomalies_found": sum(1 for r in results if _get_anomaly_count(r) > 0),
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        dump_strict_json(report, f, indent=2)
        f.write("\n")
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
