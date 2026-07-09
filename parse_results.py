import argparse
import json
import math
import sys
from json import JSONDecodeError
from pathlib import Path


ANOMALY_COUNT_KEYS = ("anomaly_count", "anomalies_detected")


def safe_float(value, default=0.0):
    """Convert a possibly missing or nonnumeric value to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def safe_int(value, default=0):
    """Convert a possibly missing or nonnumeric value to a non-negative int."""
    number = safe_float(value, None)
    if number is None:
        return default
    return max(int(number), 0)


def get_anomaly_count(result):
    """Read either anomaly_count or anomalies_detected from a result object."""
    if not isinstance(result, dict):
        return 0
    for key in ANOMALY_COUNT_KEYS:
        if key in result:
            return safe_int(result.get(key), 0)
    return 0


def as_result_list(value):
    """Return a list if value is a list, otherwise an empty list."""
    return value if isinstance(value, list) else []


def load_results(path):
    """Load a JSON results file with a context manager."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize GeoAnomalyMapper JSON results.")
    parser.add_argument("path", nargs="?", default="temp_results.json", help="Results JSON file to parse.")
    args = parser.parse_args(argv)

    path = Path(args.path)
    try:
        data = load_results(path)
    except FileNotFoundError:
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        return 1
    except JSONDecodeError as exc:
        print(f"ERROR: Could not parse JSON in {path}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: Could not read {path}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print(f"ERROR: Expected top-level JSON object in {path}", file=sys.stderr)
        return 1

    results = as_result_list(data.get("results", data.get("targets", [])))
    completed = safe_int(data.get("targets_completed", len(results)), len(results))
    print(f"Total targets scanned so far: {completed}")

    for result in results:
        if not isinstance(result, dict):
            print("- Unknown: SKIPPED malformed result entry")
            continue

        name = result.get("target_name") or result.get("target") or "Unknown"
        status = result.get("status", "")
        if status != "success":
            print(f"- {name}: FAILED or INCOMPLETE")
            continue

        anomalies = get_anomaly_count(result)
        print(f"\n=== {name} | Anomalies: {anomalies} ===")

        found_interesting = False
        for target in as_result_list(result.get("top_deep_targets", [])):
            if not isinstance(target, dict):
                continue
            conf = safe_float(target.get("fused_confidence_score", target.get("deep_target_score", 0)))
            vol = safe_float(target.get("volume_m3", 0))
            artificial = safe_float(target.get("artificiality_score", 0))

            # Print if high confidence, massive volume, or high artificiality.
            if conf > 0.6 or vol > 100000 or artificial > 0.5:
                found_interesting = True
                rank = target.get("deep_target_rank", target.get("rank", "?"))
                shape = target.get("shape_classification", target.get("shape", "?"))
                note = target.get("shape_note") or "no shape note"
                depth = safe_float(target.get("depth_m", 0))
                print(f"  Target Rank {rank}: {shape} ({note})")
                print(f"    Depth: {depth:.0f}m, Volume: {vol:,.0f} m3")
                print(f"    Confidence: {conf:.3f}, Artificiality: {artificial:.2f}")

        if not found_interesting and anomalies > 0:
            print("  (Anomalies found but did not meet the interesting threshold - likely natural/small)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
