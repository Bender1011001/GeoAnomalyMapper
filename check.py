import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path

from parse_results import as_result_list, get_anomaly_count, safe_float, safe_int


def load_results(path):
    """Load a JSON results file with a context manager."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact GeoAnomalyMapper result checker.")
    parser.add_argument("path", nargs="?", default="temp_results_fixed.json", help="Results JSON file to check.")
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
    print(f"Processed: {safe_int(data.get('targets_completed', len(results)), len(results))}")
    for result in results:
        if not isinstance(result, dict):
            print("- ?: SKIPPED malformed result entry")
            continue
        name = result.get("target_name") or result.get("target") or "?"
        count = get_anomaly_count(result)
        if count > 0:
            top_targets = as_result_list(result.get("top_deep_targets", []))
            vols = sum(safe_float(a.get("volume_m3", 0)) for a in top_targets if isinstance(a, dict))
            depths = [safe_int(a.get("depth_m", 0)) for a in top_targets if isinstance(a, dict)]
            print(f"- {name}: {count} anomalies | Depths: {depths} | Vol: {vols:,.0f} m3")
        else:
            print(f"- {name}: CLEAN")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
