#!/usr/bin/env python3
"""Merge per-experiment JSON result files into a single CSV for plotting.

Reads exp_*.json files produced by SLURM array jobs and combines them into
a sorted CSV with one row per experiment.

Usage:
    python merge_gavel_replication_results.py
    python merge_gavel_replication_results.py --results-dir /path/to/results
    python merge_gavel_replication_results.py --output /path/to/output.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def parse_figure(name):
    """Extract figure name from the first token before underscore.

    Examples:
        "fig9_max_min_fairness_perf_2.0jph_s0" -> "fig9"
        "fig10_max_min_fairness_0.4jph_multi_s1" -> "fig10"
    """
    return name.split("_")[0]


def parse_jobs_per_hr(name):
    """Extract jobs_per_hr from the number before 'jph' in the name.

    Examples:
        "fig9_max_min_fairness_perf_2.0jph_s0" -> 2.0
        "fig10_max_min_fairness_0.4jph_multi_s1" -> 0.4
    """
    match = re.search(r"(\d+(?:\.\d+)?)jph", name)
    if match:
        return float(match.group(1))
    return None


def load_result(path):
    """Load a single experiment result JSON file.

    Each file is a JSON array with one result dict.
    Returns the result dict or None on error.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to read {path.name}: {e}", file=sys.stderr)
        return None

    if isinstance(data, list):
        if len(data) == 0:
            print(f"  WARNING: Empty array in {path.name}", file=sys.stderr)
            return None
        return data[0]
    elif isinstance(data, dict):
        return data
    else:
        print(f"  WARNING: Unexpected format in {path.name}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-experiment JSON results into a combined CSV."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / ".." / "results" / "gavel_replication",
        help="Directory containing exp_*.json files "
             "(default: ../results/gavel_replication)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / ".." / "results" / "gavel_replication_combined.csv",
        help="Output CSV path "
             "(default: ../results/gavel_replication_combined.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_path = args.output.resolve()

    if not results_dir.is_dir():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Glob for result files
    json_files = sorted(results_dir.glob("exp_*.json"))
    if not json_files:
        print(f"ERROR: No exp_*.json files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} result files in {results_dir}")

    # Parse all results
    rows = []
    skipped = 0
    for path in json_files:
        result = load_result(path)
        if result is None:
            skipped += 1
            continue

        name = result.get("name", "")
        figure = parse_figure(name)
        jobs_per_hr = parse_jobs_per_hr(name)

        if jobs_per_hr is None:
            print(f"  WARNING: Could not parse jobs_per_hr from name '{name}' "
                  f"in {path.name}", file=sys.stderr)
            skipped += 1
            continue

        rows.append({
            "name": name,
            "figure": figure,
            "policy": result.get("policy", ""),
            "jobs_per_hr": jobs_per_hr,
            "seed": result.get("seed", 0),
            "jct_sec": result.get("avg_jct", ""),
            "saturated": result.get("saturated", False),
            "wall_time_sec": result.get("wall_time_seconds", ""),
            "num_completed": result.get("num_completed_jobs", 0),
            "multi_gpu": result.get("generate_multi_gpu_jobs", False),
        })

    if not rows:
        print("ERROR: No valid results found.", file=sys.stderr)
        sys.exit(1)

    # Sort by (figure, policy, jobs_per_hr, seed)
    rows.sort(key=lambda r: (r["figure"], r["policy"], r["jobs_per_hr"], r["seed"]))

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "figure", "policy", "jobs_per_hr", "seed",
        "jct_sec", "saturated", "wall_time_sec", "num_completed", "multi_gpu",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")
    if skipped:
        print(f"Skipped {skipped} files (warnings above)")

    # Summary by figure
    from collections import Counter
    figure_counts = Counter(r["figure"] for r in rows)
    print(f"\nBy figure:")
    for fig in sorted(figure_counts):
        print(f"  {fig}: {figure_counts[fig]} experiments")

    # Saturated count
    saturated_count = sum(1 for r in rows if r["saturated"])
    print(f"\nSaturated: {saturated_count}/{len(rows)}")


if __name__ == "__main__":
    main()
