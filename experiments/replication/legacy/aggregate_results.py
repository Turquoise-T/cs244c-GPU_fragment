#!/usr/bin/env python3
"""Aggregate results from cluster experiments into a summary CSV."""

import os
import re
import csv
import json
from collections import defaultdict

def parse_log_file(log_path):
    """Extract metrics from a simulation log file."""
    metrics = {
        "avg_jct": None,
        "avg_jct_low_priority": None,
        "total_duration": None,
        "utilization": None,
    }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        for line in lines[-50:]:  # Check last 50 lines
            # Average JCT
            m = re.match(r"Average job completion time: ([\d.]+) seconds", line)
            if m:
                metrics["avg_jct"] = float(m.group(1))

            # Average JCT (low priority)
            m = re.match(r"Average job completion time \(low priority\): ([\d.]+) seconds", line)
            if m:
                metrics["avg_jct_low_priority"] = float(m.group(1))

            # Total duration
            m = re.match(r"Total duration: ([\d.]+) seconds", line)
            if m:
                metrics["total_duration"] = float(m.group(1))

            # Utilization
            m = re.match(r"Cluster utilization: ([\d.]+)", line)
            if m:
                metrics["utilization"] = float(m.group(1))

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return metrics

def aggregate_results(results_dir, experiments_file):
    """Aggregate all results into a summary."""
    # Load experiments
    with open(experiments_file, "r") as f:
        experiments = json.load(f)

    results = []

    for i, exp in enumerate(experiments):
        figure = exp["figure"]
        policy = exp["policy"]
        seed = exp["seed"]
        lam = exp["lambda"]
        cluster_spec = exp["cluster_spec"]

        # Construct expected log path
        v100s, p100s, k80s = cluster_spec.split(":")
        log_path = os.path.join(
            results_dir,
            figure,
            f"v100={v100s}.p100={p100s}.k80={k80s}",
            policy,
            f"seed={seed}",
            f"lambda={lam:.6f}.log"
        )

        if os.path.exists(log_path):
            metrics = parse_log_file(log_path)
            result = {
                "experiment_id": i,
                "figure": figure,
                "policy": policy,
                "seed": seed,
                "lambda": lam,
                "input_job_rate": 3600.0 / lam,
                **metrics
            }
            results.append(result)
        else:
            print(f"Missing: {log_path}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--experiments-file", default="experiments.json", help="Experiments JSON")
    parser.add_argument("--output", default="summary.csv", help="Output CSV file")
    args = parser.parse_args()

    print(f"Aggregating results from {args.results_dir}...")
    results = aggregate_results(args.results_dir, args.experiments_file)

    if not results:
        print("No results found!")
        return

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} results to {args.output}")

    # Print summary by figure
    by_figure = defaultdict(list)
    for r in results:
        by_figure[r["figure"]].append(r)

    print("\nSummary by figure:")
    for figure, fig_results in sorted(by_figure.items()):
        policies = set(r["policy"] for r in fig_results)
        print(f"  {figure}: {len(fig_results)} results, policies: {', '.join(sorted(policies))}")

if __name__ == "__main__":
    main()
