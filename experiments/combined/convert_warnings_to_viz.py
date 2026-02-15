"""Convert scheduler WARNING logs to .viz.bin files for the gpu-scheduling-viz tool.

When experiments run with -q (WARNING level), TELEMETRY lines are suppressed.
This script extracts sim_time, active_jobs, and per-GPU-type usage from the
WARNING messages about unused GPUs and produces .viz.bin files suitable for
timeseries visualization (heatmaps will be empty since we lack per-GPU data).

Usage:
    python convert_warnings_to_viz.py <stderr_log> <output.viz.bin> [--label LABEL]
"""
import os
import re
import sys
import json
import argparse
from collections import defaultdict

# Add cs244c/ to path so `from viz.tools...` works via the viz symlink
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from viz.tools.binary_format import write_viz_file

ALIBABA_CLUSTER = {
    "G2": 4392, "T4": 840, "G3": 312,
    "P100": 264, "V100M32": 200, "V100M16": 192
}
GPU_TYPE_ORDER = ["G2", "T4", "G3", "P100", "V100M32", "V100M16"]
TOTAL_GPUS = sum(ALIBABA_CLUSTER.values())


def parse_warnings(log_path):
    """Parse WARNING lines into per-round data points."""
    pattern = re.compile(
        r'scheduler:WARNING \[([0-9.]+)\] (\d+) GPUs of type (\w+) left unused\. '
        r'Number of active jobs: (\d+)'
    )
    rounds_raw = defaultdict(lambda: {"active_jobs": 0, "unused": {}})

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            sim_time = float(m.group(1))
            unused = int(m.group(2))
            gpu_type = m.group(3)
            active_jobs = int(m.group(4))
            rounds_raw[sim_time]["active_jobs"] = active_jobs
            rounds_raw[sim_time]["unused"][gpu_type] = unused

    return rounds_raw


def convert_to_viz(log_path, output_path, policy="max_min_fairness_perf",
                   gpus_per_node=8):
    """Convert WARNING log to .viz.bin file."""
    rounds_raw = parse_warnings(log_path)
    if not rounds_raw:
        print(f"No WARNING data found in {log_path}")
        return 0

    sorted_times = sorted(rounds_raw.keys())
    print(f"  Parsed {len(sorted_times)} rounds, "
          f"sim time {sorted_times[0]:.0f} - {sorted_times[-1]:.0f}s "
          f"({sorted_times[-1]/3600:.1f} hr)")

    gpu_types = [
        {"name": name, "count": ALIBABA_CLUSTER[name], "gpus_per_node": gpus_per_node}
        for name in GPU_TYPE_ORDER
    ]
    config = {
        "policy": policy,
        "gpu_types": gpu_types,
        "measurement_window": {"start_job": 4000, "end_job": 5000},
        "job_types": [],
    }

    rounds = []
    queues = []
    for i, sim_time in enumerate(sorted_times):
        r = rounds_raw[sim_time]
        # Compute per-type GPU usage
        gpu_used = []
        for name in GPU_TYPE_ORDER:
            total = ALIBABA_CLUSTER[name]
            unused = r["unused"].get(name, 0)
            gpu_used.append(total - unused)

        total_used = sum(gpu_used)
        utilization = total_used / TOTAL_GPUS

        rounds.append({
            "round": i,
            "sim_time": sim_time,
            "utilization": utilization,
            "jobs_running": r["active_jobs"],
            "jobs_queued": 0,
            "jobs_completed": 0,
            "avg_jct": 0,
            "completion_rate": 0,
            "gpu_used": gpu_used,
            "allocations": [0] * TOTAL_GPUS,  # no per-GPU data available
        })
        queues.append([])

    # No job-level data from WARNING logs
    jobs = []

    write_viz_file(output_path, config, jobs, rounds, queues)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Wrote {output_path} ({size_mb:.1f} MB, {len(rounds)} rounds)")
    return len(rounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to stderr log file")
    parser.add_argument("output_path", help="Path for output .viz.bin file")
    parser.add_argument("--policy", default="max_min_fairness_perf")
    args = parser.parse_args()

    convert_to_viz(args.log_path, args.output_path, policy=args.policy)


if __name__ == "__main__":
    main()
