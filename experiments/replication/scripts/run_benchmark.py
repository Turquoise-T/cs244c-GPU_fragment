#!/usr/bin/env python3
"""
Benchmark runner with detailed timing instrumentation.

Captures:
- Total simulation time
- Time per scheduling round
- cvxpy solve time breakdown
- Number of active jobs per round
- Allocation computation time
"""

import argparse
import json
import os
import sys
import time
import contextlib
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run benchmark experiment with detailed timing")
    parser.add_argument("--index", type=int, required=True, help="Experiment index (0=agnostic, 1=gavel)")
    parser.add_argument("--experiments-file", default="experiments_benchmark.json")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--scheduler-dir", default="../src/scheduler")
    args = parser.parse_args()

    # Setup paths
    args.scheduler_dir = os.path.abspath(args.scheduler_dir)
    sys.path.insert(0, args.scheduler_dir)

    # Import scheduler modules
    import scheduler
    import utils
    from job_id_pair import JobIdPair

    # Load experiment config
    with open(args.experiments_file, "r") as f:
        experiments = json.load(f)

    if args.index < 0 or args.index >= len(experiments):
        print(f"Error: index {args.index} out of range")
        sys.exit(1)

    exp = experiments[args.index]

    # Generate name if not present
    if 'name' not in exp:
        multi_str = 'multi' if exp.get('multi_gpu', False) else 'single'
        exp['name'] = f"{exp['figure']}_{exp['policy']}_{exp['jobs_per_hr']}jph_{multi_str}_s{exp['seed']}"

    print("=" * 70)
    print(f"BENCHMARK EXPERIMENT: {exp['name']}")
    print("=" * 70)
    print(f"Policy: {exp['policy']}")
    print(f"Load: {exp['jobs_per_hr']} jobs/hr (lambda={exp['lambda']}s)")
    print(f"Measurement window: jobs {exp['window_start']}-{exp['window_end']}")
    print(f"Multi-GPU jobs: {exp['multi_gpu']}")
    print("=" * 70)

    # Parse config
    v100s, p100s, k80s = map(int, exp["cluster_spec"].split(":"))
    cluster_spec = {"v100": v100s, "p100": p100s, "k80": k80s}

    # Create output directory
    output_dir = os.path.join(args.output_dir, exp["name"])
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "simulation.log")
    timing_file = os.path.join(output_dir, "timing.json")
    summary_file = os.path.join(output_dir, "summary.txt")

    # Get policy
    policy = utils.get_policy(exp["policy"], solver="ECOS", seed=exp["seed"])

    # Throughputs file
    throughputs_file = os.path.join(args.scheduler_dir, "simulation_throughputs.json")

    # Jobs to complete
    jobs_to_complete = set()
    for i in range(exp["window_start"], exp["window_end"]):
        jobs_to_complete.add(JobIdPair(i, None))

    TIME_PER_ITERATION = 360  # 6-minute rounds

    # Timing data collection
    timing_data = {
        "experiment": exp,
        "start_time": datetime.now().isoformat(),
        "rounds": [],
    }

    print(f"\nStarting simulation...")
    print(f"Log file: {log_file}")

    start_time = time.time()

    with open(log_file, 'w') as f:
        with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
            sched = scheduler.Scheduler(
                policy,
                throughputs_file=throughputs_file,
                seed=exp["seed"],
                time_per_iteration=TIME_PER_ITERATION,
                simulate=True,
                profiling_percentage=1.0,
                num_reference_models=26
            )

            sched.simulate(
                cluster_spec,
                lam=exp["lambda"],
                jobs_to_complete=jobs_to_complete,
                fixed_job_duration=None,
                generate_multi_gpu_jobs=exp["multi_gpu"],
                generate_multi_priority_jobs=False,
                simulate_steady_state=True,
                checkpoint_file=None,
                checkpoint_threshold=None,
                num_gpus_per_server=None,
                ideal=False,
                # Saturation detection parameters
                completion_rate_threshold=0.1,  # Exit if completion rate < 0.1 jobs/hr
                min_simulated_time=36000,       # Wait 10h simulated time before checking
                utilization_threshold=0.99,     # Only check when utilization > 99%
                min_runtime=300                 # Wait 5 min wall-clock time before checking
            )

            saturated = sched.saturated
            partial_jct = sched.partial_jct
            if saturated:
                average_jct = partial_jct if partial_jct else float('inf')
            else:
                # Normal completion - use full measurement window
                average_jct = sched.get_average_jct(jobs_to_complete)
            utilization = sched.get_cluster_utilization()

    end_time = time.time()
    total_time = end_time - start_time

    # Save timing data
    timing_data["end_time"] = datetime.now().isoformat()
    timing_data["total_seconds"] = total_time
    timing_data["saturated"] = saturated
    timing_data["partial_jct"] = partial_jct
    timing_data["average_jct"] = average_jct
    timing_data["utilization"] = utilization

    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)

    # Print and save summary
    partial_jct_str = f"{partial_jct:.2f}" if partial_jct else "N/A"
    summary = f"""
{'=' * 70}
BENCHMARK RESULTS: {exp['name']}
{'=' * 70}

Configuration:
  Policy: {exp['policy']}
  Load: {exp['jobs_per_hr']} jobs/hr
  Cluster: {exp['cluster_spec']} (V100:P100:K80)
  Measurement window: jobs {exp['window_start']}-{exp['window_end']}

Results:
  Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)
  Saturated (early exit): {saturated}
  Partial JCT at exit: {partial_jct_str} seconds
  Average JCT: {average_jct:.2f} seconds ({average_jct/3600:.2f} hours)
  Utilization: {utilization:.2%}

{'=' * 70}
"""
    print(summary)

    with open(summary_file, 'w') as f:
        f.write(summary)

    # Also create a simple CSV line for easy comparison
    csv_file = os.path.join(args.output_dir, "results.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a') as f:
        if not csv_exists:
            f.write("name,policy,jobs_per_hr,runtime_sec,jct_sec,utilization,saturated\n")
        f.write(f"{exp['name']},{exp['policy']},{exp['jobs_per_hr']},{total_time:.2f},{average_jct:.2f},{utilization:.4f},{saturated}\n")

    print(f"Results saved to: {output_dir}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
