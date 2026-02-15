#!/usr/bin/env python3
"""Run a single experiment from the experiments.json file."""

import argparse
import json
import os
import sys
import contextlib

# Default JCT threshold: 100 hours in seconds
# Experiments exceeding this are "saturated" and exit early
DEFAULT_MAX_JCT_HOURS = 100
DEFAULT_MAX_JCT_SECONDS = DEFAULT_MAX_JCT_HOURS * 3600

def main():
    parser = argparse.ArgumentParser(description="Run a single experiment by index")
    parser.add_argument("--index", type=int, required=True, help="Experiment index (0-based)")
    parser.add_argument("--experiments-file", default="experiments.json", help="Path to experiments.json")
    parser.add_argument("--output-dir", default="results", help="Output directory for logs")
    parser.add_argument("--scheduler-dir", default="../src/scheduler", help="Path to scheduler directory")
    parser.add_argument("--max-jct", type=float, default=DEFAULT_MAX_JCT_SECONDS,
                        help=f"Max JCT in seconds before early exit (default: {DEFAULT_MAX_JCT_SECONDS} = {DEFAULT_MAX_JCT_HOURS} hours)")
    parser.add_argument("--no-max-jct", action="store_true", help="Disable JCT threshold (run until completion)")
    args = parser.parse_args()

    # Convert scheduler-dir to absolute path and add to Python path
    args.scheduler_dir = os.path.abspath(args.scheduler_dir)
    sys.path.insert(0, args.scheduler_dir)

    # Now import scheduler modules
    import scheduler
    import utils
    from job_id_pair import JobIdPair

    # Load experiments
    with open(args.experiments_file, "r") as f:
        experiments = json.load(f)

    if args.index < 0 or args.index >= len(experiments):
        print(f"Error: index {args.index} out of range (0-{len(experiments)-1})")
        sys.exit(1)

    exp = experiments[args.index]
    print(f"Running experiment {args.index}: {exp}")

    # Parse experiment config
    figure = exp["figure"]
    policy_name = exp["policy"]
    seed = exp["seed"]
    lam = exp["lambda"]
    cluster_spec_str = exp["cluster_spec"]
    window_start = exp["window_start"]
    window_end = exp["window_end"]
    generate_multi_gpu_jobs = exp["multi_gpu"]

    # Parse cluster spec
    v100s, p100s, k80s = map(int, cluster_spec_str.split(":"))
    cluster_spec = {"v100": v100s, "p100": p100s, "k80": k80s}

    # Create output directory
    output_subdir = os.path.join(
        args.output_dir,
        figure,
        f"v100={v100s}.p100={p100s}.k80={k80s}",
        policy_name,
        f"seed={seed}"
    )
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"lambda={lam:.6f}.log")

    print(f"Output: {output_file}")

    # Get policy
    policy = utils.get_policy(policy_name, solver="ECOS", seed=seed)

    # Throughputs file
    throughputs_file = os.path.join(args.scheduler_dir, "simulation_throughputs.json")

    # Jobs to complete (window_end - window_start gives measurement window)
    jobs_to_complete = set()
    for i in range(window_start, window_end):
        jobs_to_complete.add(JobIdPair(i, None))

    # Run simulation
    # Paper uses 360 second (6 minute) scheduling rounds
    TIME_PER_ITERATION = 360

    # Determine JCT threshold
    max_jct = None if args.no_max_jct else args.max_jct

    with open(output_file, 'w') as f:
        with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
            sched = scheduler.Scheduler(
                policy,
                throughputs_file=throughputs_file,
                seed=seed,
                time_per_iteration=TIME_PER_ITERATION,  # 6-minute rounds per paper Section 5
                simulate=True,
                profiling_percentage=1.0,
                num_reference_models=26
            )

            sched.simulate(
                cluster_spec,
                lam=lam,  # Inter-arrival time in seconds
                jobs_to_complete=jobs_to_complete,
                fixed_job_duration=None,
                generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                generate_multi_priority_jobs=False,
                simulate_steady_state=True,  # Pre-fill cluster for steady-state (paper Section 7)
                checkpoint_file=None,
                checkpoint_threshold=None,
                num_gpus_per_server=None,
                ideal=False,
                max_jct=max_jct
            )

            jct_exceeded = sched.jct_threshold_exceeded()
            if jct_exceeded:
                # Get partial JCT for reporting
                average_jct = float('inf')
                utilization = sched.get_cluster_utilization()
            else:
                average_jct = sched.get_average_jct(jobs_to_complete)
                utilization = sched.get_cluster_utilization()

    if jct_exceeded:
        print(f"Experiment STOPPED EARLY: JCT exceeded {max_jct/3600:.0f} hour threshold")
        print(f"Average JCT: inf (system saturated, partial JCT > {max_jct/3600:.0f} hours)")
        print(f"Utilization: {utilization:.2%}")
    else:
        print(f"Experiment completed successfully")
        print(f"Average JCT: {average_jct:.2f} seconds ({average_jct/3600:.2f} hours)")
        print(f"Utilization: {utilization:.2%}")

if __name__ == "__main__":
    main()
