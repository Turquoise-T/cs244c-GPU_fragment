#!/usr/bin/env python3
"""
Run one (or all) FGD vs Strided experiments and append results to CSV.

Usage:
  # Run a single experiment by index
  python run_fgd_experiment.py --index 0

  # Run all experiments sequentially
  python run_fgd_experiment.py --all

  # Run a range of experiments
  python run_fgd_experiment.py --range 0 10
"""

import argparse
import contextlib
import json
import os
import sys
import time

# Add scheduler to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULER_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..",
                                              "src", "scheduler"))
sys.path.insert(0, SCHEDULER_DIR)

import scheduler
import utils


def run_one_experiment(exp, output_csv, verbose=True):
    """Run a single experiment and append result to CSV."""
    name = exp["name"]

    # Parse cluster spec
    v, p, k = map(int, exp["cluster_spec"].split(":"))
    cluster_spec = {"v100": v, "p100": p, "k80": k}

    gv, gp, gk = map(int, exp["num_gpus_per_server"].split(":"))
    num_gpus_per_server = {"v100": gv, "p100": gp, "k80": gk}

    throughputs_file = os.path.join(SCHEDULER_DIR, "simulation_throughputs.json")

    if verbose:
        print(f"  [{name}] strategy={exp['placement_strategy']}, "
              f"rate={exp['jobs_per_hr']} jobs/hr, seed={exp['seed']} ... ",
              end="", flush=True)

    start = time.time()

    # Redirect scheduler logs to /dev/null
    log_path = os.path.join(os.path.dirname(output_csv), f"{name}.log")
    with open(log_path, "w") as log_f:
        with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            policy = utils.get_policy(exp["policy"], seed=exp["seed"],
                                      solver="ECOS")
            sched = scheduler.Scheduler(
                policy,
                throughputs_file=throughputs_file,
                seed=exp["seed"],
                time_per_iteration=360,
                simulate=True,
                placement_strategy=exp["placement_strategy"],
                gpu_sharing_mode=exp["gpu_sharing"],
            )

            sim_kw = dict(
                lam=exp["lambda"],
                num_total_jobs=exp["num_total_jobs"],
                num_gpus_per_server=num_gpus_per_server,
            )
            if exp["gpu_sharing"]:
                sim_kw["gpu_milli_generator_func"] = \
                    utils._generate_gpu_milli_sharing

            sched.simulate(cluster_spec, **sim_kw)

            avg_jct = sched.get_average_jct(verbose=False)
            makespan = sched.get_current_timestamp()
            utilization = sched.get_cluster_utilization()

    elapsed = time.time() - start

    # Handle None values
    if avg_jct is None:
        avg_jct = float("inf")
    if utilization is None:
        utilization = -1.0

    sched.shutdown()

    if verbose:
        print(f"done ({elapsed:.1f}s)  jct={avg_jct:.0f}s  "
              f"makespan={makespan:.0f}s  util={utilization:.2%}")

    # Append to CSV
    csv_exists = os.path.exists(output_csv)
    with open(output_csv, "a") as f:
        if not csv_exists:
            f.write("name,placement_strategy,policy,jobs_per_hr,seed,"
                    "jct_sec,makespan_sec,utilization,runtime_sec\n")
        f.write(f"{name},{exp['placement_strategy']},{exp['policy']},"
                f"{exp['jobs_per_hr']},{exp['seed']},"
                f"{avg_jct:.2f},{makespan:.2f},"
                f"{utilization:.4f},{elapsed:.2f}\n")

    # Remove log file to save space (keep CSV only)
    if os.path.exists(log_path):
        os.remove(log_path)

    return avg_jct, makespan, utilization


def main():
    parser = argparse.ArgumentParser(
        description="Run FGD vs Strided experiments")
    parser.add_argument("--index", type=int, default=None,
                        help="Run single experiment by index")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Run experiments in range [start, end)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "configs",
                                             "experiments_fgd.json"),
                        help="Experiment config JSON file")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "results",
                                             "results_fgd.csv"),
                        help="Output CSV file")
    args = parser.parse_args()

    # Load experiments
    with open(args.config) as f:
        experiments = json.load(f)

    print(f"Loaded {len(experiments)} experiments from {args.config}")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Determine which experiments to run
    if args.index is not None:
        indices = [args.index]
    elif args.range is not None:
        indices = list(range(args.range[0], args.range[1]))
    elif args.all:
        indices = list(range(len(experiments)))
    else:
        parser.error("Specify --index N, --all, or --range START END")

    print(f"Running {len(indices)} experiment(s)...")
    print(f"Output: {args.output}")
    print("=" * 70)

    total_start = time.time()
    for i, idx in enumerate(indices):
        if idx >= len(experiments):
            print(f"  Skipping index {idx} (out of range)")
            continue
        exp = experiments[idx]
        print(f"[{i+1}/{len(indices)}]", end="")
        run_one_experiment(exp, args.output)

    total_elapsed = time.time() - total_start
    print("=" * 70)
    print(f"All done. Total time: {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"Results in: {args.output}")


if __name__ == "__main__":
    main()
