#!/usr/bin/env python3
"""Generate pilot experiment configurations for Figure 9, 10, 11 replication.

Pilot: 3 data points per figure (low/mid/high), 1 seed, 2 policies each.
Total: 18 experiments for timing and validation.
"""

import json
import os
import argparse

def jobs_per_hr_to_lambda(jobs_per_hr):
    """Convert jobs/hr to inter-arrival time in seconds."""
    return 3600.0 / jobs_per_hr

# Figure 9: Single-GPU jobs, LAS without vs with Gavel's heterogeneity awareness
# Paper range: 0-8 jobs/hr
# Pilot points: 1.0, 4.0, 7.0 jobs/hr
FIG9_PILOT = {
    "name": "fig9",
    "description": "JCT vs Load - Single-GPU jobs (LAS: heterogeneity-agnostic vs Gavel)",
    "policies": ["max_min_fairness", "max_min_fairness_perf"],
    "seeds": [0],
    "cluster_spec": "36:36:36",
    "window_start": 4000,
    "window_end": 5000,
    "jobs_per_hr": [1.0, 4.0, 7.0],  # Low, mid, high
    "multi_gpu": False,
}

# Figure 10: Multi-GPU jobs, LAS without vs with Gavel's heterogeneity awareness
# Paper range: 0-3 jobs/hr
# Pilot points: 0.5, 1.5, 2.5 jobs/hr
FIG10_PILOT = {
    "name": "fig10",
    "description": "JCT vs Load - Multi-GPU jobs (LAS: heterogeneity-agnostic vs Gavel)",
    "policies": ["max_min_fairness", "max_min_fairness_perf"],
    "seeds": [0],
    "cluster_spec": "36:36:36",
    "window_start": 4000,
    "window_end": 5000,
    "jobs_per_hr": [0.5, 1.5, 2.5],  # Low, mid, high
    "multi_gpu": True,
}

# Figure 11: Multi-GPU jobs, FTF without vs with Gavel's heterogeneity awareness
# Paper range: 0-3.5 jobs/hr
# Pilot points: 0.5, 1.75, 3.0 jobs/hr
FIG11_PILOT = {
    "name": "fig11",
    "description": "JCT vs Load - Multi-GPU jobs (FTF: heterogeneity-agnostic vs Gavel)",
    "policies": ["finish_time_fairness", "finish_time_fairness_perf"],
    "seeds": [0],
    "cluster_spec": "36:36:36",
    "window_start": 4000,
    "window_end": 5000,
    "jobs_per_hr": [0.5, 1.75, 3.0],  # Low, mid, high
    "multi_gpu": True,
}

# Full sweep configurations (for later)
# Note: policies inherited from PILOT configs (heterogeneity-agnostic vs Gavel)
FIG9_FULL = {
    **FIG9_PILOT,
    "seeds": [0, 1, 2],
    # 5 points per 2 units over 0-8 range = 20 points, spacing 0.4
    "jobs_per_hr": [0.4 + 0.4*i for i in range(20)],  # 0.4 to 8.0
}

FIG10_FULL = {
    **FIG10_PILOT,
    "seeds": [0, 1, 2],
    # 5 points per 1 unit over 0-3 range = 15 points, spacing 0.2
    "jobs_per_hr": [0.2 + 0.2*i for i in range(15)],  # 0.2 to 3.0
}

FIG11_FULL = {
    **FIG11_PILOT,
    "seeds": [0, 1, 2],
    # 5 points per 1 unit over 0-3.5 range = ~17 points, spacing 0.2
    "jobs_per_hr": [0.2 + 0.2*i for i in range(17)],  # 0.2 to 3.4
}


def generate_experiments(config):
    """Generate experiment configurations from a config dict."""
    experiments = []

    for policy in config["policies"]:
        for seed in config["seeds"]:
            for jobs_hr in config["jobs_per_hr"]:
                lam = jobs_per_hr_to_lambda(jobs_hr)
                exp = {
                    "figure": config["name"],
                    "policy": policy,
                    "seed": seed,
                    "lambda": lam,
                    "jobs_per_hr": jobs_hr,
                    "cluster_spec": config["cluster_spec"],
                    "window_start": config["window_start"],
                    "window_end": config["window_end"],
                    "multi_gpu": config["multi_gpu"],
                }
                experiments.append(exp)

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configurations")
    parser.add_argument("--pilot", action="store_true", help="Generate pilot experiments only")
    parser.add_argument("--full", action="store_true", help="Generate full sweep experiments")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    args = parser.parse_args()

    if args.pilot:
        configs = [FIG9_PILOT, FIG10_PILOT, FIG11_PILOT]
        default_output = "experiments_pilot.json"
    elif args.full:
        configs = [FIG9_FULL, FIG10_FULL, FIG11_FULL]
        default_output = "experiments_full.json"
    else:
        # Default to pilot
        configs = [FIG9_PILOT, FIG10_PILOT, FIG11_PILOT]
        default_output = "experiments_pilot.json"

    all_experiments = []
    for config in configs:
        exps = generate_experiments(config)
        all_experiments.extend(exps)
        print(f"  {config['name']}: {len(exps)} experiments")

    output_file = args.output or os.path.join(os.path.dirname(__file__), default_output)
    with open(output_file, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nGenerated {len(all_experiments)} total experiments")
    print(f"Saved to: {output_file}")

    # Print summary
    print("\nExperiment breakdown:")
    for config in configs:
        n_policies = len(config["policies"])
        n_seeds = len(config["seeds"])
        n_rates = len(config["jobs_per_hr"])
        print(f"  {config['name']}: {n_policies} policies x {n_seeds} seeds x {n_rates} rates = {n_policies * n_seeds * n_rates}")


if __name__ == "__main__":
    main()
