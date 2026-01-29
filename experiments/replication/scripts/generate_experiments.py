#!/usr/bin/env python3
"""Generate experiment configurations for cluster execution."""

import json
import os
import itertools

# Figure 9: Multi-GPU jobs (primary validation figure)
FIG9_CONFIG = {
    "name": "fig9",
    "description": "JCT vs Load with multi-GPU jobs",
    "policies": ["gandiva", "max_min_fairness", "max_min_fairness_perf"],
    "seeds": [0, 1, 2],
    "cluster_spec": "36:36:36",
    "window_start": 4000,
    "window_end": 5000,
    "lambda_values": [1200, 1800, 2400, 3000, 3600, 4800, 6000, 7200, 9000, 12000],
    "multi_gpu": True,
}

# Figure 8: Single-GPU jobs
FIG8_CONFIG = {
    "name": "fig8",
    "description": "JCT vs Load with single-GPU jobs",
    "policies": ["allox", "gandiva", "max_min_fairness", "max_min_fairness_perf"],
    "seeds": [0, 1, 2],
    "cluster_spec": "36:36:36",
    "window_start": 4000,
    "window_end": 5000,
    "lambda_values": [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3600, 4200, 4800, 5400, 6000, 7200],
    "multi_gpu": False,
}

def generate_experiments(config):
    """Generate all experiment configurations for a figure."""
    experiments = []

    for policy, seed, lam in itertools.product(
        config["policies"],
        config["seeds"],
        config["lambda_values"]
    ):
        exp = {
            "figure": config["name"],
            "policy": policy,
            "seed": seed,
            "lambda": lam,
            "cluster_spec": config["cluster_spec"],
            "window_start": config["window_start"],
            "window_end": config["window_end"],
            "multi_gpu": config["multi_gpu"],
        }
        experiments.append(exp)

    return experiments

def main():
    # Generate experiments for both figures
    fig9_experiments = generate_experiments(FIG9_CONFIG)
    fig8_experiments = generate_experiments(FIG8_CONFIG)

    all_experiments = fig9_experiments + fig8_experiments

    # Write experiments to JSON file
    output_file = os.path.join(os.path.dirname(__file__), "experiments.json")
    with open(output_file, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"Generated {len(all_experiments)} experiments:")
    print(f"  - Figure 9: {len(fig9_experiments)} experiments")
    print(f"  - Figure 8: {len(fig8_experiments)} experiments")
    print(f"Saved to: {output_file}")

    # Also generate a summary
    print(f"\nFigure 9: {len(FIG9_CONFIG['policies'])} policies x {len(FIG9_CONFIG['seeds'])} seeds x {len(FIG9_CONFIG['lambda_values'])} lambdas")
    print(f"Figure 8: {len(FIG8_CONFIG['policies'])} policies x {len(FIG8_CONFIG['seeds'])} seeds x {len(FIG8_CONFIG['lambda_values'])} lambdas")

if __name__ == "__main__":
    main()
