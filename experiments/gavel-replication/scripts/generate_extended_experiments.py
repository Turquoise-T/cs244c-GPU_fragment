#!/usr/bin/env python3
"""Generate extended rate experiments for multi-GPU figures to see saturation."""

import json

def generate_experiments():
    experiments = []

    cluster_spec = "36:36:36"
    seeds = [0, 1, 2]

    # Standard measurement window
    window_start = 4000
    window_end = 5000

    # Figure 10: Max-Min Fairness, extend from 3.0 to 4.0 jph
    fig10_new_rates = [3.2, 3.4, 3.6, 3.8, 4.0]
    fig10_policies = ['max_min_fairness_perf', 'max_min_fairness']

    for rate in fig10_new_rates:
        lam = 3600 / rate  # Inter-arrival time in seconds
        for policy in fig10_policies:
            for seed in seeds:
                exp = {
                    "figure": "fig10",
                    "policy": policy,
                    "jobs_per_hr": rate,
                    "lambda": lam,
                    "cluster_spec": cluster_spec,
                    "multi_gpu": True,
                    "seed": seed,
                    "window_start": window_start,
                    "window_end": window_end,
                    "name": f"fig10_{policy}_{rate}jph_multi_s{seed}"
                }
                experiments.append(exp)

    # Figure 11: Finish-Time Fairness, extend from 3.4 to 4.4 jph
    fig11_new_rates = [3.6, 3.8, 4.0, 4.2, 4.4]
    fig11_policies = ['finish_time_fairness_perf', 'finish_time_fairness']

    for rate in fig11_new_rates:
        lam = 3600 / rate
        for policy in fig11_policies:
            for seed in seeds:
                exp = {
                    "figure": "fig11",
                    "policy": policy,
                    "jobs_per_hr": rate,
                    "lambda": lam,
                    "cluster_spec": cluster_spec,
                    "multi_gpu": True,
                    "seed": seed,
                    "window_start": window_start,
                    "window_end": window_end,
                    "name": f"fig11_{policy}_{rate}jph_multi_s{seed}"
                }
                experiments.append(exp)

    return experiments

if __name__ == "__main__":
    experiments = generate_experiments()

    print(f"Generated {len(experiments)} extended experiments:")
    print(f"  Fig 10: {len([e for e in experiments if e['figure'] == 'fig10'])} experiments (3.2-4.0 jph)")
    print(f"  Fig 11: {len([e for e in experiments if e['figure'] == 'fig11'])} experiments (3.6-4.4 jph)")

    with open("experiments_extended.json", "w") as f:
        json.dump(experiments, f, indent=2)

    print("\nSaved to experiments_extended.json")
