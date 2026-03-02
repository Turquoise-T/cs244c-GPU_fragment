#!/usr/bin/env python3
"""Generate full experiment configurations for Figures 9, 10, 11 replication."""

import json

experiments = []

# Figure 9: Single-GPU, max_min_fairness vs max_min_fairness_perf
# Range: 0.4 to 8.0 jobs/hr, 20 points, spacing 0.4
fig9_rates = [0.4 + i * 0.4 for i in range(20)]  # 0.4, 0.8, ..., 8.0
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig9_rates:
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig9",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,  # lambda = seconds between arrivals
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": 4000,
                "window_end": 5000,
                "multi_gpu": False
            })

# Figure 10: Multi-GPU, max_min_fairness vs max_min_fairness_perf
# Range: 0.2 to 3.0 jobs/hr, 15 points, spacing 0.2
fig10_rates = [0.2 + i * 0.2 for i in range(15)]  # 0.2, 0.4, ..., 3.0
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig10_rates:
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig10",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": 4000,
                "window_end": 5000,
                "multi_gpu": True
            })

# Figure 11: Multi-GPU, finish_time_fairness vs finish_time_fairness_perf
# Range: 0.2 to 3.4 jobs/hr, 17 points, spacing 0.2
fig11_rates = [0.2 + i * 0.2 for i in range(17)]  # 0.2, 0.4, ..., 3.4
for policy in ["finish_time_fairness", "finish_time_fairness_perf"]:
    for rate in fig11_rates:
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig11",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": 4000,
                "window_end": 5000,
                "multi_gpu": True
            })

print(f"Total experiments: {len(experiments)}")
print(f"  Fig 9: {len([e for e in experiments if e['figure'] == 'fig9'])}")
print(f"  Fig 10: {len([e for e in experiments if e['figure'] == 'fig10'])}")
print(f"  Fig 11: {len([e for e in experiments if e['figure'] == 'fig11'])}")

with open("experiments_full.json", "w") as f:
    json.dump(experiments, f, indent=2)

print("Written to experiments_full.json")
