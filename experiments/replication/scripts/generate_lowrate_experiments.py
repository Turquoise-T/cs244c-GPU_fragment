#!/usr/bin/env python3
"""Generate ONLY the low-rate experiments that failed due to timeout.

Only includes rates < 2.0 jph with scaled measurement windows.
These will supplement the existing successful experiments.
"""

import json

# Threshold: at 2.0 jph, takes 2000 hours to reach job 4000
RATE_THRESHOLD = 2.0

def calculate_window(rate):
    """Calculate scaled measurement window for low arrival rates."""
    # Use 1500 hours warm-up, 500 hours measurement (total 2000 hours max)
    warm_up_hours = 1500
    measurement_hours = 500

    window_start = int(rate * warm_up_hours)
    window_end = int(rate * (warm_up_hours + measurement_hours))

    # Ensure minimum window size of 100 jobs for statistical validity
    if window_end - window_start < 100:
        window_start = max(50, window_start)
        window_end = window_start + 100

    return window_start, window_end

experiments = []

# Figure 9: Single-GPU - rates < 2.0 jph
# Range: 0.4 to 1.6 jobs/hr (4 rates that failed)
fig9_low_rates = [r for r in [0.4 + i * 0.4 for i in range(20)] if r < RATE_THRESHOLD]
print(f"Fig 9 low rates: {fig9_low_rates}")
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig9_low_rates:
        window_start, window_end = calculate_window(rate)
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig9",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": window_start,
                "window_end": window_end,
                "multi_gpu": False
            })

# Figure 10: Multi-GPU - rates < 2.0 jph
# Range: 0.2 to 1.8 jobs/hr (9 rates that failed)
fig10_low_rates = [r for r in [0.2 + i * 0.2 for i in range(15)] if r < RATE_THRESHOLD]
print(f"Fig 10 low rates: {fig10_low_rates}")
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig10_low_rates:
        window_start, window_end = calculate_window(rate)
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig10",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": window_start,
                "window_end": window_end,
                "multi_gpu": True
            })

# Figure 11: Multi-GPU - rates < 2.0 jph
# Range: 0.2 to 1.8 jobs/hr (9 rates that failed)
fig11_low_rates = [r for r in [0.2 + i * 0.2 for i in range(17)] if r < RATE_THRESHOLD]
print(f"Fig 11 low rates: {fig11_low_rates}")
for policy in ["finish_time_fairness", "finish_time_fairness_perf"]:
    for rate in fig11_low_rates:
        window_start, window_end = calculate_window(rate)
        for seed in [0, 1, 2]:
            experiments.append({
                "figure": "fig11",
                "policy": policy,
                "seed": seed,
                "lambda": 3600.0 / rate,
                "jobs_per_hr": rate,
                "cluster_spec": "36:36:36",
                "window_start": window_start,
                "window_end": window_end,
                "multi_gpu": True
            })

# Print summary
print(f"\nTotal LOW-RATE experiments to rerun: {len(experiments)}")
print(f"  Fig 9: {len([e for e in experiments if e['figure'] == 'fig9'])}")
print(f"  Fig 10: {len([e for e in experiments if e['figure'] == 'fig10'])}")
print(f"  Fig 11: {len([e for e in experiments if e['figure'] == 'fig11'])}")

print("\nWindow examples:")
for rate in [0.2, 0.4, 0.8, 1.0, 1.6, 1.8]:
    ws, we = calculate_window(rate)
    print(f"  {rate:.1f} jph: window {ws}-{we} ({we-ws} jobs)")

# Write to file
with open("experiments_lowrate.json", "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nWritten to experiments_lowrate.json")
print(f"These supplement the existing high-rate results in results_full/")
