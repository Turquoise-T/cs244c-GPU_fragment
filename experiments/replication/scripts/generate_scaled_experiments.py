#!/usr/bin/env python3
"""Generate experiment configurations with rate-proportional measurement windows.

Only scales the measurement window for low arrival rates that cannot reach
the standard window (4000-5000) within the 2000-hour simulation timeout.

- Rates >= 2.0 jph: Standard window 4000-5000 (reachable within 2000 hours)
- Rates < 2.0 jph: Scaled window based on rate to fit within timeout
"""

import json

# Threshold: at 2.0 jph, takes 2000 hours to reach job 4000
RATE_THRESHOLD = 2.0
STANDARD_WINDOW_START = 4000
STANDARD_WINDOW_END = 5000

def calculate_window(rate):
    """Calculate measurement window based on job arrival rate.

    Args:
        rate: Job arrival rate in jobs/hour

    Returns:
        (window_start, window_end): Job indices for measurement window
    """
    if rate >= RATE_THRESHOLD:
        # Standard window for high rates
        return STANDARD_WINDOW_START, STANDARD_WINDOW_END
    else:
        # Scaled window for low rates
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

# Figure 9: Single-GPU, max_min_fairness vs max_min_fairness_perf
# Range: 0.4 to 8.0 jobs/hr, 20 points, spacing 0.4
fig9_rates = [0.4 + i * 0.4 for i in range(20)]  # 0.4, 0.8, ..., 8.0
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig9_rates:
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

# Figure 10: Multi-GPU, max_min_fairness vs max_min_fairness_perf
# Range: 0.2 to 3.0 jobs/hr, 15 points, spacing 0.2
fig10_rates = [0.2 + i * 0.2 for i in range(15)]  # 0.2, 0.4, ..., 3.0
for policy in ["max_min_fairness", "max_min_fairness_perf"]:
    for rate in fig10_rates:
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

# Figure 11: Multi-GPU, finish_time_fairness vs finish_time_fairness_perf
# Range: 0.2 to 3.4 jobs/hr, 17 points, spacing 0.2
fig11_rates = [0.2 + i * 0.2 for i in range(17)]  # 0.2, 0.4, ..., 3.4
for policy in ["finish_time_fairness", "finish_time_fairness_perf"]:
    for rate in fig11_rates:
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
print(f"Total experiments: {len(experiments)}")
print(f"  Fig 9: {len([e for e in experiments if e['figure'] == 'fig9'])}")
print(f"  Fig 10: {len([e for e in experiments if e['figure'] == 'fig10'])}")
print(f"  Fig 11: {len([e for e in experiments if e['figure'] == 'fig11'])}")

print(f"\nWindow scaling (threshold = {RATE_THRESHOLD} jph):")
print("  Rates >= 2.0 jph: standard window 4000-5000")
print("  Rates < 2.0 jph: scaled window")
print("\nExamples:")
for rate in [0.2, 0.4, 0.8, 1.0, 1.6, 2.0, 3.0, 8.0]:
    ws, we = calculate_window(rate)
    scaled = "SCALED" if rate < RATE_THRESHOLD else "standard"
    print(f"  {rate:.1f} jph: window {ws}-{we} ({we-ws} jobs) [{scaled}]")

# Write to file
with open("experiments_scaled.json", "w") as f:
    json.dump(experiments, f, indent=2)

print("\nWritten to experiments_scaled.json")
