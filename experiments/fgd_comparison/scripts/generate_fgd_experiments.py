#!/usr/bin/env python3
"""
Generate experiment configurations for FGD vs Strided comparison
under GPU spatial sharing in Gavel.

Output: configs/experiments_fgd.json
"""

import json
import os

experiments = []

# Experiment parameters
CLUSTER_SPEC = "8:4:4"            # 8 V100 + 4 P100 + 4 K80 = 16 GPUs
NUM_GPUS_PER_SERVER = "4:4:4"     # 4 GPUs per server → 4 servers
POLICY = "fifo"
NUM_TOTAL_JOBS = 100
SEEDS = [0, 1, 2]
PLACEMENT_STRATEGIES = ["strided", "fgd"]

# Job arrival rates (jobs per hour)
# lam = 3600 / jobs_per_hr (seconds between arrivals)
JOBS_PER_HR_RATES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

for strategy in PLACEMENT_STRATEGIES:
    for rate in JOBS_PER_HR_RATES:
        for seed in SEEDS:
            lam = 3600.0 / rate
            name = f"fgd_{strategy}_{rate:.1f}jph_s{seed}"
            experiments.append({
                "name": name,
                "placement_strategy": strategy,
                "policy": POLICY,
                "jobs_per_hr": rate,
                "lambda": lam,
                "seed": seed,
                "cluster_spec": CLUSTER_SPEC,
                "num_gpus_per_server": NUM_GPUS_PER_SERVER,
                "num_total_jobs": NUM_TOTAL_JOBS,
                "gpu_sharing": True,
            })

print(f"Total experiments: {len(experiments)}")
print(f"  Strategies: {PLACEMENT_STRATEGIES}")
print(f"  Rates: {JOBS_PER_HR_RATES}")
print(f"  Seeds: {SEEDS}")
print(f"  Cluster: {CLUSTER_SPEC}, {NUM_GPUS_PER_SERVER} per server")
print(f"  Jobs per experiment: {NUM_TOTAL_JOBS}")

output_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "experiments_fgd.json")

with open(output_path, "w") as f:
    json.dump(experiments, f, indent=2)

print(f"\nWritten to {output_path}")
