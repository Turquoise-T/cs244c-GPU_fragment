#!/usr/bin/env python3
"""Generate phase_fgd_replication_fifo.json -- 180 experiments with FIFO policy.

Same structure as phase_fgd_replication.json but with FIFO scheduling
to match the FGD paper's original evaluation methodology.

4 placements x 15 arrival rates x 3 seeds.
"""
import json
import os


PLACEMENTS = [
    {'label': 'strided',  'enable_fgd': False, 'fgd_placement_mode': 'fgd'},
    {'label': 'random',   'enable_fgd': True,  'fgd_placement_mode': 'random'},
    {'label': 'bestfit',  'enable_fgd': True,  'fgd_placement_mode': 'bestfit'},
    {'label': 'fgd',      'enable_fgd': True,  'fgd_placement_mode': 'fgd'},
]

RATES = [5, 10, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 360]

SEEDS = [0, 1, 2]

COMMON = {
    "policy": "fifo",
    "mode": "steady_state",
    "window_start": 4000,
    "window_end": 5000,
    "time_per_iteration": 600,
    "generate_multi_gpu_jobs": True,
    "workload_mode": "alibaba",
    "fgd_workload_mode": "alibaba",
    "enable_migration_penalty": False,
    "enable_gpu_sharing": False,
    "completion_rate_threshold": 0.1,
    "throughputs_file": "simulation_throughputs_alibaba_split.json",
    "reference_worker_type": "V100M32_8",
    "cluster_spec": {
        "G2_8": 4392, "T4_2": 774, "G3_8": 312, "P100_2": 262,
        "V100M32_8": 168, "V100M16_4": 112, "T4_4": 68, "V100M16_8": 64,
        "V100M32_4": 36, "V100M16_1": 19, "P100_1": 3, "A10_1": 2
    },
    "num_gpus_per_server": {
        "G2_8": 8, "T4_2": 2, "G3_8": 8, "P100_2": 2,
        "V100M32_8": 8, "V100M16_4": 4, "T4_4": 4, "V100M16_8": 8,
        "V100M32_4": 4, "V100M16_1": 1, "P100_1": 1, "A10_1": 1
    }
}


def generate_experiments():
    experiments = []
    for placement in PLACEMENTS:
        for rate in RATES:
            lam = 3600.0 / rate
            for seed in SEEDS:
                name = f"fifo_{placement['label']}_{rate}jph_s{seed}"
                exp = {
                    "name": name,
                    "seed": seed,
                    "lam": lam,
                    "enable_fgd": placement["enable_fgd"],
                    "fgd_placement_mode": placement["fgd_placement_mode"],
                }
                experiments.append(exp)
    return experiments


def main():
    experiments = generate_experiments()
    assert len(experiments) == 180

    config = {
        "description": (
            f"FGD replication with FIFO policy: {len(experiments)} experiments. "
            f"4 placements x {len(RATES)} rates x {len(SEEDS)} seeds. "
            f"Alibaba split cluster (12 sub-types, 6212 GPUs)."
        ),
        "common": COMMON,
        "experiments": experiments,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(configs_dir, exist_ok=True)
    output_path = os.path.join(configs_dir, "phase_fgd_replication_fifo.json")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Written {len(experiments)} experiments to {output_path}")


if __name__ == "__main__":
    main()
