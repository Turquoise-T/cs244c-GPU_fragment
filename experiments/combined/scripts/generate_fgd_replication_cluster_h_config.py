#!/usr/bin/env python3
"""Generate phase_fgd_replication_cluster_h.json -- FGD paper's Cluster H topology.

Uses the FGD paper's actual cluster: 1200 nodes, single "generic" GPU type,
mixed node sizes (462x8, 310x4, 228x2, 200x1 = 5592 GPUs).

This matches the paper's experimental setup exactly, unlike the Alibaba split
config which uses 12 heterogeneous GPU sub-types with uniform servers.

4 placements x 15 arrival rates x 3 seeds x 2 policies = 360 experiments.
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

POLICIES = [
    {'label': 'fifo', 'policy': 'fifo'},
    {'label': 'mmf',  'policy': 'max_min_fairness'},
]

# FGD paper Cluster H: 1200 nodes, single generic GPU type, mixed node sizes.
# Source: src/fgd/configs/cluster_h.json
COMMON = {
    "mode": "steady_state",
    "window_start": 4000,
    "window_end": 5000,
    "time_per_iteration": 600,
    "generate_multi_gpu_jobs": True,
    "workload_mode": "alibaba",
    "fgd_workload_mode": "alibaba",
    "enable_migration_penalty": False,
    "enable_gpu_sharing": False,
    "solver": "ECOS",
    "completion_rate_threshold": 0.1,
    "throughputs_file": "simulation_throughputs_cluster_h.json",
    "reference_worker_type": "generic",
    "cluster_spec": {
        "generic": 5592,
    },
    "num_gpus_per_server": {
        "generic": {"8": 462, "4": 310, "2": 228, "1": 200},
    },
}


def generate_experiments():
    experiments = []
    for pol in POLICIES:
        for placement in PLACEMENTS:
            for rate in RATES:
                lam = 3600.0 / rate
                for seed in SEEDS:
                    name = (f"ch_{pol['label']}_{placement['label']}_"
                            f"{rate}jph_s{seed}")
                    exp = {
                        "name": name,
                        "policy": pol["policy"],
                        "seed": seed,
                        "lam": lam,
                        "enable_fgd": placement["enable_fgd"],
                        "fgd_placement_mode": placement["fgd_placement_mode"],
                    }
                    experiments.append(exp)
    return experiments


def main():
    experiments = generate_experiments()
    expected = len(POLICIES) * len(PLACEMENTS) * len(RATES) * len(SEEDS)
    assert len(experiments) == expected, \
        f"Expected {expected}, got {len(experiments)}"

    config = {
        "description": (
            f"FGD replication with Cluster H topology: {len(experiments)} "
            f"experiments. {len(POLICIES)} policies x {len(PLACEMENTS)} "
            f"placements x {len(RATES)} rates x {len(SEEDS)} seeds. "
            f"Single 'generic' GPU type, 1200 nodes (462x8 + 310x4 + "
            f"228x2 + 200x1 = 5592 GPUs)."
        ),
        "common": COMMON,
        "experiments": experiments,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(configs_dir, exist_ok=True)
    output_path = os.path.join(configs_dir,
                               "phase_fgd_replication_cluster_h.json")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Written {len(experiments)} experiments to {output_path}")


if __name__ == "__main__":
    main()
