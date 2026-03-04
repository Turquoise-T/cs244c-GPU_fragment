#!/usr/bin/env python3
"""Generate phase_fgd_replication.json -- 180 experiments.

4 placements x 15 arrival rates x 3 seeds.

Placements:
  strided  (enable_fgd=False) -- baseline, no FGD
  random   (enable_fgd=True)  -- FGD with random placement
  bestfit  (enable_fgd=True)  -- FGD with best-fit placement
  fgd      (enable_fgd=True)  -- FGD with fragmentation-aware placement
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
    "policy": "max_min_fairness",
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
    """Build the full list of 180 experiments."""
    experiments = []
    for placement in PLACEMENTS:
        for rate in RATES:
            lam = 3600.0 / rate
            for seed in SEEDS:
                name = f"fgd_{placement['label']}_{rate}jph_s{seed}"
                exp = {
                    "name": name,
                    "seed": seed,
                    "lam": lam,
                    "enable_fgd": placement["enable_fgd"],
                    "fgd_placement_mode": placement["fgd_placement_mode"],
                }
                experiments.append(exp)
    return experiments


def build_config(experiments):
    """Wrap experiments in the standard config envelope."""
    return {
        "description": (
            f"FGD replication via Gavel: {len(experiments)} experiments. "
            f"4 placements x {len(RATES)} rates x {len(SEEDS)} seeds. "
            f"Alibaba split cluster (12 sub-types, 6212 GPUs)."
        ),
        "common": COMMON,
        "experiments": experiments,
    }


def validate(experiments):
    """Run assertions and print summary."""
    # Total count
    assert len(experiments) == 180, (
        f"Expected 180 experiments, got {len(experiments)}"
    )

    # No duplicate names
    names = [e["name"] for e in experiments]
    assert len(names) == len(set(names)), (
        f"Duplicate names found: {len(names)} total, {len(set(names))} unique"
    )

    # Strided experiments have enable_fgd=False
    strided = [e for e in experiments if "_strided_" in e["name"]]
    assert len(strided) == len(RATES) * len(SEEDS), (
        f"Expected {len(RATES) * len(SEEDS)} strided experiments, got {len(strided)}"
    )
    for e in strided:
        assert e["enable_fgd"] is False, (
            f"{e['name']}: strided should have enable_fgd=False"
        )

    # Non-strided experiments have enable_fgd=True
    non_strided = [e for e in experiments if "_strided_" not in e["name"]]
    for e in non_strided:
        assert e["enable_fgd"] is True, (
            f"{e['name']}: non-strided should have enable_fgd=True"
        )

    # All 4 placement modes present
    modes = set(e["fgd_placement_mode"] for e in experiments)
    assert modes == {"fgd", "random", "bestfit"}, (
        f"Expected placement modes {{fgd, random, bestfit}}, got {modes}"
    )

    # Per-placement counts
    for placement in PLACEMENTS:
        label = placement["label"]
        group = [e for e in experiments if f"_{label}_" in e["name"]]
        expected = len(RATES) * len(SEEDS)
        assert len(group) == expected, (
            f"Placement '{label}': expected {expected}, got {len(group)}"
        )

    # Lambda values
    for e in experiments:
        parts = e["name"].split("_")
        rate_str = [p for p in parts if p.endswith("jph")][0]
        rate = float(rate_str.replace("jph", ""))
        expected_lam = 3600.0 / rate
        assert abs(e["lam"] - expected_lam) < 1e-4, (
            f"{e['name']}: lam={e['lam']}, expected {expected_lam}"
        )

    # Print summary
    print("Validation passed.")
    print(f"  Total experiments: {len(experiments)}")
    print()

    for placement in PLACEMENTS:
        label = placement["label"]
        group = [e for e in experiments if f"_{label}_" in e["name"]]
        rates = sorted(set(
            float([p for p in e["name"].split("_") if p.endswith("jph")][0].replace("jph", ""))
            for e in group
        ))
        print(f"  {label} (enable_fgd={placement['enable_fgd']}, "
              f"mode={placement['fgd_placement_mode']}): {len(group)} experiments")
        print(f"    Rates: {rates[0]:.0f} to {rates[-1]:.0f} jph ({len(rates)} points)")
        print(f"    Seeds: {sorted(set(e['seed'] for e in group))}")
        print()


def main():
    experiments = generate_experiments()

    # Validate before writing
    validate(experiments)

    config = build_config(experiments)

    # Write to configs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(configs_dir, exist_ok=True)
    output_path = os.path.join(configs_dir, "phase_fgd_replication.json")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
