#!/usr/bin/env python3
"""Generate experiment config for replicating Gavel OSDI'20 Figures 9, 10, 11.

Produces experiments/combined/configs/phase_gavel_replication.json with 417
experiments across three figures, all using Philly traces on the standard
36:36:36 cluster.

Experiment matrix:
  Fig 9  (single-GPU, LAS): 3 policies x 20 rates x 3 seeds = 180
  Fig 10 (multi-GPU, LAS):  3 policies x 15 rates x 3 seeds = 135
  Fig 11 (multi-GPU, FTF):  2 policies x 17 rates x 3 seeds = 102
  Total: 417
"""

import json
import os
import sys


def generate_experiments():
    """Build the full list of 417 experiments."""
    experiments = []

    seeds = [0, 1, 2]

    # ------------------------------------------------------------------
    # Fig 9: Single-GPU workload, Least Attained Service (LAS) policies
    # Rates: 0.4 to 8.0 jph, step 0.4 (20 points)
    # ------------------------------------------------------------------
    fig9_policies = [
        "max_min_fairness",
        "max_min_fairness_perf",
        "max_min_fairness_packed",
    ]
    fig9_rates = [round(0.4 + i * 0.4, 1) for i in range(20)]

    for policy in fig9_policies:
        for rate in fig9_rates:
            lam = round(3600.0 / rate, 6)
            for seed in seeds:
                name = f"fig9_{policy}_{rate}jph_single_s{seed}"
                experiments.append({
                    "name": name,
                    "policy": policy,
                    "lam": lam,
                    "seed": seed,
                    "generate_multi_gpu_jobs": False,
                })

    # ------------------------------------------------------------------
    # Fig 10: Multi-GPU workload, LAS policies
    # Rates: 0.2 to 3.0 jph, step 0.2 (15 points)
    # ------------------------------------------------------------------
    fig10_policies = [
        "max_min_fairness",
        "max_min_fairness_perf",
        "max_min_fairness_packed",
    ]
    fig10_rates = [round(0.2 + i * 0.2, 1) for i in range(15)]

    for policy in fig10_policies:
        for rate in fig10_rates:
            lam = round(3600.0 / rate, 6)
            for seed in seeds:
                name = f"fig10_{policy}_{rate}jph_multi_s{seed}"
                experiments.append({
                    "name": name,
                    "policy": policy,
                    "lam": lam,
                    "seed": seed,
                    "generate_multi_gpu_jobs": True,
                })

    # ------------------------------------------------------------------
    # Fig 11: Multi-GPU workload, Finish Time Fairness (FTF) policies
    # Rates: 0.2 to 3.4 jph, step 0.2 (17 points)
    # ------------------------------------------------------------------
    fig11_policies = [
        "finish_time_fairness",
        "finish_time_fairness_perf",
    ]
    fig11_rates = [round(0.2 + i * 0.2, 1) for i in range(17)]

    for policy in fig11_policies:
        for rate in fig11_rates:
            lam = round(3600.0 / rate, 6)
            for seed in seeds:
                name = f"fig11_{policy}_{rate}jph_multi_s{seed}"
                experiments.append({
                    "name": name,
                    "policy": policy,
                    "lam": lam,
                    "seed": seed,
                    "generate_multi_gpu_jobs": True,
                })

    return experiments


def build_config(experiments):
    """Wrap experiments in the standard config envelope."""
    return {
        "description": (
            "Gavel OSDI'20 replication: Figures 9, 10, 11. "
            "417 experiments (3 figures x policies x load sweep x 3 seeds). "
            "Fig 9: single-GPU LAS (180). "
            "Fig 10: multi-GPU LAS (135). "
            "Fig 11: multi-GPU FTF (102)."
        ),
        "experiments": experiments,
        "common": {
            "cluster_spec": {"v100": 36, "p100": 36, "k80": 36},
            "num_gpus_per_server": None,
            "mode": "steady_state",
            "window_start": 4000,
            "window_end": 5000,
            "max_jct": 360000,
            "time_per_iteration": 360,
            "enable_fgd": False,
            "enable_migration_penalty": False,
            "enable_gpu_sharing": False,
            "solver": "ECOS",
        },
    }


def validate(experiments):
    """Run assertions and print summary."""
    # Total count
    assert len(experiments) == 417, (
        f"Expected 417 experiments, got {len(experiments)}"
    )

    # No duplicate names
    names = [e["name"] for e in experiments]
    assert len(names) == len(set(names)), (
        f"Duplicate names found: {len(names)} total, {len(set(names))} unique"
    )

    # Per-figure counts
    fig9 = [e for e in experiments if e["name"].startswith("fig9_")]
    fig10 = [e for e in experiments if e["name"].startswith("fig10_")]
    fig11 = [e for e in experiments if e["name"].startswith("fig11_")]
    assert len(fig9) == 180, f"Fig 9: expected 180, got {len(fig9)}"
    assert len(fig10) == 135, f"Fig 10: expected 135, got {len(fig10)}"
    assert len(fig11) == 102, f"Fig 11: expected 102, got {len(fig11)}"

    # Lambda values: lam = 3600 / rate
    for e in experiments:
        # Recover rate from name (e.g. "fig9_..._4.0jph_single_s0")
        parts = e["name"].split("_")
        rate_str = [p for p in parts if p.endswith("jph")][0]
        rate = float(rate_str.replace("jph", ""))
        expected_lam = round(3600.0 / rate, 6)
        assert abs(e["lam"] - expected_lam) < 1e-4, (
            f"{e['name']}: lam={e['lam']}, expected {expected_lam}"
        )

    # Multi-GPU flags
    for e in fig9:
        assert e["generate_multi_gpu_jobs"] is False, (
            f"{e['name']}: should be single-GPU"
        )
    for e in fig10 + fig11:
        assert e["generate_multi_gpu_jobs"] is True, (
            f"{e['name']}: should be multi-GPU"
        )

    # Print summary
    print("Validation passed.")
    print(f"  Total experiments: {len(experiments)}")
    print()

    for label, group in [("Fig 9", fig9), ("Fig 10", fig10), ("Fig 11", fig11)]:
        policies = sorted(set(e["policy"] for e in group))
        rates = sorted(set(
            float([p for p in e["name"].split("_") if p.endswith("jph")][0].replace("jph", ""))
            for e in group
        ))
        multi = "multi-GPU" if group[0]["generate_multi_gpu_jobs"] else "single-GPU"
        print(f"  {label} ({multi}): {len(group)} experiments")
        print(f"    Policies: {', '.join(policies)}")
        print(f"    Rates: {rates[0]} to {rates[-1]} jph "
              f"({len(rates)} points, step {round(rates[1] - rates[0], 1)})")
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
    output_path = os.path.join(configs_dir, "phase_gavel_replication.json")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
