#!/usr/bin/env python3
"""
Print sample jobs from Gavel's generators for inspection.

Usage (run from src/scheduler):
  python scripts/print_sample_jobs.py [--count 20] [--mix default|fragmentation] [--throughputs path]

  --mix fragmentation: use Alibaba-like mix (many 1/2-GPU, some 4-GPU) to create
                       placement fragmentation so FGD vs strided can show difference.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


def main():
    parser = argparse.ArgumentParser(description="Print sample generated jobs")
    parser.add_argument("-n", "--count", type=int, default=20, help="Number of jobs to generate")
    parser.add_argument(
        "--mix",
        choices=["default", "fragmentation", "both"],
        default="both",
        help="default=Philly mix (70%% 1-GPU, 10%% 2, 15%% 4, 5%% 8); "
        "fragmentation=Alibaba-like (55%% 1, 30%% 2, 15%% 4) to see FGD vs strided difference",
    )
    parser.add_argument(
        "--throughputs-file",
        type=str,
        default="simulation_throughputs.json",
        help="Path to throughputs JSON (oracle)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--raw", action="store_true", help="Print Job.__str__ (trace) format only")
    args = parser.parse_args()

    if not os.path.isfile(args.throughputs_file):
        print("Error: throughputs file not found:", args.throughputs_file, file=sys.stderr)
        print("Run from src/scheduler or pass --throughputs-file.", file=sys.stderr)
        sys.exit(1)

    # Use same loader as scheduler so keys are (job_type, scale_factor) tuples
    throughputs = utils.read_all_throughputs_json_v2(args.throughputs_file)

    rng = __import__("random").Random(args.seed)

    def run(mix_name, scale_factor_func, multi_gpu=True):
        jobs = []
        for i in range(args.count):
            job = utils.generate_job(
                throughputs=throughputs,
                reference_worker_type="v100",
                rng=rng,
                job_id=None,
                fixed_job_duration=None,
                generate_multi_gpu_jobs=multi_gpu,
                scale_factor_generator_func=scale_factor_func,
            )
            jobs.append(job)
        return jobs

    if args.raw:
        if args.mix in ("default", "both"):
            jobs = run("default", utils._generate_scale_factor, multi_gpu=True)
            for j in jobs:
                print(j)
        if args.mix in ("fragmentation", "both"):
            if args.mix == "both":
                print("--- fragmentation mix ---")
            jobs = run("fragmentation", utils._generate_scale_factor_fragmentation_friendly, multi_gpu=True)
            for j in jobs:
                print(j)
        return

    if args.mix in ("default", "both"):
        title = "Default (Philly-style) job mix — mostly 1-GPU, some 2/4/8; whole GPU only."
        print("=" * 72)
        print(title)
        print("=" * 72)
        jobs = run("default", utils._generate_scale_factor, multi_gpu=True)
        scale_counts = {}
        for j in jobs:
            s = j.scale_factor
            scale_counts[s] = scale_counts.get(s, 0) + 1
        print("Scale factor distribution:", dict(sorted(scale_counts.items())))
        for i, j in enumerate(jobs):
            print(utils.format_job_for_print(j, index=i + 1))
        print()

    if args.mix in ("fragmentation", "both"):
        title = "Fragmentation-friendly (Alibaba-like + Gavel) job mix — 55% 1-GPU, 30% 2-GPU, 15% 4-GPU."
        print("=" * 72)
        print(title)
        print("=" * 72)
        jobs = run("fragmentation", utils._generate_scale_factor_fragmentation_friendly, multi_gpu=True)
        scale_counts = {}
        for j in jobs:
            s = j.scale_factor
            scale_counts[s] = scale_counts.get(s, 0) + 1
        print("Scale factor distribution:", dict(sorted(scale_counts.items())))
        for i, j in enumerate(jobs):
            print(utils.format_job_for_print(j, index=i + 1))
        print()
        print("Use with run_sweep_static.py: --job-mix fragmentation --placement-strategy fgd vs strided to see difference.")


if __name__ == "__main__":
    main()
