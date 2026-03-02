#!/usr/bin/env python3
"""
Build .viz.bin files from FGD experiment logs for the GPU Scheduling Visualizer.

Requires that experiments were run with --keep-logs so that
results/logs/<name>/simulation.log exist. Converts one strided + one fgd
run (same jobs_per_hr and seed) into two .viz.bin files you can load
side-by-side in gpu-scheduling-viz/index.html.

Usage:
  # Build viz for one (rate, seed) pair (e.g. 5 jobs/hr, seed 0)
  python build_fgd_viz.py --rate 5 --seed 0

  # Build viz for all available (rate, seed) pairs that have both logs
  python build_fgd_viz.py --all

  # List which pairs have logs (no conversion)
  python build_fgd_viz.py --list
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
VIZ_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "viz")
PREPROCESS_VIZ = os.path.join(SCRIPT_DIR, "..", "..", "..", "gpu-scheduling-viz", "preprocess_viz.py")


def get_config():
    config_path = os.path.join(SCRIPT_DIR, "..", "configs", "experiments_fgd.json")
    with open(config_path) as f:
        return json.load(f)


def find_log_pairs():
    """Find (rate, seed) pairs that have both strided and fgd logs."""
    if not os.path.isdir(LOGS_DIR):
        return []
    experiments = get_config()
    by_key = {}  # (rate, seed) -> { 'strided': path, 'fgd': path }
    for exp in experiments:
        name = exp["name"]
        log_dir = os.path.join(LOGS_DIR, name)
        sim_log = os.path.join(log_dir, "simulation.log")
        if not os.path.isfile(sim_log):
            continue
        key = (exp["jobs_per_hr"], exp["seed"])
        if key not in by_key:
            by_key[key] = {}
        by_key[key][exp["placement_strategy"]] = sim_log
    pairs = []
    for (rate, seed), strategies in by_key.items():
        if "strided" in strategies and "fgd" in strategies:
            pairs.append((rate, seed, strategies["strided"], strategies["fgd"]))
    return sorted(pairs)


def build_one(rate, seed, strided_log, fgd_log, out_dir):
    """Run preprocess_viz.py on both logs and write .viz.bin to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    base = f"fgd_{rate}jph_s{seed}"
    out_strided = os.path.join(out_dir, f"{base}_strided.viz.bin")
    out_fgd = os.path.join(out_dir, f"{base}_fgd.viz.bin")
    for log_path, out_path, policy in [
        (strided_log, out_strided, "Strided (worst-fit)"),
        (fgd_log, out_fgd, "FGD (fragmentation-aware)"),
    ]:
        cmd = [
            sys.executable,
            PREPROCESS_VIZ,
            log_path,
            "-o", out_path,
            "--policy", policy,
        ]
        subprocess.run(cmd, check=True)
    return out_strided, out_fgd


def main():
    parser = argparse.ArgumentParser(
        description="Build .viz.bin from FGD experiment logs for the visualizer"
    )
    parser.add_argument("--rate", type=float, default=None,
                        help="Jobs per hour (e.g. 5.0); with --seed build this pair only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed (e.g. 0); with --rate build this pair only")
    parser.add_argument("--all", action="store_true",
                        help="Build .viz.bin for all (rate, seed) pairs that have both logs")
    parser.add_argument("--list", action="store_true",
                        help="List (rate, seed) pairs that have logs; no conversion")
    parser.add_argument("--output-dir", type=str, default=VIZ_DIR,
                        help="Directory for .viz.bin files (default: figures/viz/)")
    args = parser.parse_args()

    pairs = find_log_pairs()
    if not pairs:
        print("No log pairs found. Run experiments with --keep-logs first, e.g.:")
        print("  python run_fgd_experiment.py --range 0 2 --keep-logs   # strided 1.0jph s0,s1")
        print("  python run_fgd_experiment.py --range 24 26 --keep-logs  # fgd 1.0jph s0,s1")
        sys.exit(1)

    if args.list:
        print("Available (rate, seed) pairs with both strided and fgd logs:")
        for rate, seed, _, _ in pairs:
            print(f"  rate={rate} jobs/hr  seed={seed}")
        return

    if args.all:
        for rate, seed, strided_log, fgd_log in pairs:
            print(f"Building viz for rate={rate} seed={seed}...")
            build_one(rate, seed, strided_log, fgd_log, args.output_dir)
        print(f"Done. .viz.bin files written to {args.output_dir}")
        print("Open gpu-scheduling-viz/index.html and load two files (e.g. *_strided.viz.bin and *_fgd.viz.bin).")
        return

    if args.rate is None or args.seed is None:
        parser.error("Specify --rate and --seed for one pair, or --all, or --list")
        return

    match = [(r, s, sl, fl) for r, s, sl, fl in pairs if r == args.rate and s == args.seed]
    if not match:
        print(f"No logs found for rate={args.rate} seed={args.seed}. Available pairs:")
        for r, s, _, _ in pairs:
            print(f"  rate={r} seed={s}")
        sys.exit(1)
    rate, seed, strided_log, fgd_log = match[0]
    build_one(rate, seed, strided_log, fgd_log, args.output_dir)
    print(f"Wrote {args.output_dir}/fgd_{rate}jph_s{seed}_strided.viz.bin and _fgd.viz.bin")
    print("Open gpu-scheduling-viz/index.html and load both files in the file pickers.")


if __name__ == "__main__":
    main()
