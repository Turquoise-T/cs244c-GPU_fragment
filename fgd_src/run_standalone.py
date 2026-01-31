#!/usr/bin/env python3
"""CLI entry point for standalone FGD experiments (Phase A).

Usage examples:

  # Monte-Carlo inflation on Cluster H with FGD placement
  python run_standalone.py --mode inflation --placement fgd --config configs/cluster_h.json

  # Trace replay (first 1000 tasks)
  python run_standalone.py --mode trace --trace-dir /path/to/alibaba/trace --max-tasks 1000

  # Compare all placement strategies
  python run_standalone.py --mode inflation --placement all --config configs/cluster_h.json
"""

import argparse
import json
import os
import sys

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fgd import Task, Workload
from simulator import FGDSimulator
from alibaba_trace_parser import (
    parse_alibaba_trace, derive_workload_distribution, generate_synthetic_trace
)

ALL_PLACEMENTS = ['fgd', 'bestfit', 'firstfit', 'random', 'dotprod', 'gpupacking', 'gpuclustering']


def build_workload_from_distribution(distribution):
    """Convert distribution tuples into FGD Workload object.

    Accepts both old 3-tuple (gpu, cpu, pop) and new 5-tuple
    (gpu, cpu, mem, gpu_type, pop) formats.
    """
    workload = Workload()
    for entry in distribution:
        gpu_req = entry[0]
        cpu_req = entry[1]
        mem_req = entry[2] if len(entry) > 3 else 0.0
        gpu_type = entry[3] if len(entry) > 4 else None
        pop = entry[-1]
        task_id = f'{gpu_req}gpu'
        workload.add_task_type(
            Task(id=task_id, cpu_request=cpu_req, gpu_request=gpu_req,
                 memory_request=mem_req, gpu_type=gpu_type),
            popularity=pop,
        )
    workload.normalize_popularity()
    return workload


def run_inflation(args):
    """Run Monte-Carlo workload inflation experiment."""
    with open(args.config, 'r') as f:
        cluster_config = json.load(f)

    # Use Philly distribution or derive from trace
    if args.trace_dir:
        trace_tasks = parse_alibaba_trace(args.trace_dir, max_tasks=args.max_tasks)
        distribution = derive_workload_distribution(trace_tasks)
        print(f"Derived distribution from {len(trace_tasks)} trace tasks:")
    else:
        distribution = [
            (1.0, 4.0, 0.70),
            (2.0, 8.0, 0.10),
            (4.0, 16.0, 0.15),
            (8.0, 32.0, 0.05),
        ]
        print("Using default Philly-like distribution:")

    for entry in distribution:
        gpu, cpu, pop = entry[0], entry[1], entry[-1]
        print(f"  {gpu:.2f} GPU, {cpu:.0f} CPU: {pop:.1%}")

    workload = build_workload_from_distribution(distribution)

    placements = [args.placement] if args.placement != 'all' else ALL_PLACEMENTS

    all_results = {}
    for placement in placements:
        print(f"\n{'='*60}")
        print(f"Running inflation with placement={placement}")
        print(f"{'='*60}")

        sim = FGDSimulator(
            cluster_config=cluster_config,
            placement=placement,
            seed=args.seed,
            workload=workload,
        )
        results = sim.run_inflation(
            distribution=distribution,
            target_utilization=args.target_utilization,
            batch_size=args.batch_size,
            max_tasks=args.max_inflate_tasks,
            seed=args.seed,
        )
        all_results[placement] = results

        # Print summary
        if results:
            last = results[-1]
            print(f"  Final state: {last['tasks_submitted']} tasks submitted")
            print(f"  GPU utilization: {last['utilization']:.2%}")
            print(f"  Unallocated GPUs: {last['unallocated_gpus']:.1f}")
            print(f"  Fragmentation F_N(M): {last['fragmentation']:.2f}")
            print(f"  Demand fraction: {last['demand_fraction']:.2f}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return all_results


def run_trace(args):
    """Run trace replay experiment."""
    with open(args.config, 'r') as f:
        cluster_config = json.load(f)

    if not args.trace_dir:
        print("Error: --trace-dir required for trace mode")
        sys.exit(1)

    trace_tasks = parse_alibaba_trace(args.trace_dir, max_tasks=args.max_tasks)
    print(f"Loaded {len(trace_tasks)} tasks from trace")

    distribution = derive_workload_distribution(trace_tasks)
    workload = build_workload_from_distribution(distribution)

    placements = [args.placement] if args.placement != 'all' else ALL_PLACEMENTS

    for placement in placements:
        print(f"\n{'='*60}")
        print(f"Running trace replay with placement={placement}")
        print(f"{'='*60}")

        sim = FGDSimulator(
            cluster_config=cluster_config,
            placement=placement,
            seed=args.seed,
            workload=workload,
        )
        sim.run(trace_tasks, metrics_interval=args.metrics_interval)

        # Print summary
        if sim.metrics_history:
            last = sim.metrics_history[-1]
            print(f"  Completed tasks: {last.completed_tasks}")
            print(f"  Final GPU utilization: {last.gpu_utilization:.2%}")
            print(f"  Final fragmentation: {last.fragmentation:.2f}")
            print(f"  Peak pending: {max(m.pending_tasks for m in sim.metrics_history)}")


def main():
    parser = argparse.ArgumentParser(description='Standalone FGD experiments')
    parser.add_argument('--mode', choices=['inflation', 'trace'], default='inflation',
                        help='Experiment mode')
    parser.add_argument('--placement',
                        choices=ALL_PLACEMENTS + ['all'],
                        default='fgd', help='Placement strategy')
    parser.add_argument('--config', default=os.path.join(
        os.path.dirname(__file__), 'configs', 'cluster_h.json'),
                        help='Cluster config JSON')
    parser.add_argument('--trace-dir', default=None,
                        help='Directory containing Alibaba trace CSVs')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Max tasks to load from trace')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file for results')
    parser.add_argument('--metrics-interval', type=float, default=60.0,
                        help='Metrics recording interval (seconds)')
    parser.add_argument('--target-utilization', type=float, default=1.0,
                        help='Target GPU utilization for inflation mode')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for inflation mode')
    parser.add_argument('--max-inflate-tasks', type=int, default=50000,
                        help='Max tasks for inflation mode')

    args = parser.parse_args()

    if args.mode == 'inflation':
        run_inflation(args)
    elif args.mode == 'trace':
        run_trace(args)


if __name__ == '__main__':
    main()
