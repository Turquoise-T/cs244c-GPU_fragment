#!/usr/bin/env python3
"""Evaluation runner for FGD paper replication.

Runs the inflation experiment for 6 policies across multiple seeds on the
Alibaba trace, matching the methodology from the Go-based reference
implementation (kubernetes-scheduler-simulator).

Usage:
    python run_evaluation.py \
        --node-csv data/alibaba-gpu-v2023/openb_node_list_gpu_node.csv \
        --pod-csv  data/alibaba-gpu-v2023/openb_pod_list_default.csv \
        --seeds 42 43 44 \
        --policies random dotprod gpuclustering gpupacking bestfit fgd \
        --demand-target 1.3 \
        --output-dir results/
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'fgd'))

from fgd import Node, Task, Workload
from simulator import FGDSimulator
from alibaba_trace_parser import (
    parse_alibaba_trace, parse_node_list, derive_workload_distribution,
    count_non_gpu_tasks,
)


ALL_POLICIES = ['random', 'dotprod', 'gpuclustering', 'gpupacking', 'bestfit', 'fgd']


def build_workload_from_distribution(distribution):
    """Convert distribution tuples into FGD Workload object."""
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


def deep_copy_nodes(nodes):
    """Create independent copies of node list for each experiment."""
    new_nodes = []
    for n in nodes:
        node = Node(
            id=n.id,
            total_cpu=n.total_cpu,
            total_memory=n.total_memory,
            gpus=n.gpus.copy(),
            gpu_type=n.gpu_type,
        )
        # Reset allocation state
        node.allocated_cpu = 0.0
        node.allocated_memory = 0.0
        new_nodes.append(node)
    return new_nodes


def build_task_list_from_trace(trace_tasks, seed, total_gpus, demand_target):
    """Build a shuffled list of Task objects from trace tasks.

    Matches reference impl: shuffle-pod=true with given seed.
    Repeats task list if needed to reach demand_target * total_gpus.
    Tasks get unique IDs per seed/repetition to avoid collisions.
    """
    # Calculate how many repetitions we need
    trace_demand = sum(tt.task.gpu_request for tt in trace_tasks)
    needed_demand = total_gpus * demand_target
    reps = max(1, int(needed_demand / trace_demand) + 1)

    tasks = []
    for rep in range(reps):
        for i, tt in enumerate(trace_tasks):
            task = Task(
                id=f's{seed}-r{rep}-{i}',
                cpu_request=tt.task.cpu_request,
                gpu_request=tt.task.gpu_request,
                memory_request=tt.task.memory_request,
                gpu_type=tt.task.gpu_type,
            )
            tasks.append(task)

    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks


def run_single_experiment(args_tuple):
    """Run a single experiment (policy, seed) combination.

    Designed for use with multiprocessing.Pool.
    """
    policy, seed, base_nodes, trace_tasks, distribution, demand_target, record_interval = args_tuple

    t0 = time.time()

    # Fresh node copies
    nodes = deep_copy_nodes(base_nodes)
    total_gpus = sum(n.num_gpus for n in nodes)

    # Build workload for fragmentation measurement
    workload = build_workload_from_distribution(distribution)

    # Build shuffled task list (repeated to reach demand target)
    tasks = build_task_list_from_trace(trace_tasks, seed, total_gpus, demand_target)

    # Create simulator with direct node list
    sim = FGDSimulator(
        nodes=nodes,
        placement=policy,
        seed=seed,
        workload=workload,
    )

    # Run inflation
    curve = sim.run_inflation_from_tasks(
        tasks=tasks,
        target_demand_fraction=demand_target,
        record_interval=record_interval,
    )

    elapsed = time.time() - t0

    result = {
        'policy': policy,
        'seed': seed,
        'total_gpus': total_gpus,
        'num_tasks': len(tasks),
        'demand_target': demand_target,
        'elapsed_seconds': round(elapsed, 1),
        'curve': curve,
    }

    print(f"  {policy:15s} seed={seed}: "
          f"{len(curve)} points, "
          f"rejected={curve[-1]['rejected'] if curve else 0}, "
          f"{elapsed:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description='FGD paper replication evaluation')
    parser.add_argument('--node-csv', required=True,
                        help='Path to openb_node_list_gpu_node.csv')
    parser.add_argument('--pod-csv', required=True,
                        help='Path to openb_pod_list_default.csv')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    parser.add_argument('--policies', nargs='+', default=ALL_POLICIES,
                        choices=ALL_POLICIES,
                        help='Policies to evaluate')
    parser.add_argument('--demand-target', type=float, default=1.3,
                        help='Target demand fraction (default: 1.3)')
    parser.add_argument('--output-dir', default='results/',
                        help='Output directory for results')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--record-interval', type=int, default=10,
                        help='Record curve point every N tasks (default: 10)')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit number of trace tasks loaded')

    args = parser.parse_args()

    # Parse trace data
    print("Loading trace data...")
    base_nodes = parse_node_list(args.node_csv)
    trace_tasks = parse_alibaba_trace(args.pod_csv, max_tasks=args.max_tasks)
    non_gpu_stats = count_non_gpu_tasks(args.pod_csv)
    distribution = derive_workload_distribution(trace_tasks, non_gpu_stats)

    total_gpus = sum(n.num_gpus for n in base_nodes)
    print(f"  Nodes: {len(base_nodes)}, Total GPUs: {total_gpus}")
    print(f"  Tasks: {len(trace_tasks)} GPU + {non_gpu_stats['count']} non-GPU")
    print(f"  Distribution ({len(distribution)} buckets):")
    for entry in distribution:
        gpu, cpu, pop = entry[0], entry[1], entry[-1]
        print(f"    {gpu:.3f} GPU, {cpu:.1f} CPU: {pop:.1%}")

    # Build experiment list
    experiments = []
    for policy in args.policies:
        for seed in args.seeds:
            experiments.append((
                policy, seed, base_nodes, trace_tasks, distribution,
                args.demand_target, args.record_interval,
            ))

    print(f"\nRunning {len(experiments)} experiments "
          f"({len(args.policies)} policies x {len(args.seeds)} seeds)...")

    # Run experiments
    if args.workers == 1:
        results = [run_single_experiment(exp) for exp in experiments]
    else:
        with Pool(processes=args.workers) as pool:
            results = pool.map(run_single_experiment, experiments)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual results per policy
    by_policy = {}
    for r in results:
        policy = r['policy']
        if policy not in by_policy:
            by_policy[policy] = []
        by_policy[policy].append(r)

    for policy, policy_results in by_policy.items():
        out_path = os.path.join(args.output_dir, f'{policy}.json')
        with open(out_path, 'w') as f:
            json.dump(policy_results, f, indent=2)
        print(f"Saved {out_path}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, 'all_results.json')
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {combined_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
