#!/usr/bin/env python3
"""
Evaluate FGD and baseline scheduling policies on Alibaba production traces.

Uses the event-driven simulator to replay Alibaba v2023 traces with real
job arrival/departure timestamps, measuring fragmentation rate, utilization,
and scheduling success over time.

Usage:
    # Quick test (subset)
    python run_alibaba_experiment.py --max-tasks 500 --cluster-scale 10

    # Full run
    python run_alibaba_experiment.py

    # Custom schedulers
    python run_alibaba_experiment.py --schedulers fgd,random,bestfit
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict

from simulator import EventDrivenSimulator, SimulationMetrics, JobRecord, Node
from trace_loader import AlibabaTraceLoader, print_trace_statistics
from schedulers import (
    Scheduler, FGDScheduler, WindowedFGDScheduler,
    ClusteringScheduler, get_scheduler, get_all_schedulers,
)


def run_single(loader, scheduler, tasks, task_distribution,
               sample_interval, show_progress):
    """Run a single scheduler on the trace."""
    # Create fresh cluster
    cluster = loader.create_cluster(task_distribution)

    # Reset stateful schedulers
    if isinstance(scheduler, ClusteringScheduler):
        scheduler.reset()
    if isinstance(scheduler, WindowedFGDScheduler):
        scheduler.reset()
        for t in tasks[:scheduler.window_size]:
            scheduler.observe_task(t)

    sim = EventDrivenSimulator(cluster, tasks)
    results = sim.run(
        scheduler=scheduler,
        sample_interval_pct=sample_interval,
        show_progress=show_progress,
    )
    return results, sim.job_records


def build_schedulers(names, task_distribution, window_size):
    """Build scheduler instances from names."""
    schedulers = []
    for name in names:
        name_lower = name.lower().strip()
        if name_lower == 'w-fgd' or name_lower.startswith('w-fgd'):
            s = WindowedFGDScheduler(window_size=window_size)
            schedulers.append(s)
        elif name_lower == 'fgd-full':
            s = FGDScheduler(scheduling_task_types=task_distribution.get_task_types())
            s.name = "FGD-Full"
            schedulers.append(s)
        else:
            schedulers.append(get_scheduler(name_lower))
    return schedulers


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FGD on Alibaba GPU traces (event-driven simulation)")
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Path to trace data directory")
    parser.add_argument("--schedulers",
                        default="random,bestfit,dotprod,packing,clustering,fgd",
                        help="Comma-separated scheduler names")
    parser.add_argument("--sample-interval", type=float, default=5.0,
                        help="Fragmentation sampling interval as %% of GPU capacity")
    parser.add_argument("--cluster-scale", type=float, default=100.0,
                        help="Use N%% of original cluster nodes")
    parser.add_argument("--window-size", type=int, default=500,
                        help="Sliding window size for W-FGD")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "alibaba_results"),
                        help="Results output directory")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Limit number of tasks (0 = all)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Override cluster to N homogeneous GPUs (4 per node). "
                             "0 = use trace cluster")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    print("=" * 60)
    print("FGD Evaluation on Alibaba Production Traces")
    print("=" * 60)

    # Load trace
    loader = AlibabaTraceLoader(args.data_dir)
    loader.load_nodes()
    loader.load_tasks(gpu_only=True)

    print_trace_statistics(loader)

    # Cluster sizing
    if args.num_gpus > 0:
        # Create homogeneous cluster with specified GPU count
        gpus_per_node = 4
        num_nodes = (args.num_gpus + gpus_per_node - 1) // gpus_per_node
        loader.nodes = [
            Node(node_id=i, total_cpu=96.0, num_gpus=gpus_per_node,
                 gpu_model='GPU', memory_mib=262144)
            for i in range(num_nodes)
        ]
        actual_gpus = num_nodes * gpus_per_node
        print(f"\nCustom cluster: {num_nodes} nodes, {actual_gpus} GPUs "
              f"({gpus_per_node} per node)")
    elif args.cluster_scale < 100.0:
        print(f"\nScaling cluster to {args.cluster_scale}%...")
        loader.nodes = loader.scale_cluster(loader.nodes, args.cluster_scale)
        print(f"  Scaled to {len(loader.nodes)} nodes, "
              f"{sum(n.num_gpus for n in loader.nodes)} GPUs")

    # Limit tasks if requested
    tasks = loader.tasks
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
        print(f"\nLimited to first {len(tasks)} tasks")

    # Compute task distribution from full trace (for FGD scheduling decisions)
    task_distribution = loader.compute_task_distribution()
    total_gpus = sum(n.num_gpus for n in loader.nodes)

    print(f"\nCluster: {len(loader.nodes)} nodes, {total_gpus} GPUs")
    print(f"Tasks: {len(tasks)}")
    print(f"Task distribution: {len(task_distribution.distribution)} types")

    # Build schedulers
    scheduler_names = [s.strip() for s in args.schedulers.split(",")]
    schedulers = build_schedulers(scheduler_names, task_distribution, args.window_size)

    print(f"Schedulers: {[s.name for s in schedulers]}")
    print(f"Sample interval: {args.sample_interval}%")
    print()

    # Run experiments
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for scheduler in schedulers:
        print(f"--- {scheduler.name} ---")
        t0 = time.time()

        # Feed tasks to windowed scheduler
        if isinstance(scheduler, WindowedFGDScheduler):
            for t in tasks[:scheduler.window_size]:
                scheduler.observe_task(t)

        results, job_records = run_single(
            loader, scheduler, tasks, task_distribution,
            args.sample_interval, not args.no_progress,
        )
        elapsed = time.time() - t0

        # Summary
        final = results[-1] if results else None
        completed_jcts = [r.jct for r in job_records if r.jct >= 0]
        avg_jct = sum(completed_jcts) / len(completed_jcts) if completed_jcts else 0
        if final:
            print(f"  Frag={final.fragmentation_rate:.1f}%  "
                  f"Util={final.utilization_pct:.1f}%  "
                  f"Scheduled={final.tasks_scheduled}/{final.tasks_arrived}  "
                  f"Queued={final.tasks_queued}  "
                  f"AvgJCT={avg_jct:.0f}s  "
                  f"Time={elapsed:.1f}s")

        all_results[scheduler.name] = {
            'curve': [asdict(m) for m in results],
            'summary': asdict(final) if final else {},
            'wall_time_sec': elapsed,
            'job_records': [asdict(r) for r in job_records],
        }
        print()

    # Save results
    config = {
        'nodes': len(loader.nodes),
        'total_gpus': total_gpus,
        'tasks': len(tasks),
        'cluster_scale': args.cluster_scale,
        'sample_interval': args.sample_interval,
        'window_size': args.window_size,
    }

    # JSON
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump({'config': config, 'results': all_results}, f, indent=2)
    print(f"Results saved to {json_path}")

    # CSV (fragmentation curves)
    csv_path = os.path.join(args.output_dir, "fragmentation_curves.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scheduler', 'sim_time', 'arrived_workload_pct', 'active_workload_pct',
            'fragmentation_rate', 'utilization_pct', 'tasks_arrived',
            'tasks_scheduled', 'tasks_completed', 'tasks_failed', 'tasks_active',
            'tasks_queued', 'avg_jct',
        ])
        for sched_name, data in all_results.items():
            for m in data['curve']:
                writer.writerow([
                    sched_name,
                    m['sim_time'],
                    m['arrived_workload_pct'],
                    m['active_workload_pct'],
                    m['fragmentation_rate'],
                    m['utilization_pct'],
                    m['tasks_arrived'],
                    m['tasks_scheduled'],
                    m['tasks_completed'],
                    m['tasks_failed'],
                    m['tasks_active'],
                    m.get('tasks_queued', 0),
                    m.get('avg_jct', 0),
                ])
    print(f"Curves saved to {csv_path}")

    # CSV (per-job records with JCT)
    jobs_csv_path = os.path.join(args.output_dir, "job_records.csv")
    with open(jobs_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scheduler', 'task_id', 'gpu_demand', 'creation_time',
            'trace_duration', 'placement_time', 'completion_time',
            'wait_time', 'jct', 'frag_at_placement',
        ])
        for sched_name, data in all_results.items():
            for rec in data['job_records']:
                jct = rec['completion_time'] - rec['creation_time'] if rec['completion_time'] >= 0 else -1
                wait = rec['placement_time'] - rec['creation_time'] if rec['placement_time'] >= 0 else -1
                writer.writerow([
                    sched_name,
                    rec['task_id'],
                    rec['gpu_demand'],
                    rec['creation_time'],
                    rec['trace_duration'],
                    rec['placement_time'],
                    rec['completion_time'],
                    wait,
                    jct,
                    rec['frag_at_placement'],
                ])
    print(f"Job records saved to {jobs_csv_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'Scheduler':<16} {'Frag%':>8} {'Util%':>8} {'Sched':>8} {'Queued':>8} "
          f"{'AvgJCT':>10} {'AvgWait':>10} {'Time':>8}")
    print("-" * 90)
    for sched_name, data in all_results.items():
        s = data['summary']
        records = data['job_records']
        completed = [r for r in records if r['completion_time'] >= 0]
        avg_jct = sum(r['completion_time'] - r['creation_time'] for r in completed) / len(completed) if completed else 0
        waited = [r for r in completed if r['placement_time'] > r['creation_time']]
        avg_wait = sum(r['placement_time'] - r['creation_time'] for r in waited) / len(waited) if waited else 0
        if s:
            print(f"{sched_name:<16} {s['fragmentation_rate']:>8.1f} "
                  f"{s['utilization_pct']:>8.1f} {s['tasks_scheduled']:>8} "
                  f"{s.get('tasks_queued', 0):>8} "
                  f"{avg_jct:>9.0f}s {avg_wait:>9.0f}s "
                  f"{data['wall_time_sec']:>7.1f}s")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    sys.exit(main())
