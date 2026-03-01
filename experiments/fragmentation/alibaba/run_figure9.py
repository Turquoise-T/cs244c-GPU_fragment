#!/usr/bin/env python3
"""
Recreate Figure 9 from the FGD paper (ATC'23, page 11).

Uses Monte-Carlo workload inflation: tasks are randomly sampled WITH
REPLACEMENT from the Alibaba trace and submitted to the cluster until
cumulative GPU requests exceed a threshold. No departures. Repeated
across multiple trials and averaged.

Figure 9 subplots:
  (a) Unallocatable GPU (%) vs arrived workloads (%)
  (b) Occupied nodes vs arrived workloads (%)
  (c) GPU requests of failed tasks at 96% capacity (stacked bar)
  (d) Fragmentation breakdown into 3 causes (stacked bar)

Usage:
    # Quick test
    python3 run_figure9.py --num-trials 2 --max-tasks 500 --num-gpus 32

    # Full experiment (matches paper setup)
    python3 run_figure9.py --num-trials 10

    # Custom schedulers
    python3 run_figure9.py --schedulers fgd,random,bestfit --num-trials 5
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

from simulator import Task, Node, Cluster, TaskDistribution
from trace_loader import AlibabaTraceLoader
from schedulers import (
    Scheduler, FGDScheduler, ClusteringScheduler,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# Policy styling (consistent with plot_results.py)
# ---------------------------------------------------------------------------

POLICY_STYLES = {
    'FGD':        {'color': '#DC143C', 'marker': 'x', 'linestyle': '-'},
    'BestFit':    {'color': '#9400D3', 'marker': 'v', 'linestyle': '--'},
    'Packing':    {'color': '#FF8C00', 'marker': 'D', 'linestyle': '-'},
    'Clustering': {'color': '#228B22', 'marker': '^', 'linestyle': ':'},
    'DotProd':    {'color': '#1E90FF', 'marker': 's', 'linestyle': '-.'},
    'Random':     {'color': '#808080', 'marker': 'o', 'linestyle': '-.'},
}

# Bar chart order matches paper: FGD first (leftmost)
POLICY_ORDER = ['FGD', 'BestFit', 'Packing', 'Clustering', 'DotProd', 'Random']
# Legend order for line plots: Random at top, FGD at bottom (matches paper Fig 9a)
LEGEND_ORDER = ['Random', 'DotProd', 'Clustering', 'Packing', 'BestFit', 'FGD']


def get_style(name):
    if name in POLICY_STYLES:
        return POLICY_STYLES[name]
    return {'color': 'black', 'marker': '.', 'linestyle': '-'}


# ---------------------------------------------------------------------------
# Workload inflation simulator
# ---------------------------------------------------------------------------

@dataclass
class InflationSnapshot:
    """Metrics at a point during workload inflation."""
    arrived_workload_pct: float
    unalloc_gpu_pct: float
    occupied_nodes: int
    total_failed_gpu: float          # cumulative GPU demand of failed tasks
    failed_by_type: Dict[str, float] # GPU demand of failed tasks by category


def count_occupied_nodes(cluster: Cluster) -> int:
    """Count nodes with at least one GPU allocation."""
    count = 0
    for node in cluster.nodes:
        if any(g < 1.0 for g in node.gpu_remaining):
            count += 1
    return count


def compute_fragmentation_breakdown(
    cluster: Cluster,
    task_distribution: TaskDistribution,
) -> Tuple[float, float, float]:
    """Decompose F_N(M) into three fragmentation causes.

    Returns (non_gpu_frac, stranded_frac, deficient_frac) as percentages
    of total unallocated GPU capacity.

    - non_gpu: contribution from task types requesting 0 GPUs
    - stranded: GPU tasks that can't run due to CPU shortage
    - deficient: GPU tasks where GPU slots are insufficient
    """
    task_types = task_distribution.get_task_types()
    total_unalloc = cluster.total_unallocated_gpu
    if total_unalloc <= 0:
        return (0.0, 0.0, 0.0)

    f_non_gpu = 0.0
    f_stranded = 0.0
    f_deficient = 0.0

    for node in cluster.nodes:
        node_unalloc = node.total_unallocated_gpu
        if node_unalloc <= 0:
            continue

        for (cpu_demand, gpu_demand), popularity in task_types:
            dummy = Task(task_id=-1, cpu_demand=cpu_demand, gpu_demand=gpu_demand)
            frag_for_task = node.get_fragmentation_for_task(dummy)
            weighted = popularity * frag_for_task

            if gpu_demand == 0:
                # Non-GPU task: all unallocated GPU is fragmented from its POV
                f_non_gpu += weighted
            elif node.remaining_cpu < cpu_demand:
                # GPU task blocked by CPU shortage → stranded
                f_stranded += weighted
            else:
                # GPU task blocked by insufficient GPU slots → deficient
                f_deficient += weighted

    # Convert to percentages of total unallocated GPU
    non_gpu_pct = (f_non_gpu / total_unalloc) * 100
    stranded_pct = (f_stranded / total_unalloc) * 100
    deficient_pct = (f_deficient / total_unalloc) * 100

    return (non_gpu_pct, stranded_pct, deficient_pct)


def categorize_gpu_demand(gpu_demand: float) -> str:
    """Categorize a task's GPU demand for the stacked bar chart."""
    if gpu_demand >= 8:
        return '8'
    elif gpu_demand >= 4:
        return '4'
    elif gpu_demand >= 2:
        return '2'
    elif gpu_demand >= 1:
        return '1'
    else:
        return '<1'


def run_workload_inflation(
    loader: AlibabaTraceLoader,
    tasks: List[Task],
    scheduler: Scheduler,
    task_distribution: TaskDistribution,
    max_workload_pct: float = 130.0,
    sample_interval_pct: float = 1.0,
) -> Tuple[List[InflationSnapshot], Dict[str, float], Tuple[float, float, float]]:
    """Run a single workload inflation trial.

    Returns:
        snapshots: metrics at each sample point
        failed_by_type_at_96: GPU demand of failed tasks at 96% workload
        frag_breakdown_at_96: (non_gpu, stranded, deficient) at 96% workload
    """
    cluster = loader.create_cluster(task_distribution)
    total_gpu = cluster.total_gpu_capacity

    snapshots: List[InflationSnapshot] = []
    failed_tasks: List[Task] = []
    cumulative_gpu_demand = 0.0
    next_sample_pct = sample_interval_pct
    task_counter = 0

    # Snapshot at 96% for subplots (c) and (d)
    failed_by_type_at_96: Dict[str, float] = defaultdict(float)
    frag_breakdown_at_96 = (0.0, 0.0, 0.0)
    captured_96 = False

    while (cumulative_gpu_demand / total_gpu) * 100 < max_workload_pct:
        # Sample task with replacement
        src_task = random.choice(tasks)
        task_copy = Task(
            task_id=task_counter,
            cpu_demand=src_task.cpu_demand,
            gpu_demand=src_task.gpu_demand,
            name=src_task.name,
            gpu_spec=src_task.gpu_spec,
        )
        task_counter += 1
        cumulative_gpu_demand += task_copy.gpu_demand

        success = scheduler.schedule(task_copy, cluster)
        if not success:
            failed_tasks.append(task_copy)

        arrived_pct = (cumulative_gpu_demand / total_gpu) * 100

        # Capture 96% snapshot for subplots (c) and (d)
        if not captured_96 and arrived_pct >= 96.0:
            for ft in failed_tasks:
                cat = categorize_gpu_demand(ft.gpu_demand)
                failed_by_type_at_96[cat] += ft.gpu_demand
            frag_breakdown_at_96 = compute_fragmentation_breakdown(
                cluster, task_distribution)
            captured_96 = True

        # Sample at intervals
        if arrived_pct >= next_sample_pct:
            unalloc_pct = (cluster.total_unallocated_gpu / total_gpu) * 100
            occupied = count_occupied_nodes(cluster)

            # Failed tasks by type
            fbt = defaultdict(float)
            for ft in failed_tasks:
                cat = categorize_gpu_demand(ft.gpu_demand)
                fbt[cat] += ft.gpu_demand

            snapshots.append(InflationSnapshot(
                arrived_workload_pct=arrived_pct,
                unalloc_gpu_pct=unalloc_pct,
                occupied_nodes=occupied,
                total_failed_gpu=sum(ft.gpu_demand for ft in failed_tasks),
                failed_by_type=dict(fbt),
            ))
            next_sample_pct = arrived_pct + sample_interval_pct

    return snapshots, dict(failed_by_type_at_96), frag_breakdown_at_96


# ---------------------------------------------------------------------------
# Monte-Carlo experiment
# ---------------------------------------------------------------------------

def run_monte_carlo(
    loader: AlibabaTraceLoader,
    tasks: List[Task],
    schedulers: List[Scheduler],
    task_distribution: TaskDistribution,
    num_trials: int = 10,
    max_workload_pct: float = 130.0,
    base_seed: int = 42,
) -> Dict:
    """Run Monte-Carlo workload inflation for all schedulers.

    Returns dict: scheduler_name -> {
        'curves': {workload_pct -> {'unalloc_gpu_pct': [values], 'occupied_nodes': [values]}},
        'failed_at_96': [{type -> gpu_demand}],
        'frag_breakdown_at_96': [(non_gpu, stranded, deficient)],
    }
    """
    results = {}

    for scheduler in schedulers:
        name = scheduler.name
        print(f"\n{'='*60}")
        print(f"  {name}  ({num_trials} trials)")
        print(f"{'='*60}")

        trial_snapshots = []
        trial_failed_96 = []
        trial_frag_96 = []

        for trial in range(num_trials):
            random.seed(base_seed + trial)
            np.random.seed(base_seed + trial)

            # Reset stateful schedulers
            if isinstance(scheduler, ClusteringScheduler):
                scheduler.reset()

            t0 = time.time()
            snapshots, failed_96, frag_96 = run_workload_inflation(
                loader, tasks, scheduler, task_distribution,
                max_workload_pct=max_workload_pct,
            )
            elapsed = time.time() - t0

            trial_snapshots.append(snapshots)
            trial_failed_96.append(failed_96)
            trial_frag_96.append(frag_96)

            print(f"  Trial {trial+1}/{num_trials}: "
                  f"{len(snapshots)} samples, {elapsed:.1f}s")

        # Aggregate across trials: bin by arrived_workload_pct (1% bins)
        curves = defaultdict(lambda: {'unalloc_gpu_pct': [], 'occupied_nodes': []})

        for snapshots in trial_snapshots:
            for snap in snapshots:
                bin_pct = round(snap.arrived_workload_pct)
                curves[bin_pct]['unalloc_gpu_pct'].append(snap.unalloc_gpu_pct)
                curves[bin_pct]['occupied_nodes'].append(snap.occupied_nodes)

        results[name] = {
            'curves': dict(curves),
            'failed_at_96': trial_failed_96,
            'frag_breakdown_at_96': trial_frag_96,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure9(results: Dict, num_nodes: int, output_dir: str):
    """Generate Figure 9 with 4 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # Determine plot order
    ordered_names = [n for n in POLICY_ORDER if n in results]
    # Add any extra schedulers not in the predefined order
    for n in results:
        if n not in ordered_names:
            ordered_names.append(n)

    # --- Subplot (a): Unallocatable GPU (%) vs Arrived workloads ---
    # Draw in LEGEND_ORDER so FGD (last) draws on top
    draw_order = [n for n in LEGEND_ORDER if n in ordered_names]
    for n in ordered_names:
        if n not in draw_order:
            draw_order.append(n)
    y_max_a = 0
    for name in draw_order:
        data = results[name]
        style = get_style(name)
        curves = data['curves']

        pcts = sorted(curves.keys())
        x_vals = []
        y_mean = []
        y_std = []
        for pct in pcts:
            if pct < 78 or pct > 122:
                continue
            vals = curves[pct]['unalloc_gpu_pct']
            if vals:
                x_vals.append(pct)
                y_mean.append(np.mean(vals))
                y_std.append(np.std(vals))

        if x_vals:
            markevery = max(1, len(x_vals) // 12)
            ax_a.plot(x_vals, y_mean, label=name,
                      color=style['color'], linestyle=style['linestyle'],
                      marker=style['marker'], markersize=4,
                      markevery=markevery, linewidth=1.5)
            if max(y_std) > 0.01:
                ax_a.fill_between(x_vals,
                                  np.maximum(0, np.array(y_mean) - np.array(y_std)),
                                  np.array(y_mean) + np.array(y_std),
                                  alpha=0.07, color=style['color'])
            y_max_a = max(y_max_a, max(y_mean) * 1.15)

    # Ideal line: perfect packing means unalloc = max(0, 100 - arrived)
    ideal_x = list(range(78, 123))
    ideal_y = [max(0, 100 - x) for x in ideal_x]
    ax_a.plot(ideal_x, ideal_y, label='Ideal', color='gray',
              linestyle=':', linewidth=1.5)

    ax_a.set_xlabel('Arrived workloads (in % of cluster GPU capacity)')
    ax_a.set_ylabel('Unalloc. GPU (%)')
    ax_a.set_xlim(78, 122)
    ax_a.set_ylim(0, max(25, y_max_a))
    # Reorder legend: Random at top, FGD at bottom (matches paper)
    handles, labels = ax_a.get_legend_handles_labels()
    label_order = [n for n in LEGEND_ORDER if n in labels] + ['Ideal']
    ordered_handles = [handles[labels.index(l)] for l in label_order if l in labels]
    ordered_labels = [l for l in label_order if l in labels]
    ax_a.legend(ordered_handles, ordered_labels, fontsize=7, loc='upper right')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_title('(a) Unallocatable GPUs given arriving workloads')

    # --- Subplot (b): Occupied nodes vs Arrived workloads ---
    for name in draw_order:
        data = results[name]
        style = get_style(name)
        curves = data['curves']

        pcts = sorted(curves.keys())
        x_vals = []
        y_mean = []
        for pct in pcts:
            if pct > 105:
                continue
            vals = curves[pct]['occupied_nodes']
            if vals:
                x_vals.append(pct)
                y_mean.append(np.mean(vals))

        if x_vals:
            markevery = max(1, len(x_vals) // 15)
            ax_b.plot(x_vals, y_mean, label=name,
                      color=style['color'], linestyle=style['linestyle'],
                      marker=style['marker'], markersize=5,
                      markevery=markevery, linewidth=1.5)

    ax_b.set_xlabel('Arrived workloads (in % of cluster GPU capacity)')
    ax_b.set_ylabel('Occupied nodes')
    ax_b.set_xlim(0, 105)
    ax_b.set_ylim(0, num_nodes * 1.1)
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    order_b = [n for n in LEGEND_ORDER if n in labels_b]
    ax_b.legend([handles_b[labels_b.index(l)] for l in order_b if l in labels_b],
                [l for l in order_b if l in labels_b],
                fontsize=7, loc='lower right')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_title('(b) GPU nodes occupied during scheduling')

    # --- Subplot (c): GPU requests of failed tasks (stacked bar) ---
    gpu_categories = ['<1', '1', '2', '4', '8']
    cat_colors = {
        '<1': '#4472C4',
        '1':  '#ED7D31',
        '2':  '#A5A5A5',
        '4':  '#FFC000',
        '8':  '#5B9BD5',
    }

    bar_width = 0.6
    x_positions = np.arange(len(ordered_names))

    for name_idx, name in enumerate(ordered_names):
        data = results[name]
        # Average across trials
        avg_failed = defaultdict(float)
        n_trials = len(data['failed_at_96'])
        for trial_failed in data['failed_at_96']:
            for cat, val in trial_failed.items():
                avg_failed[cat] += val / n_trials

        bottom = 0.0
        for cat in gpu_categories:
            val = avg_failed.get(cat, 0.0)
            if val > 0 or name_idx == 0:  # draw all categories for legend
                ax_c.bar(x_positions[name_idx], val, bar_width,
                         bottom=bottom,
                         color=cat_colors.get(cat, 'gray'),
                         label=cat if name_idx == 0 else None,
                         edgecolor='white', linewidth=0.5)
            bottom += val

    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels(ordered_names, fontsize=8)
    ax_c.set_ylabel('Sum of Requesting Task GPUs')
    ax_c.set_title('(c) GPU requests of failed tasks (at 96% capacity)')
    # Legend with title
    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(handles, labels, title='Task GPU Req', fontsize=7,
                title_fontsize=7, loc='upper right')
    ax_c.grid(True, alpha=0.3, axis='y')

    # --- Subplot (d): Fragmentation breakdown (stacked bar) ---
    breakdown_colors = {
        'non-gpu':   '#228B22',   # green (matches paper)
        'stranded':  '#ED7D31',   # orange
        'deficient': '#A5A5A5',   # gray
    }

    for name_idx, name in enumerate(ordered_names):
        data = results[name]
        # Average across trials
        avg_non_gpu = np.mean([fb[0] for fb in data['frag_breakdown_at_96']])
        avg_stranded = np.mean([fb[1] for fb in data['frag_breakdown_at_96']])
        avg_deficient = np.mean([fb[2] for fb in data['frag_breakdown_at_96']])

        bottom = 0.0
        for label_name, val in [('non-gpu', avg_non_gpu),
                                ('stranded', avg_stranded),
                                ('deficient', avg_deficient)]:
            ax_d.bar(x_positions[name_idx], val, bar_width,
                     bottom=bottom,
                     color=breakdown_colors[label_name],
                     label=label_name if name_idx == 0 else None,
                     edgecolor='white', linewidth=0.5)
            bottom += val

    ax_d.set_xticks(x_positions)
    ax_d.set_xticklabels(ordered_names, fontsize=8)
    ax_d.set_ylabel('Fragmented GPUs (%)')
    ax_d.set_ylim(0, 100)
    ax_d.set_title('(d) Fragmentation breakdown into three causes')
    ax_d.legend(fontsize=7, loc='upper right')
    ax_d.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Figure 9: Performance comparison of FGD and baselines\n'
                 '(Monte-Carlo Workload Inflation on Alibaba Trace)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Save
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f'figure9.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recreate Figure 9 from FGD paper (Monte-Carlo Workload Inflation)")
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Path to trace data directory")
    parser.add_argument("--schedulers",
                        default="fgd,bestfit,packing,clustering,dotprod,random",
                        help="Comma-separated scheduler names")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of Monte-Carlo trials (default: 10)")
    parser.add_argument("--max-workload-pct", type=float, default=130.0,
                        help="Stop inflation at this %% of GPU capacity")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Override cluster to N homogeneous GPUs (4/node). "
                             "0 = use trace cluster")
    parser.add_argument("--cluster-scale", type=float, default=100.0,
                        help="Use N%% of original cluster nodes (preserves "
                             "heterogeneity). Default: 100%%")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Limit task pool size (0 = all)")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "figure9_results"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Figure 9: Monte-Carlo Workload Inflation")
    print("=" * 60)

    # Load trace
    loader = AlibabaTraceLoader(args.data_dir)
    loader.load_nodes()
    # Load ALL tasks (including non-GPU) for accurate distribution
    loader.load_tasks(gpu_only=False)

    total_tasks_loaded = len(loader.tasks)
    gpu_tasks = [t for t in loader.tasks if t.gpu_demand > 0]
    non_gpu_tasks = [t for t in loader.tasks if t.gpu_demand == 0]
    print(f"\nTrace: {total_tasks_loaded} tasks "
          f"({len(gpu_tasks)} GPU, {len(non_gpu_tasks)} non-GPU)")

    # Compute distribution from ALL tasks (matches paper Table 1)
    task_distribution = loader.compute_task_distribution()
    print(f"Task distribution: {len(task_distribution.distribution)} types")

    # For sampling during inflation, use ONLY GPU tasks.
    # Non-GPU tasks remain in the distribution M (for fragmentation
    # calculation in subplots a and d), but are not scheduled during
    # inflation -- they add 0 to cumulative GPU demand (x-axis) while
    # consuming CPU and inflating occupied node counts.
    sample_tasks = gpu_tasks
    if args.max_tasks > 0:
        sample_tasks = sample_tasks[:args.max_tasks]
        print(f"Limited task pool to {len(sample_tasks)} tasks")
    print(f"  GPU tasks in sample pool: {len(sample_tasks)}")

    # Cluster setup
    if args.num_gpus > 0:
        gpus_per_node = 4
        num_nodes = (args.num_gpus + gpus_per_node - 1) // gpus_per_node
        loader.nodes = [
            Node(node_id=i, total_cpu=96.0, num_gpus=gpus_per_node,
                 gpu_model='GPU', memory_mib=262144)
            for i in range(num_nodes)
        ]
        actual_gpus = num_nodes * gpus_per_node
        print(f"\nCustom cluster: {num_nodes} nodes, {actual_gpus} GPUs")
    elif args.cluster_scale < 100.0:
        loader.nodes = loader.scale_cluster(loader.nodes, args.cluster_scale)
        total = sum(n.num_gpus for n in loader.nodes)
        print(f"\nScaled cluster ({args.cluster_scale}%): "
              f"{len(loader.nodes)} nodes, {total} GPUs")
    else:
        print(f"\nTrace cluster: {len(loader.nodes)} nodes, "
              f"{sum(n.num_gpus for n in loader.nodes)} GPUs")

    total_gpus = sum(n.num_gpus for n in loader.nodes)
    num_nodes = len(loader.nodes)

    # Build schedulers
    scheduler_names = [s.strip() for s in args.schedulers.split(",")]
    schedulers = []
    for name in scheduler_names:
        name_lower = name.lower()
        if name_lower == 'fgd':
            s = FGDScheduler()
            schedulers.append(s)
        else:
            schedulers.append(get_scheduler(name_lower))

    print(f"Schedulers: {[s.name for s in schedulers]}")
    print(f"Trials: {args.num_trials}")
    print(f"Max workload: {args.max_workload_pct}%")
    print(f"Base seed: {args.seed}")

    # Run experiment
    os.makedirs(args.output_dir, exist_ok=True)
    t0_total = time.time()

    results = run_monte_carlo(
        loader, sample_tasks, schedulers, task_distribution,
        num_trials=args.num_trials,
        max_workload_pct=args.max_workload_pct,
        base_seed=args.seed,
    )

    elapsed_total = time.time() - t0_total
    print(f"\nTotal experiment time: {elapsed_total:.1f}s")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Scheduler':<14} {'Unalloc@100%':>14} {'Occupied@96%':>14} "
          f"{'Failed@96%':>12} {'Frag@96%':>10}")
    print(f"{'-'*80}")

    def lookup_nearest(curves, target_pct, field):
        """Find data from the bin nearest to target_pct."""
        if not curves:
            return []
        bins = sorted(curves.keys())
        nearest = min(bins, key=lambda b: abs(b - target_pct))
        return curves[nearest].get(field, [])

    for name in [s.name for s in schedulers]:
        data = results[name]
        curves = data['curves']

        # Unallocatable GPU at 100%
        unalloc_100 = lookup_nearest(curves, 100, 'unalloc_gpu_pct')
        unalloc_str = f"{np.mean(unalloc_100):.1f} +/- {np.std(unalloc_100):.1f}" if unalloc_100 else "N/A"

        # Occupied nodes at 96%
        occ_96 = lookup_nearest(curves, 96, 'occupied_nodes')
        occ_str = f"{np.mean(occ_96):.0f}" if occ_96 else "N/A"

        # Total failed GPU at 96%
        total_failed = [sum(f96.values()) for f96 in data['failed_at_96']]
        failed_str = f"{np.mean(total_failed):.0f}" if total_failed else "N/A"

        # Total fragmentation at 96%
        total_frag = [sum(fb) for fb in data['frag_breakdown_at_96']]
        frag_str = f"{np.mean(total_frag):.1f}%" if total_frag else "N/A"

        print(f"{name:<14} {unalloc_str:>14} {occ_str:>14} "
              f"{failed_str:>12} {frag_str:>10}")

    print(f"{'='*80}")

    # Save raw results
    json_path = os.path.join(args.output_dir, "figure9_results.json")
    # Convert numpy types to native python for JSON serialization
    serializable = {}
    for sched_name, data in results.items():
        serializable[sched_name] = {
            'curves': {
                str(k): {
                    'unalloc_gpu_pct': v['unalloc_gpu_pct'],
                    'occupied_nodes': v['occupied_nodes'],
                }
                for k, v in data['curves'].items()
            },
            'failed_at_96': data['failed_at_96'],
            'frag_breakdown_at_96': [list(fb) for fb in data['frag_breakdown_at_96']],
        }
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'num_nodes': num_nodes,
                'total_gpus': total_gpus,
                'num_trials': args.num_trials,
                'max_workload_pct': args.max_workload_pct,
                'seed': args.seed,
                'num_tasks_in_pool': len(sample_tasks),
            },
            'results': serializable,
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Plot
    print("\nGenerating Figure 9...")
    plot_figure9(results, num_nodes, args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
