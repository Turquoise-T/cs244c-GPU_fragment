"""
Experiment 2: Replicates Figure 9 from Section 6.2

Figure 9 has 4 sub-plots:
  9(a): Unallocated GPU % vs arrived workloads (80-120% range)
  9(b): Occupied nodes vs arrived workloads (0-100% range)
  9(c): Failed tasks by GPU request category at 96% arrival (bar chart)
  9(d): Fragmentation breakdown into 3 causes (bar chart)

Methodology: Monte-Carlo workload inflation (Section 6.1)
  - Sample tasks from trace with replacement
  - Submit until cumulative GPU requests reach target (120%)
  - Repeat N runs, average results
"""

import os
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from simulator import Task, Node, Cluster, TaskDistribution
from schedulers import (
    Scheduler, get_all_schedulers,
    ClusteringScheduler, FGDScheduler
)
from trace_loader import AlibabaTraceLoader


@dataclass
class Figure9Result:
    """Results from a single Monte-Carlo run for Figure 9"""
    scheduler_name: str
    # 9(a): (arrived_pct, unallocated_gpu_pct)
    unalloc_curve: List[Tuple[float, float]] = field(default_factory=list)
    # 9(b): (arrived_pct, occupied_nodes)
    occupied_curve: List[Tuple[float, int]] = field(default_factory=list)
    # 9(c): failed tasks at 96% by GPU category: {category: sum_of_gpu_demand}
    failed_by_category: Dict[str, float] = field(default_factory=dict)
    # 9(d): fragmentation breakdown: {cause: pct}
    frag_breakdown: Dict[str, float] = field(default_factory=dict)
    # Summary
    tasks_scheduled: int = 0
    tasks_failed: int = 0


class Figure9Experiment:
    """
    Replicates Figure 9: Multi-metric evaluation using Monte-Carlo workload inflation.

    Tasks are randomly sampled from the trace with replacement and submitted
    until cumulative GPU requests reach max_arrival_pct% of cluster capacity.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.loader = AlibabaTraceLoader(data_dir)
        self.loader.load_nodes()
        self.loader.load_tasks()

        self.task_distribution = self.loader.compute_task_distribution()
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded trace: {len(self.loader.nodes)} nodes, {self.total_gpu_capacity} GPUs")
        print(f"Tasks in trace: {len(self.loader.tasks)}")
        print(f"Task types: {len(self.task_distribution.get_task_types())}")

    def create_fresh_cluster(self) -> Cluster:
        """Create a fresh cluster from trace nodes"""
        cluster = Cluster()
        for i, orig_node in enumerate(self.loader.nodes):
            node = Node(
                node_id=i,
                total_cpu=orig_node.total_cpu,
                num_gpus=orig_node.num_gpus,
                name=orig_node.name,
                gpu_model=orig_node.gpu_model,
                memory_mib=orig_node.memory_mib
            )
            cluster.add_node(node)
        cluster.set_task_distribution(self.task_distribution)
        return cluster

    @staticmethod
    def _gpu_category(gpu_demand: float) -> str:
        """Categorize GPU demand for Figure 9(c)"""
        if gpu_demand < 1:
            return "<1"
        elif gpu_demand == 1:
            return "1"
        elif gpu_demand == 2:
            return "2"
        else:
            return "8"

    @staticmethod
    def _compute_frag_breakdown(cluster: Cluster) -> Dict[str, float]:
        """
        Decompose fragmentation into 3 causes for Figure 9(d).
        Uses per-task-type statistical measure from Section 3.2, Figure 4b.

        For each node n and task type m (weighted by popularity p_m):
        - Deficient (Q-I/Q-II): D_m^GPU > R_n^GPU — insufficient GPU capacity
        - Stranded (Q-IV): D_m^GPU <= R_n^GPU but D_m^CPU > R_n^CPU — GPU stranded by CPU shortage
        - Non-GPU (Case 3): D_m^GPU = 0 — all unallocated GPUs wasted
        - Q-III (node can run task, individual GPUs too small): counted as Deficient
        """
        if cluster.task_distribution is None:
            return {"deficient": 0, "stranded": 0, "non_gpu": 0}

        task_types = cluster.task_distribution.get_task_types()

        deficient = 0.0
        stranded = 0.0
        non_gpu = 0.0

        for node in cluster.nodes:
            if node.total_unallocated_gpu == 0:
                continue

            for (cpu_demand, gpu_demand), popularity in task_types:
                task = Task(task_id=-1, cpu_demand=cpu_demand, gpu_demand=gpu_demand)
                frag = node.get_fragmentation_for_task(task)

                if frag == 0:
                    continue

                weighted_frag = popularity * frag

                if gpu_demand == 0:
                    # Case 3: Non-GPU task
                    non_gpu += weighted_frag
                elif gpu_demand > node.scalar_gpu_capacity:
                    # Q-I/Q-II: Insufficient GPU → Deficient
                    deficient += weighted_frag
                elif cpu_demand > node.remaining_cpu:
                    # Q-IV: Has GPU but not enough CPU → Stranded
                    stranded += weighted_frag
                else:
                    # Q-III: Can run task but individual GPUs too small → Deficient
                    deficient += weighted_frag

        total = deficient + stranded + non_gpu
        if total == 0:
            return {"deficient": 0, "stranded": 0, "non_gpu": 0}

        return {
            "deficient": deficient / total * 100,
            "stranded": stranded / total * 100,
            "non_gpu": non_gpu / total * 100,
        }

    @staticmethod
    def _count_occupied_nodes(cluster: Cluster) -> int:
        """Count nodes with at least one task (any allocation)"""
        count = 0
        for node in cluster.nodes:
            if node.allocated_cpu > 0 or any(g < 1.0 for g in node.gpu_remaining):
                count += 1
        return count

    def run_single(
        self,
        scheduler: Scheduler,
        seed: int = 42,
        sample_interval_pct: float = 2.0,
        max_arrival_pct: float = 120.0,
        snapshot_pct: float = 96.0,
        show_progress: bool = True
    ) -> Figure9Result:
        """
        Run a single Monte-Carlo experiment.

        Args:
            seed: Random seed for sampling
            sample_interval_pct: How often to record metrics
            max_arrival_pct: Stop when cumulative GPU requests reach this % of capacity
            snapshot_pct: Record failed task breakdown at this arrival %
        """
        from tqdm import tqdm

        rng = random.Random(seed)
        cluster = self.create_fresh_cluster()
        result = Figure9Result(scheduler_name=scheduler.name)

        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        cumulative_gpu_demand = 0.0
        next_sample_pct = sample_interval_pct
        max_gpu_demand = self.total_gpu_capacity * max_arrival_pct / 100.0
        task_pool = self.loader.tasks

        # Track ALL failed tasks up to snapshot point
        all_failed_tasks: List[Task] = []
        snapshot_taken = False

        task_count = 0
        estimated_tasks = int(len(task_pool) * max_arrival_pct / 100.0)

        pbar = tqdm(
            total=estimated_tasks,
            desc=f"{scheduler.name:12} s={seed}",
            unit="task",
            disable=not show_progress,
            ncols=90
        )

        while cumulative_gpu_demand < max_gpu_demand:
            # Sample a random task from the trace
            orig_task = rng.choice(task_pool)
            task = Task(
                task_id=task_count,
                cpu_demand=orig_task.cpu_demand,
                gpu_demand=orig_task.gpu_demand
            )
            task_count += 1
            cumulative_gpu_demand += task.gpu_demand
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100

            # Try to schedule
            if scheduler.schedule(task, cluster):
                result.tasks_scheduled += 1
            else:
                result.tasks_failed += 1
                if not snapshot_taken:
                    all_failed_tasks.append(task)

            # Take snapshot at snapshot_pct (all failed tasks up to this point)
            if not snapshot_taken and arrived_pct >= snapshot_pct:
                snapshot_taken = True
                cat_sums: Dict[str, float] = defaultdict(float)
                for t in all_failed_tasks:
                    cat = self._gpu_category(t.gpu_demand)
                    cat_sums[cat] += t.gpu_demand
                result.failed_by_category = dict(cat_sums)

            # Record metrics at intervals
            if arrived_pct >= next_sample_pct:
                unalloc_pct = (cluster.total_unallocated_gpu / self.total_gpu_capacity) * 100
                result.unalloc_curve.append((next_sample_pct, unalloc_pct))

                occupied = self._count_occupied_nodes(cluster)
                result.occupied_curve.append((next_sample_pct, occupied))

                next_sample_pct += sample_interval_pct

            pbar.update(1)

        pbar.close()

        # 9(d): Fragmentation breakdown at end
        result.frag_breakdown = self._compute_frag_breakdown(cluster)

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
        num_runs: int = 10,
        seed: int = 42,
        sample_interval_pct: float = 2.0,
        max_arrival_pct: float = 120.0,
        show_progress: bool = True
    ) -> Dict[str, List[Figure9Result]]:
        """Run Monte-Carlo experiment for multiple schedulers, multiple runs."""
        if schedulers is None:
            schedulers = get_all_schedulers()

        results: Dict[str, List[Figure9Result]] = {s.name: [] for s in schedulers}

        for scheduler in schedulers:
            for run_idx in range(num_runs):
                run_seed = seed + run_idx
                result = self.run_single(
                    scheduler,
                    seed=run_seed,
                    sample_interval_pct=sample_interval_pct,
                    max_arrival_pct=max_arrival_pct,
                    show_progress=show_progress
                )
                results[scheduler.name].append(result)

            # Print average for this scheduler
            avg_sched = sum(r.tasks_scheduled for r in results[scheduler.name]) / num_runs
            avg_fail = sum(r.tasks_failed for r in results[scheduler.name]) / num_runs
            print(f"  {scheduler.name:12}: Avg Scheduled={avg_sched:.0f}, Avg Failed={avg_fail:.0f}")

        return results


def plot_figure9(results: Dict[str, List[Figure9Result]], total_nodes: int,
                 total_gpu: int, output_dir: str = None):
    """Plot all 4 sub-figures of Figure 9."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed. Skipping plot.")
        return

    styles = {
        'Random': {'color': 'gray', 'linestyle': '--', 'marker': 'o'},
        'DotProd': {'color': 'blue', 'linestyle': '-.', 'marker': 's'},
        'Clustering': {'color': 'green', 'linestyle': ':', 'marker': '^'},
        'Packing': {'color': 'orange', 'linestyle': '-', 'marker': 'D'},
        'BestFit': {'color': 'purple', 'linestyle': '--', 'marker': 'v'},
        'FGD': {'color': 'red', 'linestyle': '-', 'marker': 'x'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 9(a): Unallocated GPU % ---
    ax = axes[0, 0]
    # Ideal line: max(0, 100 - arrived_pct)
    ideal_x = list(range(80, 125, 2))
    ideal_y = [max(0, 100 - x) for x in ideal_x]
    ax.plot(ideal_x, ideal_y, color='gray', linestyle=':', linewidth=1.5, label='Ideal')

    for name, result_list in results.items():
        # Average curves across runs
        all_curves = [r.unalloc_curve for r in result_list]
        avg = _average_curves(all_curves)
        if avg:
            x_vals = [p[0] for p in avg]
            y_vals = [p[1] for p in avg]
            # Filter to 80-120% range
            filtered = [(x, y) for x, y in zip(x_vals, y_vals) if 80 <= x <= 120]
            if filtered:
                style = styles.get(name, {'color': 'black', 'linestyle': '-', 'marker': '.'})
                ax.plot([p[0] for p in filtered], [p[1] for p in filtered],
                        label=name, color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], markersize=4, markevery=2)

    ax.set_xlabel('Arrived workloads (% of GPU capacity)')
    ax.set_ylabel('Unalloc. GPU (%)')
    ax.set_title('(a) Unallocated GPUs')
    ax.legend(fontsize=8)
    ax.set_xlim(80, 120)
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)

    # --- 9(b): Occupied nodes ---
    ax = axes[0, 1]
    for name, result_list in results.items():
        all_curves = [r.occupied_curve for r in result_list]
        avg = _average_curves(all_curves)
        if avg:
            x_vals = [p[0] for p in avg]
            y_vals = [p[1] for p in avg]
            filtered = [(x, y) for x, y in zip(x_vals, y_vals) if x <= 100]
            if filtered:
                style = styles.get(name, {'color': 'black', 'linestyle': '-', 'marker': '.'})
                ax.plot([p[0] for p in filtered], [p[1] for p in filtered],
                        label=name, color=style['color'], linestyle=style['linestyle'],
                        marker=style['marker'], markersize=4, markevery=2)

    ax.set_xlabel('Arrived workloads (% of GPU capacity)')
    ax.set_ylabel('Occupied nodes')
    ax.set_title('(b) Occupied Nodes')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total_nodes + 50)
    ax.grid(True, alpha=0.3)

    # Fixed scheduler order for bar charts (9c, 9d)
    bar_order = ['FGD', 'BestFit', 'Packing', 'Clustering', 'DotProd', 'Random']
    scheduler_names = [n for n in bar_order if n in results]
    x_pos = np.arange(len(scheduler_names))

    # --- 9(c): Failed tasks by GPU category (stacked bar) ---
    ax = axes[1, 0]
    categories = ['<1', '1', '2', '8']
    cat_colors = {'<1': '#4e79a7', '1': '#f28e2b', '2': '#e15759', '8': '#76b7b2'}

    bottoms = np.zeros(len(scheduler_names))
    for cat in categories:
        values = []
        for name in scheduler_names:
            avg_val = sum(r.failed_by_category.get(cat, 0) for r in results[name]) / len(results[name])
            values.append(avg_val)
        ax.bar(x_pos, values, bottom=bottoms, label=f'GPU {cat}', color=cat_colors[cat])
        bottoms += np.array(values)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(scheduler_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sum of Requesting Task GPUs')
    ax.set_title('(c) Failed Tasks at 96% Arrival')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- 9(d): Fragmentation breakdown (stacked bar) ---
    ax = axes[1, 1]
    causes = ['deficient', 'stranded', 'non_gpu']
    cause_colors = {'deficient': '#4e79a7', 'stranded': '#f28e2b', 'non_gpu': '#e15759'}
    cause_labels = {'deficient': 'Deficient', 'stranded': 'Stranded', 'non_gpu': 'Non-GPU'}

    bottoms = np.zeros(len(scheduler_names))
    for cause in causes:
        values = []
        for name in scheduler_names:
            avg_val = sum(r.frag_breakdown.get(cause, 0) for r in results[name]) / len(results[name])
            values.append(avg_val)
        ax.bar(x_pos, values, bottom=bottoms, label=cause_labels[cause],
               color=cause_colors[cause])
        bottoms += np.array(values)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(scheduler_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fragmented GPUs (%)')
    ax.set_title('(d) Fragmentation Breakdown')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Figure 9: Scheduling Evaluation (Monte-Carlo)', fontsize=14)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'figure9.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path}")

    plt.show()


def _average_curves(curves: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Average multiple curves by x-value."""
    if not curves:
        return []
    by_x: Dict[float, List[float]] = defaultdict(list)
    for curve in curves:
        for x, y in curve:
            by_x[x].append(y)
    return sorted([(x, sum(ys) / len(ys)) for x, ys in by_x.items()])


def save_results_to_csv(results: Dict[str, List[Figure9Result]], output_dir: str):
    """Save all Figure 9 data to CSV files."""
    import csv

    # 9(a): Unallocated GPU curves
    path = os.path.join(output_dir, 'figure9a_unalloc.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scheduler', 'arrived_pct', 'unalloc_gpu_pct', 'run'])
        for name, result_list in results.items():
            for run_idx, r in enumerate(result_list):
                for x, y in r.unalloc_curve:
                    writer.writerow([name, x, y, run_idx])

    # 9(b): Occupied nodes curves
    path = os.path.join(output_dir, 'figure9b_occupied.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scheduler', 'arrived_pct', 'occupied_nodes', 'run'])
        for name, result_list in results.items():
            for run_idx, r in enumerate(result_list):
                for x, y in r.occupied_curve:
                    writer.writerow([name, x, y, run_idx])

    # 9(c): Failed task breakdown
    path = os.path.join(output_dir, 'figure9c_failed.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scheduler', 'gpu_category', 'sum_gpu_demand', 'run'])
        for name, result_list in results.items():
            for run_idx, r in enumerate(result_list):
                for cat, val in r.failed_by_category.items():
                    writer.writerow([name, cat, val, run_idx])

    # 9(d): Fragmentation breakdown
    path = os.path.join(output_dir, 'figure9d_breakdown.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scheduler', 'cause', 'pct', 'run'])
        for name, result_list in results.items():
            for run_idx, r in enumerate(result_list):
                for cause, pct in r.frag_breakdown.items():
                    writer.writerow([name, cause, pct, run_idx])

    print(f"CSV files saved to {output_dir}")


def format_summary(results: Dict[str, List[Figure9Result]]) -> str:
    """Format summary statistics"""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT SUMMARY (Figure 9)")
    lines.append("=" * 70)
    lines.append(f"\n{'Scheduler':<12} {'Scheduled':>10} {'Failed':>8} "
                 f"{'Unalloc%':>9} {'Occupied':>9} {'Deficient':>10} {'Stranded':>9} {'Non-GPU':>8}")
    lines.append("-" * 80)

    for name, result_list in results.items():
        n = len(result_list)
        avg_sched = sum(r.tasks_scheduled for r in result_list) / n
        avg_fail = sum(r.tasks_failed for r in result_list) / n

        # Final unallocated %
        avg_unalloc = 0
        for r in result_list:
            if r.unalloc_curve:
                avg_unalloc += r.unalloc_curve[-1][1]
        avg_unalloc /= n

        # Final occupied nodes
        avg_occupied = 0
        for r in result_list:
            if r.occupied_curve:
                avg_occupied += r.occupied_curve[-1][1]
        avg_occupied /= n

        # Frag breakdown
        avg_def = sum(r.frag_breakdown.get('deficient', 0) for r in result_list) / n
        avg_str = sum(r.frag_breakdown.get('stranded', 0) for r in result_list) / n
        avg_ng = sum(r.frag_breakdown.get('non_gpu', 0) for r in result_list) / n

        lines.append(f"{name:<12} {avg_sched:>10.0f} {avg_fail:>8.0f} "
                     f"{avg_unalloc:>9.1f} {avg_occupied:>9.0f} "
                     f"{avg_def:>10.1f} {avg_str:>9.1f} {avg_ng:>8.1f}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Figure 9 Replication - Monte-Carlo")
    parser.add_argument('--num-runs', type=int, default=10, help='Number of Monte-Carlo runs (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed (default: 42)')
    parser.add_argument('--sample-interval', type=float, default=2.0, help='Sampling interval %% (default: 2)')
    parser.add_argument('--max-arrival', type=float, default=120.0, help='Max arrival %% (default: 120)')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'alibaba_traces', 'cluster-trace-gpu-v2023')

    print("=" * 60)
    print("Figure 9 Replication - Monte-Carlo Workload Inflation")
    print(f"  runs={args.num_runs}, seed={args.seed}")
    print(f"  interval={args.sample_interval}%, max_arrival={args.max_arrival}%")
    print("=" * 60)

    experiment = Figure9Experiment(data_dir)
    schedulers = get_all_schedulers()

    results = experiment.run_experiment(
        schedulers=schedulers,
        num_runs=args.num_runs,
        seed=args.seed,
        sample_interval_pct=args.sample_interval,
        max_arrival_pct=args.max_arrival
    )

    # Create result directory
    result_name = f"exp2-runs{args.num_runs}-seed{args.seed}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)
    os.makedirs(result_dir, exist_ok=True)

    # Print & save summary
    summary = format_summary(results)
    print("\n" + summary)

    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write("Experiment: Figure 9 Replication - Monte-Carlo\n")
        f.write(f"Result: {result_name}\n")
        f.write(f"Runs: {args.num_runs}\n")
        f.write(f"Base seed: {args.seed}\n")
        f.write(f"Sample interval: {args.sample_interval}%\n")
        f.write(f"Max arrival: {args.max_arrival}%\n")
        f.write(f"Nodes: {len(experiment.loader.nodes)}\n")
        f.write(f"GPUs: {experiment.total_gpu_capacity}\n")
        f.write(f"Task pool: {len(experiment.loader.tasks)}\n\n")
        f.write(summary + "\n")
    print(f"Summary log saved to {log_path}")

    # Save CSV data
    save_results_to_csv(results, result_dir)

    # Plot
    plot_figure9(results, len(experiment.loader.nodes),
                 experiment.total_gpu_capacity, result_dir)
