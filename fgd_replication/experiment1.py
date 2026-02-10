"""
Experiment Runner for FGD Replication

Replicates Figure 7(a): Fragmentation rate vs arrived workloads
Using Monte-Carlo workload inflation approach from Section 6.1
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter

from simulator import Task, Node, Cluster, TaskDistribution
from schedulers import (
    Scheduler, get_all_schedulers, get_scheduler,
    ClusteringScheduler, FGDScheduler, WindowedFGDScheduler
)
from trace_loader import AlibabaTraceLoader


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    scheduler_name: str
    # List of (arrived_workload_pct, fragmentation_rate) tuples
    fragmentation_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Final metrics
    final_frag_rate: float = 0.0
    final_gpu_alloc_rate: float = 0.0
    tasks_scheduled: int = 0
    tasks_failed: int = 0


class Figure7aExperiment:
    """
    Replicates Figure 7(a): Fragmentation rate grows to 100% as more resources are allocated.

    Trace Replay mode: processes tasks in creation_time order from the trace.
    """

    def __init__(self, data_dir: str, window_size: int = 500, task_order: str = 'trace',
                 cluster_scale: float = 100.0):
        self.data_dir = data_dir
        self.window_size = window_size
        self.task_order = task_order
        self.cluster_scale = cluster_scale
        self.loader = AlibabaTraceLoader(data_dir)

        # Load trace data
        self.loader.load_nodes()
        self.loader.load_tasks()

        # Reduce cluster size if requested
        self.original_node_count = len(self.loader.nodes)
        if cluster_scale < 100.0:
            self.loader.nodes = self._scale_cluster(self.loader.nodes, cluster_scale)
        self.scaled_node_count = len(self.loader.nodes)

        # Sort tasks by type if requested (creates skewed arrival pattern)
        if task_order != 'trace':
            reverse = (task_order == 'descending')
            self.loader.tasks.sort(
                key=lambda t: (round(t.gpu_demand, 2), round(t.cpu_demand / 4) * 4),
                reverse=reverse
            )

        # Compute TWO task distributions:
        # 1. Full distribution (all tasks) - for original FGD baseline
        # 2. First-N distribution - for penalized FGD and W-FGD starting point
        self.full_task_distribution = self.loader.compute_task_distribution()
        self.first_n_distribution = self._compute_initial_distribution(window_size)

        # Get cluster capacity
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded trace: {len(self.loader.nodes)} nodes, {self.total_gpu_capacity} GPUs")
        print(f"Tasks in trace: {len(self.loader.tasks)}")
        print(f"Task order: {task_order}")
        print(f"Full distribution: {len(self.full_task_distribution.get_task_types())} task types")
        print(f"First-{window_size} distribution: {len(self.first_n_distribution.get_task_types())} task types")
        print("\n" + self.format_distribution_comparison() + "\n")

    def _compute_initial_distribution(self, n: int) -> TaskDistribution:
        """Compute task distribution from first N tasks of the trace"""
        subset = self.loader.tasks[:n]
        dist = TaskDistribution()
        type_counts: Counter = Counter()
        for task in subset:
            gpu_rounded = round(task.gpu_demand, 2)
            cpu_bucket = round(task.cpu_demand / 4) * 4
            type_counts[(cpu_bucket, gpu_rounded)] += 1
        total = sum(type_counts.values())
        for (cpu, gpu), count in type_counts.items():
            dist.add_task_type(cpu, gpu, count / total)
        return dist

    def format_distribution_comparison(self) -> str:
        """Format side-by-side comparison of full vs first-N distributions"""
        n = self.window_size
        full_types = {(cpu, gpu): pop for (cpu, gpu), pop in self.full_task_distribution.get_task_types()}
        first_n_types = {(cpu, gpu): pop for (cpu, gpu), pop in self.first_n_distribution.get_task_types()}

        all_keys = sorted(set(full_types) | set(first_n_types), key=lambda k: (-full_types.get(k, 0)))

        lines = []
        lines.append(f"{'Task Type (cpu,gpu)':<22} {'Full%':>8} {'First-'+str(n)+'%':>10} {'Diff':>8}")
        lines.append("-" * 52)
        for cpu, gpu in all_keys:
            f = full_types.get((cpu, gpu), 0) * 100
            p = first_n_types.get((cpu, gpu), 0) * 100
            diff = p - f
            marker = " *" if abs(diff) > 3 else ""
            lines.append(f"  ({cpu:>4}, {gpu:>5})       {f:>7.1f}  {p:>9.1f}  {diff:>+7.1f}{marker}")
        lines.append(f"  {'Types present:':<20} {len(full_types):>7}  {len(first_n_types):>9}")
        return "\n".join(lines)

    @staticmethod
    def _scale_cluster(nodes: list, scale_pct: float) -> list:
        """Keep scale_pct% of nodes per type (same cpu, mem, gpu count, gpu model). At least 1 per type."""
        from collections import defaultdict
        import math

        # Group nodes by type
        type_groups: dict = defaultdict(list)
        for node in nodes:
            key = (node.total_cpu, node.memory_mib, node.num_gpus, node.gpu_model)
            type_groups[key].append(node)

        scaled_nodes = []
        print(f"\nCluster scaling to {scale_pct}%:")
        for key, group in sorted(type_groups.items()):
            keep = max(1, math.ceil(len(group) * scale_pct / 100.0))
            scaled_nodes.extend(group[:keep])
            print(f"  {key}: {len(group)} -> {keep}")

        print(f"  Total: {len(nodes)} -> {len(scaled_nodes)} nodes")
        return scaled_nodes

    def create_fresh_cluster(self) -> Cluster:
        """Create a fresh cluster with full distribution (always used for evaluation)"""
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

        # Always use full distribution for evaluation (compute_fragmentation_rate)
        cluster.set_task_distribution(self.full_task_distribution)
        return cluster

    def run_single(
        self,
        scheduler: Scheduler,
        sample_interval_pct: float = 5.0,
        show_progress: bool = True
    ) -> ExperimentResult:
        """
        Run a single trace-replay experiment with one scheduler.

        Tasks are processed in creation_time order until the trace is exhausted.
        Cluster always uses full distribution for evaluation.
        Schedulers carry their own scheduling distribution if needed.
        """
        from tqdm import tqdm

        cluster = self.create_fresh_cluster()
        result = ExperimentResult(scheduler_name=scheduler.name)

        tasks = self.loader.tasks  # already sorted by creation_time

        # Reset stateful schedulers
        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()
        if isinstance(scheduler, WindowedFGDScheduler):
            scheduler.reset()
            # Pre-populate window with first N tasks
            for t in tasks[:scheduler.window_size]:
                scheduler.observe_task(t)

        cumulative_gpu_demand = 0.0
        next_sample_pct = sample_interval_pct

        pbar = tqdm(
            total=len(tasks),
            desc=f"{scheduler.name:16}",
            unit="task",
            disable=not show_progress,
            ncols=90
        )

        for task in tasks:
            cumulative_gpu_demand += task.gpu_demand
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100

            # Feed task to windowed scheduler before scheduling
            if isinstance(scheduler, WindowedFGDScheduler):
                scheduler.observe_task(task)

            # Try to schedule
            if scheduler.schedule(task, cluster):
                result.tasks_scheduled += 1
            else:
                result.tasks_failed += 1

            # Record fragmentation at intervals
            if arrived_pct >= next_sample_pct:
                frag_rate = cluster.compute_fragmentation_rate()
                result.fragmentation_curve.append((next_sample_pct, frag_rate))
                next_sample_pct += sample_interval_pct

            pbar.update(1)

        pbar.close()

        # Record final metrics
        result.final_frag_rate = cluster.compute_fragmentation_rate()
        result.final_gpu_alloc_rate = cluster.gpu_allocation_rate

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
        sample_interval_pct: float = 5.0,
        show_progress: bool = True
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run trace-replay experiment for multiple schedulers.
        Single run per scheduler (deterministic — no random sampling).
        """
        if schedulers is None:
            schedulers = get_all_schedulers()

        results: Dict[str, List[ExperimentResult]] = {s.name: [] for s in schedulers}

        for scheduler in schedulers:
            result = self.run_single(
                scheduler,
                sample_interval_pct=sample_interval_pct,
                show_progress=show_progress
            )
            results[scheduler.name].append(result)

            print(f"  {scheduler.name:16}: Frag={result.final_frag_rate:.1f}%, "
                  f"Alloc={result.final_gpu_alloc_rate:.1f}%, "
                  f"Scheduled={result.tasks_scheduled}, Failed={result.tasks_failed}")

        return results


def plot_figure7a(results: Dict[str, List[ExperimentResult]], output_path: str = None):
    """
    Plot Figure 7(a): Fragmentation rate vs arrived workloads.

    Args:
        results: Dict mapping scheduler name to list of results
        output_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))

    # Color and style mapping to match paper
    styles = {
        'Random': {'color': 'gray', 'linestyle': '--', 'marker': 'o'},
        'DotProd': {'color': 'blue', 'linestyle': '-.', 'marker': 's'},
        'Clustering': {'color': 'green', 'linestyle': ':', 'marker': '^'},
        'Packing': {'color': 'orange', 'linestyle': '-', 'marker': 'D'},
        'BestFit': {'color': 'purple', 'linestyle': '--', 'marker': 'v'},
        'FGD-Full': {'color': 'red', 'linestyle': '-', 'marker': 'x'},
    }

    for name, result_list in results.items():
        # Single run in replay mode, just use its curve directly
        curve = result_list[0].fragmentation_curve if result_list else []
        if curve:
            x_vals = [p[0] for p in curve]
            y_vals = [p[1] for p in curve]

            # Match known styles
            if name.startswith('W-FGD'):
                style = {'color': 'darkgreen', 'linestyle': '-', 'marker': '*'}
            elif name.startswith('FGD-') and name != 'FGD-Full':
                style = {'color': 'crimson', 'linestyle': '--', 'marker': 'P'}
            else:
                style = styles.get(name, {'color': 'black', 'linestyle': '-', 'marker': '.'})
            plt.plot(x_vals, y_vals, label=name,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=4,
                    markevery=2)

    plt.xlabel('Arrived workloads (in % of cluster GPU capacity)', fontsize=12)
    plt.ylabel('Frag Rate (%)', fontsize=12)
    plt.title('Figure 7(a): Fragmentation Rate vs Arrived Workloads', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 120)
    plt.ylim(0, 100)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    plt.show()


def save_results_to_csv(results: Dict[str, List[ExperimentResult]], output_path: str):
    """
    Save experiment results to CSV file.

    CSV format: scheduler,arrived_workload_pct,frag_rate,run
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scheduler', 'arrived_workload_pct', 'frag_rate', 'run'])

        for scheduler_name, result_list in results.items():
            for run_idx, result in enumerate(result_list):
                for arrived_pct, frag_rate in result.fragmentation_curve:
                    writer.writerow([scheduler_name, arrived_pct, frag_rate, run_idx])

    print(f"Results saved to {output_path}")


def load_results_from_csv(csv_path: str) -> Dict[str, List[ExperimentResult]]:
    """
    Load experiment results from CSV file.

    Returns:
        Dict mapping scheduler name to list of ExperimentResult
    """
    import csv
    from collections import defaultdict

    # Temporary storage: scheduler -> run -> [(x, y), ...]
    data = defaultdict(lambda: defaultdict(list))

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheduler = row['scheduler']
            run = int(row['run'])
            x = float(row['arrived_workload_pct'])
            y = float(row['frag_rate'])
            data[scheduler][run].append((x, y))

    # Convert to ExperimentResult objects
    results = {}
    for scheduler, runs in data.items():
        results[scheduler] = []
        for run_idx in sorted(runs.keys()):
            result = ExperimentResult(scheduler_name=scheduler)
            result.fragmentation_curve = sorted(runs[run_idx], key=lambda p: p[0])
            results[scheduler].append(result)

    return results


def format_summary(results: Dict[str, List[ExperimentResult]]) -> str:
    """Format summary statistics as a string"""
    lines = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\n{'Scheduler':<16} {'Avg Frag%':>10} {'Avg Alloc%':>12} {'Scheduled':>12} {'Failed':>10}")
    lines.append("-" * 64)

    for name, result_list in results.items():
        avg_frag = sum(r.final_frag_rate for r in result_list) / len(result_list)
        avg_alloc = sum(r.final_gpu_alloc_rate for r in result_list) / len(result_list)
        total_scheduled = sum(r.tasks_scheduled for r in result_list) / len(result_list)
        total_failed = sum(r.tasks_failed for r in result_list) / len(result_list)

        lines.append(f"{name:<16} {avg_frag:>10.1f} {avg_alloc:>12.1f} {total_scheduled:>12.0f} {total_failed:>10.0f}")

    return "\n".join(lines)


def print_summary(results: Dict[str, List[ExperimentResult]]):
    """Print summary statistics"""
    print("\n" + format_summary(results))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Figure 7(a) Replication - Trace Replay")
    parser.add_argument('--sample-interval', type=float, default=5.0, help='Fragmentation sampling interval %% (default: 5)')
    parser.add_argument('--window-size', type=int, default=500, help='Sliding window size for W-FGD (default: 500)')
    parser.add_argument('--task-order', choices=['trace', 'ascending', 'descending'], default='trace',
                        help='Task arrival order: trace (original), ascending/descending (sorted by type)')
    parser.add_argument('--cluster-scale', type=float, default=100.0,
                        help='Cluster size as %% of original (e.g., 50 keeps 50%% of each node type)')
    args = parser.parse_args()

    # Run the experiment
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'alibaba_traces', 'cluster-trace-gpu-v2023')

    print("=" * 60)
    print("Figure 7(a) Replication - Trace Replay")
    print(f"  mode=replay, interval={args.sample_interval}%")
    print(f"  window_size={args.window_size}")
    print(f"  task_order={args.task_order}")
    print(f"  cluster_scale={args.cluster_scale}%")
    print("=" * 60)

    experiment = Figure7aExperiment(
        data_dir, window_size=args.window_size,
        task_order=args.task_order, cluster_scale=args.cluster_scale
    )

    # Build schedulers:
    # - Baselines: don't use FGD, distribution irrelevant
    # - FGD-Full: original FGD, schedules with full trace distribution (perfect knowledge)
    # - FGD-N: penalized FGD, schedules with first-N distribution only
    # - W-FGD-N: windowed FGD, schedules with sliding window of last N tasks
    # All evaluated with full distribution (cluster always has full dist)

    # FGD-Full: no override → uses cluster's full distribution for scheduling
    fgd_full = FGDScheduler()
    fgd_full.name = "FGD-Full"

    # FGD-N: override scheduling distribution to first-N only
    fgd_n = FGDScheduler(
        scheduling_task_types=experiment.first_n_distribution.get_task_types()
    )
    fgd_n.name = f"FGD-{args.window_size}"

    w_fgd = WindowedFGDScheduler(window_size=args.window_size)

    # Non-FGD baselines only (exclude FGD from get_all_schedulers)
    baselines = [s for s in get_all_schedulers() if not isinstance(s, FGDScheduler)]

    schedulers = baselines + [fgd_full, fgd_n, w_fgd]

    # Run trace replay
    results = experiment.run_experiment(
        schedulers=schedulers,
        sample_interval_pct=args.sample_interval
    )

    # Create result directory
    scale_str = f"{args.cluster_scale:g}"
    result_name = f"exp1-{args.window_size}-{args.task_order}-{scale_str}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)
    os.makedirs(result_dir, exist_ok=True)

    # Print summary
    print_summary(results)

    # Save summary log
    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write(f"Experiment: Figure 7(a) Replication - Trace Replay\n")
        f.write(f"Result: {result_name}\n")
        f.write(f"Mode: replay\n")
        f.write(f"Sample interval: {args.sample_interval}%\n")
        f.write(f"Window size (W-FGD): {args.window_size}\n")
        f.write(f"Task order: {args.task_order}\n")
        f.write(f"Cluster: {experiment.original_node_count} nodes -> {experiment.scaled_node_count} nodes ({args.cluster_scale}%)\n")
        f.write(f"Full distribution: {len(experiment.full_task_distribution.get_task_types())} task types\n")
        f.write(f"First-{args.window_size} distribution: {len(experiment.first_n_distribution.get_task_types())} task types\n\n")
        f.write(experiment.format_distribution_comparison() + "\n\n")
        f.write(format_summary(results) + "\n")
    print(f"Summary log saved to {log_path}")

    # Save results to CSV
    csv_path = os.path.join(result_dir, 'figure7a_results.csv')
    save_results_to_csv(results, csv_path)

    # Plot results
    plot_path = os.path.join(result_dir, 'figure7a.png')
    plot_figure7a(results, plot_path)
