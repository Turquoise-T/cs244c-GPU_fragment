"""
Experiment Runner for FGD Replication

Replicates Figure 7(a) and Figure 7(b):
  7(a): Fragmentation rate vs arrived workloads
  7(b): Fragmented GPUs / total resources vs arrived workloads
Using Monte-Carlo workload inflation approach from Section 6.1
"""

import os
import random
import copy
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from simulator import Task, Node, Cluster, TaskDistribution
from schedulers import (
    Scheduler, get_all_schedulers, get_scheduler,
    ClusteringScheduler
)
from trace_loader import AlibabaTraceLoader


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    scheduler_name: str
    # Figure 7(a): List of (arrived_workload_pct, fragmentation_rate) tuples
    fragmentation_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Figure 7(b): List of (arrived_workload_pct, fragmented/total_gpu_pct) tuples
    frag_total_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Final metrics
    final_frag_rate: float = 0.0
    final_frag_total_pct: float = 0.0
    final_gpu_alloc_rate: float = 0.0
    tasks_scheduled: int = 0
    tasks_failed: int = 0


class Figure7aExperiment:
    """
    Replicates Figure 7(a) and Figure 7(b).

    Methodology (Monte-Carlo Workload Inflation from Section 6.1):
    - Randomly sample tasks from trace with replacement
    - Submit tasks until cumulative GPU requests reach target % of cluster capacity
    - Track:
        Figure 7(a): fragmentation rate (% of unallocated GPUs)
        Figure 7(b): fragmented GPUs / total cluster GPUs (%)
    """

    def __init__(self, data_dir: str, seed: int = 42):
        """
        Initialize the experiment.

        Args:
            data_dir: Path to data directory with Alibaba trace
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.seed = seed
        self.loader = AlibabaTraceLoader(data_dir)

        # Load trace data
        self.loader.load_nodes()
        self.loader.load_tasks()

        # Compute task distribution from trace
        self.task_distribution = self.loader.compute_task_distribution()

        # Get cluster capacity
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded trace: {len(self.loader.nodes)} nodes, {self.total_gpu_capacity} GPUs")
        print(f"Task pool: {len(self.loader.tasks)} tasks")

    def create_fresh_cluster(self) -> Cluster:
        """Create a fresh cluster with the same nodes as the trace"""
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

    def sample_task(self, task_id: int) -> Task:
        """Sample a random task from the trace (with replacement)"""
        orig = random.choice(self.loader.tasks)
        return Task(
            task_id=task_id,
            cpu_demand=orig.cpu_demand,
            gpu_demand=orig.gpu_demand,
            name=f"sampled-{task_id}",
            gpu_spec=orig.gpu_spec
        )

    def run_single_experiment(
        self,
        scheduler: Scheduler,
        max_workload_pct: float = 120.0,
        sample_interval_pct: float = 5.0,
        show_progress: bool = True
    ) -> ExperimentResult:
        """
        Run a single experiment with one scheduler.

        Args:
            scheduler: The scheduling policy to use
            max_workload_pct: Stop when arrived workload reaches this % of GPU capacity
            sample_interval_pct: Record fragmentation every N% of arrived workload
            show_progress: Whether to show progress bar

        Returns:
            ExperimentResult with fragmentation curve
        """
        from tqdm import tqdm

        cluster = self.create_fresh_cluster()
        result = ExperimentResult(scheduler_name=scheduler.name)

        # Reset clustering scheduler state if needed
        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        cumulative_gpu_demand = 0.0
        task_id = 0
        next_sample_pct = sample_interval_pct

        # Estimate total tasks needed (based on average GPU demand)
        avg_gpu_demand = sum(t.gpu_demand for t in self.loader.tasks) / len(self.loader.tasks)
        estimated_tasks = int((max_workload_pct / 100) * self.total_gpu_capacity / max(avg_gpu_demand, 0.1))

        pbar = tqdm(
            total=int(max_workload_pct),
            desc=f"{scheduler.name:12}",
            unit="%",
            disable=not show_progress,
            ncols=80
        )

        last_pct = 0
        while True:
            # Sample a task
            task = self.sample_task(task_id)
            cumulative_gpu_demand += task.gpu_demand
            task_id += 1

            # Calculate arrived workload percentage
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100

            # Update progress bar
            if int(arrived_pct) > last_pct:
                pbar.update(int(arrived_pct) - last_pct)
                last_pct = int(arrived_pct)

            # Try to schedule
            if scheduler.schedule(task, cluster):
                result.tasks_scheduled += 1
            else:
                result.tasks_failed += 1

            # Record fragmentation at intervals
            if arrived_pct >= next_sample_pct:
                frag_rate = cluster.compute_fragmentation_rate()
                frag_total_pct = (
                    cluster.compute_cluster_fragmentation()
                    / cluster.total_gpu_capacity
                ) * 100.0
                result.fragmentation_curve.append((next_sample_pct, frag_rate))
                result.frag_total_curve.append((next_sample_pct, frag_total_pct))
                next_sample_pct += sample_interval_pct

            # Stop condition
            if arrived_pct >= max_workload_pct:
                break

        pbar.close()

        # Record final metrics
        result.final_frag_rate = cluster.compute_fragmentation_rate()
        result.final_frag_total_pct = (
            cluster.compute_cluster_fragmentation() / cluster.total_gpu_capacity
        ) * 100.0
        result.final_gpu_alloc_rate = cluster.gpu_allocation_rate

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
        num_runs: int = 10,
        max_workload_pct: float = 120.0,
        sample_interval_pct: float = 5.0,
        show_progress: bool = True
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run experiments for multiple schedulers with multiple runs.

        Args:
            schedulers: List of schedulers to test (default: all)
            num_runs: Number of runs per scheduler for averaging
            max_workload_pct: Stop when arrived workload reaches this %
            sample_interval_pct: Record fragmentation every N%
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping scheduler name to list of results
        """
        if schedulers is None:
            schedulers = get_all_schedulers()

        results: Dict[str, List[ExperimentResult]] = {s.name: [] for s in schedulers}

        for run in range(num_runs):
            # Set seed for this run (different seed per run, but consistent across schedulers)
            random.seed(self.seed + run)

            print(f"\n=== Run {run + 1}/{num_runs} ===")

            for scheduler in schedulers:
                # Reset seed to ensure same task sequence for fair comparison
                random.seed(self.seed + run)

                # Reset scheduler state
                if isinstance(scheduler, ClusteringScheduler):
                    scheduler.reset()

                result = self.run_single_experiment(
                    scheduler,
                    max_workload_pct=max_workload_pct,
                    sample_interval_pct=sample_interval_pct,
                    show_progress=show_progress
                )
                results[scheduler.name].append(result)

                print(f"  {scheduler.name:12}: Frag={result.final_frag_rate:.1f}%, "
                      f"Frag/Total={result.final_frag_total_pct:.1f}%, "
                      f"Alloc={result.final_gpu_alloc_rate:.1f}%, "
                      f"Scheduled={result.tasks_scheduled}, Failed={result.tasks_failed}")

        return results

    @staticmethod
    def average_curves(results: List[ExperimentResult]) -> List[Tuple[float, float]]:
        """Average fragmentation curves across multiple runs"""
        if not results:
            return []

        # Get all x-values
        all_x = set()
        for r in results:
            for x, _ in r.fragmentation_curve:
                all_x.add(x)

        # Average y-values for each x
        avg_curve = []
        for x in sorted(all_x):
            y_values = []
            for r in results:
                for rx, ry in r.fragmentation_curve:
                    if rx == x:
                        y_values.append(ry)
                        break
            if y_values:
                avg_curve.append((x, sum(y_values) / len(y_values)))

        return avg_curve


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
        'Random': {'color': 'brown', 'linestyle': '-.'},
        'DotProd': {'color': 'purple', 'linestyle': '--'},
        'Clustering': {'color': 'red', 'linestyle': '--'},
        'Packing': {'color': 'darkgreen', 'linestyle': ':'},
        'BestFit': {'color': 'orange', 'linestyle': '--'},
        'FGD': {'color': 'blue', 'linestyle': '-'},
    }

    for name, result_list in results.items():
        avg_curve = Figure7aExperiment.average_curves(result_list)
        if avg_curve:
            x_vals = [p[0] for p in avg_curve]
            y_vals = [p[1] for p in avg_curve]

            style = styles.get(name, {'color': 'black', 'linestyle': '-'})
            plt.plot(x_vals, y_vals, label=name,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=2)

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


def plot_figure7b(results: Dict[str, List[ExperimentResult]], output_path: str = None):
    """
    Plot Figure 7(b): Fragmented GPUs / total resources (%) vs arrived workloads.

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

    styles = {
        'Random': {'color': 'brown', 'linestyle': '-.'},
        'DotProd': {'color': 'purple', 'linestyle': '--'},
        'Clustering': {'color': 'red', 'linestyle': '--'},
        'Packing': {'color': 'darkgreen', 'linestyle': ':'},
        'BestFit': {'color': 'orange', 'linestyle': '--'},
        'FGD': {'color': 'blue', 'linestyle': '-'},
    }

    for name, result_list in results.items():
        # Average Figure 7(b) curve across runs
        all_x = set()
        for r in result_list:
            for x, _ in r.frag_total_curve:
                all_x.add(x)
        avg_curve = []
        for x in sorted(all_x):
            y_values = []
            for r in result_list:
                for rx, ry in r.frag_total_curve:
                    if rx == x:
                        y_values.append(ry)
                        break
            if y_values:
                avg_curve.append((x, sum(y_values) / len(y_values)))

        if avg_curve:
            x_vals = [p[0] for p in avg_curve]
            y_vals = [p[1] for p in avg_curve]

            style = styles.get(name, {'color': 'black', 'linestyle': '-'})
            plt.plot(x_vals, y_vals, label=name,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=2)

    plt.xlabel('Arrived workloads (in % of cluster GPU capacity)', fontsize=12)
    plt.ylabel('Frag / Total (%)', fontsize=12)
    plt.title('Figure 7(b): Fragmented GPUs / Total Resources', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 120)
    plt.ylim(bottom=0)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    plt.show()


def save_results_to_csv(results: Dict[str, List[ExperimentResult]], output_path: str):
    """
    Save experiment results to CSV file.

    CSV format: scheduler,arrived_workload_pct,frag_rate,frag_total_pct,run
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scheduler',
            'arrived_workload_pct',
            'frag_rate',
            'frag_total_pct',
            'run',
        ])

        for scheduler_name, result_list in results.items():
            for run_idx, result in enumerate(result_list):
                # Curves are sampled at the same x-values.
                by_x_total = {x: y for x, y in result.frag_total_curve}
                for arrived_pct, frag_rate in result.fragmentation_curve:
                    frag_total = by_x_total.get(arrived_pct, '')
                    writer.writerow([
                        scheduler_name,
                        arrived_pct,
                        frag_rate,
                        frag_total,
                        run_idx,
                    ])

    print(f"Results saved to {output_path}")


def load_results_from_csv(csv_path: str) -> Dict[str, List[ExperimentResult]]:
    """
    Load experiment results from CSV file.

    Returns:
        Dict mapping scheduler name to list of ExperimentResult
    """
    import csv
    from collections import defaultdict

    # Temporary storage:
    # scheduler -> run -> {'frag_rate': [(x, y)], 'frag_total': [(x, y)]}
    data = defaultdict(lambda: defaultdict(lambda: {
        'frag_rate': [],
        'frag_total': [],
    }))

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheduler = row['scheduler']
            run = int(row['run'])
            x = float(row['arrived_workload_pct'])
            y = float(row['frag_rate'])
            data[scheduler][run]['frag_rate'].append((x, y))

            # Backward compatibility: older CSVs may not have frag_total_pct.
            frag_total_raw = row.get('frag_total_pct', '')
            if frag_total_raw != '':
                y_total = float(frag_total_raw)
                data[scheduler][run]['frag_total'].append((x, y_total))

    # Convert to ExperimentResult objects
    results = {}
    for scheduler, runs in data.items():
        results[scheduler] = []
        for run_idx in sorted(runs.keys()):
            result = ExperimentResult(scheduler_name=scheduler)
            result.fragmentation_curve = sorted(
                runs[run_idx]['frag_rate'], key=lambda p: p[0])
            result.frag_total_curve = sorted(
                runs[run_idx]['frag_total'], key=lambda p: p[0])
            results[scheduler].append(result)

    return results


def format_summary(results: Dict[str, List[ExperimentResult]]) -> str:
    """Format summary statistics as a string"""
    lines = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\n{'Scheduler':<12} {'Avg Frag%':>10} {'Frag/Total%':>12} {'Avg Alloc%':>12} {'Scheduled':>12} {'Failed':>10}")
    lines.append("-" * 60)

    for name, result_list in results.items():
        avg_frag = sum(r.final_frag_rate for r in result_list) / len(result_list)
        avg_frag_total = sum(r.final_frag_total_pct for r in result_list) / len(result_list)
        avg_alloc = sum(r.final_gpu_alloc_rate for r in result_list) / len(result_list)
        total_scheduled = sum(r.tasks_scheduled for r in result_list) / len(result_list)
        total_failed = sum(r.tasks_failed for r in result_list) / len(result_list)

        lines.append(f"{name:<12} {avg_frag:>10.1f} {avg_frag_total:>12.1f} {avg_alloc:>12.1f} {total_scheduled:>12.0f} {total_failed:>10.0f}")

    return "\n".join(lines)


def print_summary(results: Dict[str, List[ExperimentResult]]):
    """Print summary statistics"""
    print("\n" + format_summary(results))


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Figure 7(a)/(b) Replication Experiment")
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per scheduler (default: 3, paper uses 10)')
    parser.add_argument('--max-workload', type=float, default=120.0, help='Max arrived workload %% (default: 120)')
    parser.add_argument('--sample-interval', type=float, default=5.0, help='Fragmentation sampling interval %% (default: 5)')
    parser.add_argument('--plot-csv', type=str, default=None,
                        help='Plot from existing CSV file instead of running experiment')
    parser.add_argument('--schedulers', type=str, default='all',
                        help='Comma-separated scheduler names to run (default: all). '
                             'Available: Random,BestFit,DotProd,Packing,Clustering,FGD')
    args = parser.parse_args()

    # Plot-only mode
    if args.plot_csv:
        results = load_results_from_csv(args.plot_csv)
        print(f"Loaded {len(results)} schedulers from {args.plot_csv}")
        plot_dir = os.path.dirname(args.plot_csv)
        plot_path = os.path.join(plot_dir, 'figure7a.png')
        plot_figure7a(results, plot_path)
        # Figure 7(b) can be drawn only from new CSVs that include frag_total_pct.
        if any(r.frag_total_curve for runs in results.values() for r in runs):
            plot_figure7b(results, os.path.join(plot_dir, 'figure7b.png'))
        else:
            print("INFO: CSV has no 'frag_total_pct' column; skipping Figure 7(b) plot.")
        exit(0)

    # Run the experiment
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'alibaba_traces', 'cluster-trace-gpu-v2023')

    print("=" * 60)
    print("Figure 7(a)/(b) Replication Experiment")
    print(f"  seed={args.seed}, runs={args.num_runs}, "
          f"max_workload={args.max_workload}%, interval={args.sample_interval}%")
    print("=" * 60)

    experiment = Figure7aExperiment(data_dir, seed=args.seed)

    # Filter schedulers
    all_sched_map = {s.name: s for s in get_all_schedulers()}
    if args.schedulers == 'all':
        schedulers = get_all_schedulers()
    else:
        selected = [s.strip() for s in args.schedulers.split(',')]
        schedulers = []
        for name in selected:
            if name in all_sched_map:
                schedulers.append(all_sched_map[name])
            else:
                print(f"WARNING: Unknown scheduler '{name}'. Available: {list(all_sched_map.keys())}")
        if not schedulers:
            print("No valid schedulers selected. Exiting.")
            exit(1)

    results = experiment.run_experiment(
        schedulers=schedulers,
        num_runs=args.num_runs,
        max_workload_pct=args.max_workload,
        sample_interval_pct=args.sample_interval
    )

    # Create result directory
    result_name = f"fig7-runs{args.num_runs}-seed{args.seed}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)
    os.makedirs(result_dir, exist_ok=True)

    # Print summary
    print_summary(results)

    # Save summary log
    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write(f"Experiment: Figure 7(a) Replication\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Num runs: {args.num_runs}\n")
        f.write(f"Max workload: {args.max_workload}%\n")
        f.write(f"Sample interval: {args.sample_interval}%\n\n")
        f.write(format_summary(results) + "\n")
    print(f"Summary log saved to {log_path}")

    # Save results to CSV
    csv_path = os.path.join(result_dir, 'figure7_results.csv')
    save_results_to_csv(results, csv_path)

    # Plot results
    plot_path = os.path.join(result_dir, 'figure7a.png')
    plot_figure7a(results, plot_path)
    plot_path_b = os.path.join(result_dir, 'figure7b.png')
    plot_figure7b(results, plot_path_b)
