"""
Experiment Runner for FGD Replication

Replicates Figure 7(a): Fragmentation rate vs arrived workloads
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

    Methodology (Monte-Carlo Workload Inflation from Section 6.1):
    - Randomly sample tasks from trace with replacement
    - Submit tasks until cumulative GPU requests reach target % of cluster capacity
    - Track fragmentation rate at regular intervals
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
        sample_interval_pct: float = 5.0
    ) -> ExperimentResult:
        """
        Run a single experiment with one scheduler.

        Args:
            scheduler: The scheduling policy to use
            max_workload_pct: Stop when arrived workload reaches this % of GPU capacity
            sample_interval_pct: Record fragmentation every N% of arrived workload

        Returns:
            ExperimentResult with fragmentation curve
        """
        cluster = self.create_fresh_cluster()
        result = ExperimentResult(scheduler_name=scheduler.name)

        # Reset clustering scheduler state if needed
        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        cumulative_gpu_demand = 0.0
        task_id = 0
        next_sample_pct = sample_interval_pct

        while True:
            # Sample a task
            task = self.sample_task(task_id)
            cumulative_gpu_demand += task.gpu_demand
            task_id += 1

            # Calculate arrived workload percentage
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100

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

            # Stop condition
            if arrived_pct >= max_workload_pct:
                break

        # Record final metrics
        result.final_frag_rate = cluster.compute_fragmentation_rate()
        result.final_gpu_alloc_rate = cluster.gpu_allocation_rate

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
        num_runs: int = 10,
        max_workload_pct: float = 120.0,
        sample_interval_pct: float = 5.0
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run experiments for multiple schedulers with multiple runs.

        Args:
            schedulers: List of schedulers to test (default: all)
            num_runs: Number of runs per scheduler for averaging
            max_workload_pct: Stop when arrived workload reaches this %
            sample_interval_pct: Record fragmentation every N%

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
                    sample_interval_pct=sample_interval_pct
                )
                results[scheduler.name].append(result)

                print(f"  {scheduler.name:12}: Frag={result.final_frag_rate:.1f}%, "
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
        'Random': {'color': 'gray', 'linestyle': '--', 'marker': 'o'},
        'DotProd': {'color': 'blue', 'linestyle': '-.', 'marker': 's'},
        'Clustering': {'color': 'green', 'linestyle': ':', 'marker': '^'},
        'Packing': {'color': 'orange', 'linestyle': '-', 'marker': 'D'},
        'BestFit': {'color': 'purple', 'linestyle': '--', 'marker': 'v'},
        'FGD': {'color': 'red', 'linestyle': '-', 'marker': 'x'},
    }

    for name, result_list in results.items():
        avg_curve = Figure7aExperiment.average_curves(result_list)
        if avg_curve:
            x_vals = [p[0] for p in avg_curve]
            y_vals = [p[1] for p in avg_curve]

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


def print_summary(results: Dict[str, List[ExperimentResult]]):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\n{'Scheduler':<12} {'Avg Frag%':>10} {'Avg Alloc%':>12} {'Scheduled':>12} {'Failed':>10}")
    print("-" * 60)

    for name, result_list in results.items():
        avg_frag = sum(r.final_frag_rate for r in result_list) / len(result_list)
        avg_alloc = sum(r.final_gpu_alloc_rate for r in result_list) / len(result_list)
        total_scheduled = sum(r.tasks_scheduled for r in result_list) / len(result_list)
        total_failed = sum(r.tasks_failed for r in result_list) / len(result_list)

        print(f"{name:<12} {avg_frag:>10.1f} {avg_alloc:>12.1f} {total_scheduled:>12.0f} {total_failed:>10.0f}")


if __name__ == "__main__":
    # Run the experiment
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    print("=" * 60)
    print("Figure 7(a) Replication Experiment")
    print("=" * 60)

    experiment = Figure7aExperiment(data_dir, seed=42)

    # Run with all schedulers
    results = experiment.run_experiment(
        num_runs=3,  # Use 3 runs for quick test (paper uses 10)
        max_workload_pct=120.0,
        sample_interval_pct=5.0
    )

    # Print summary
    print_summary(results)

    # Plot results
    output_path = os.path.join(os.path.dirname(__file__), 'figure7a.png')
    plot_figure7a(results, output_path)
