"""
Experiment 3: Sensitivity Analysis (Figures 11-14)

Figure 11 (Section 6.3): Varying GPU-sharing task proportion
Figure 12 (Section 6.4): Varying multi-GPU task proportion
Figure 13 (Section 6.5): Varying GPU-type constrained task proportion
Figure 14 (Section 6.6): Varying non-GPU task proportion

Each figure loads pre-built trace files from the Alibaba cluster-trace-gpu-v2023
dataset that already encode the desired workload mix:
  Fig 11: openb_pod_list_gpushare{40,60,80,100}.csv
  Fig 12: openb_pod_list_multigpu{20,30,40,50}.csv
  Fig 13: openb_pod_list_gpuspec{10,20,25,33}.csv
  Fig 14: openb_pod_list_cpu{050,100,200,250}.csv  (5/10/20/25% non-GPU)

All figures use Monte-Carlo workload inflation:
- Sample tasks with replacement until GPU requests reach 100% of cluster capacity
- Measure unallocated GPU %
- Repeat N runs, average results
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from simulator import Task, Node, Cluster, TaskDistribution
from schedulers import (
    Scheduler, get_all_schedulers,
    ClusteringScheduler, FGDScheduler
)
from trace_loader import AlibabaTraceLoader


# ---------------------------------------------------------------------------
# GPU-type aware cluster (Figure 13)
# ---------------------------------------------------------------------------

class GpuTypeAwareCluster(Cluster):
    """Cluster subclass that enforces GPU-type placement constraints.

    When a task has gpu_spec set, it can only be scheduled on nodes
    whose gpu_model matches.
    """

    def get_eligible_nodes(self, task: Task) -> List[Node]:
        eligible = super().get_eligible_nodes(task)
        if task.gpu_spec:
            eligible = [n for n in eligible if n.gpu_model == task.gpu_spec]
        return eligible


# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------

FIGURE_FILES: Dict[int, Dict[int, str]] = {
    11: {
        40:  'openb_pod_list_gpushare40.csv',
        60:  'openb_pod_list_gpushare60.csv',
        80:  'openb_pod_list_gpushare80.csv',
        100: 'openb_pod_list_gpushare100.csv',
    },
    12: {
        20: 'openb_pod_list_multigpu20.csv',
        30: 'openb_pod_list_multigpu30.csv',
        40: 'openb_pod_list_multigpu40.csv',
        50: 'openb_pod_list_multigpu50.csv',
    },
    13: {
        10: 'openb_pod_list_gpuspec10.csv',
        20: 'openb_pod_list_gpuspec20.csv',
        25: 'openb_pod_list_gpuspec25.csv',
        33: 'openb_pod_list_gpuspec33.csv',
    },
    14: {
        5:  'openb_pod_list_cpu050.csv',
        10: 'openb_pod_list_cpu100.csv',
        20: 'openb_pod_list_cpu200.csv',
        25: 'openb_pod_list_cpu250.csv',
    },
}

FIGURE_CONFIG = {
    11: {
        'name': 'GPU-Sharing Task Proportion',
        'section': '6.3',
        'xlabel': 'Proportion of GPU-sharing workloads\n(% of GPU requests)',
        'proportions': [40, 60, 80, 100],
        'gpu_type_aware': False,
    },
    12: {
        'name': 'Multi-GPU Task Proportion',
        'section': '6.4',
        'xlabel': 'Proportion of multi-GPU workloads\n(% of GPU requests)',
        'proportions': [20, 30, 40, 50],
        'gpu_type_aware': False,
    },
    13: {
        'name': 'GPU-Type Constrained Task Proportion',
        'section': '6.5',
        'xlabel': 'Proportion with GPU type constraints\n(% of GPU requests)',
        'proportions': [10, 20, 25, 33],
        'gpu_type_aware': True,
    },
    14: {
        'name': 'Non-GPU Task Proportion',
        'section': '6.6',
        'xlabel': 'Proportion of non-GPU workloads\n(% of task number)',
        'proportions': [5, 10, 20, 25],
        'gpu_type_aware': False,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result for one (scheduler, proportion) combination, averaged over runs."""
    scheduler_name: str
    proportion: float
    unalloc_gpu_pct: float
    unalloc_std: float = 0.0


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class SensitivityExperiment:
    """Runs sensitivity experiments for Figures 11-14."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, 'csv')

        # Load nodes once (shared across all figures)
        self.loader = AlibabaTraceLoader(data_dir)
        self.loader.load_nodes()

        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded cluster: {len(self.loader.nodes)} nodes, "
              f"{self.total_gpu_capacity} GPUs")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_tasks(self, filename: str) -> List[Task]:
        """Load tasks from a specific trace CSV file."""
        self.loader.load_tasks(filename=filename)
        return self.loader.tasks

    def _compute_task_distribution(self, tasks: List[Task]) -> TaskDistribution:
        """Compute task type distribution from a task list."""
        from collections import Counter
        type_counts: Counter = Counter()
        for t in tasks:
            gpu_rounded = round(t.gpu_demand, 2)
            cpu_bucket = round(t.cpu_demand / 4) * 4
            type_counts[(cpu_bucket, gpu_rounded)] += 1
        total = sum(type_counts.values())
        dist = TaskDistribution()
        for (cpu, gpu), count in type_counts.items():
            dist.add_task_type(cpu, gpu, count / total)
        return dist

    def create_fresh_cluster(self, gpu_type_aware: bool = False) -> Cluster:
        cls = GpuTypeAwareCluster if gpu_type_aware else Cluster
        cluster = cls()
        for i, orig in enumerate(self.loader.nodes):
            cluster.add_node(Node(
                node_id=i, total_cpu=orig.total_cpu,
                num_gpus=orig.num_gpus, name=orig.name,
                gpu_model=orig.gpu_model, memory_mib=orig.memory_mib))
        return cluster

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_single(self, scheduler: Scheduler, tasks: List[Task],
                   dist: TaskDistribution, seed: int,
                   gpu_type_aware: bool = False) -> float:
        """Run single Monte-Carlo inflation until 100% GPU arrival.

        Returns unallocated GPU % at that point.
        """
        rng = random.Random(seed)
        cluster = self.create_fresh_cluster(gpu_type_aware=gpu_type_aware)
        cluster.set_task_distribution(dist)

        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        cumulative_gpu = 0.0
        max_gpu = float(self.total_gpu_capacity)
        task_count = 0

        while cumulative_gpu < max_gpu:
            orig = rng.choice(tasks)
            task = Task(
                task_id=task_count,
                cpu_demand=orig.cpu_demand,
                gpu_demand=orig.gpu_demand,
                gpu_spec=orig.gpu_spec if gpu_type_aware else '')
            task_count += 1
            cumulative_gpu += task.gpu_demand
            scheduler.schedule(task, cluster)

        return (cluster.total_unallocated_gpu / self.total_gpu_capacity) * 100

    def run_figure(self, figure_num: int, schedulers: List[Scheduler],
                   num_runs: int = 10, seed: int = 42,
                   proportions: List[float] = None,
                   show_progress: bool = True
                   ) -> Dict[float, List[SensitivityResult]]:
        """Run one figure experiment across all proportions and schedulers."""
        from tqdm import tqdm

        config = FIGURE_CONFIG[figure_num]
        gpu_type_aware = config['gpu_type_aware']
        if proportions is None:
            proportions = config['proportions']

        file_map = FIGURE_FILES[figure_num]

        results: Dict[float, List[SensitivityResult]] = {}
        total_runs = len(proportions) * len(schedulers) * num_runs
        pbar = tqdm(total=total_runs, desc=f"Figure {figure_num}",
                    disable=not show_progress, ncols=90)

        for pct in proportions:
            filename = file_map[pct]
            tasks = self.load_tasks(filename)
            dist = self._compute_task_distribution(tasks)
            print(f"\n  [{figure_num}] {pct}% — {filename} ({len(tasks)} tasks)")

            results[pct] = []

            for scheduler in schedulers:
                unallocs = []
                for run_idx in range(num_runs):
                    run_seed = seed + run_idx
                    val = self.run_single(
                        scheduler, tasks, dist, run_seed,
                        gpu_type_aware=gpu_type_aware)
                    unallocs.append(val)
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"{pct}% {scheduler.name}", refresh=False)

                avg = sum(unallocs) / len(unallocs)
                std = (sum((x - avg) ** 2 for x in unallocs)
                       / len(unallocs)) ** 0.5
                results[pct].append(SensitivityResult(
                    scheduler_name=scheduler.name,
                    proportion=pct,
                    unalloc_gpu_pct=avg,
                    unalloc_std=std))

        pbar.close()
        return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity(results: Dict[float, List[SensitivityResult]],
                     figure_num: int, output_dir: str = None):
    """Grouped bar chart for one figure (paper style)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed. Skipping plot.")
        return

    config = FIGURE_CONFIG[figure_num]

    colors = {
        'FGD': '#1f77b4', 'BestFit': '#ff7f0e', 'Packing': '#2ca02c',
        'Clustering': '#d62728', 'DotProd': '#9467bd', 'Random': '#8c564b',
    }
    hatches = {
        'FGD': '//', 'BestFit': '//', 'Packing': '//',
        'Clustering': '//', 'DotProd': '//', 'Random': '//',
    }
    sched_order = ['FGD', 'BestFit', 'Packing', 'Clustering',
                   'DotProd', 'Random']

    proportions = sorted(results.keys())
    available = [r.scheduler_name for r in results[proportions[0]]]
    sched_names = [s for s in sched_order if s in available]

    n_groups = len(proportions)
    n_bars = len(sched_names)
    bar_width = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(sched_names):
        x_positions = []
        values = []
        for j, pct in enumerate(proportions):
            r = next((r for r in results[pct]
                       if r.scheduler_name == name), None)
            if r:
                x_positions.append(j + (i - n_bars / 2 + 0.5) * bar_width)
                values.append(r.unalloc_gpu_pct)

        bars = ax.bar(x_positions, values, bar_width,
                      label=name,
                      color=colors.get(name, 'gray'),
                      hatch=hatches.get(name, ''),
                      edgecolor='black', linewidth=0.5)

        # Annotate FGD bars with values
        if name == 'FGD':
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f'{int(p)}%' for p in proportions])
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel('Unallocated GPU (%)')
    ax.set_title(f"Figure {figure_num}: {config['name']}")
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.set_ylim(0, 25)
    ax.set_yticks(range(0, 26, 5))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, f'figure{figure_num}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# CSV / summary helpers
# ---------------------------------------------------------------------------

def save_results_to_csv(results: Dict[float, List[SensitivityResult]],
                        figure_num: int, output_dir: str):
    import csv
    path = os.path.join(output_dir, f'figure{figure_num}_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['proportion', 'scheduler', 'unalloc_gpu_pct', 'std'])
        for pct in sorted(results.keys()):
            for r in results[pct]:
                writer.writerow([pct, r.scheduler_name,
                                 f'{r.unalloc_gpu_pct:.2f}',
                                 f'{r.unalloc_std:.2f}'])
    print(f"  CSV saved to {path}")


def load_results_from_csv(csv_path: str
                          ) -> Tuple[int, Dict[float, List[SensitivityResult]]]:
    """Load results from a CSV file produced by save_results_to_csv.

    Returns (figure_num, results_dict).
    figure_num is inferred from the filename (e.g. figure12_results.csv -> 12).
    """
    import csv
    import re

    basename = os.path.basename(csv_path)
    m = re.search(r'figure(\d+)', basename)
    figure_num = int(m.group(1)) if m else 0

    results: Dict[float, List[SensitivityResult]] = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pct = float(row['proportion'])
            if pct not in results:
                results[pct] = []
            results[pct].append(SensitivityResult(
                scheduler_name=row['scheduler'],
                proportion=pct,
                unalloc_gpu_pct=float(row['unalloc_gpu_pct']),
                unalloc_std=float(row.get('std', 0))))

    return figure_num, results


def format_summary(results: Dict[float, List[SensitivityResult]],
                   figure_num: int) -> str:
    config = FIGURE_CONFIG[figure_num]
    proportions = sorted(results.keys())
    sched_names = [r.scheduler_name for r in results[proportions[0]]]

    lines = []
    lines.append(f"Figure {figure_num} ({config['section']}): "
                 f"{config['name']}")
    lines.append("=" * 70)

    header = f"{'Scheduler':<12}"
    for pct in proportions:
        header += f"  {pct:>5.0f}%"
    lines.append(header)
    lines.append("-" * (12 + 8 * len(proportions)))

    for name in sched_names:
        row = f"{name:<12}"
        for pct in proportions:
            r = next((r for r in results[pct]
                       if r.scheduler_name == name), None)
            if r:
                row += f"  {r.unalloc_gpu_pct:>5.1f}%"
            else:
                row += f"  {'N/A':>5}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis (Figures 11-14)")
    parser.add_argument('--figures', type=str, default='11,12,13,14',
                        help='Comma-separated figure numbers '
                             '(default: 11,12,13,14)')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Monte-Carlo runs per configuration (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--plot-csv', type=str, nargs='+', default=None,
                        help='Plot from existing CSV file(s) instead of '
                             'running experiments. '
                             'e.g. --plot-csv result/exp3/figure11_results.csv')
    args = parser.parse_args()

    # ---- Plot-only mode ----
    if args.plot_csv:
        for csv_path in args.plot_csv:
            fig_num, results = load_results_from_csv(csv_path)
            if fig_num == 0:
                print(f"Warning: could not infer figure number from {csv_path}")
                continue
            output_dir = os.path.dirname(csv_path) or '.'
            summary = format_summary(results, fig_num)
            print(summary)
            plot_sensitivity(results, fig_num, output_dir)
        exit(0)

    # ---- Full experiment mode ----
    figures = [int(x) for x in args.figures.split(',')]

    data_dir = os.path.join(os.path.dirname(__file__), '..',
                            'alibaba_traces', 'cluster-trace-gpu-v2023')

    print("=" * 60)
    print("Sensitivity Analysis (Figures 11-14)")
    print(f"  Figures: {figures}")
    print(f"  Runs per config: {args.num_runs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    experiment = SensitivityExperiment(data_dir)
    schedulers = get_all_schedulers()

    # Result directory
    fig_str = '-'.join(str(f) for f in figures)
    result_name = f"fig{fig_str}-runs{args.num_runs}-seed{args.seed}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)
    os.makedirs(result_dir, exist_ok=True)

    all_summaries = []

    for fig_num in figures:
        config = FIGURE_CONFIG[fig_num]
        print(f"\n{'=' * 60}")
        print(f"Figure {fig_num} ({config['section']}): {config['name']}")
        print(f"  Proportions: {config['proportions']}")
        print(f"{'=' * 60}")

        results = experiment.run_figure(
            fig_num, schedulers,
            num_runs=args.num_runs, seed=args.seed)

        summary = format_summary(results, fig_num)
        print("\n" + summary)
        all_summaries.append(summary)

        save_results_to_csv(results, fig_num, result_dir)
        plot_sensitivity(results, fig_num, result_dir)

    # Combined summary log
    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write("Experiment: Sensitivity Analysis (Figures 11-14)\n")
        f.write(f"Result: {result_name}\n")
        f.write(f"Figures: {figures}\n")
        f.write(f"Runs: {args.num_runs}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Nodes: {len(experiment.loader.nodes)}\n")
        f.write(f"GPUs: {experiment.total_gpu_capacity}\n\n")
        for s in all_summaries:
            f.write(s + "\n\n")
    print(f"\nSummary log saved to {log_path}")

    # Cleanup FGD pools
    for s in schedulers:
        if isinstance(s, FGDScheduler):
            s.cleanup()
