"""
Experiment 3: Sensitivity Analysis (Figures 11-14)

Figure 11 (Section 6.3): Varying GPU-sharing task proportion
Figure 12 (Section 6.4): Varying multi-GPU task proportion
Figure 13 (Section 6.5): Varying GPU-type constrained task proportion
Figure 14 (Section 6.6): Varying non-GPU task proportion

All figures use Monte-Carlo workload inflation:
- Modify task distribution by adjusting sampling weights
- Sample tasks with replacement until GPU requests reach 100% of cluster capacity
- Measure unallocated GPU %
- Repeat N runs, average results
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict

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
# Weighted two-pool sampler
# ---------------------------------------------------------------------------

class TaskSampler:
    """Two-pool weighted sampler for modified task distributions.

    Samples from group_a with probability p_a, from group_b with (1 - p_a).
    """

    def __init__(self, group_a: List[Task], group_b: List[Task], p_a: float):
        self.group_a = group_a
        self.group_b = group_b
        self.p_a = max(0.0, min(1.0, p_a))

    def sample(self, rng: random.Random) -> Task:
        if self.group_a and rng.random() < self.p_a:
            return rng.choice(self.group_a)
        elif self.group_b:
            return rng.choice(self.group_b)
        elif self.group_a:
            return rng.choice(self.group_a)
        raise ValueError("Both task groups are empty")


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

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


class SensitivityExperiment:
    """Runs sensitivity experiments for Figures 11-14."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.loader = AlibabaTraceLoader(data_dir)
        self.loader.load_nodes()
        self.loader.load_tasks()

        self.task_distribution = self.loader.compute_task_distribution()
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        self.all_tasks = self.loader.tasks

        # ---- Categorise tasks ----
        self.sharing_tasks = [t for t in self.all_tasks if 0 < t.gpu_demand < 1]
        self.nonsharing_tasks = [t for t in self.all_tasks
                                  if t.gpu_demand == 0 or t.gpu_demand >= 1]
        self.multigpu_tasks = [t for t in self.all_tasks if t.gpu_demand >= 2]
        self.nonmultigpu_tasks = [t for t in self.all_tasks if t.gpu_demand < 2]
        self.nogpu_tasks = [t for t in self.all_tasks if t.gpu_demand == 0]
        self.gpu_tasks = [t for t in self.all_tasks if t.gpu_demand > 0]

        self.total_gpu_demand = sum(t.gpu_demand for t in self.all_tasks)

        # ---- Synthetic GPU-type constrained tasks (Figure 13) ----
        # The public Alibaba trace has empty gpu_spec for all tasks.
        # We synthetically assign gpu_spec proportional to cluster GPU-model
        # distribution so that Figure 13 can vary the constrained fraction.
        self.constrained_tasks, self.unconstrained_tasks = \
            self._build_gpu_type_pools()

        # ---- Print summary ----
        print(f"Loaded trace: {len(self.loader.nodes)} nodes, "
              f"{self.total_gpu_capacity} GPUs")
        print(f"Tasks: {len(self.all_tasks)} total")
        print(f"  GPU-sharing (0<gpu<1): {len(self.sharing_tasks)} "
              f"({self._gpu_pct(self.sharing_tasks):.1f}% of GPU reqs)")
        print(f"  Multi-GPU (gpu>=2):    {len(self.multigpu_tasks)} "
              f"({self._gpu_pct(self.multigpu_tasks):.1f}% of GPU reqs)")
        print(f"  1-GPU:                 "
              f"{len([t for t in self.all_tasks if t.gpu_demand == 1])} "
              f"({self._gpu_pct([t for t in self.all_tasks if t.gpu_demand == 1]):.1f}%"
              f" of GPU reqs)")
        print(f"  No-GPU:                {len(self.nogpu_tasks)} "
              f"({100 * len(self.nogpu_tasks) / len(self.all_tasks):.1f}% of tasks)")
        print(f"  Constrained (synth):   {len(self.constrained_tasks)} "
              f"({self._gpu_pct(self.constrained_tasks):.1f}% of GPU reqs)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gpu_pct(self, tasks: List[Task]) -> float:
        if self.total_gpu_demand == 0:
            return 0.0
        return sum(t.gpu_demand for t in tasks) / self.total_gpu_demand * 100

    def _build_gpu_type_pools(self):
        """Create constrained / unconstrained task copies for Figure 13.

        Every GPU task gets a copy with gpu_spec assigned (sampled from the
        cluster's GPU-model distribution, weighted by GPU count per model).
        Non-GPU tasks go into the unconstrained pool only.
        """
        model_gpus: Counter = Counter()
        for node in self.loader.nodes:
            if node.gpu_model:
                model_gpus[node.gpu_model] += node.num_gpus
        models = list(model_gpus.keys())
        weights = [model_gpus[m] for m in models]

        rng = random.Random(0)  # deterministic
        constrained: List[Task] = []
        unconstrained: List[Task] = []

        for t in self.all_tasks:
            # unconstrained copy (always)
            unconstrained.append(Task(
                task_id=t.task_id, cpu_demand=t.cpu_demand,
                gpu_demand=t.gpu_demand, gpu_spec=''))
            # constrained copy (GPU tasks only)
            if t.gpu_demand > 0:
                spec = rng.choices(models, weights=weights, k=1)[0]
                constrained.append(Task(
                    task_id=t.task_id, cpu_demand=t.cpu_demand,
                    gpu_demand=t.gpu_demand, gpu_spec=spec))

        return constrained, unconstrained

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
    # Task distribution from sampler (for FGD)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sampler_distribution(sampler: TaskSampler) -> TaskDistribution:
        """Compute effective task distribution from the sampler's weighted pools."""
        type_counts: Counter = Counter()

        if sampler.group_a and sampler.p_a > 0:
            w = sampler.p_a / len(sampler.group_a)
            for t in sampler.group_a:
                key = (round(t.cpu_demand / 4) * 4, round(t.gpu_demand, 2))
                type_counts[key] += w

        if sampler.group_b and sampler.p_a < 1:
            w = (1 - sampler.p_a) / len(sampler.group_b)
            for t in sampler.group_b:
                key = (round(t.cpu_demand / 4) * 4, round(t.gpu_demand, 2))
                type_counts[key] += w

        total = sum(type_counts.values())
        dist = TaskDistribution()
        for (cpu, gpu), weight in type_counts.items():
            dist.add_task_type(cpu, gpu, weight / total if total > 0 else 0)
        return dist

    # ------------------------------------------------------------------
    # Pool builders
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_p_a(group_a: List[Task], group_b: List[Task],
                   target_pct: float) -> float:
        """Compute sampling probability p_a so that GPU-request proportion
        from group_a equals target_pct (0-100).

        Derivation:
          target = p_a * avg_a / (p_a * avg_a + (1-p_a) * avg_b)
          =>  p_a = target * avg_b / (target * avg_b + (1-target) * avg_a)
        """
        target = target_pct / 100.0
        if target >= 1.0:
            return 1.0
        if target <= 0.0:
            return 0.0
        if not group_a or not group_b:
            return 1.0 if group_a else 0.0

        avg_a = sum(t.gpu_demand for t in group_a) / len(group_a)
        avg_b = sum(t.gpu_demand for t in group_b) / len(group_b)

        denom = target * avg_b + (1 - target) * avg_a
        if denom <= 0:
            return 0.5
        return target * avg_b / denom

    def build_sharing_sampler(self, target_gpu_pct: float) -> TaskSampler:
        """Figure 11: GPU-sharing tasks as target % of GPU requests."""
        p_a = self._solve_p_a(
            self.sharing_tasks, self.nonsharing_tasks, target_gpu_pct)
        return TaskSampler(self.sharing_tasks, self.nonsharing_tasks, p_a)

    def build_multigpu_sampler(self, target_gpu_pct: float) -> TaskSampler:
        """Figure 12: Multi-GPU tasks as target % of GPU requests."""
        p_a = self._solve_p_a(
            self.multigpu_tasks, self.nonmultigpu_tasks, target_gpu_pct)
        return TaskSampler(self.multigpu_tasks, self.nonmultigpu_tasks, p_a)

    def build_gpu_type_sampler(self, target_gpu_pct: float) -> TaskSampler:
        """Figure 13: GPU-type-constrained tasks as target % of GPU requests."""
        p_a = self._solve_p_a(
            self.constrained_tasks, self.unconstrained_tasks, target_gpu_pct)
        return TaskSampler(self.constrained_tasks, self.unconstrained_tasks, p_a)

    def build_nogpu_sampler(self, target_task_pct: float) -> TaskSampler:
        """Figure 14: Non-GPU tasks as target % of task count."""
        # Task-count proportion: p_a = target directly
        return TaskSampler(
            self.nogpu_tasks, self.gpu_tasks, target_task_pct / 100.0)

    # ------------------------------------------------------------------
    # Run methods
    # ------------------------------------------------------------------

    def run_single(self, scheduler: Scheduler, sampler: TaskSampler,
                   seed: int, gpu_type_aware: bool = False) -> float:
        """Run single Monte-Carlo inflation until 100% GPU arrival.

        Returns unallocated GPU % at that point.
        """
        rng = random.Random(seed)
        cluster = self.create_fresh_cluster(gpu_type_aware=gpu_type_aware)

        dist = self._compute_sampler_distribution(sampler)
        cluster.set_task_distribution(dist)

        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        cumulative_gpu = 0.0
        max_gpu = float(self.total_gpu_capacity)  # 100% arrival
        task_count = 0

        while cumulative_gpu < max_gpu:
            orig = sampler.sample(rng)
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

        builder = {
            11: self.build_sharing_sampler,
            12: self.build_multigpu_sampler,
            13: self.build_gpu_type_sampler,
            14: self.build_nogpu_sampler,
        }[figure_num]

        results: Dict[float, List[SensitivityResult]] = {}
        total_runs = len(proportions) * len(schedulers) * num_runs
        pbar = tqdm(total=total_runs, desc=f"Figure {figure_num}",
                    disable=not show_progress, ncols=90)

        for pct in proportions:
            sampler = builder(pct)
            results[pct] = []

            for scheduler in schedulers:
                unallocs = []
                for run_idx in range(num_runs):
                    run_seed = seed + run_idx
                    val = self.run_single(
                        scheduler, sampler, run_seed,
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
        'FGD': '#e15759', 'BestFit': '#9467bd', 'Packing': '#ff7f0e',
        'Clustering': '#2ca02c', 'DotProd': '#1f77b4', 'Random': '#7f7f7f',
    }
    hatches = {
        'FGD': '', 'BestFit': '//', 'Packing': '\\\\',
        'Clustering': 'xx', 'DotProd': '..', 'Random': '--',
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
                      edgecolor='white', linewidth=0.5)

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
    args = parser.parse_args()

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
    result_name = f"exp3-fig{fig_str}-runs{args.num_runs}-seed{args.seed}"
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
        f.write(f"GPUs: {experiment.total_gpu_capacity}\n")
        f.write(f"Tasks in trace: {len(experiment.all_tasks)}\n")
        f.write(f"Total GPU demand: {experiment.total_gpu_demand:.1f}\n\n")
        for s in all_summaries:
            f.write(s + "\n\n")
    print(f"\nSummary log saved to {log_path}")

    # Cleanup FGD pools
    for s in schedulers:
        if isinstance(s, FGDScheduler):
            s.cleanup()
