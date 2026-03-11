"""
Distribution-Shift Experiment Runner for FGD Replication

Processes tasks in trace order and measures fragmentation under distribution shift.
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from simulator import Task, Node, Cluster, TaskDistribution
from schedulers import (
    Scheduler, get_all_schedulers, get_scheduler,
    ClusteringScheduler, FGDScheduler, WindowedFGDScheduler,
    BayesianFGDScheduler
)
from trace_loader import AlibabaTraceLoader


def build_scheduler(name: str, experiment,
                    prior_strength: float = 10.0,
                    min_gpu_tasks: int = 50) -> 'Scheduler':
    """Create a scheduler instance by name, parsing N/M from FGD-N / W-FGD-M.

    Supported names:
      Random, BestFit, DotProd, Packing, Clustering — baseline schedulers
      FGD-Full  — FGD with full trace distribution
      FGD-N     — FGD with first-N-task distribution  (N any positive int)
      W-FGD-M   — Windowed FGD with sliding window size M
      B-FGD     — Bayesian FGD
      U-FGD     — FGD with uniform grid distribution

    Returns None if the name is not recognised.
    """
    import re

    # FGD-N: static first-N distribution (N encoded in the name)
    m = re.fullmatch(r'FGD-(\d+)', name)
    if m:
        n = int(m.group(1))
        dist = experiment._compute_initial_distribution(n)
        sched = FGDScheduler(scheduling_task_types=dist.get_task_types())
        sched.name = name
        return sched

    # W-FGD-M: sliding window with size M encoded in the name
    m = re.fullmatch(r'W-FGD-(\d+)', name)
    if m:
        window_size = int(m.group(1))
        return WindowedFGDScheduler(window_size=window_size)

    # Fixed-name variants
    if name == 'FGD-Full':
        sched = FGDScheduler()
        sched.name = 'FGD-Full'
        return sched

    if name == 'U-FGD':
        uniform_types = build_uniform_task_types(experiment.loader.nodes)
        sched = FGDScheduler(scheduling_task_types=uniform_types)
        sched.name = 'U-FGD'
        return sched

    if name == 'B-FGD':
        return BayesianFGDScheduler(prior_strength=prior_strength,
                                    min_gpu_tasks=min_gpu_tasks)

    # Baseline schedulers (non-FGD)
    baselines = {s.name: s for s in get_all_schedulers()
                 if not isinstance(s, FGDScheduler)}
    if name in baselines:
        return baselines[name]

    return None


def build_uniform_task_types(nodes) -> List[Tuple[Tuple[float, float], float]]:
    """Build uniform distribution over (cpu, gpu) grid from cluster node specs."""
    max_cpu = int(max(n.total_cpu for n in nodes))
    max_gpu = int(max(n.num_gpus for n in nodes))
    cpu_values = list(range(0, max_cpu + 1, 4))
    gpu_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0..1.0
    gpu_values += list(range(2, max_gpu + 1))              # 2, 3, ..., max_gpu
    n_types = len(cpu_values) * len(gpu_values)
    weight = 1.0 / n_types
    task_types = [((cpu, gpu), weight) for cpu in cpu_values for gpu in gpu_values]
    print(f"  U-FGD uniform grid: {len(cpu_values)} CPU x {len(gpu_values)} GPU = {n_types} types")
    return task_types


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    scheduler_name: str
    # Figure 7(a): (arrived_pct, frag_rate)
    fragmentation_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Figure 7(b): (arrived_pct, fragmented_gpu_over_total_gpu_pct)
    frag_total_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Figure 9(a): (arrived_pct, unallocated_gpu_pct)
    unalloc_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Figure 9(b): (arrived_pct, occupied_nodes)
    occupied_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Figure 9(c): failed tasks at snapshot by GPU category
    failed_by_category: Dict[str, float] = field(default_factory=dict)
    # Figure 9(d): fragmentation breakdown by cause at end
    frag_breakdown: Dict[str, float] = field(default_factory=dict)
    final_frag_rate: float = 0.0
    final_frag_total_pct: float = 0.0
    final_gpu_alloc_rate: float = 0.0
    tasks_scheduled: int = 0
    tasks_failed: int = 0
    elapsed_sec: float = 0.0


class DistShiftExperiment:
    """
    Distribution-Shift Experiment: processes tasks in creation_time order from the trace
    and measures fragmentation under various scheduling policies.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 500,
        task_order: str = 'trace',
        cluster_scale: float = 100.0,
        tier_order: List[int] = None,
        fgd_popularity_threshold: float = 95
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.task_order = task_order
        self.cluster_scale = cluster_scale
        self.fgd_popularity_threshold = fgd_popularity_threshold
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
        if task_order in ('ascending', 'descending'):
            reverse = (task_order == 'descending')
            self.loader.tasks.sort(
                key=lambda t: (round(t.gpu_demand, 2), round(t.cpu_demand / 4) * 4),
                reverse=reverse
            )
        elif task_order == 'phased':
            self.loader.tasks, self.phase_info = self._phased_order(self.loader.tasks, tier_order or [0,1,2,3,4])

        # Compute TWO task distributions:
        # 1. Full distribution (all tasks) - for original FGD baseline
        # 2. First-N distribution - for penalized FGD and W-FGD starting point
        self.full_task_distribution = self.loader.compute_task_distribution()
        self.fgd_scoring_distribution = self.loader.compute_task_distribution(
            popularity_threshold=self.fgd_popularity_threshold
        )
        self.first_n_distribution = self._compute_initial_distribution(window_size)

        # Get cluster capacity
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded trace: {len(self.loader.nodes)} nodes, {self.total_gpu_capacity} GPUs")
        print(f"Tasks in trace: {len(self.loader.tasks)}")
        print(f"Task order: {task_order}")
        print(f"Full distribution: {len(self.full_task_distribution.get_task_types())} task types")
        print(
            f"FGD scoring distribution (top {self.fgd_popularity_threshold:.0f}%): "
            f"{len(self.fgd_scoring_distribution.get_task_types())} task types"
        )

    @staticmethod
    def _gpu_category(gpu_demand: float) -> str:
        """Categorize GPU demand for Figure 9(c)."""
        if gpu_demand < 1:
            return "<1"
        if gpu_demand == 1:
            return "1"
        if gpu_demand == 2:
            return "2"
        return "8"

    @staticmethod
    def _count_occupied_nodes(cluster: Cluster) -> int:
        """Count nodes with any allocated resource."""
        count = 0
        for node in cluster.nodes:
            if node.allocated_cpu > 0 or any(g < 1.0 for g in node.gpu_remaining):
                count += 1
        return count

    @staticmethod
    def _compute_frag_breakdown(cluster: Cluster) -> Dict[str, float]:
        """
        Decompose fragmentation into 3 causes for Figure 9(d):
        deficient, stranded, non_gpu.
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
                    non_gpu += weighted_frag
                elif gpu_demand > node.scalar_gpu_capacity:
                    deficient += weighted_frag
                elif cpu_demand > node.remaining_cpu:
                    stranded += weighted_frag
                else:
                    deficient += weighted_frag

        total = deficient + stranded + non_gpu
        if total == 0:
            return {"deficient": 0, "stranded": 0, "non_gpu": 0}

        return {
            "deficient": deficient / total * 100,
            "stranded": stranded / total * 100,
            "non_gpu": non_gpu / total * 100,
        }

    def _compute_initial_distribution(self, n: int) -> TaskDistribution:
        """
        Compute task distribution from first N tasks of the trace.

        Applies the same typical-pod popularity filtering (top K%) as FGD-Full,
        but on the first-N subset, then renormalizes.
        """
        subset = self.loader.tasks[:n]
        dist = TaskDistribution()
        type_counts: Counter = Counter()
        for task in subset:
            gpu_rounded = round(task.gpu_demand, 2)
            cpu_bucket = round(task.cpu_demand / 4) * 4
            type_counts[(cpu_bucket, gpu_rounded)] += 1

        total = sum(type_counts.values())
        if total == 0:
            return dist

        # Keep only top types whose cumulative count reaches threshold%.
        threshold = self.fgd_popularity_threshold
        expected = threshold * total / 100.0
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        selected: Dict[Tuple[float, float], int] = {}
        cum = 0
        for type_key, count in sorted_types:
            selected[type_key] = count
            cum += count
            if cum >= expected:
                break

        for (cpu, gpu), count in selected.items():
            dist.add_task_type(cpu, gpu, count / cum)
        return dist

    @staticmethod
    def _phased_order(tasks: list, tier_order: List[int]) -> list:
        """
        Phased arrival order: tasks grouped by GPU demand tier, tiers ordered by tier_order.
        Within each tier, tasks are shuffled (seeded).

        Tiers:
          0: gpu == 0        (CPU-only / no-GPU tasks)
          1: 0 < gpu < 0.5   (small fractional GPU)
          2: 0.5 <= gpu < 1  (large fractional GPU)
          3: gpu == 1         (single full GPU)
          4: gpu > 1          (multi-GPU)
        """
        import random as rng

        tiers = {0: [], 1: [], 2: [], 3: [], 4: []}
        for task in tasks:
            g = task.gpu_demand
            if g == 0:
                tier = 0
            elif g < 0.5:
                tier = 1
            elif g < 1.0:
                tier = 2
            elif g == 1.0:
                tier = 3
            else:
                tier = 4
            tiers[tier].append(task)

        # Shuffle within each tier (seeded for reproducibility)
        r = rng.Random(42)
        for tier_tasks in tiers.values():
            r.shuffle(tier_tasks)

        lines = [f"Tier order: {tier_order}"]
        result = []
        for phase_idx, tier_id in enumerate(tier_order):
            tier_tasks = tiers[tier_id]
            if tier_tasks:
                lines.append(f"  Phase {phase_idx}: Tier {tier_id} - {len(tier_tasks)} tasks "
                             f"(gpu range: {min(t.gpu_demand for t in tier_tasks):.2f}"
                             f"-{max(t.gpu_demand for t in tier_tasks):.2f})")
            result.extend(tier_tasks)

        phase_info = "\n".join(lines)
        print(phase_info)
        return result, phase_info

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
        sample_interval_pct: float = 2.0,
        max_arrival_pct: float = 120.0,
        snapshot_pct: float = 96.0,
        show_progress: bool = True
    ) -> ExperimentResult:
        """
        Run a single trace-replay experiment with one scheduler.

        Tasks are processed in creation_time order until the trace is exhausted.
        Cluster always uses full distribution for evaluation.
        Schedulers carry their own scheduling distribution if needed.
        """
        import time
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
        if isinstance(scheduler, BayesianFGDScheduler):
            scheduler.reset()
            # Uniform prior from cluster specs
            max_cpu = max(n.total_cpu for n in self.loader.nodes)
            max_gpu = max(n.num_gpus for n in self.loader.nodes)
            scheduler.set_uniform_prior(max_cpu, max_gpu)
        if isinstance(scheduler, FGDScheduler) and not isinstance(
            scheduler, (WindowedFGDScheduler, BayesianFGDScheduler)
        ):
            # Apply typical-pod scoring only to plain FGD variants that did not
            # come with an explicit scheduling distribution.
            # This preserves intended behavior for FGD-N / U-FGD.
            if scheduler.scheduling_task_types is None:
                scheduler.scheduling_task_types = self.fgd_scoring_distribution.get_task_types()

        pbar = tqdm(
            total=len(tasks),
            desc=f"{scheduler.name:16}",
            unit="task",
            disable=not show_progress,
            ncols=90
        )

        t_start = time.monotonic()
        cumulative_gpu_demand = 0.0
        max_gpu_demand = self.total_gpu_capacity * max_arrival_pct / 100.0
        next_sample_pct = sample_interval_pct
        snapshot_taken = False
        failed_tasks_until_snapshot: List[Task] = []

        # Figure 7 should start from the true initial cluster state (x=0),
        # not from a synthetic point copied from the first sampled value.
        initial_frag_rate = cluster.compute_fragmentation_rate()
        initial_frag_total_pct = (
            cluster.compute_cluster_fragmentation() / self.total_gpu_capacity
        ) * 100.0
        result.fragmentation_curve.append((0.0, initial_frag_rate))
        result.frag_total_curve.append((0.0, initial_frag_total_pct))

        for task in tasks:
            cumulative_gpu_demand += task.gpu_demand
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100

            # Feed task to adaptive schedulers before scheduling
            if isinstance(scheduler, WindowedFGDScheduler):
                scheduler.observe_task(task)
            if isinstance(scheduler, BayesianFGDScheduler):
                scheduler.observe_task(task)

            # Try to schedule
            if scheduler.schedule(task, cluster):
                result.tasks_scheduled += 1
            else:
                result.tasks_failed += 1
                if not snapshot_taken:
                    failed_tasks_until_snapshot.append(task)

            # 9(c) snapshot at specified arrived percentage.
            if not snapshot_taken and arrived_pct >= snapshot_pct:
                snapshot_taken = True
                cat_sums: Dict[str, float] = defaultdict(float)
                for failed_task in failed_tasks_until_snapshot:
                    cat = self._gpu_category(failed_task.gpu_demand)
                    cat_sums[cat] += failed_task.gpu_demand
                result.failed_by_category = dict(cat_sums)

            # 9(a)/(b) sampled curves.
            while arrived_pct >= next_sample_pct and next_sample_pct <= max_arrival_pct:
                frag_rate = cluster.compute_fragmentation_rate()
                frag_total_pct = (
                    cluster.compute_cluster_fragmentation() / self.total_gpu_capacity
                ) * 100.0
                result.fragmentation_curve.append((next_sample_pct, frag_rate))
                result.frag_total_curve.append((next_sample_pct, frag_total_pct))

                unalloc_pct = (cluster.total_unallocated_gpu / self.total_gpu_capacity) * 100
                occupied = self._count_occupied_nodes(cluster)
                result.unalloc_curve.append((next_sample_pct, unalloc_pct))
                result.occupied_curve.append((next_sample_pct, occupied))
                next_sample_pct += sample_interval_pct

            pbar.update(1)

            if cumulative_gpu_demand >= max_gpu_demand:
                break

        pbar.close()

        if not snapshot_taken:
            cat_sums: Dict[str, float] = defaultdict(float)
            for failed_task in failed_tasks_until_snapshot:
                cat = self._gpu_category(failed_task.gpu_demand)
                cat_sums[cat] += failed_task.gpu_demand
            result.failed_by_category = dict(cat_sums)

        result.frag_breakdown = self._compute_frag_breakdown(cluster)

        # Record final metrics
        result.elapsed_sec = time.monotonic() - t_start
        result.final_frag_rate = cluster.compute_fragmentation_rate()
        result.final_frag_total_pct = (
            cluster.compute_cluster_fragmentation() / self.total_gpu_capacity
        ) * 100.0
        result.final_gpu_alloc_rate = cluster.gpu_allocation_rate

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
        sample_interval_pct: float = 2.0,
        max_arrival_pct: float = 120.0,
        snapshot_pct: float = 96.0,
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
                max_arrival_pct=max_arrival_pct,
                snapshot_pct=snapshot_pct,
                show_progress=show_progress
            )
            results[scheduler.name].append(result)

            avg_frag = sum(r.final_frag_rate for r in results[scheduler.name]) / len(results[scheduler.name])
            avg_alloc = sum(r.final_gpu_alloc_rate for r in results[scheduler.name]) / len(results[scheduler.name])
            avg_sched = sum(r.tasks_scheduled for r in results[scheduler.name]) / len(results[scheduler.name])
            avg_failed = sum(r.tasks_failed for r in results[scheduler.name]) / len(results[scheduler.name])
            print(
                f"  {scheduler.name:16}: Frag={avg_frag:.1f}%, "
                f"Alloc={avg_alloc:.1f}%, Scheduled={avg_sched:.0f}, Failed={avg_failed:.0f}"
            )

        return results



def format_summary(results: Dict[str, List[ExperimentResult]]) -> str:
    """Format summary statistics as a string"""
    lines = []
    lines.append("=" * 76)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 76)
    lines.append(f"\n{'Scheduler':<16} {'Avg Frag%':>10} {'Avg Alloc%':>12} {'Scheduled':>12} {'Failed':>10} {'Time(s)':>10}")
    lines.append("-" * 76)

    for name, result_list in results.items():
        avg_frag = sum(r.final_frag_rate for r in result_list) / len(result_list)
        avg_alloc = sum(r.final_gpu_alloc_rate for r in result_list) / len(result_list)
        total_scheduled = sum(r.tasks_scheduled for r in result_list) / len(result_list)
        total_failed = sum(r.tasks_failed for r in result_list) / len(result_list)
        avg_elapsed = sum(r.elapsed_sec for r in result_list) / len(result_list)

        lines.append(f"{name:<16} {avg_frag:>10.1f} {avg_alloc:>12.1f} {total_scheduled:>12.0f} {total_failed:>10.0f} {avg_elapsed:>10.1f}")

    return "\n".join(lines)


def print_summary(results: Dict[str, List[ExperimentResult]]):
    """Print summary statistics"""
    print("\n" + format_summary(results))


def _average_curves(curves: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Average multiple curves by x-value."""
    if not curves:
        return []
    by_x: Dict[float, List[float]] = defaultdict(list)
    for curve in curves:
        for x, y in curve:
            by_x[x].append(y)
    return sorted([(x, sum(ys) / len(ys)) for x, ys in by_x.items()])


def _average_curve_by_key(
    result_list: List[ExperimentResult],
    curve_key: str,
) -> List[Tuple[float, float]]:
    """Average one curve type across runs by x-value."""
    curves = [getattr(r, curve_key) for r in result_list]
    return _average_curves(curves)


def save_figure7_results_to_csv(results: Dict[str, List[ExperimentResult]], output_dir: str):
    """Save Figure 7 artifacts as a single CSV file."""
    import csv

    path = os.path.join(output_dir, 'figure7_results.csv')
    with open(path, 'w', newline='') as f:
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
                by_x_total = {x: y for x, y in result.frag_total_curve}
                for arrived_pct, frag_rate in result.fragmentation_curve:
                    writer.writerow([
                        scheduler_name,
                        arrived_pct,
                        frag_rate,
                        by_x_total.get(arrived_pct, ''),
                        run_idx,
                    ])

    print(f"Figure 7 CSV saved to {path}")


def load_figure7_results_from_csv(csv_path: str) -> Dict[str, List[ExperimentResult]]:
    """Load Figure 7 CSV into ExperimentResult structures."""
    import csv

    # scheduler -> run -> curves
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

            frag_total_raw = row.get('frag_total_pct', '')
            if frag_total_raw != '':
                data[scheduler][run]['frag_total'].append((x, float(frag_total_raw)))

    results: Dict[str, List[ExperimentResult]] = {}
    for scheduler, runs in data.items():
        results[scheduler] = []
        for run_idx in sorted(runs.keys()):
            r = ExperimentResult(scheduler_name=scheduler)
            r.fragmentation_curve = sorted(runs[run_idx]['frag_rate'], key=lambda p: p[0])
            r.frag_total_curve = sorted(runs[run_idx]['frag_total'], key=lambda p: p[0])
            results[scheduler].append(r)

    return results


def plot_figure7(results: Dict[str, List[ExperimentResult]], output_dir: str):
    """
    Plot Figure 7 as one image with two sub-panels:
      (a) Fragmentation rate
      (b) Fragmented GPUs / total resources
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping Figure 7 plot.")
        return

    import re

    matplotlib.rcdefaults()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams.update({"font.size": 16})
    matplotlib.rcParams['lines.linewidth'] = 3
    matplotlib.rcParams['savefig.bbox'] = None
    matplotlib.rcParams['savefig.pad_inches'] = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(8.33, 6.00), dpi=100)

    styles = {
        'Random': {'color': 'brown', 'linestyle': '-.'},
        'DotProd': {'color': 'purple', 'linestyle': '--'},
        'Clustering': {'color': 'red', 'linestyle': '--'},
        'Packing': {'color': 'darkgreen', 'linestyle': ':'},
        'BestFit': {'color': 'orange', 'linestyle': '--'},
        'BestFit-PN': {'color': '#fdbf6f', 'linestyle': '--'},
        'FGD': {'color': 'black', 'linestyle': '-'},
        'FGD-Full': {'color': '#1f77b4', 'linestyle': '-'},
        'B-FGD': {'color': '#e377c2', 'linestyle': '--'},
        'U-FGD': {'color': '#17becf', 'linestyle': '-.'},
    }

    variant_colors = [
        '#000000', '#1f77b4', '#17becf', '#e377c2', '#7f7f7f',
        '#bcbd22', '#2ca02c', '#9467bd', '#8c564b', '#ff7f0e'
    ]
    variant_styles = ['-', '--', '-.', ':']
    variant_style_map: Dict[str, Dict[str, str]] = {}
    variant_names = sorted([
        n for n in results.keys()
        if re.fullmatch(r'FGD-\d+', n) or re.fullmatch(r'W-FGD-\d+', n)
        or (n.startswith('FGD') and n not in styles)
    ])
    for i, name in enumerate(variant_names):
        variant_style_map[name] = {
            'color': variant_colors[i % len(variant_colors)],
            'linestyle': variant_styles[(i // len(variant_colors)) % len(variant_styles)]
        }

    def _style_for(name: str) -> Dict[str, str]:
        if name in styles:
            return styles[name]
        if name in variant_style_map:
            return variant_style_map[name]
        return {'color': 'black', 'linestyle': '-'}

    order = ['Random', 'DotProd', 'Clustering', 'Packing', 'BestFit', 'BestFit-PN', 'FGD', 'FGD-Full', 'B-FGD', 'U-FGD']
    names = [n for n in order if n in results]
    names.extend([n for n in results.keys() if n not in names])

    # 7(a)
    ax = axes[0]
    for name in names:
        avg_curve = _average_curve_by_key(results[name], 'fragmentation_curve')
        if not avg_curve:
            continue
        x_vals = [p[0] for p in avg_curve]
        y_vals = [p[1] for p in avg_curve]
        style = _style_for(name)
        ax.plot(x_vals, y_vals, label=name, color=style['color'], linestyle=style['linestyle'])

    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)', fontsize=14)
    ax.set_ylabel('Frag Rate (%)', fontsize=14)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(linestyle='-.', alpha=0.65)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.03), fontsize=12, frameon=False)

    # 7(b)
    ax = axes[1]
    max_y_7b = 0.0
    for name in names:
        avg_curve = _average_curve_by_key(results[name], 'frag_total_curve')
        if not avg_curve:
            continue
        x_vals = [p[0] for p in avg_curve]
        y_vals = [p[1] for p in avg_curve]
        if y_vals:
            max_y_7b = max(max_y_7b, max(y_vals))
        style = _style_for(name)
        ax.plot(x_vals, y_vals, label=name, color=style['color'], linestyle=style['linestyle'])

    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)', fontsize=14)
    ax.set_ylabel('Frag / Total (%)', fontsize=14)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    # Dynamic y-range from actual data, with 5 intervals (6 ticks) from 0 to max.
    import math
    y_max_7b = max(5.0, max_y_7b * 1.05)
    y_max_7b = math.ceil(y_max_7b / 5.0) * 5.0
    ax.set_ylim(0, y_max_7b)
    ax.set_yticks(np.linspace(0, y_max_7b, 6))
    ax.grid(linestyle='-.', alpha=0.65)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.03), fontsize=12, frameon=False)

    fig.subplots_adjust(left=0.10, right=0.77, top=0.97, bottom=0.20, hspace=0.72)
    captions = [
        "(a) Fragmentation rate grows to 100% as more resources are allocated.",
        "(b) Percentage of fragmented GPUs to total resources under our measure.",
    ]
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    fig.text(0.5, p0.y0 - 0.11, captions[0], ha='center', va='top', fontsize=13, family='serif')
    fig.text(0.5, p1.y0 - 0.11, captions[1], ha='center', va='top', fontsize=13, family='serif')

    path = os.path.join(output_dir, 'figure7.png')
    plt.savefig(path, dpi=100, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print(f"Figure 7 plot saved to {path}")


def save_figure9_results_to_csv(results: Dict[str, List[ExperimentResult]], output_dir: str):
    """Save Figure 9 artifacts as CSV files."""
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

    print(f"Figure 9 CSV files saved to {output_dir}")


def load_figure9_results_from_csv(csv_dir: str) -> Dict[str, List[ExperimentResult]]:
    """Load Figure 9 artifacts from CSV files in a directory."""
    import csv

    def load_curve_csv(filename: str, x_col: str, y_col: str):
        path = os.path.join(csv_dir, filename)
        if not os.path.exists(path):
            return {}
        data = defaultdict(lambda: defaultdict(list))
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['scheduler']
                run = int(row['run'])
                data[name][run].append((float(row[x_col]), float(row[y_col])))
        return data

    def load_cat_csv(filename: str, cat_col: str, val_col: str):
        path = os.path.join(csv_dir, filename)
        if not os.path.exists(path):
            return {}
        data = defaultdict(lambda: defaultdict(dict))
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['scheduler']
                run = int(row['run'])
                data[name][run][row[cat_col]] = float(row[val_col])
        return data

    unalloc = load_curve_csv('figure9a_unalloc.csv', 'arrived_pct', 'unalloc_gpu_pct')
    occupied = load_curve_csv('figure9b_occupied.csv', 'arrived_pct', 'occupied_nodes')
    failed = load_cat_csv('figure9c_failed.csv', 'gpu_category', 'sum_gpu_demand')
    breakdown = load_cat_csv('figure9d_breakdown.csv', 'cause', 'pct')

    results: Dict[str, List[ExperimentResult]] = {}
    all_names = set(unalloc) | set(occupied) | set(failed) | set(breakdown)
    for name in all_names:
        runs = set()
        for data in (unalloc, occupied, failed, breakdown):
            if name in data:
                runs |= set(data[name].keys())

        results[name] = []
        for run_idx in sorted(runs):
            r = ExperimentResult(scheduler_name=name)
            if name in unalloc and run_idx in unalloc[name]:
                r.unalloc_curve = sorted(unalloc[name][run_idx])
            if name in occupied and run_idx in occupied[name]:
                r.occupied_curve = sorted(occupied[name][run_idx])
            if name in failed and run_idx in failed[name]:
                r.failed_by_category = failed[name][run_idx]
            if name in breakdown and run_idx in breakdown[name]:
                r.frag_breakdown = breakdown[name][run_idx]
            results[name].append(r)

    return results


def plot_figure9(results: Dict[str, List[ExperimentResult]], output_dir: str):
    """Plot Figure 9 (a-d) and save to figure9.png."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import math
    except ImportError:
        print("matplotlib/numpy not installed. Skipping figure9 plot.")
        return

    import re

    styles = {
        'Random': {'color': 'brown', 'linestyle': '-.'},
        'DotProd': {'color': 'purple', 'linestyle': '-.'},
        'Clustering': {'color': 'red', 'linestyle': '--'},
        'Packing': {'color': 'green', 'linestyle': ':'},
        'BestFit': {'color': 'orange', 'linestyle': '--'},
        'BestFit-PN': {'color': '#fdbf6f', 'linestyle': '--'},
        'FGD': {'color': 'black', 'linestyle': '-'},
        'FGD-Full': {'color': '#1f77b4', 'linestyle': '-'},
        'B-FGD': {'color': '#e377c2', 'linestyle': '--'},
        'U-FGD': {'color': '#17becf', 'linestyle': '-.'},
    }

    # Distinct palette for FGD variants (FGD-<N>, W-FGD-<M>, and other FGD-prefixed names).
    fgd_variant_colors = [
        '#000000', '#1f77b4', '#17becf', '#e377c2', '#7f7f7f',
        '#bcbd22', '#2ca02c', '#9467bd', '#8c564b', '#ff7f0e'
    ]
    fgd_variant_styles = ['-', '--', '-.', ':']
    variant_style_map: Dict[str, Dict[str, str]] = {}

    variant_names = sorted(
        [
            n for n in results.keys()
            if re.fullmatch(r'FGD-\d+', n) or re.fullmatch(r'W-FGD-\d+', n)
            or (n.startswith('FGD') and n not in styles)
        ]
    )
    for i, name in enumerate(variant_names):
        variant_style_map[name] = {
            'color': fgd_variant_colors[i % len(fgd_variant_colors)],
            'linestyle': fgd_variant_styles[(i // len(fgd_variant_colors)) % len(fgd_variant_styles)]
        }

    def _style_for(name: str) -> Dict[str, str]:
        """Return plotting style, ensuring FGD variants are visually distinct."""
        if name in styles:
            return styles[name]
        if name in variant_style_map:
            return variant_style_map[name]
        return {'color': 'black', 'linestyle': '-'}

    # Reset any global style changes (e.g., from plot_figure7) and set Figure 9 sizes explicitly.
    matplotlib.rcdefaults()
    fig, axes = plt.subplots(4, 1, figsize=(5.9, 9), dpi=100)

    # 9(a): Unallocated GPU %
    ax = axes[0]
    all_x_9a: List[float] = []
    all_y_9a: List[float] = []
    averaged_curves_9a: Dict[str, List[Tuple[float, float]]] = {}

    for name, result_list in results.items():
        avg = _average_curves([r.unalloc_curve for r in result_list])
        if not avg:
            continue
        averaged_curves_9a[name] = avg
        for x, y in avg:
            all_x_9a.append(x)
            all_y_9a.append(y)

    if all_x_9a:
        x_min_9a = math.floor(min(all_x_9a) / 5.0) * 5
        x_max_9a = math.ceil(max(all_x_9a) / 5.0) * 5
    else:
        x_min_9a, x_max_9a = 0, 120

    ideal_x = list(range(int(x_min_9a), int(x_max_9a) + 1))
    ideal_y = [max(0, 100 - x) for x in ideal_x]
    ax.plot(ideal_x, ideal_y, color='gray', linestyle=':', linewidth=1.5, label='Ideal')

    for name, avg in averaged_curves_9a.items():
        filtered = [(x, y) for x, y in avg if x_min_9a <= x <= x_max_9a]
        if not filtered:
            continue
        style = _style_for(name)
        ax.plot(
            [p[0] for p in filtered],
            [p[1] for p in filtered],
            label=name,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2
        )

    # Dynamic axis range for 9(a)
    if all_y_9a:
        y_min_9a = 0
        y_max_9a = max(5, math.ceil((max(all_y_9a) * 1.10) / 5.0) * 5)
    else:
        y_min_9a, y_max_9a = 0, 25

    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)')
    ax.set_ylabel('Unalloc. GPU (%)')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlim(40, 100)
    ax.set_xticks(np.arange(40, 100.1, 10))
    ax.set_ylim(0, 70)
    ax.set_yticks(np.arange(0, 70.1, 10))
    ax.grid(True, linestyle='--', alpha=0.35)

    # 9(b): Occupied nodes
    ax = axes[1]
    for name, result_list in results.items():
        avg = _average_curves([r.occupied_curve for r in result_list])
        if not avg:
            continue
        filtered = [(x, y) for x, y in avg if x >= 0]
        if not filtered:
            continue
        if filtered[0][0] > 0:
            filtered = [(0.0, 0.0)] + filtered
        style = _style_for(name)
        ax.plot(
            [p[0] for p in filtered],
            [p[1] for p in filtered],
            label=name,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2
        )

    # Dynamic axis range for 9(b)
    all_x_9b: List[float] = []
    all_y_9b: List[float] = []
    for result_list in results.values():
        avg = _average_curves([r.occupied_curve for r in result_list])
        for x, y in avg:
            all_x_9b.append(x)
            all_y_9b.append(y)
    if all_x_9b:
        x_min_9b = max(0.0, math.floor(min(all_x_9b) / 5.0) * 5)
        x_max_9b = math.ceil(max(all_x_9b) / 5.0) * 5
    else:
        x_min_9b, x_max_9b = 0, 100
    if all_y_9b:
        y_min_9b = 0
        y_max_9b = max(10, math.ceil((max(all_y_9b) * 1.10) / 50.0) * 50)
    else:
        y_min_9b, y_max_9b = 0, 1000

    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)')
    ax.set_ylabel('Occupied nodes')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_xlim(x_min_9b, x_max_9b)
    xtick_step_9b = 10 if (x_max_9b - x_min_9b) <= 80 else 20
    ax.set_xticks(np.arange(x_min_9b, x_max_9b + 0.1, xtick_step_9b))
    ax.set_ylim(y_min_9b, y_max_9b)
    ytick_step_9b = 100 if y_max_9b <= 800 else 250
    ax.set_yticks(np.arange(y_min_9b, y_max_9b + 0.1, ytick_step_9b))
    ax.grid(True, linestyle='--', alpha=0.35)

    # Fixed scheduler order for bar charts
    bar_order = ['FGD', 'FGD-Full', 'B-FGD', 'U-FGD', 'BestFit', 'BestFit-PN', 'Packing', 'Clustering', 'DotProd', 'Random']
    scheduler_names = [n for n in bar_order if n in results]
    # Always append any additional schedulers (e.g., FGD-500, W-FGD-500, custom names).
    scheduler_names.extend([n for n in results.keys() if n not in scheduler_names])
    x_pos = np.arange(len(scheduler_names))

    # 9(c): Failed tasks by category
    ax = axes[2]
    categories = ['<1', '1', '2', '8']  # bottom -> top
    cat_colors = {'<1': 'orange', '1': 'green', '2': 'red', '8': '#8c564b'}

    bottoms = np.zeros(len(scheduler_names))
    cat_handles = {}
    for cat in categories:
        values = []
        for name in scheduler_names:
            avg_val = sum(r.failed_by_category.get(cat, 0) for r in results[name]) / len(results[name])
            values.append(avg_val)
        cat_handles[cat] = ax.bar(
            x_pos, values, bottom=bottoms, label=cat,
            color=cat_colors[cat], edgecolor='0', linewidth=0.3
        )
        bottoms += np.array(values)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(scheduler_names, rotation=0, fontsize=10)
    ax.set_ylabel('Sum of Pending Task GPUs')
    ax.set_title('When arrived workloads equals 96% GPU capacity', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    max_stack_9c = float(max(bottoms)) if len(bottoms) > 0 else 0.0
    y_max_9c = max(10, math.ceil((max_stack_9c * 1.10) / 50.0) * 50)
    ax.set_ylim(0, y_max_9c)
    # Keep exactly 10 intervals on y-axis (11 ticks including 0 and max).
    ax.set_yticks(np.linspace(0, y_max_9c, 11))
    legend_order_top_to_bottom = ['8', '2', '1', '<1']
    ax.legend(
        handles=[cat_handles[c][0] for c in legend_order_top_to_bottom],
        labels=legend_order_top_to_bottom,
        title='Task GPU Req',
        fontsize=8,
        title_fontsize=9,
        loc='upper left',
        frameon=True,
        borderpad=0.6,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.4,
    )
    ax.grid(True, linestyle='--', alpha=0.35, axis='y')

    # 9(d): fragmentation breakdown
    ax = axes[3]
    causes = ['deficient', 'stranded', 'non_gpu']
    cause_colors = {'deficient': 'blue', 'stranded': 'orange', 'non_gpu': 'green'}
    cause_labels = {'deficient': 'Deficient', 'stranded': 'Stranded', 'non_gpu': 'Non-GPU'}

    bottoms = np.zeros(len(scheduler_names))
    for cause in causes:
        values = []
        for name in scheduler_names:
            avg_val = sum(r.frag_breakdown.get(cause, 0) for r in results[name]) / len(results[name])
            values.append(avg_val)
        ax.bar(
            x_pos, values, bottom=bottoms, label=cause_labels[cause],
            color=cause_colors[cause], edgecolor='0', linewidth=0.3
        )
        bottoms += np.array(values)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(scheduler_names, rotation=0, fontsize=10)
    ax.set_ylabel('Fragmented GPUs (%)')
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=9, loc='lower left', frameon=True)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, linestyle='--', alpha=0.35, axis='y')

    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.07, hspace=1.05)

    # 9(a)/(b) slightly narrower to leave legend room
    left = 0.10
    right_line = 0.82
    right_bar = 0.98
    for i in [0, 1]:
        p = axes[i].get_position()
        axes[i].set_position([left, p.y0, right_line - left, p.height])
    for i in [2, 3]:
        p = axes[i].get_position()
        axes[i].set_position([left, p.y0, right_bar - left, p.height])

    captions = [
        "(a) The percentage of unallocated GPUs given arriving workloads.",
        "(b) The number of GPU nodes occupied during the scheduling.",
        "(c) GPU requests of failed tasks when the cluster is almost full\n(i.e., cumulative GPU requests reach 96% of the cluster capacity).",
        "(d) The breakdown of GPU fragmentation into three causes.",
    ]
    caption_offsets = [0.060, 0.060, 0.050, 0.045]
    for i, cap in enumerate(captions):
        p = axes[i].get_position()
        fig.text(0.5, p.y0 - caption_offsets[i], cap, ha='center', va='top', fontsize=9)

    path = os.path.join(output_dir, 'figure9.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 9 plot saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distribution-Shift Experiment - Trace Replay")
    parser.add_argument('--task-order', choices=['trace', 'ascending', 'descending', 'phased'], default='trace',
                        help='Task arrival order: trace (original), ascending/descending (sorted by GPU type), phased (GPU tier phases)')
    parser.add_argument('--cluster-scale', type=float, default=100.0,
                        help='Cluster size as %% of original (e.g., 50 keeps 50%% of each node type)')
    parser.add_argument('--tier-order', type=str, default='0,1,2,3,4',
                        help='Tier order for phased mode (comma-separated, e.g., 3,2,1,4,0)')
    parser.add_argument('--prior-strength', type=float, default=10.0,
                        help='Prior strength for B-FGD (pseudo-count total, default: 10)')
    parser.add_argument('--min-gpu-tasks', type=int, default=50,
                        help='B-FGD uses Packing fallback until this many GPU tasks observed (default: 50)')
    parser.add_argument('--fgd-popularity-threshold', type=float, default=95,
                        help='Top-N%% typical pod popularity threshold used for FGD scoring (default: 95)')
    parser.add_argument('--sample-interval-pct', type=float, default=2.0,
                        help='Figure 9 sample interval in arrived GPU %% (default: 2)')
    parser.add_argument('--max-arrival-pct', type=float, default=120.0,
                        help='Stop replay when cumulative GPU requests reach this %% of cluster capacity (default: 120)')
    parser.add_argument('--snapshot-pct', type=float, default=96.0,
                        help='Figure 9(c) snapshot point in arrived GPU %% (default: 96)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating figure9.png (still saves CSVs)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate figure9.png from existing CSV files')
    parser.add_argument('--plot-csv', type=str, default=None,
                        help='CSV directory for --plot-only (default: inferred result dir)')
    parser.add_argument('--schedulers', type=str, default='all',
                        help='Comma-separated scheduler names to run (default: all). '
                             'Baselines: Random,BestFit,DotProd,Packing,Clustering. '
                             'FGD variants: FGD-Full, FGD-<N> (first-N distribution, e.g. FGD-500), '
                             'W-FGD-<M> (sliding window size M, e.g. W-FGD-200), B-FGD, U-FGD. '
                             'For FGD-<N>/W-FGD-<M>, N/M are parsed from the name.')
    args = parser.parse_args()

    # Parse tier order
    args.tier_order_list = [int(x) for x in args.tier_order.split(',')]

    # Result directory naming (shared by normal and plot-only mode)
    scale_str = f"{args.cluster_scale:g}"
    order_str = args.task_order
    if args.task_order == 'phased':
        order_str = f"phased-{''.join(str(x) for x in args.tier_order_list)}"
    result_name = f"dist-shift-{order_str}-{scale_str}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)

    # Plot-only mode: load CSV artifacts and regenerate figure9.png.
    if args.plot_only:
        csv_dir = args.plot_csv if args.plot_csv else result_dir
        plotted_any = False

        fig7_csv = os.path.join(csv_dir, 'figure7_results.csv')
        if os.path.exists(fig7_csv):
            fig7_results = load_figure7_results_from_csv(fig7_csv)
            if fig7_results:
                print(f"Loaded {len(fig7_results)} schedulers for Figure 7 from {fig7_csv}")
                plot_figure7(fig7_results, csv_dir)
                plotted_any = True

        fig9_results = load_figure9_results_from_csv(csv_dir)
        if fig9_results:
            print(f"Loaded {len(fig9_results)} schedulers for Figure 9 from {csv_dir}")
            plot_figure9(fig9_results, csv_dir)
            plotted_any = True

        if not plotted_any:
            print(f"No Figure 7/9 CSV data found in {csv_dir}")
            exit(1)
        exit(0)

    # Run the experiment
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'alibaba_traces', 'cluster-trace-gpu-v2023')

    print("=" * 60)
    print("Distribution-Shift Experiment - Trace Replay")
    print(f"  task_order={args.task_order}")
    if args.task_order == 'phased':
        print(f"  tier_order={args.tier_order}")
    print(f"  cluster_scale={args.cluster_scale}%")
    print("=" * 60)

    experiment = DistShiftExperiment(
        data_dir,
        task_order=args.task_order, cluster_scale=args.cluster_scale,
        tier_order=args.tier_order_list,
        fgd_popularity_threshold=args.fgd_popularity_threshold
    )

    # Build scheduler list
    if args.schedulers == 'all':
        fgd_full = FGDScheduler()
        fgd_full.name = "FGD-Full"
        fgd_n = FGDScheduler(
            scheduling_task_types=experiment.first_n_distribution.get_task_types()
        )
        fgd_n.name = f"FGD-{experiment.window_size}"
        w_fgd = WindowedFGDScheduler(window_size=experiment.window_size)
        b_fgd = BayesianFGDScheduler(prior_strength=args.prior_strength,
                                      min_gpu_tasks=args.min_gpu_tasks)
        uniform_types = build_uniform_task_types(experiment.loader.nodes)
        u_fgd = FGDScheduler(scheduling_task_types=uniform_types)
        u_fgd.name = "U-FGD"
        baselines = [s for s in get_all_schedulers() if not isinstance(s, FGDScheduler)]
        schedulers = baselines + [fgd_full, fgd_n, w_fgd, b_fgd, u_fgd]
    else:
        # For explicit names, N/M are parsed directly from the scheduler name.
        selected = [s.strip() for s in args.schedulers.split(',')]
        schedulers = []
        for name in selected:
            sched = build_scheduler(name, experiment,
                                    args.prior_strength, args.min_gpu_tasks)
            if sched is not None:
                schedulers.append(sched)
            else:
                print(f"WARNING: Unknown scheduler '{name}'. "
                      f"Available: Random, BestFit, DotProd, Packing, Clustering, "
                      f"FGD-Full, FGD-<N>, W-FGD-<M>, B-FGD, U-FGD")
        if not schedulers:
            print("No valid schedulers selected. Exiting.")
            exit(1)

    import re as _re
    for s in schedulers:
        m = _re.fullmatch(r'FGD-(\d+)', s.name)
        if m:
            n = int(m.group(1))
            n_types = len(experiment._compute_initial_distribution(n).get_task_types())
            print(f"First-{n} distribution ({s.name}): {n_types} task types")
    if any(isinstance(s, BayesianFGDScheduler) for s in schedulers):
        print(f"  prior_strength={args.prior_strength}, min_gpu_tasks={args.min_gpu_tasks}")

    # Run trace replay
    results = experiment.run_experiment(
        schedulers=schedulers,
        sample_interval_pct=args.sample_interval_pct,
        max_arrival_pct=args.max_arrival_pct,
        snapshot_pct=args.snapshot_pct,
    )

    # Create result directory
    os.makedirs(result_dir, exist_ok=True)

    # Print summary
    print_summary(results)

    # Save summary log
    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write(f"Experiment: Distribution-Shift - Trace Replay\n")
        f.write(f"Result: {result_name}\n")
        f.write(f"Mode: replay\n")
        f.write(f"FGD popularity threshold: {args.fgd_popularity_threshold}\n")
        f.write(f"Figure9 sample interval (%): {args.sample_interval_pct}\n")
        f.write(f"Figure9 max arrival (%): {args.max_arrival_pct}\n")
        f.write(f"Figure9 snapshot (%): {args.snapshot_pct}\n")
        if any(isinstance(s, BayesianFGDScheduler) for s in schedulers):
            f.write(f"Prior strength (B-FGD): {args.prior_strength}\n")
            f.write(f"Min GPU tasks (B-FGD): {args.min_gpu_tasks}\n")
        f.write(f"Task order: {args.task_order}\n")
        if args.task_order == 'phased':
            f.write(experiment.phase_info + "\n")
        f.write(f"Cluster: {experiment.original_node_count} nodes -> {experiment.scaled_node_count} nodes ({args.cluster_scale}%)\n")
        f.write(f"Full distribution: {len(experiment.full_task_distribution.get_task_types())} task types\n")
        f.write(
            f"FGD scoring distribution (top {args.fgd_popularity_threshold:g}%): "
            f"{len(experiment.fgd_scoring_distribution.get_task_types())} task types\n"
        )
        import re as _re
        for s in schedulers:
            m = _re.fullmatch(r'FGD-(\d+)', s.name)
            if m:
                n = int(m.group(1))
                n_types = len(experiment._compute_initial_distribution(n).get_task_types())
                f.write(
                    f"First-{n} distribution ({s.name}, top {args.fgd_popularity_threshold:g}%): "
                    f"{n_types} task types\n"
                )
        f.write("\n")
        f.write(format_summary(results) + "\n")
    print(f"Summary log saved to {log_path}")

    # Save and plot Figure 7 artifacts from the same replay.
    save_figure7_results_to_csv(results, result_dir)

    # Save and plot Figure 9 artifacts from the same replay.
    save_figure9_results_to_csv(results, result_dir)
    if not args.no_plot:
        plot_figure7(results, result_dir)
        plot_figure9(results, result_dir)
