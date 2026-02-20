"""
Distribution-Shift Experiment Runner for FGD Replication

Processes tasks in trace order and measures fragmentation under distribution shift.
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

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
    final_frag_rate: float = 0.0
    final_gpu_alloc_rate: float = 0.0
    tasks_scheduled: int = 0
    tasks_failed: int = 0
    elapsed_sec: float = 0.0


class DistShiftExperiment:
    """
    Distribution-Shift Experiment: processes tasks in creation_time order from the trace
    and measures fragmentation under various scheduling policies.
    """

    def __init__(self, data_dir: str, window_size: int = 500, task_order: str = 'trace',
                 cluster_scale: float = 100.0, tier_order: List[int] = None):
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
        self.first_n_distribution = self._compute_initial_distribution(window_size)

        # Get cluster capacity
        self.total_gpu_capacity = sum(n.num_gpus for n in self.loader.nodes)

        print(f"Loaded trace: {len(self.loader.nodes)} nodes, {self.total_gpu_capacity} GPUs")
        print(f"Tasks in trace: {len(self.loader.tasks)}")
        print(f"Task order: {task_order}")
        print(f"Full distribution: {len(self.full_task_distribution.get_task_types())} task types")

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

        pbar = tqdm(
            total=len(tasks),
            desc=f"{scheduler.name:16}",
            unit="task",
            disable=not show_progress,
            ncols=90
        )

        t_start = time.monotonic()
        for task in tasks:
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

            pbar.update(1)

        pbar.close()

        # Record final metrics
        result.elapsed_sec = time.monotonic() - t_start
        result.final_frag_rate = cluster.compute_fragmentation_rate()
        result.final_gpu_alloc_rate = cluster.gpu_allocation_rate

        return result

    def run_experiment(
        self,
        schedulers: List[Scheduler] = None,
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
                show_progress=show_progress
            )
            results[scheduler.name].append(result)

            print(f"  {scheduler.name:16}: Frag={result.final_frag_rate:.1f}%, "
                  f"Alloc={result.final_gpu_alloc_rate:.1f}%, "
                  f"Scheduled={result.tasks_scheduled}, Failed={result.tasks_failed}")

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
    parser.add_argument('--schedulers', type=str, default='all',
                        help='Comma-separated scheduler names to run (default: all). '
                             'Baselines: Random,BestFit,DotProd,Packing,Clustering. '
                             'FGD variants: FGD-Full, FGD-<N> (first-N distribution, e.g. FGD-500), '
                             'W-FGD-<M> (sliding window size M, e.g. W-FGD-200), B-FGD, U-FGD. '
                             'For FGD-<N>/W-FGD-<M>, N/M are parsed from the name.')
    args = parser.parse_args()

    # Parse tier order
    args.tier_order_list = [int(x) for x in args.tier_order.split(',')]

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
        tier_order=args.tier_order_list
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
    results = experiment.run_experiment(schedulers=schedulers)

    # Create result directory
    scale_str = f"{args.cluster_scale:g}"
    order_str = args.task_order
    if args.task_order == 'phased':
        order_str = f"phased-{''.join(str(x) for x in args.tier_order_list)}"
    result_name = f"dist-shift-{order_str}-{scale_str}"
    result_dir = os.path.join(os.path.dirname(__file__), 'result', result_name)
    os.makedirs(result_dir, exist_ok=True)

    # Print summary
    print_summary(results)

    # Save summary log
    log_path = os.path.join(result_dir, 'experiment_summary.log')
    with open(log_path, 'w') as f:
        f.write(f"Experiment: Distribution-Shift - Trace Replay\n")
        f.write(f"Result: {result_name}\n")
        f.write(f"Mode: replay\n")
        if any(isinstance(s, BayesianFGDScheduler) for s in schedulers):
            f.write(f"Prior strength (B-FGD): {args.prior_strength}\n")
            f.write(f"Min GPU tasks (B-FGD): {args.min_gpu_tasks}\n")
        f.write(f"Task order: {args.task_order}\n")
        if args.task_order == 'phased':
            f.write(experiment.phase_info + "\n")
        f.write(f"Cluster: {experiment.original_node_count} nodes -> {experiment.scaled_node_count} nodes ({args.cluster_scale}%)\n")
        f.write(f"Full distribution: {len(experiment.full_task_distribution.get_task_types())} task types\n")
        import re as _re
        for s in schedulers:
            m = _re.fullmatch(r'FGD-(\d+)', s.name)
            if m:
                n = int(m.group(1))
                n_types = len(experiment._compute_initial_distribution(n).get_task_types())
                f.write(f"First-{n} distribution ({s.name}): {n_types} task types\n")
        f.write("\n")
        f.write(format_summary(results) + "\n")
    print(f"Summary log saved to {log_path}")
