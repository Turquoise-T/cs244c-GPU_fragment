"""
Scheduling Policies for GPU Cluster Simulation

Implements 6 scheduling policies from the FGD paper (Section 6.1):
1. Random - Random node selection
2. BestFit - Node with least remaining resources
3. DotProd - Smallest dot-product between remaining resources and task demands
4. Packing - Prioritize occupied GPUs, then idle GPUs on occupied nodes
5. Clustering - Pack tasks with same GPU request together
6. FGD - Fragmentation Gradient Descent
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from collections import deque, Counter
import math
import random

from simulator import Task, Node, Cluster, TaskDistribution


class Scheduler(ABC):
    """Abstract base class for all scheduling policies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        """
        Select a node for the given task.

        Args:
            task: The task to schedule
            cluster: Current cluster state

        Returns:
            Node ID to schedule on, or None if no suitable node found
        """
        pass

    def schedule(self, task: Task, cluster: Cluster) -> bool:
        """
        Schedule a task on the cluster.

        Returns:
            True if scheduled successfully, False otherwise
        """
        node_id = self.select_node(task, cluster)
        if node_id is not None:
            return cluster.schedule_task(task, node_id)
        return False


class RandomScheduler(Scheduler):
    """
    Random-fit: Distributes tasks randomly to any node that meets requirements.
    """

    def __init__(self):
        super().__init__("Random")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None
        return random.choice(eligible).node_id


class BestFitScheduler(Scheduler):
    """
    Best-fit: Assigns tasks to the node with the least remaining resources.
    Score = 0.5 * free_cpu / MaxSpecCpu + 0.5 * free_gpu / MaxSpecGpu
    matching the author's formula (best_fit_score.go), where MaxSpec values
    are the global cluster-wide maximums computed once on first scheduling call.
    """

    def __init__(self):
        super().__init__("BestFit")
        self._max_cpu: Optional[float] = None
        self._max_gpu: Optional[float] = None

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Compute global max specs once and cache (cluster topology is fixed)
        if self._max_cpu is None:
            self._max_cpu = max(n.total_cpu for n in cluster.nodes)
            self._max_gpu = max(n.num_gpus for n in cluster.nodes)

        best_node = None
        best_score = float('inf')

        for node in eligible:
            cpu_score = 0.5 * node.remaining_cpu / self._max_cpu if self._max_cpu > 0 else 0
            gpu_score = 0.5 * node.total_unallocated_gpu / self._max_gpu if self._max_gpu > 0 else 0
            score = cpu_score + gpu_score

            if score < best_score:
                best_score = score
                best_node = node

        return best_node.node_id if best_node else None


class BestFitLocalScheduler(Scheduler):
    """
    Best-fit with per-node normalization (original/naive implementation).

    Score = remaining_cpu / node.total_cpu + total_unallocated_gpu / node.num_gpus

    Kept for comparison against BestFitScheduler (global normalization).
    Per-node normalization treats a 4-GPU node at 50% as equivalent to an
    8-GPU node at 50%, which is incorrect for heterogeneous clusters.
    """

    def __init__(self):
        super().__init__("BestFit-PN")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        best_node = None
        best_score = float('inf')

        for node in eligible:
            cpu_score = node.remaining_cpu / node.total_cpu if node.total_cpu > 0 else 0
            gpu_score = node.total_unallocated_gpu / node.num_gpus if node.num_gpus > 0 else 0
            score = cpu_score + gpu_score

            if score < best_score:
                best_score = score
                best_node = node

        return best_node.node_id if best_node else None


class DotProdScheduler(Scheduler):
    """
    Dot-product: Allocates to node with smallest dot-product between
    remaining resources and task demands.
    """

    def __init__(self):
        super().__init__("DotProd")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        best_node = None
        best_score = float('inf')

        # Normalize demands
        max_cpu = max(n.total_cpu for n in cluster.nodes)
        max_gpu = max(n.num_gpus for n in cluster.nodes)

        task_cpu_norm = task.cpu_demand / max_cpu if max_cpu > 0 else 0
        task_gpu_norm = task.gpu_demand / max_gpu if max_gpu > 0 else 0

        for node in eligible:
            # Normalize remaining resources
            cpu_norm = node.remaining_cpu / max_cpu if max_cpu > 0 else 0
            gpu_norm = node.total_unallocated_gpu / max_gpu if max_gpu > 0 else 0

            # Dot product
            score = cpu_norm * task_cpu_norm + gpu_norm * task_gpu_norm

            if score < best_score:
                best_score = score
                best_node = node

        return best_node.node_id if best_node else None


class PackingScheduler(Scheduler):
    """
    GPU Packing: Prioritizes task assignment to:
    1. Occupied GPUs (partial GPUs on nodes with some allocation)
    2. Idle GPUs on occupied nodes
    3. Fully idle nodes

    The intuition is to reserve available resources for multi-GPU tasks.
    """

    def __init__(self):
        super().__init__("Packing")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Categorize nodes
        occupied_partial = []  # Nodes with partial GPUs
        occupied_full = []     # Occupied nodes with only full GPUs available
        idle = []              # Fully idle nodes

        for node in eligible:
            has_partial = any(0 < g < 1.0 for g in node.gpu_remaining)
            is_occupied = node.allocated_cpu > 0 or any(g < 1.0 for g in node.gpu_remaining)

            if has_partial:
                occupied_partial.append(node)
            elif is_occupied:
                occupied_full.append(node)
            else:
                idle.append(node)

        # Priority: occupied with partial GPUs > occupied > idle
        # Within each category, prefer node with less remaining (pack tighter)
        def sort_key(n):
            return n.total_unallocated_gpu

        if occupied_partial:
            occupied_partial.sort(key=sort_key)
            return occupied_partial[0].node_id
        elif occupied_full:
            occupied_full.sort(key=sort_key)
            return occupied_full[0].node_id
        elif idle:
            idle.sort(key=sort_key)
            return idle[0].node_id

        return None


class ClusteringScheduler(Scheduler):
    """
    GPU Clustering: Packs tasks requesting the same GPU amount together.
    Avoids heterogeneous distribution of task resource requirements on the same node.
    """

    def __init__(self):
        super().__init__("Clustering")
        # Track which nodes have which GPU request patterns
        self.node_gpu_patterns: dict = {}  # node_id -> set of gpu_demands seen

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Categorize: nodes that already have this GPU pattern vs others
        matching_nodes = []
        other_nodes = []

        for node in eligible:
            if node.node_id in self.node_gpu_patterns:
                patterns = self.node_gpu_patterns[node.node_id]
                if task.gpu_demand in patterns:
                    matching_nodes.append(node)
                else:
                    other_nodes.append(node)
            else:
                other_nodes.append(node)

        # Prefer nodes with matching patterns, then others
        # Within each group, use best-fit (least remaining resources)
        def sort_key(n):
            return n.total_unallocated_gpu

        if matching_nodes:
            matching_nodes.sort(key=sort_key)
            selected = matching_nodes[0]
        elif other_nodes:
            other_nodes.sort(key=sort_key)
            selected = other_nodes[0]
        else:
            return None

        # Update pattern tracking
        if selected.node_id not in self.node_gpu_patterns:
            self.node_gpu_patterns[selected.node_id] = set()
        self.node_gpu_patterns[selected.node_id].add(task.gpu_demand)

        return selected.node_id

    def reset(self):
        """Reset pattern tracking for new simulation"""
        self.node_gpu_patterns = {}


class FGDScheduler(Scheduler):
    """
    Fragmentation Gradient Descent (FGD):
    Schedules tasks towards the steepest descent of fragmentation.

    For each task, evaluates all nodes and selects the one that
    causes the minimum increase in fragmentation.

    Algorithm 1 from the paper.
    """

    def __init__(self, num_workers: int = None, scheduling_task_types=None):
        super().__init__("FGD")
        self.num_workers = num_workers
        self._pool = None
        # If set, use this for scheduling decisions instead of cluster's distribution
        self.scheduling_task_types = scheduling_task_types
        # If set, use GPU-type-aware distribution: [((cpu, gpu, gpu_spec), popularity)]
        self.typed_task_types = None
        # GPU slot index chosen by select_node for the upcoming schedule() call.
        # Valid only for partial GPU tasks; -1 means use default allocate_task().
        self._pending_slot: int = -1

    @staticmethod
    def _compute_frag_delta_for_node(args: Tuple) -> Tuple[int, float, int]:
        """
        Worker function to compute fragmentation delta for a single node.
        Used for parallel evaluation.

        Args:
            args: (node_id, remaining_cpu, gpu_remaining, num_gpus, total_cpu,
                   task_cpu, task_gpu, task_types, node_gpu_model)
                  task_types entries may be ((cpu, gpu), popularity) or
                  ((cpu, gpu, gpu_spec), popularity) — the latter enables
                  GPU-type-aware fragmentation computation.

        Returns:
            (node_id, fragmentation_delta)
        """
        (node_id, remaining_cpu, gpu_remaining, num_gpus, total_cpu,
         task_cpu, task_gpu, task_types, node_gpu_model) = args

        # Reconstruct node state
        node = Node(
            node_id=node_id,
            total_cpu=total_cpu,
            num_gpus=num_gpus,
            allocated_cpu=total_cpu - remaining_cpu,
            gpu_remaining=list(gpu_remaining)
        )

        task = Task(task_id=-1, cpu_demand=task_cpu, gpu_demand=task_gpu)

        def _frag_for_types(node_state):
            frag = 0.0
            for type_key, popularity in task_types:
                cpu, gpu = type_key[0], type_key[1]
                gpu_spec = type_key[2] if len(type_key) > 2 else ''
                # GPU type compatibility: if task requires a specific type and
                # this node's GPU model is incompatible, all unallocated GPUs
                # on this node are fragmented from that task's perspective.
                if gpu_spec and node_gpu_model:
                    allowed = set(gpu_spec.split('|'))
                    if node_gpu_model not in allowed:
                        frag += popularity * node_state.total_unallocated_gpu
                        continue
                dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
                frag += popularity * node_state.get_fragmentation_for_task(dummy)
            return frag

        # Compute fragmentation before (only for this node)
        frag_before = _frag_for_types(node)

        if task.is_partial_gpu():
            # For partial GPU tasks, evaluate each eligible GPU slot separately
            # and track which slot produces the minimum fragmentation delta.
            # The slot index is returned so FGD can place the task on the same
            # slot that was evaluated, keeping scoring and placement consistent.
            best_delta = float('inf')
            best_slot = -1
            node.allocated_cpu += task_cpu  # CPU is always consumed
            for i in range(num_gpus):
                if gpu_remaining[i] >= task_gpu:
                    node.gpu_remaining = list(gpu_remaining)
                    node.gpu_remaining[i] -= task_gpu
                    delta = _frag_for_types(node) - frag_before
                    if delta < best_delta:
                        best_delta = delta
                        best_slot = i
            return (node_id, best_delta, best_slot)
        else:
            # Full-GPU and no-GPU tasks have a single allocation path
            node.allocate_task(task)
            frag_after = _frag_for_types(node)
            return (node_id, frag_after - frag_before, -1)

    def _get_pool(self):
        """Lazy initialization of process pool"""
        if self._pool is None:
            from multiprocessing import Pool, cpu_count
            workers = self.num_workers or cpu_count()
            self._pool = Pool(workers)
        return self._pool

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Priority: typed (GPU-type-aware) > explicit override > cluster distribution
        if self.typed_task_types is not None:
            task_types = self.typed_task_types
        elif self.scheduling_task_types is not None:
            task_types = self.scheduling_task_types
        else:
            task_types = cluster.task_distribution.get_task_types()

        # Prepare arguments for parallel workers
        args_list = [
            (
                node.node_id,
                node.remaining_cpu,
                tuple(node.gpu_remaining),
                node.num_gpus,
                node.total_cpu,
                task.cpu_demand,
                task.gpu_demand,
                task_types,
                node.gpu_model,
            )
            for node in eligible
        ]

        # Parallel execution
        pool = self._get_pool()
        results = pool.map(FGDScheduler._compute_frag_delta_for_node, args_list)

        # Paper scores: int(sigmoid(-delta) * 100), higher = better.
        best_node_id = None
        best_score = -1
        self._pending_slot = -1
        for node_id, delta, slot in results:
            score = int(100.0 / (1.0 + math.exp(delta)))  # sigmoid(-delta)*100
            if score > best_score:
                best_score = score
                best_node_id = node_id
                self._pending_slot = slot

        return best_node_id

    def schedule(self, task: Task, cluster: Cluster) -> bool:
        """
        Override base schedule() to place partial GPU tasks on the exact slot
        that was scored in select_node(), not the best-fit slot from allocate_task().
        This ensures scoring and placement are consistent.
        """
        node_id = self.select_node(task, cluster)
        if node_id is None:
            return False

        if task.is_partial_gpu() and self._pending_slot >= 0:
            node = cluster.nodes[node_id]
            if node.allocate_to_slot(task, self._pending_slot):
                cluster.scheduled_tasks.append((task, node_id))
                return True
            return False
        else:
            return cluster.schedule_task(task, node_id)

    def cleanup(self):
        """Clean up the process pool"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


class WindowedFGDScheduler(FGDScheduler):
    """
    Distribution-shift-aware FGD using a sliding window.

    Instead of using the global task distribution (which assumes perfect
    future knowledge), this variant estimates the distribution online
    from the last `window_size` tasks observed.

    This addresses FGD's key assumption: that the task popularity
    distribution is known in advance. In production, distributions
    shift over time, so a sliding window provides a more realistic
    and adaptive estimate.
    """

    def __init__(self, window_size: int = 500, num_workers: int = None):
        super().__init__(num_workers=num_workers)
        self.name = f"W-FGD-{window_size}"
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._cached_task_types = None
        self._cache_dirty = True

    def observe_task(self, task: Task):
        """Record a task into the sliding window"""
        gpu_rounded = round(task.gpu_demand, 2)
        cpu_bucket = round(task.cpu_demand / 4) * 4
        self._window.append((cpu_bucket, gpu_rounded))
        self._cache_dirty = True

    def _get_windowed_task_types(self) -> List[Tuple[Tuple[float, float], float]]:
        """Compute task distribution from the sliding window"""
        if not self._cache_dirty and self._cached_task_types is not None:
            return self._cached_task_types

        if not self._window:
            return []

        counts = Counter(self._window)
        total = sum(counts.values())
        self._cached_task_types = [
            ((cpu, gpu), count / total) for (cpu, gpu), count in counts.items()
        ]
        self._cache_dirty = False
        return self._cached_task_types

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Use windowed distribution instead of global
        task_types = self._get_windowed_task_types()

        # Fall back to global distribution if window is empty
        if not task_types:
            if cluster.task_distribution is not None:
                task_types = cluster.task_distribution.get_task_types()
            else:
                return BestFitScheduler().select_node(task, cluster)

        # Prepare arguments for parallel workers
        args_list = [
            (
                node.node_id,
                node.remaining_cpu,
                tuple(node.gpu_remaining),
                node.num_gpus,
                node.total_cpu,
                task.cpu_demand,
                task.gpu_demand,
                task_types,
                node.gpu_model,
            )
            for node in eligible
        ]

        # Parallel execution
        pool = self._get_pool()
        results = pool.map(FGDScheduler._compute_frag_delta_for_node, args_list)

        best_node_id = None
        best_score = -1
        self._pending_slot = -1
        for node_id, delta, slot in results:
            score = int(100.0 / (1.0 + math.exp(delta)))
            if score > best_score:
                best_score = score
                best_node_id = node_id
                self._pending_slot = slot

        return best_node_id

    def reset(self):
        """Reset the sliding window"""
        self._window.clear()
        self._cached_task_types = None
        self._cache_dirty = True


class BayesianFGDScheduler(FGDScheduler):
    """
    Bayesian FGD: Starts with a uniform prior over a (cpu, gpu) grid derived
    from cluster specs, then performs Bayesian updates as tasks arrive.

    Grid: CPU bucketed by 4 ({0, 4, 8, ..., max_cpu}),
          GPU bucketed by 0.1 ({0.0, 0.1, 0.2, ..., max_gpu}).
    Prior: uniform weight across all grid cells, total = prior_strength.
    Update: each observed task adds 1 count to its bucket.

    Equivalent to Dirichlet-Multinomial posterior:
        p_m = (prior_count_m + observed_count_m) / total_count
    """

    def __init__(self, prior_strength: float = 10.0, min_gpu_tasks: int = 50,
                 num_workers: int = None):
        super().__init__(num_workers=num_workers)
        self.name = "B-FGD"
        self.prior_strength = prior_strength
        self.min_gpu_tasks = min_gpu_tasks  # Use Packing until this many GPU tasks observed
        self._type_counts: Counter = Counter()
        self._total_count: float = 0.0
        self._gpu_task_count: int = 0  # Number of GPU-demanding tasks observed
        self._cached_task_types = None
        self._cache_dirty = True
        self._fallback = PackingScheduler()

    def set_uniform_prior(self, max_cpu: int, max_gpu: int):
        """Initialize uniform prior over (cpu, gpu) grid from cluster specs.

        Grid: cpu in {0, 4, 8, ..., max_cpu}
              gpu in {0.0, 0.1, 0.2, ..., 1.0, 2.0, 3.0, ..., max_gpu}
        Each cell gets equal weight, total pseudo-counts = prior_strength.
        """
        self._type_counts.clear()
        max_cpu = int(max_cpu)
        max_gpu = int(max_gpu)
        cpu_values = list(range(0, max_cpu + 1, 4))
        # GPU: 0.0-1.0 by 0.1, then 2.0-max_gpu by 1.0
        gpu_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0..1.0
        gpu_values += list(range(2, max_gpu + 1))              # 2, 3, ..., max_gpu

        n_types = len(cpu_values) * len(gpu_values)
        weight_per_type = self.prior_strength / n_types

        for cpu in cpu_values:
            for gpu in gpu_values:
                self._type_counts[(cpu, gpu)] = weight_per_type

        self._total_count = self.prior_strength
        self._cache_dirty = True
        print(f"  B-FGD uniform prior: {len(cpu_values)} CPU x {len(gpu_values)} GPU = "
              f"{n_types} types, prior_strength={self.prior_strength}, "
              f"weight/type={weight_per_type:.6f}")

    def observe_task(self, task: Task):
        """Bayesian update: bucket task by cpu/4 and gpu (0.1 for <1, 1.0 for >=1)."""
        if task.gpu_demand >= 1.0:
            gpu_bucketed = round(task.gpu_demand)
        else:
            gpu_bucketed = round(round(task.gpu_demand / 0.1) * 0.1, 1)
        cpu_bucket = round(task.cpu_demand / 4) * 4
        self._type_counts[(cpu_bucket, gpu_bucketed)] += 1
        self._total_count += 1
        if task.gpu_demand > 0:
            self._gpu_task_count += 1
        self._cache_dirty = True

    def _get_bayesian_task_types(self) -> List[Tuple[Tuple[float, float], float]]:
        """Compute task distribution from accumulated counts."""
        if not self._cache_dirty and self._cached_task_types is not None:
            return self._cached_task_types

        if self._total_count == 0:
            return []

        self._cached_task_types = [
            ((cpu, gpu), count / self._total_count)
            for (cpu, gpu), count in self._type_counts.items()
        ]
        self._cache_dirty = False
        return self._cached_task_types

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        # Fall back to Packing until enough GPU tasks observed
        if self._gpu_task_count < self.min_gpu_tasks:
            return self._fallback.select_node(task, cluster)

        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        task_types = self._get_bayesian_task_types()

        if not task_types:
            return self._fallback.select_node(task, cluster)

        args_list = [
            (
                node.node_id, node.remaining_cpu,
                tuple(node.gpu_remaining), node.num_gpus, node.total_cpu,
                task.cpu_demand, task.gpu_demand, task_types,
                node.gpu_model,
            )
            for node in eligible
        ]

        pool = self._get_pool()
        results = pool.map(FGDScheduler._compute_frag_delta_for_node, args_list)

        best_node_id = None
        best_score = -1
        self._pending_slot = -1
        for node_id, delta, slot in results:
            score = int(100.0 / (1.0 + math.exp(delta)))
            if score > best_score:
                best_score = score
                best_node_id = node_id
                self._pending_slot = slot

        return best_node_id

    def reset(self):
        """Reset all counts and cache."""
        self._type_counts.clear()
        self._total_count = 0.0
        self._gpu_task_count = 0
        self._cached_task_types = None
        self._cache_dirty = True


def get_scheduler(name: str) -> Scheduler:
    """Factory function to get scheduler by name"""
    schedulers = {
        'random': RandomScheduler,
        'bestfit': BestFitScheduler,
        'bestfit-pn': BestFitLocalScheduler,
        'dotprod': DotProdScheduler,
        'packing': PackingScheduler,
        'clustering': ClusteringScheduler,
        'fgd': FGDScheduler,
    }

    name_lower = name.lower()
    if name_lower not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    return schedulers[name_lower]()


def get_all_schedulers() -> List[Scheduler]:
    """Return instances of all schedulers (global-norm BestFit only)."""
    return [
        RandomScheduler(),
        BestFitScheduler(),
        DotProdScheduler(),
        PackingScheduler(),
        ClusteringScheduler(),
        FGDScheduler(),
    ]


def get_all_schedulers_with_bestfit_variants() -> List[Scheduler]:
    """Return all schedulers including both BestFit normalization variants.
    Used by exp_fig11_14 to compare per-node vs global normalization.
    """
    return [
        RandomScheduler(),
        BestFitScheduler(),
        BestFitLocalScheduler(),
        DotProdScheduler(),
        PackingScheduler(),
        ClusteringScheduler(),
        FGDScheduler(),
    ]


if __name__ == "__main__":
    from simulator import Cluster, Task, create_alibaba_like_distribution

    print("=== Scheduler Test ===\n")

    # Create test tasks
    tasks = [
        Task(task_id=i, cpu_demand=8, gpu_demand=1.0) for i in range(5)
    ] + [
        Task(task_id=i+5, cpu_demand=4, gpu_demand=0.5) for i in range(5)
    ]

    # Test each scheduler
    for scheduler in get_all_schedulers():
        # Fresh cluster for each scheduler
        cluster = Cluster()
        cluster.create_homogeneous_cluster(num_nodes=4, cpu_per_node=64, gpus_per_node=4)
        cluster.set_task_distribution(create_alibaba_like_distribution())

        # Reset clustering scheduler's state
        if isinstance(scheduler, ClusteringScheduler):
            scheduler.reset()

        scheduled = 0
        for task in tasks:
            if scheduler.schedule(task, cluster):
                scheduled += 1

        print(f"{scheduler.name:12} - Scheduled: {scheduled}/{len(tasks)}, "
              f"Frag Rate: {cluster.compute_fragmentation_rate():.2f}%, "
              f"GPU Alloc: {cluster.gpu_allocation_rate:.2f}%")
