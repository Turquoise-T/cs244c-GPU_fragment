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
    Computed as weighted sum of all resource dimensions.
    """

    def __init__(self):
        super().__init__("BestFit")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Select node with minimum remaining resources
        # Score = remaining_cpu + remaining_gpu (normalized)
        best_node = None
        best_score = float('inf')

        for node in eligible:
            # Normalize by max capacity in cluster for fair comparison
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

    @staticmethod
    def _compute_frag_delta_for_node(args: Tuple) -> Tuple[int, float]:
        """
        Worker function to compute fragmentation delta for a single node.
        Used for parallel evaluation.

        Args:
            args: (node_id, remaining_cpu, gpu_remaining, num_gpus, total_cpu,
                   task_cpu, task_gpu, task_types)

        Returns:
            (node_id, fragmentation_delta)
        """
        (node_id, remaining_cpu, gpu_remaining, num_gpus, total_cpu,
         task_cpu, task_gpu, task_types) = args

        # Reconstruct node state
        node = Node(
            node_id=node_id,
            total_cpu=total_cpu,
            num_gpus=num_gpus,
            allocated_cpu=total_cpu - remaining_cpu,
            gpu_remaining=list(gpu_remaining)
        )

        task = Task(task_id=-1, cpu_demand=task_cpu, gpu_demand=task_gpu)

        # Compute fragmentation before (only for this node)
        frag_before = 0.0
        for (cpu, gpu), popularity in task_types:
            dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
            frag_before += popularity * node.get_fragmentation_for_task(dummy)

        # Hypothetically allocate
        node.allocate_task(task)

        # Compute fragmentation after
        frag_after = 0.0
        for (cpu, gpu), popularity in task_types:
            dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
            frag_after += popularity * node.get_fragmentation_for_task(dummy)

        return (node_id, frag_after - frag_before)

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

        # Use override distribution for scheduling if set, otherwise cluster's
        if self.scheduling_task_types is not None:
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
                task_types
            )
            for node in eligible
        ]

        # Parallel execution
        pool = self._get_pool()
        results = pool.map(FGDScheduler._compute_frag_delta_for_node, args_list)

        # Find best node
        best_node_id = None
        best_delta = float('inf')
        for node_id, delta in results:
            if delta < best_delta:
                best_delta = delta
                best_node_id = node_id

        return best_node_id

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
                task_types
            )
            for node in eligible
        ]

        # Parallel execution
        pool = self._get_pool()
        results = pool.map(FGDScheduler._compute_frag_delta_for_node, args_list)

        # Find best node
        best_node_id = None
        best_delta = float('inf')
        for node_id, delta in results:
            if delta < best_delta:
                best_delta = delta
                best_node_id = node_id

        return best_node_id

    def reset(self):
        """Reset the sliding window"""
        self._window.clear()
        self._cached_task_types = None
        self._cache_dirty = True


def get_scheduler(name: str) -> Scheduler:
    """Factory function to get scheduler by name"""
    schedulers = {
        'random': RandomScheduler,
        'bestfit': BestFitScheduler,
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
    """Return instances of all available schedulers"""
    return [
        RandomScheduler(),
        BestFitScheduler(),
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
