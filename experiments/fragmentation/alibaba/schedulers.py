"""
Scheduling Policies for GPU Cluster Simulation

Implements scheduling policies from the FGD paper (Section 6.1):
1. Random     - Random node selection
2. BestFit    - Node with least remaining resources
3. DotProd    - Smallest dot-product of remaining resources and task demands
4. Packing    - Prioritize occupied GPUs, then idle GPUs on occupied nodes
5. Clustering - Pack tasks with same GPU request together
6. FGD        - Fragmentation Gradient Descent (same algorithm as src/scheduler/policies/fgd.py)
7. W-FGD      - Windowed FGD with sliding-window distribution estimate

Adapted from fgd_replication/schedulers.py on the clubzip/fgd-alibaba-trace-loader branch.
Multiprocessing removed for simplicity.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from collections import deque, Counter
import random

from simulator import Task, Node, Cluster, TaskDistribution


class Scheduler(ABC):
    """Abstract base class for scheduling policies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        """Select a node for the task. Returns node_id or None."""
        pass

    def schedule(self, task: Task, cluster: Cluster) -> bool:
        """Schedule a task on the cluster."""
        node_id = self.select_node(task, cluster)
        if node_id is not None:
            return cluster.schedule_task(task, node_id)
        return False


class RandomScheduler(Scheduler):
    """Random-fit: pick any eligible node at random."""

    def __init__(self):
        super().__init__("Random")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None
        return random.choice(eligible).node_id


class BestFitScheduler(Scheduler):
    """Best-fit: node with least remaining resources (weighted CPU+GPU)."""

    def __init__(self):
        super().__init__("BestFit")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        # Normalize by max capacity in cluster (paper Section 6.1)
        max_cpu = max(n.total_cpu for n in cluster.nodes)
        max_gpu = max(n.num_gpus for n in cluster.nodes)

        best_node = None
        best_score = float('inf')
        for node in eligible:
            cpu_score = node.remaining_cpu / max_cpu if max_cpu > 0 else 0
            gpu_score = node.total_unallocated_gpu / max_gpu if max_gpu > 0 else 0
            score = cpu_score + gpu_score
            if score < best_score:
                best_score = score
                best_node = node

        return best_node.node_id if best_node else None


class DotProdScheduler(Scheduler):
    """Dot-product: allocate to node with smallest dot-product of remaining
    resources and task demands."""

    def __init__(self):
        super().__init__("DotProd")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        max_cpu = max(n.total_cpu for n in cluster.nodes)
        max_gpu = max(n.num_gpus for n in cluster.nodes)
        task_cpu_norm = task.cpu_demand / max_cpu if max_cpu > 0 else 0
        task_gpu_norm = task.gpu_demand / max_gpu if max_gpu > 0 else 0

        best_node = None
        best_score = float('inf')
        for node in eligible:
            cpu_norm = node.remaining_cpu / max_cpu if max_cpu > 0 else 0
            gpu_norm = node.total_unallocated_gpu / max_gpu if max_gpu > 0 else 0
            score = cpu_norm * task_cpu_norm + gpu_norm * task_gpu_norm
            if score < best_score:
                best_score = score
                best_node = node

        return best_node.node_id if best_node else None


class PackingScheduler(Scheduler):
    """GPU Packing: prioritize occupied GPUs > idle GPUs on occupied nodes > idle nodes."""

    def __init__(self):
        super().__init__("Packing")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        occupied_partial = []
        occupied_full = []
        idle = []

        for node in eligible:
            has_partial = any(0 < g < 1.0 for g in node.gpu_remaining)
            is_occupied = node.allocated_cpu > 0 or any(g < 1.0 for g in node.gpu_remaining)

            if has_partial:
                occupied_partial.append(node)
            elif is_occupied:
                occupied_full.append(node)
            else:
                idle.append(node)

        def sort_key(n):
            return n.total_unallocated_gpu

        for group in [occupied_partial, occupied_full, idle]:
            if group:
                group.sort(key=sort_key)
                return group[0].node_id

        return None


class ClusteringScheduler(Scheduler):
    """GPU Clustering: pack tasks with same GPU demand together."""

    def __init__(self):
        super().__init__("Clustering")
        self.node_gpu_patterns: dict = {}

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        matching = []
        other = []
        for node in eligible:
            if node.node_id in self.node_gpu_patterns:
                if task.gpu_demand in self.node_gpu_patterns[node.node_id]:
                    matching.append(node)
                else:
                    other.append(node)
            else:
                other.append(node)

        def sort_key(n):
            return n.total_unallocated_gpu

        if matching:
            matching.sort(key=sort_key)
            selected = matching[0]
        elif other:
            other.sort(key=sort_key)
            selected = other[0]
        else:
            return None

        if selected.node_id not in self.node_gpu_patterns:
            self.node_gpu_patterns[selected.node_id] = set()
        self.node_gpu_patterns[selected.node_id].add(task.gpu_demand)

        return selected.node_id

    def reset(self):
        self.node_gpu_patterns = {}


class FGDScheduler(Scheduler):
    """Fragmentation Gradient Descent: schedule to minimize fragmentation increase.

    Same algorithm as src/scheduler/policies/fgd.py (FragmentationCalculator +
    greedy min-delta placement), extended with CPU-awareness for Alibaba traces.

    Per-GPU evaluation: for partial-GPU tasks, tries each candidate GPU
    individually and picks the (node, GPU) pair with the smallest delta
    (paper Algorithm 1 footnote).

    Performance: groups eligible nodes by state to avoid redundant evaluations.
    """

    def __init__(self, scheduling_task_types=None):
        super().__init__("FGD")
        self.scheduling_task_types = scheduling_task_types
        self._chosen_gpu_idx = None  # best GPU index for partial-GPU tasks

    def _compute_frag_for_state(self, node, task_type_list):
        """Compute F_n(M) using precomputed task type list."""
        frag = 0.0
        for dummy_task, popularity in task_type_list:
            frag += popularity * node.get_fragmentation_for_task(dummy_task)
        return frag

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        if self.scheduling_task_types is not None:
            task_types = self.scheduling_task_types
        elif cluster.task_distribution is not None:
            task_types = cluster.task_distribution.get_task_types()
        else:
            return eligible[0].node_id

        # Precompute Task objects for the distribution (Opt B)
        task_type_list = [
            (Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu), popularity)
            for (cpu, gpu), popularity in task_types
        ]

        best_node_id = None
        best_delta = float('inf')
        best_gpu_idx = None

        # Group eligible nodes by state to avoid redundant evals (Opt A)
        from collections import defaultdict
        groups = defaultdict(list)
        for node in eligible:
            key = (node.total_cpu, node.num_gpus, node.allocated_cpu,
                   tuple(sorted(node.gpu_remaining)))
            groups[key].append(node)

        for _key, group_nodes in groups.items():
            node = group_nodes[0]  # evaluate representative

            # F_n(M) before placement
            frag_before = self._compute_frag_for_state(node, task_type_list)

            if task.is_partial_gpu():
                # Per-GPU evaluation: try each feasible GPU (paper Alg 1 footnote)
                for gpu_idx, g in enumerate(node.gpu_remaining):
                    if g < task.gpu_demand:
                        continue
                    # Hypothetical placement on this specific GPU
                    saved_cpu = node.allocated_cpu
                    saved_gpu = list(node.gpu_remaining)
                    node.allocated_cpu += task.cpu_demand
                    node.gpu_remaining[gpu_idx] -= task.gpu_demand

                    frag_after = self._compute_frag_for_state(node, task_type_list)
                    delta = frag_after - frag_before

                    # Restore
                    node.allocated_cpu = saved_cpu
                    node.gpu_remaining = saved_gpu

                    if delta < best_delta:
                        best_delta = delta
                        best_node_id = node.node_id
                        best_gpu_idx = gpu_idx
            else:
                # Full-GPU or no-GPU tasks: single hypothetical placement
                saved_cpu = node.allocated_cpu
                saved_gpu = list(node.gpu_remaining)

                gpu_indices = node.allocate_task(task)
                if gpu_indices is None:
                    continue

                frag_after = self._compute_frag_for_state(node, task_type_list)
                delta = frag_after - frag_before

                # Restore
                node.allocated_cpu = saved_cpu
                node.gpu_remaining = saved_gpu

                if delta < best_delta:
                    best_delta = delta
                    best_node_id = node.node_id
                    best_gpu_idx = None

        # If we picked a representative from a group, map back to an actual node
        if best_node_id is not None:
            # The best_node_id is the representative's ID.
            # Find the group it belongs to and pick any node from that group.
            for _key, group_nodes in groups.items():
                if group_nodes[0].node_id == best_node_id:
                    # Pick the first node in the group (all have same state)
                    best_node_id = group_nodes[0].node_id
                    break

        self._chosen_gpu_idx = best_gpu_idx
        return best_node_id

    def schedule(self, task: Task, cluster: Cluster) -> bool:
        """Schedule task, using chosen GPU index for partial-GPU tasks."""
        node_id = self.select_node(task, cluster)
        if node_id is None:
            return False
        if self._chosen_gpu_idx is not None and task.is_partial_gpu():
            node = cluster.nodes[node_id]
            node.allocated_cpu += task.cpu_demand
            node.gpu_remaining[self._chosen_gpu_idx] -= task.gpu_demand
            cluster.placement_map[task.task_id] = (node_id, [self._chosen_gpu_idx])
            cluster._active_tasks[task.task_id] = task
            self._chosen_gpu_idx = None
            return True
        return cluster.schedule_task(task, node_id)


class WindowedFGDScheduler(FGDScheduler):
    """FGD with sliding-window distribution estimate.

    Instead of using the global distribution (perfect future knowledge),
    estimates the distribution from the last `window_size` observed tasks.
    """

    def __init__(self, window_size: int = 500):
        super().__init__()
        self.name = f"W-FGD-{window_size}"
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._cached_task_types = None
        self._cache_dirty = True

    def observe_task(self, task: Task):
        """Record a task into the sliding window."""
        gpu_rounded = round(task.gpu_demand, 2)
        cpu_bucket = round(task.cpu_demand / 4) * 4
        self._window.append((cpu_bucket, gpu_rounded))
        self._cache_dirty = True

    def _get_windowed_task_types(self):
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

        task_types = self._get_windowed_task_types()
        if not task_types:
            if cluster.task_distribution is not None:
                task_types = cluster.task_distribution.get_task_types()
            else:
                return eligible[0].node_id

        best_node_id = None
        best_delta = float('inf')

        for node in eligible:
            frag_before = 0.0
            for (cpu, gpu), popularity in task_types:
                dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
                frag_before += popularity * node.get_fragmentation_for_task(dummy)

            saved_cpu = node.allocated_cpu
            saved_gpu = list(node.gpu_remaining)

            gpu_indices = node.allocate_task(task)
            if gpu_indices is None:
                continue

            frag_after = 0.0
            for (cpu, gpu), popularity in task_types:
                dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
                frag_after += popularity * node.get_fragmentation_for_task(dummy)

            delta = frag_after - frag_before

            node.allocated_cpu = saved_cpu
            node.gpu_remaining = saved_gpu

            if delta < best_delta:
                best_delta = delta
                best_node_id = node.node_id

        return best_node_id

    def reset(self):
        self._window.clear()
        self._cached_task_types = None
        self._cache_dirty = True


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_all_schedulers() -> List[Scheduler]:
    """Return instances of all baseline schedulers (no FGD variants)."""
    return [
        RandomScheduler(),
        BestFitScheduler(),
        DotProdScheduler(),
        PackingScheduler(),
        ClusteringScheduler(),
        FGDScheduler(),
    ]


def get_scheduler(name: str) -> Scheduler:
    """Get a scheduler by name."""
    registry = {
        'random': RandomScheduler,
        'bestfit': BestFitScheduler,
        'dotprod': DotProdScheduler,
        'packing': PackingScheduler,
        'clustering': ClusteringScheduler,
        'fgd': FGDScheduler,
    }
    name_lower = name.lower()
    if name_lower not in registry:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(registry.keys())}")
    return registry[name_lower]()
