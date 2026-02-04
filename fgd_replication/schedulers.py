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
import random

from simulator import Task, Node, Cluster


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

    def __init__(self):
        super().__init__("FGD")

    def select_node(self, task: Task, cluster: Cluster) -> Optional[int]:
        eligible = cluster.get_eligible_nodes(task)
        if not eligible:
            return None

        if cluster.task_distribution is None:
            # Fall back to best-fit if no distribution available
            return BestFitScheduler().select_node(task, cluster)

        best_node = None
        best_delta = float('inf')

        # Current fragmentation
        current_frag = cluster.compute_cluster_fragmentation()

        for node in eligible:
            # Hypothetically assign task to this node
            # Create a copy of the node state
            original_cpu = node.allocated_cpu
            original_gpus = node.gpu_remaining.copy()

            # Perform hypothetical allocation
            node.allocate_task(task)

            # Compute new fragmentation
            new_frag = cluster.compute_cluster_fragmentation()
            delta = new_frag - current_frag

            # Restore node state
            node.allocated_cpu = original_cpu
            node.gpu_remaining = original_gpus

            if delta < best_delta:
                best_delta = delta
                best_node = node

        return best_node.node_id if best_node else None


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
