"""
Lightweight GPU Cluster Simulator for FGD Replication

Based on: "Beware of Fragmentation: Scheduling GPU-Sharing Workloads
with Fragmentation Gradient Descent" (ATC'23)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy


@dataclass
class Task:
    """
    Represents a task/pod requesting resources.

    Attributes:
        task_id: Unique identifier
        cpu_demand: Number of CPUs requested (in cores)
        gpu_demand: GPU request - either partial (0,1) or full integer
                   e.g., 0.3 GPUs, 1 GPU, 2 GPUs
        name: Task name from trace (optional)
        creation_time: Timestamp of creation in seconds (optional)
        scheduled_time: Timestamp of scheduling in seconds (optional)
        deletion_time: Timestamp of deletion in seconds (optional)
        gpu_spec: GPU type constraint from trace (optional)
    """
    task_id: int
    cpu_demand: float
    gpu_demand: float
    name: str = ""
    creation_time: int = 0
    scheduled_time: int = 0
    deletion_time: int = 0
    gpu_spec: str = ""

    def is_partial_gpu(self) -> bool:
        """Check if task requests partial GPU (0 < demand < 1)"""
        return 0 < self.gpu_demand < 1

    def is_full_gpu(self) -> bool:
        """Check if task requests one or more full GPUs"""
        return self.gpu_demand >= 1 and self.gpu_demand == int(self.gpu_demand)

    def is_no_gpu(self) -> bool:
        """Check if task requests no GPU"""
        return self.gpu_demand == 0


@dataclass
class Node:
    """
    Represents a node in the cluster with CPU and multiple GPUs.

    Each GPU tracks its remaining capacity independently (0 to 1).
    GPU sharing allows multiple tasks on the same GPU if capacity permits.

    Attributes:
        node_id: Unique identifier
        total_cpu: Total CPU cores on the node
        num_gpus: Number of GPUs on the node
        allocated_cpu: Currently allocated CPU
        gpu_remaining: List of remaining capacity for each GPU [0,1]
        name: Node name from trace (optional)
        gpu_model: GPU type from trace (optional)
        memory_mib: Memory capacity in MiB from trace (optional)
    """
    node_id: int
    total_cpu: float
    num_gpus: int
    allocated_cpu: float = 0.0
    gpu_remaining: List[float] = field(default_factory=list)
    name: str = ""
    gpu_model: str = ""
    memory_mib: int = 0

    def __post_init__(self):
        if not self.gpu_remaining:
            # Initialize all GPUs as fully available (1.0 each)
            self.gpu_remaining = [1.0] * self.num_gpus

    @property
    def remaining_cpu(self) -> float:
        """Available CPU on the node"""
        return self.total_cpu - self.allocated_cpu

    @property
    def fully_unallocated_gpus(self) -> int:
        """Count of GPUs with 100% capacity remaining (f in paper)"""
        return sum(1 for g in self.gpu_remaining if g == 1.0)

    @property
    def max_partial_gpu(self) -> float:
        """Maximum remaining capacity among partial GPUs (p in paper)"""
        partial = [g for g in self.gpu_remaining if 0 < g < 1.0]
        return max(partial) if partial else 0.0

    @property
    def scalar_gpu_capacity(self) -> float:
        """
        Scalar representation of unallocated GPU capacity: u = f + p
        (Equation from Section 3.1)
        """
        return self.fully_unallocated_gpus + self.max_partial_gpu

    @property
    def total_unallocated_gpu(self) -> float:
        """Sum of all remaining GPU capacity"""
        return sum(self.gpu_remaining)

    def can_fit_task(self, task: Task) -> bool:
        """Check if node has sufficient resources for the task"""
        # Check CPU
        if self.remaining_cpu < task.cpu_demand:
            return False

        # Check GPU
        if task.is_no_gpu():
            return True

        if task.is_partial_gpu():
            # Need at least one GPU with enough remaining capacity
            return any(g >= task.gpu_demand for g in self.gpu_remaining)

        if task.is_full_gpu():
            # Need enough fully unallocated GPUs
            return self.fully_unallocated_gpus >= int(task.gpu_demand)

        return False

    def allocate_task(self, task: Task) -> bool:
        """
        Allocate resources to a task. Returns True if successful.
        For partial GPU tasks, assigns to GPU with least remaining capacity
        that can still fit the task (best-fit within node).
        """
        if not self.can_fit_task(task):
            return False

        # Allocate CPU
        self.allocated_cpu += task.cpu_demand

        # Allocate GPU
        if task.is_no_gpu():
            return True

        if task.is_partial_gpu():
            # Find GPU with minimum remaining capacity that can fit
            best_idx = -1
            best_remaining = float('inf')
            for i, g in enumerate(self.gpu_remaining):
                if g >= task.gpu_demand and g < best_remaining:
                    best_remaining = g
                    best_idx = i
            if best_idx >= 0:
                self.gpu_remaining[best_idx] -= task.gpu_demand
            return True

        if task.is_full_gpu():
            # Allocate from fully unallocated GPUs
            gpus_needed = int(task.gpu_demand)
            allocated = 0
            for i in range(len(self.gpu_remaining)):
                if self.gpu_remaining[i] == 1.0 and allocated < gpus_needed:
                    self.gpu_remaining[i] = 0.0
                    allocated += 1
            return True

        return False

    def get_fragmentation_for_task(self, task: Task) -> float:
        """
        Calculate F_n(m): fragmented GPUs on this node for a specific task.
        (Section 3.2, Equations 2-3)

        Returns the amount of GPU resources that cannot be allocated to the task.
        """
        # Case 3: Task requests no GPU - all unallocated GPUs are fragments
        if task.is_no_gpu():
            return self.total_unallocated_gpu

        # Case 1: Task cannot run due to insufficient CPU or GPU
        if self.remaining_cpu < task.cpu_demand:
            return self.total_unallocated_gpu

        if task.is_full_gpu():
            gpus_needed = int(task.gpu_demand)
            if self.fully_unallocated_gpus < gpus_needed:
                # Insufficient full GPUs - all unallocated GPUs are fragments
                return self.total_unallocated_gpu
            else:
                # Can run the task - no fragmentation from this task's view
                return 0.0

        if task.is_partial_gpu():
            # Case 2 (Q-III): Check each GPU
            # GPUs with insufficient capacity are considered fragmented
            if not self.can_fit_task(task):
                return self.total_unallocated_gpu

            # Count fragmented capacity: GPUs that can't fit this task
            fragmented = 0.0
            for g in self.gpu_remaining:
                if g < task.gpu_demand:
                    # This GPU cannot fit the task - its remaining capacity is fragmented
                    fragmented += g
            return fragmented

        return 0.0


@dataclass
class TaskDistribution:
    """
    Represents the target workload distribution M.
    Maps task types to their popularity (probability).

    Task types are defined by (cpu_demand, gpu_demand) tuples.
    """
    # Dict mapping (cpu_demand, gpu_demand) -> popularity
    distribution: Dict[Tuple[float, float], float] = field(default_factory=dict)

    def add_task_type(self, cpu_demand: float, gpu_demand: float, popularity: float):
        """Add a task type with its popularity"""
        self.distribution[(cpu_demand, gpu_demand)] = popularity

    def normalize(self):
        """Normalize popularities to sum to 1"""
        total = sum(self.distribution.values())
        if total > 0:
            for key in self.distribution:
                self.distribution[key] /= total

    def get_task_types(self) -> List[Tuple[Tuple[float, float], float]]:
        """Return list of ((cpu, gpu), popularity) tuples"""
        return list(self.distribution.items())


class Cluster:
    """
    Represents a GPU cluster with multiple nodes.

    Attributes:
        nodes: List of nodes in the cluster
        task_distribution: Target workload distribution for fragmentation calculation
    """

    def __init__(self):
        self.nodes: List[Node] = []
        self.task_distribution: Optional[TaskDistribution] = None
        self.scheduled_tasks: List[Tuple[Task, int]] = []  # (task, node_id)

    def add_node(self, node: Node):
        """Add a node to the cluster"""
        self.nodes.append(node)

    def create_homogeneous_cluster(self, num_nodes: int, cpu_per_node: float,
                                    gpus_per_node: int):
        """Create a cluster with identical nodes"""
        self.nodes = []
        for i in range(num_nodes):
            self.nodes.append(Node(
                node_id=i,
                total_cpu=cpu_per_node,
                num_gpus=gpus_per_node
            ))

    def set_task_distribution(self, distribution: TaskDistribution):
        """Set the target workload distribution"""
        self.task_distribution = distribution

    @property
    def total_gpu_capacity(self) -> int:
        """Total number of GPUs in the cluster"""
        return sum(n.num_gpus for n in self.nodes)

    @property
    def total_cpu_capacity(self) -> float:
        """Total CPU cores in the cluster"""
        return sum(n.total_cpu for n in self.nodes)

    @property
    def total_unallocated_gpu(self) -> float:
        """Total unallocated GPU capacity in the cluster"""
        return sum(n.total_unallocated_gpu for n in self.nodes)

    @property
    def total_allocated_gpu(self) -> float:
        """Total allocated GPU capacity"""
        return self.total_gpu_capacity - self.total_unallocated_gpu

    @property
    def gpu_allocation_rate(self) -> float:
        """Percentage of GPU capacity allocated"""
        if self.total_gpu_capacity == 0:
            return 0.0
        return (self.total_allocated_gpu / self.total_gpu_capacity) * 100

    def get_eligible_nodes(self, task: Task) -> List[Node]:
        """Get all nodes that can fit the task"""
        return [n for n in self.nodes if n.can_fit_task(task)]

    def compute_node_fragmentation(self, node: Node) -> float:
        """
        Compute F_n(M): fragmentation on node n for the entire workload M.
        (Equation 1)

        F_n(M) = sum over m in M of: p_m * F_n(m)
        """
        if self.task_distribution is None:
            return 0.0

        fragmentation = 0.0
        for (cpu, gpu), popularity in self.task_distribution.get_task_types():
            dummy_task = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
            fragmentation += popularity * node.get_fragmentation_for_task(dummy_task)

        return fragmentation

    def compute_cluster_fragmentation(self) -> float:
        """
        Compute F_N(M): total fragmentation across all nodes.
        (Equation 5)
        """
        return sum(self.compute_node_fragmentation(n) for n in self.nodes)

    def compute_fragmentation_rate(self) -> float:
        """
        Compute f_N(M): fragmentation rate (percentage).
        (Equation 6)

        f_N(M) = F_N(M) / total_unallocated_gpu * 100
        """
        total_unallocated = self.total_unallocated_gpu
        if total_unallocated == 0:
            return 100.0  # All GPUs allocated means 100% of remaining (0) is fragmented

        return (self.compute_cluster_fragmentation() / total_unallocated) * 100

    def deep_copy(self) -> 'Cluster':
        """Create a deep copy of the cluster state"""
        return copy.deepcopy(self)

    def schedule_task(self, task: Task, node_id: int) -> bool:
        """Schedule a task on a specific node"""
        node = self.nodes[node_id]
        if node.allocate_task(task):
            self.scheduled_tasks.append((task, node_id))
            return True
        return False


def create_alibaba_like_distribution() -> TaskDistribution:
    """
    Create a task distribution similar to Alibaba cluster H (Table 1).

    GPU Request per Task: 0, (0,1), 1, 2, 4, 8
    Task Population (%): 13.3, 37.8, 48.0, 0.2, 0.2, 0.5

    For simplicity, we use average values for partial GPU tasks.
    CPU demands are estimated based on typical CPU:GPU ratios.
    """
    dist = TaskDistribution()

    # (cpu_demand, gpu_demand) -> popularity
    # Assuming ~8 CPUs per GPU ratio on average
    dist.add_task_type(cpu_demand=4.0, gpu_demand=0.0, popularity=0.133)   # no GPU
    dist.add_task_type(cpu_demand=4.0, gpu_demand=0.5, popularity=0.378)   # partial GPU (avg 0.5)
    dist.add_task_type(cpu_demand=8.0, gpu_demand=1.0, popularity=0.480)   # 1 GPU
    dist.add_task_type(cpu_demand=16.0, gpu_demand=2.0, popularity=0.002)  # 2 GPUs
    dist.add_task_type(cpu_demand=32.0, gpu_demand=4.0, popularity=0.002)  # 4 GPUs
    dist.add_task_type(cpu_demand=64.0, gpu_demand=8.0, popularity=0.005)  # 8 GPUs

    dist.normalize()
    return dist


if __name__ == "__main__":
    # Simple test
    print("=== Simulator Test ===\n")

    # Create a small cluster
    cluster = Cluster()
    cluster.create_homogeneous_cluster(num_nodes=3, cpu_per_node=64, gpus_per_node=4)
    cluster.set_task_distribution(create_alibaba_like_distribution())

    print(f"Cluster: {len(cluster.nodes)} nodes, {cluster.total_gpu_capacity} GPUs")
    print(f"Initial fragmentation rate: {cluster.compute_fragmentation_rate():.2f}%")

    # Schedule some tasks
    tasks = [
        Task(task_id=0, cpu_demand=8, gpu_demand=1.0),
        Task(task_id=1, cpu_demand=4, gpu_demand=0.3),
        Task(task_id=2, cpu_demand=4, gpu_demand=0.5),
        Task(task_id=3, cpu_demand=8, gpu_demand=1.0),
    ]

    for task in tasks:
        eligible = cluster.get_eligible_nodes(task)
        if eligible:
            cluster.schedule_task(task, eligible[0].node_id)
            print(f"Scheduled task {task.task_id} (GPU: {task.gpu_demand}) -> Node {eligible[0].node_id}")

    print(f"\nAfter scheduling:")
    print(f"  GPU allocation rate: {cluster.gpu_allocation_rate:.2f}%")
    print(f"  Fragmentation rate: {cluster.compute_fragmentation_rate():.2f}%")

    # Show node states
    print(f"\nNode states:")
    for node in cluster.nodes:
        print(f"  Node {node.node_id}: CPU {node.remaining_cpu:.0f}/{node.total_cpu:.0f}, "
              f"GPUs {node.gpu_remaining}")
