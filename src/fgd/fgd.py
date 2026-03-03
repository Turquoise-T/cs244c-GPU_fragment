from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    cpu_request: float
    gpu_request: float  # either partial GPU (0-1) or full GPUs (1, 2, 4, 8, etc.)
    memory_request: float = 0.0
    gpu_type: Optional[str] = None  # e.g., "A100", "V100", None for any
    
    def __repr__(self):
        return f"Task({self.id}, CPU={self.cpu_request}, GPU={self.gpu_request})"


@dataclass
class Node:
    id: str
    total_cpu: float
    total_memory: float
    gpus: List[float]  # list where each element is available capacity per GPU (0.0-1.0)
    gpu_type: str = "generic"
    
    def __post_init__(self):
        self.allocated_cpu = 0.0
        self.allocated_memory = 0.0
    
    @property
    def available_cpu(self) -> float:
        return self.total_cpu - self.allocated_cpu
    
    @property
    def available_memory(self) -> float:
        return self.total_memory - self.allocated_memory
    
    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
    
    def get_gpu_scalar(self) -> float:
        """
        Map GPU vector to scalar: u = f + p
        where f = number of fully unallocated GPUs
              p = maximum unallocated partial GPU
        """
        full_gpus = sum(1 for gpu in self.gpus if gpu == 1.0)
        partial_gpus = [gpu for gpu in self.gpus if 0 < gpu < 1.0]
        max_partial = max(partial_gpus) if partial_gpus else 0.0
        return full_gpus + max_partial
    
    def can_fit_task(self, task: Task) -> bool:
        # Check CPU and memory
        if self.available_cpu < task.cpu_request:
            return False
        if self.available_memory < task.memory_request:
            return False
        
        # Check GPU type constraint (pipe-delimited spec like "V100M16|V100M32")
        if task.gpu_type is not None:
            acceptable = set(task.gpu_type.split('|'))
            if self.gpu_type not in acceptable:
                return False
        
        # Check GPU availability
        gpu_scalar = self.get_gpu_scalar()
        if gpu_scalar < task.gpu_request:
            return False
        
        return True
    
    def find_suitable_gpus(self, task: Task) -> Optional[List[int]]:
        # Find which GPU(s) can accommodate the task and return list of GPU indices or None if not possible.
        if task.gpu_request == 0:
            return []
        
        # For partial GPU request (0 < gpu < 1)
        if 0 < task.gpu_request < 1:
            suitable = []
            for idx, available in enumerate(self.gpus):
                if available >= task.gpu_request:
                    suitable.append(idx)
            return suitable if suitable else None
        
        # For full GPU request (1, 2, 4, 8, etc.)
        num_gpus_needed = int(task.gpu_request)
        full_gpus = [idx for idx, avail in enumerate(self.gpus) if avail == 1.0]
        
        if len(full_gpus) >= num_gpus_needed:
            return full_gpus[:num_gpus_needed]
        
        return None
    
    def __repr__(self):
        return f"Node({self.id}, CPU={self.available_cpu}/{self.total_cpu}, GPUs={self.gpus})"


class Workload:
    # Represent the target workload with task popularity distribution
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.popularity: Dict[str, float] = {}  # Normalized popularity (sums to 1)
    
    def add_task_type(self, task: Task, popularity: float):
        """Add a task type with its popularity"""
        self.tasks[task.id] = task
        self.popularity[task.id] = popularity
    
    def normalize_popularity(self):
        """Ensure popularity sums to 1"""
        total = sum(self.popularity.values())
        if total > 0:
            for task_id in self.popularity:
                self.popularity[task_id] /= total


class FragmentationCalculator:
    # Calculate fragmentation measure for nodes
    
    @staticmethod
    def compute_node_fragmentation(node: Node, task: Task) -> float:
        # Compute F_n(m): fragmentation of node n measured by task m and return the amount of GPU resources that cannot be allocated to task m.
        available_cpu = node.available_cpu
        available_memory = node.available_memory
        available_gpu_scalar = node.get_gpu_scalar()

        # Case 1: Task cannot run (Q-I, Q-II, Q-IV, or x-axis)
        # All unallocated GPUs are fragmented
        # Check CPU, memory, GPU type, and GPU scalar constraints
        if (task.cpu_request > available_cpu or
            task.memory_request > available_memory or
            task.gpu_request > available_gpu_scalar or
            task.gpu_request == 0):
            return sum(node.gpus)

        # GPU type constraint: if task requires specific type, check match
        if task.gpu_type is not None:
            acceptable = set(task.gpu_type.split('|'))
            if node.gpu_type not in acceptable:
                return sum(node.gpus)
        
        # Case 2: Task can run and requests GPU (Q-III)
        # Check each GPU individually
        fragmented = 0.0
        for gpu_available in node.gpus:
            # GPU is fragmented if it has insufficient capacity
            min_needed = min(task.gpu_request, 1.0)
            if gpu_available < min_needed:
                fragmented += gpu_available
        
        return fragmented
    
    @staticmethod
    def compute_node_fragmentation_for_workload(node: Node, workload: Workload) -> float:
        # Compute F_n(M): expected fragmentation for workload M.
        # F_n(M) = Σ p_m * F_n(m) for all tasks m in workload M
        total_frag = 0.0
        for task_id, task in workload.tasks.items():
            popularity = workload.popularity.get(task_id, 0.0)
            task_frag = FragmentationCalculator.compute_node_fragmentation(node, task)
            total_frag += popularity * task_frag
        
        return total_frag
    
    @staticmethod
    def compute_cluster_fragmentation(nodes: List[Node], workload: Workload) -> float:
        # Compute F_N(M): cluster-level fragmentation.
        # F_N(M) = Σ F_n(M) for all nodes n in cluster N
        return sum(
            FragmentationCalculator.compute_node_fragmentation_for_workload(node, workload)
            for node in nodes
        )


class FGDScheduler:
    # Fragmentation Gradient Descent Scheduler
    
    def __init__(self, nodes: List[Node], workload: Workload):
        self.nodes = nodes
        self.workload = workload
        self.task_queue = []
        self.scheduled_tasks: Dict[str, Tuple[str, List[int]]] = {}  # task_id -> (node_id, gpu_indices)
    
    def schedule_task(self, task: Task) -> Tuple[Optional[Node], Optional[List[int]]]:
        """
        Schedule a single task using FGD algorithm.
        Returns (selected_node, gpu_indices) or (None, None) if cannot schedule.
        
        Algorithm:
        1. Filter unavailable nodes
        2. For each available node, hypothetically assign task
        3. Calculate fragmentation increment Δ
        4. Select node with minimum Δ
        """
        best_node = None
        best_gpu_indices = None
        min_delta = float('inf')
        
        # Track candidate nodes and their scores
        candidates = []
        
        for node in self.nodes:
            # Filter: Check if node has sufficient resources
            if not node.can_fit_task(task):
                continue
            
            # Find suitable GPU(s) for this task
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue
            
            # If partial GPU task, try each suitable GPU
            if 0 < task.gpu_request < 1:
                for gpu_idx in gpu_indices:
                    delta = self._compute_fragmentation_delta(
                        node, task, [gpu_idx]
                    )
                    candidates.append((node, [gpu_idx], delta))
            else:
                # Full GPU(s) task
                delta = self._compute_fragmentation_delta(
                    node, task, gpu_indices
                )
                candidates.append((node, gpu_indices, delta))
        
        # Select node with minimum fragmentation increment
        if candidates:
            best_node, best_gpu_indices, min_delta = min(
                candidates, key=lambda x: x[2]
            )
        
        return best_node, best_gpu_indices
    
    def _compute_fragmentation_delta(
        self, 
        node: Node, 
        task: Task, 
        gpu_indices: List[int]
    ) -> float:
        # Compute fragmentation increment: Δ = F_n'(M) - F_n(M) where n' is the node state after hypothetically assigning the task.

        # Calculate current fragmentation
        current_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
            node, self.workload
        )
        
        # Create hypothetical node state
        hypothetical_node = self._create_hypothetical_assignment(node, task, gpu_indices)
        
        # Calculate fragmentation after assignment
        new_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
            hypothetical_node, self.workload
        )
        
        # Return the delta
        return new_frag - current_frag
    
    def _create_hypothetical_assignment(
        self, 
        node: Node, 
        task: Task, 
        gpu_indices: List[int]
    ) -> Node:
        # Create a copy of the node with the task hypothetically assigned

        # Deep copy the node
        hyp_node = Node(
            id=node.id,
            total_cpu=node.total_cpu,
            total_memory=node.total_memory,
            gpus=node.gpus.copy(),
            gpu_type=node.gpu_type
        )
        hyp_node.allocated_cpu = node.allocated_cpu
        hyp_node.allocated_memory = node.allocated_memory
        
        # Allocate resources
        hyp_node.allocated_cpu += task.cpu_request
        hyp_node.allocated_memory += task.memory_request
        
        # Allocate GPU(s)
        if task.gpu_request > 0:
            if 0 < task.gpu_request < 1:
                # Partial GPU
                hyp_node.gpus[gpu_indices[0]] -= task.gpu_request
            else:
                # Full GPU(s)
                for idx in gpu_indices:
                    hyp_node.gpus[idx] = 0.0
        
        return hyp_node
    
    def allocate_task(self, task: Task, node: Node, gpu_indices: List[int]):
        # Actually allocate the task to the node
        node.allocated_cpu += task.cpu_request
        node.allocated_memory += task.memory_request
        
        if task.gpu_request > 0:
            if 0 < task.gpu_request < 1:
                # Partial GPU
                node.gpus[gpu_indices[0]] -= task.gpu_request
            else:
                # Full GPU(s)
                for idx in gpu_indices:
                    node.gpus[idx] = 0.0
        
        self.scheduled_tasks[task.id] = (node.id, gpu_indices)