from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

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
    
    def num_allocated_gpus(self) -> int:
        """Return the number of fully or partially allocated GPUs."""
        return sum(1 for gpu in self.gpus if gpu < 1.0)

    def num_free_gpus(self) -> int:
        """Return the number of completely free GPUs."""
        return sum(1 for gpu in self.gpus if gpu == 1.0)

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

    def filter_by_popularity_threshold(self, threshold_pct: float = 60.0) -> 'Workload':
        """Apply paper's GetTypicalPods filter.

        Keep only the most popular task types that together cover threshold_pct
        of all tasks (sorted by popularity descending), then renormalize.

        Paper default: 60% (DefaultTypicalPodPopularityThreshold).

        Args:
            threshold_pct: Percentage threshold (0-100). Default 60.

        Returns:
            New Workload with filtered and renormalized task types.
        """
        if not self.tasks:
            return Workload()

        # Sort tasks by popularity descending
        sorted_tasks = sorted(
            self.popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Accumulate until we reach threshold
        target = threshold_pct / 100.0
        cumulative = 0.0
        selected_ids = []

        for task_id, pop in sorted_tasks:
            selected_ids.append(task_id)
            cumulative += pop
            if cumulative >= target:
                break

        # Create filtered workload and renormalize
        filtered = Workload()
        for task_id in selected_ids:
            filtered.tasks[task_id] = self.tasks[task_id]
            filtered.popularity[task_id] = self.popularity[task_id]

        filtered.normalize_popularity()
        return filtered


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

    def __init__(self, nodes: List[Node], workload: Workload,
                 use_paper_scoring: bool = False,
                 popularity_threshold: Optional[float] = None,
                 use_buddy_tiebreak: bool = True,
                 use_cluster_fragmentation: bool = False):
        """Initialize FGD Scheduler.

        Args:
            nodes: List of cluster nodes.
            workload: Target workload distribution for fragmentation scoring.
            use_paper_scoring: If True, use paper's sigmoid scoring with integer
                quantization: score = int(100 / (1 + exp(delta))). Creates ties
                that can be broken by buddy score. Default False uses raw delta.
            popularity_threshold: If set, filter workload to only top task types
                covering this percentage (paper default: 60%). None = use all.
            use_buddy_tiebreak: If True, use buddy-aware tie-breaking when scores
                tie (prefer leaving 2^n free GPUs). Default True.
            use_cluster_fragmentation: If True, compute fragmentation delta across
                the entire cluster (sum of all nodes). If False (default), compute
                delta only for the candidate node (per-node). Per-cluster considers
                global impact but is more expensive to compute.
        """
        self.nodes = nodes
        self.use_paper_scoring = use_paper_scoring
        self.use_buddy_tiebreak = use_buddy_tiebreak
        self.use_cluster_fragmentation = use_cluster_fragmentation

        # Apply popularity threshold filter if specified
        if popularity_threshold is not None:
            self.workload = workload.filter_by_popularity_threshold(popularity_threshold)
        else:
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
        
        # Track candidate nodes and their scores.
        # Each candidate stores:
        #   (node, gpu_indices, delta_frag, new_frag_after_assignment)
        candidates = []
        
        for node in self.nodes:
            # Filter: Check if node has sufficient resources
            if not node.can_fit_task(task):
                continue
            
            # Find suitable GPU(s) for this task.
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue
            
            # If partial GPU task, try each suitable GPU.
            if 0 < task.gpu_request < 1:
                for gpu_idx in gpu_indices:
                    delta, new_frag = self._compute_fragmentation_delta(
                        node, task, [gpu_idx]
                    )
                    candidates.append((node, [gpu_idx], delta, new_frag))
            else:
                # Full GPU(s) task.
                delta, new_frag = self._compute_fragmentation_delta(
                    node, task, gpu_indices
                )
                candidates.append((node, gpu_indices, delta, new_frag))
        
        # Select best node based on scoring mode
        if candidates:
            if self.use_paper_scoring:
                # Paper's approach: sigmoid scoring with integer quantization.
                # score = int(100 / (1 + exp(delta))) - higher is better.
                # Integer quantization creates ties, broken by buddy score.
                def paper_key(x):
                    delta = x[2]
                    # Sigmoid score: higher = better, so negate for min()
                    sigmoid_score = int(100.0 / (1.0 + math.exp(delta)))
                    neg_score = -sigmoid_score  # negate so min() finds highest

                    if self.use_buddy_tiebreak:
                        buddy = self._compute_buddy_score(x[0], x[1])
                    else:
                        buddy = 0

                    return (neg_score, buddy, x[3], x[0].id)

                best_node, best_gpu_indices, _, _ = min(candidates, key=paper_key)
            else:
                # Original approach: raw delta comparison with buddy tie-breaking.
                # Tie-breaking (Buddy-Aware FGD):
                #   1) smaller fragmentation delta (primary FGD objective)
                #   2) smaller buddy score (prefer 2^n free GPUs after allocation)
                #   3) smaller final fragmentation on the chosen node
                #   4) lexicographically smaller node id (stable deterministic choice)
                def original_key(x):
                    if self.use_buddy_tiebreak:
                        buddy = self._compute_buddy_score(x[0], x[1])
                    else:
                        buddy = 0
                    return (x[2], buddy, x[3], x[0].id)

                best_node, best_gpu_indices, _, _ = min(candidates, key=original_key)

        return best_node, best_gpu_indices
    
    def _compute_buddy_score(self, node: Node, gpu_indices: List[int]) -> int:
        """Compute buddy alignment score (lower = better).

        Prefers allocations that leave 2^n GPUs free on the node,
        making it easier to accommodate future multi-GPU jobs.

        The buddy system in memory allocation keeps blocks in powers of 2,
        which reduces external fragmentation. We apply the same principle:
        leaving 0, 1, 2, 4, or 8 free GPUs is preferred over odd numbers
        like 3, 5, 6, 7.

        Args:
            node: The node being considered for allocation.
            gpu_indices: The GPU indices that would be allocated.

        Returns:
            Integer score representing distance to nearest buddy size.
            Lower is better (0 = perfect alignment).
        """
        gpus_per_node = len(node.gpus)
        # Count GPUs that will be free after this allocation
        currently_free = node.num_free_gpus()
        gpus_to_allocate = len(gpu_indices)
        free_after = currently_free - gpus_to_allocate

        # Buddy sizes: powers of 2 from 0 to gpus_per_node
        # For 8-GPU nodes: [0, 1, 2, 4, 8]
        buddy_sizes = [0]
        power = 1
        while power <= gpus_per_node:
            buddy_sizes.append(power)
            power *= 2

        # Return minimum distance to any buddy size
        return min(abs(free_after - b) for b in buddy_sizes)

    def _compute_fragmentation_delta(
        self,
        node: Node,
        task: Task,
        gpu_indices: List[int]
    ) -> Tuple[float, float]:
        """Compute fragmentation increment and new fragmentation for a candidate.

        When use_cluster_fragmentation=False (default, per-node):
            delta_frag = F_n'(M) - F_n(M)  (only the candidate node)
            new_frag   = F_n'(M)

        When use_cluster_fragmentation=True (per-cluster):
            delta_frag = F_N'(M) - F_N(M)  (sum over all nodes)
            new_frag   = F_N'(M)
            Note: Only the target node changes; other nodes contribute the same
            fragmentation before and after, so they cancel out in delta.
            But new_frag is the total cluster fragmentation.

        Returns:
            (delta_frag, new_frag)
        """
        # Create hypothetical node state
        hypothetical_node = self._create_hypothetical_assignment(node, task, gpu_indices)

        if self.use_cluster_fragmentation:
            # Per-cluster: compute fragmentation across all nodes
            # Current cluster fragmentation
            current_cluster_frag = sum(
                FragmentationCalculator.compute_node_fragmentation_for_workload(n, self.workload)
                for n in self.nodes
            )

            # New cluster fragmentation: replace target node with hypothetical
            new_cluster_frag = 0.0
            for n in self.nodes:
                if n.id == node.id:
                    new_cluster_frag += FragmentationCalculator.compute_node_fragmentation_for_workload(
                        hypothetical_node, self.workload
                    )
                else:
                    new_cluster_frag += FragmentationCalculator.compute_node_fragmentation_for_workload(
                        n, self.workload
                    )

            delta = new_cluster_frag - current_cluster_frag
            return delta, new_cluster_frag
        else:
            # Per-node (default): compute fragmentation only for candidate node
            current_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
                node, self.workload
            )
            new_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
                hypothetical_node, self.workload
            )
            delta = new_frag - current_frag
            return delta, new_frag
    
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