import copy
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from node import Node
from job import Job
from lease import Lease
from workload import Workload
from fragmentation import FragmentationCalculator

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

class FGDScheduler:
    """
    Fragmentation Gradient Descent Scheduler
    Based on the ATC'23 paper
    """
    
    def __init__(self, time_per_iteration: int = 360, simulate: bool = False):
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
        
        # Simulation vs real mode
        self._simulate = simulate
        if self._simulate:
            self._current_timestamp = 0
        else:
            self._current_timestamp = time.time()
        
        # Round-based scheduling (similar to Gavel)
        self._time_per_iteration = time_per_iteration
        self._num_completed_rounds = 0
        
        # Cluster state
        self._nodes: List[Node] = []
        self._node_dict: Dict[str, Node] = {}
        
        # Job state
        self._jobs: Dict[str, Job] = {}
        self._pending_jobs: List[Job] = []
        self._running_jobs: Dict[str, Tuple[str, List[int]]] = {}  # job_id -> (node_id, gpu_indices)
        self._completed_jobs = set()
        
        # Workload for fragmentation calculation
        self._workload = Workload()
        
        # Job assignment tracking
        self._current_worker_assignments: Dict[str, Tuple[str, List[int]]] = {}
        
        # Synchronization
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        
        # Job ID counter
        self._job_id_counter = 0
        
        # Leases (similar to Gavel)
        self._leases: Dict[str, Lease] = {}

    def _clone_nodes(self) -> Dict[str, Node]:
        """Deep-copy nodes for dry-run scheduling (Gavel-style)."""
        return {node.node_id: copy.deepcopy(node) for node in self._nodes}

    
    def register_node(self, node_id: str, total_cpu: float, total_memory: float,
                     num_gpus: int, gpu_type: str = "V100"):
        """Register a new node with the scheduler"""
        with self._scheduler_lock:
            node = Node(node_id, total_cpu, total_memory, num_gpus, gpu_type)
            self._nodes.append(node)
            self._node_dict[node_id] = node
            self._logger.info(f'Registered node {node_id} with {num_gpus} GPUs')
    
    def add_job(self, job: Job, timestamp: Optional[float] = None):
        """Add a new job to the scheduler"""
        with self._scheduler_lock:
            job_id = f"job_{self._job_id_counter}"
            self._job_id_counter += 1
            job._job_id = job_id
            
            self._jobs[job_id] = job
            self._pending_jobs.append(job)
            
            # Update workload for fragmentation calculation
            # Assume equal popularity for now
            self._update_workload()
            
            self._logger.info(f'Added job {job_id}: {job}')
            self._scheduler_cv.notifyAll()
            
            return job_id
    
    def _update_workload(self):
        """Update workload based on current active jobs"""
        self._workload = Workload()
        for job in list(self._jobs.values()):
            self._workload.add_job_type(job, 1.0)
        self._workload.normalize_popularity()
    
    def _compute_fragmentation_delta(self, node: Node, job: Job,
                                    gpu_indices: List[int]) -> float:
        """
        Compute fragmentation increment: Δ = F_n'(M) - F_n(M)
        where n' is the node state after hypothetically assigning the job.
        """
        # Calculate current fragmentation
        current_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
            node, self._workload
        )
        
        # Create hypothetical node state
        hypothetical_node = self._create_hypothetical_assignment(
            node, job, gpu_indices
        )
        
        # Calculate fragmentation after assignment
        new_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
            hypothetical_node, self._workload
        )
        
        # Return the delta
        return new_frag - current_frag
    
    def _create_hypothetical_assignment(self, node: Node, job: Job,
                                       gpu_indices: List[int]) -> Node:
        """Create a copy of the node with the job hypothetically assigned"""
        # Deep copy the node
        state = node.get_state_dict()
        hyp_node = Node(
            node_id=state['node_id'],
            total_cpu=node.total_cpu,
            total_memory=node.total_memory,
            num_gpus=node.num_gpus,
            gpu_type=node.gpu_type
        )
        hyp_node.allocated_cpu = node.allocated_cpu
        hyp_node.allocated_memory = node.allocated_memory
        hyp_node.gpu_capacities = state['gpu_capacities'].copy()
        
        # Allocate resources
        hyp_node.allocated_cpu += job.cpu_request
        hyp_node.allocated_memory += job.memory_request
        
        # Allocate GPU(s)
        if job.gpu_request > 0:
            if 0 < job.gpu_request < 1:
                # Partial GPU
                hyp_node.gpu_capacities[gpu_indices[0]] -= job.gpu_request
            else:
                # Full GPU(s)
                for idx in gpu_indices:
                    hyp_node.gpu_capacities[idx] = 0.0
        
        return hyp_node
    
    # PAPER[§4.2] FGD Algorithm: schedule towards steepest descent of fragmentation
    def _schedule_job_fgd(self, job: Job, nodes: Dict[str, Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        """
        Schedule a single job using FGD algorithm.
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
        
        candidates = []
        
        for node in nodes.values():
            # Filter: Check if node has sufficient resources
            if not node.can_fit_task(job.job_id, job.cpu_request,
                                    job.gpu_request, job.memory_request,
                                    job.gpu_type):
                continue
            
            # Find suitable GPU(s) for this task
            gpu_indices = node.find_suitable_gpus(job.gpu_request)
            if gpu_indices is None:
                continue
            
            # If partial GPU task, try each suitable GPU
            if 0 < job.gpu_request < 1:
                for gpu_idx in gpu_indices:
                    delta = self._compute_fragmentation_delta(
                        node, job, [gpu_idx]
                    )
                    candidates.append((node, [gpu_idx], delta))
            else:
                # Full GPU(s) task
                delta = self._compute_fragmentation_delta(
                    node, job, gpu_indices
                )
                candidates.append((node, gpu_indices, delta))
        
        # Select node with minimum fragmentation increment
        if candidates:
            best_node, best_gpu_indices, min_delta = min(
                candidates, key=lambda x: x[2]
            )
            self._logger.debug(
                f'Selected node {best_node.node_id} with Δ={min_delta:.4f} '
                f'for job {job.job_id}'
            )
        
        return best_node, best_gpu_indices
    
    def _schedule_jobs(self) -> Dict[str, Tuple[str, List[int]]]:
        """
        Schedule all pending jobs using FGD.
        Returns mapping of job_id -> (node_id, gpu_indices)
        """
        scheduled_assignments = {}
        
        # dry-run cluster state
        dry_run_nodes = {n.node_id: n for n in self._nodes}

        # Sort pending jobs by priority (can be customized)
        pending_jobs = sorted(self._pending_jobs, 
                            key=lambda j: j.priority_weight, 
                            reverse=True)
        
        still_pending = []
        
        for job in pending_jobs:
            node, gpu_indices = self._schedule_job_fgd(job, dry_run_nodes)

            if node is not None:
                scheduled_assignments[job.job_id] = (node.node_id, gpu_indices)

                # mutate round view immediately
                node.allocate_resources(
                    job.job_id,
                    job.cpu_request,
                    job.gpu_request,
                    job.memory_request,
                    gpu_indices
                )

                self._logger.info(
                    f"Scheduled job {job.job_id} on node {node.node_id}, GPUs {gpu_indices}"
                )
            else:
                still_pending.append(job)

        self._pending_jobs = still_pending
        return scheduled_assignments
    
    def _allocate_job(self, job: Job, node: Node, gpu_indices: List[int]):
        """Actually allocate the job to the node"""
        node.allocate_resources(
            job.job_id, job.cpu_request, job.gpu_request,
            job.memory_request, gpu_indices
        )
        
        self._running_jobs[job.job_id] = (node.node_id, gpu_indices)
        self._current_worker_assignments[job.job_id] = (node.node_id, gpu_indices)
        
        # Create lease for the job
        remaining_steps = job.remaining_steps
        lease = Lease(remaining_steps, self._time_per_iteration)
        self._leases[job.job_id] = lease
    
    def _schedule_round(self):
        """Execute one scheduling round (similar to Gavel)"""
        with self._scheduler_lock:
            self._logger.info(f'*** START ROUND {self._num_completed_rounds} ***')
            
            # Schedule pending jobs
            scheduled_assignments = self._schedule_jobs()
            
            # Allocate resources for scheduled jobs
            for job_id, (node_id, gpu_indices) in scheduled_assignments.items():
                job = self._jobs[job_id]
                node = self._node_dict[node_id]
                self._allocate_job(job, node, gpu_indices)
            
            # Print cluster state
            self._print_cluster_state()
            
            self._num_completed_rounds += 1
            self._logger.info(f'*** END ROUND {self._num_completed_rounds - 1} ***')
    
    def _print_cluster_state(self):
        """Print current cluster state for debugging"""
        self._logger.info('=== Cluster State ===')
        for node in self._nodes:
            state = node.get_state_dict()
            self._logger.info(
                f"Node {state['node_id']}: "
                f"CPU {state['available_cpu']:.1f}/{node.total_cpu}, "
                f"GPUs {[f'{c:.2f}' for c in state['gpu_capacities']]}"
            )
        
        total_frag = FragmentationCalculator.compute_cluster_fragmentation(
            self._nodes, self._workload
        )
        self._logger.info(f'Total cluster fragmentation: {total_frag:.2f} GPUs')
    
    def complete_job(self, job_id: str):
        """Mark a job as completed and release its resources"""
        with self._scheduler_lock:
            if job_id not in self._running_jobs:
                self._logger.warning(f'Job {job_id} is not running')
                return
            
            node_id, gpu_indices = self._running_jobs[job_id]
            node = self._node_dict[node_id]
            job = self._jobs[job_id]
            
            # Release resources
            node.release_resources(
                job_id, job.cpu_request, job.gpu_request,
                job.memory_request, gpu_indices
            )
            
            # Update state
            del self._running_jobs[job_id]
            del self._current_worker_assignments[job_id]
            del self._jobs[job_id]
            del self._leases[job_id]
            self._completed_jobs.add(job_id)
            
            # Update workload
            self._update_workload()
            
            self._logger.info(f'Completed job {job_id}')
            self._scheduler_cv.notifyAll()
    
    def get_current_timestamp(self) -> float:
        """Get current timestamp"""
        if self._simulate:
            return self._current_timestamp
        else:
            return time.time()
    
    def simulate(self, jobs: List[Job], cluster_spec: Dict[str, Dict]):
        """
        Simulate scheduler execution
        
        Args:
            jobs: List of jobs to schedule
            cluster_spec: Dict of node_id -> {cpu, memory, num_gpus, gpu_type}
        """
        # Set up cluster
        for node_id, spec in cluster_spec.items():
            self.register_node(
                node_id, spec['cpu'], spec['memory'],
                spec['num_gpus'], spec.get('gpu_type', 'V100')
            )
        
        # Add all jobs
        for job in jobs:
            self.add_job(job)
        
        # Run scheduling rounds until all jobs complete
        while self._jobs or self._pending_jobs:
            self._schedule_round()
            
            # Simulate time passing
            self._current_timestamp += self._time_per_iteration
            
            # Simulate job completion (for now, complete jobs after one round)
            jobs_to_complete = list(self._running_jobs.keys())
            for job_id in jobs_to_complete:
                self.complete_job(job_id)
        
        self._logger.info('Simulation complete!')
        self._logger.info(f'Total rounds: {self._num_completed_rounds}')
        self._logger.info(f'Total jobs completed: {len(self._completed_jobs)}')