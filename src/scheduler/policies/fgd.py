"""
FGD (Fragmentation Gradient Descent) Placement Algorithm
=========================================================

Based on: "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with
          Fragmentation Gradient Descent" (ATC 2023)

FGD is a PLACEMENT algorithm that minimizes GPU fragmentation when placing
jobs that use FRACTIONAL GPUs (GPU sharing).

Key Concepts:
-------------
1. GPU Sharing: Multiple jobs can share the same GPU
   - Job A needs 0.3 GPU, Job B needs 0.5 GPU → can share 1 GPU
   - Represented as gpu_milli (0-1000, where 1000 = 1.0 GPU)

2. Fragmentation: When GPU resources are split across multiple GPUs
   - GPU-A has 0.3 free, GPU-B has 0.4 free
   - Total free = 0.7 GPU, but a job needing 0.5 GPU can't use it efficiently
   - FGD minimizes this "scattered" free space

3. Fragmentation Gradient (Δfrag):
   - For each candidate GPU, compute how much fragmentation would increase
   - Pick the GPU with minimum Δfrag

Integration with Gavel:
-----------------------
This module provides GPU-sharing-aware placement for Gavel's scheduler.
Enable with placement_strategy='fgd' and set gpu_sharing_mode=True.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# GPU State for Fractional GPU Sharing
# =============================================================================

@dataclass
class GPUState:
    """
    Represents the state of a single GPU with fractional allocation support.
    
    Attributes:
        gpu_id: Unique identifier for this GPU
        server_id: Which server this GPU belongs to
        total_milli: Total capacity in milli-GPU (typically 1000)
        used_milli: Currently used capacity in milli-GPU
        job_assignments: List of (job_id, milli) tuples for jobs on this GPU
    """
    gpu_id: int
    server_id: int
    total_milli: int = 1000  # 1000 milli = 1.0 GPU
    used_milli: int = 0
    job_assignments: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def free_milli(self) -> int:
        """Available capacity in milli-GPU."""
        return self.total_milli - self.used_milli
    
    @property
    def free_fraction(self) -> float:
        """Available capacity as fraction (0.0-1.0)."""
        return self.free_milli / 1000.0
    
    @property
    def used_fraction(self) -> float:
        """Used capacity as fraction (0.0-1.0)."""
        return self.used_milli / 1000.0
    
    @property
    def is_empty(self) -> bool:
        """GPU has no allocations."""
        return self.used_milli == 0
    
    @property
    def is_full(self) -> bool:
        """GPU is fully allocated."""
        return self.free_milli == 0
    
    @property
    def is_partial(self) -> bool:
        """GPU has some but not full allocation (fragmented)."""
        return not self.is_empty and not self.is_full
    
    def can_fit(self, milli_needed: int) -> bool:
        """Check if this GPU can accommodate the requested milli-GPU."""
        return self.free_milli >= milli_needed


@dataclass
class ServerState:
    """Represents the state of a single server (for whole-GPU allocation)."""
    server_id: int
    total_gpus: int
    allocated_gpus: int
    available_gpus: int
    
    @property
    def is_empty(self) -> bool:
        return self.allocated_gpus == 0
    
    @property
    def is_full(self) -> bool:
        return self.available_gpus == 0
    
    @property
    def is_partial(self) -> bool:
        return not self.is_empty and not self.is_full


# =============================================================================
# Fragmentation Metrics for GPU Sharing
# =============================================================================

def compute_gpu_fragmentation(gpu_states: List[GPUState]) -> float:
    """
    Compute cluster-wide GPU fragmentation metric.
    
    Fragmentation = sum of "unusable" free space on partial GPUs.
    A GPU is fragmented if it has some free space but not enough for
    typical workloads.
    
    Args:
        gpu_states: List of GPUState objects.
        
    Returns:
        Total fragmentation score (lower is better).
    """
    fragmentation = 0.0
    for gpu in gpu_states:
        if gpu.is_partial:
            # Fragmentation contribution = free space that's "trapped"
            # Weight by how small the free space is (smaller = more fragmented)
            fragmentation += gpu.free_fraction
    return fragmentation


def compute_fragmentation_increment_gpu(
    gpu: GPUState,
    milli_to_allocate: int
) -> float:
    """
    Compute the fragmentation increment (Δfrag) if we allocate on this GPU.
    
    Based on FGD paper Algorithm 1:
    Δ = F_n-(M) - F_n(M)
    where F is the fragmentation function, n- is after allocation, n is before.
    
    Args:
        gpu: Current GPU state.
        milli_to_allocate: milli-GPU to allocate (0-1000).
        
    Returns:
        Fragmentation increment. Lower (or negative) is better.
    """
    if not gpu.can_fit(milli_to_allocate):
        return float('inf')
    
    # Current fragmentation contribution
    if gpu.is_empty:
        old_frag = 0.0
    elif gpu.is_partial:
        old_frag = gpu.free_fraction
    else:  # full
        old_frag = 0.0
    
    # New state after allocation
    new_free_milli = gpu.free_milli - milli_to_allocate
    new_used_milli = gpu.used_milli + milli_to_allocate
    
    # New fragmentation contribution
    if new_free_milli == 0:  # Will be full - no fragmentation
        new_frag = 0.0
    elif new_used_milli == 0:  # Still empty - no fragmentation (impossible here)
        new_frag = 0.0
    else:  # Partial
        new_frag = new_free_milli / 1000.0
    
    return new_frag - old_frag


# =============================================================================
# FGD Placement Algorithm for GPU Sharing
# =============================================================================

def fgd_select_gpu(
    gpu_states: List[GPUState],
    milli_needed: int
) -> Optional[int]:
    """
    Select the best GPU for placement using FGD algorithm.
    
    FGD Algorithm (from paper):
    1. For each GPU that can fit the job
    2. Compute Δfrag = fragmentation increment if we place here
    3. Select GPU with minimum Δfrag
    
    Tie-breaking: Prefer GPUs with less free space (pack tightly)
    
    Args:
        gpu_states: Current state of all GPUs.
        milli_needed: milli-GPU required by the job.
        
    Returns:
        GPU ID to place the job on, or None if no GPU can fit it.
    """
    best_gpu = None
    best_delta = float('inf')
    best_free = float('inf')  # For tie-breaking
    
    for gpu in gpu_states:
        if not gpu.can_fit(milli_needed):
            continue
        
        delta = compute_fragmentation_increment_gpu(gpu, milli_needed)
        
        # Select GPU with minimum fragmentation increment
        # Tie-break by preferring GPUs with less free space (pack tightly)
        if (delta < best_delta or 
            (delta == best_delta and gpu.free_milli < best_free)):
            best_delta = delta
            best_gpu = gpu.gpu_id
            best_free = gpu.free_milli
    
    return best_gpu


def fgd_select_gpus_multi(
    gpu_states: List[GPUState],
    milli_needed: int,
    num_gpus: int
) -> Optional[List[int]]:
    """
    Select multiple GPUs for a job that needs multiple GPUs.
    
    For jobs that need multiple GPUs (scale_factor > 1), we need to
    select num_gpus GPUs, each with milli_needed capacity.
    
    Args:
        gpu_states: Current state of all GPUs.
        milli_needed: milli-GPU required per GPU.
        num_gpus: Number of GPUs needed.
        
    Returns:
        List of GPU IDs, or None if cannot fit.
    """
    # Make a copy to simulate allocations
    simulated = [
        GPUState(g.gpu_id, g.server_id, g.total_milli, g.used_milli, list(g.job_assignments))
        for g in gpu_states
    ]
    
    selected = []
    for _ in range(num_gpus):
        gpu_id = fgd_select_gpu(simulated, milli_needed)
        if gpu_id is None:
            return None
        
        selected.append(gpu_id)
        # Update simulated state
        simulated[gpu_id].used_milli += milli_needed
    
    return selected


# =============================================================================
# Baseline Placement Algorithms for Comparison
# =============================================================================

def bestfit_select_gpu(
    gpu_states: List[GPUState],
    milli_needed: int
) -> Optional[int]:
    """
    Best-fit placement: Select GPU with least free space that can fit the job.
    
    This is a common baseline that minimizes wasted space per-GPU but
    can lead to fragmentation across the cluster.
    """
    best_gpu = None
    best_free = float('inf')
    
    for gpu in gpu_states:
        if gpu.can_fit(milli_needed) and gpu.free_milli < best_free:
            best_free = gpu.free_milli
            best_gpu = gpu.gpu_id
    
    return best_gpu


def worstfit_select_gpu(
    gpu_states: List[GPUState],
    milli_needed: int
) -> Optional[int]:
    """
    Worst-fit placement: Select GPU with most free space.
    
    This spreads load evenly but can lead to more fragmentation.
    """
    best_gpu = None
    best_free = -1
    
    for gpu in gpu_states:
        if gpu.can_fit(milli_needed) and gpu.free_milli > best_free:
            best_free = gpu.free_milli
            best_gpu = gpu.gpu_id
    
    return best_gpu


def firstfit_select_gpu(
    gpu_states: List[GPUState],
    milli_needed: int
) -> Optional[int]:
    """
    First-fit placement: Select first GPU that can fit the job.
    """
    for gpu in gpu_states:
        if gpu.can_fit(milli_needed):
            return gpu.gpu_id
    return None


# =============================================================================
# GPU Sharing Cluster State Manager
# =============================================================================

class GPUSharingCluster:
    """
    Manages GPU state for a cluster with GPU sharing support.
    
    This class tracks per-GPU usage and provides placement decisions
    using FGD or baseline algorithms.
    """
    
    def __init__(self, num_servers: int, gpus_per_server: int):
        """
        Initialize cluster state.
        
        Args:
            num_servers: Number of servers in the cluster.
            gpus_per_server: Number of GPUs per server.
        """
        self.num_servers = num_servers
        self.gpus_per_server = gpus_per_server
        self.total_gpus = num_servers * gpus_per_server
        
        # Initialize GPU states
        self.gpu_states: List[GPUState] = []
        gpu_id = 0
        for server_id in range(num_servers):
            for _ in range(gpus_per_server):
                self.gpu_states.append(GPUState(
                    gpu_id=gpu_id,
                    server_id=server_id,
                    total_milli=1000,
                    used_milli=0,
                    job_assignments=[]
                ))
                gpu_id += 1
    
    def place_job(self, job_id: int, gpu_milli: int, num_gpus: int = 1,
                  strategy: str = 'fgd') -> Optional[List[int]]:
        """
        Place a job on the cluster using the specified strategy.
        
        Args:
            job_id: Unique job identifier.
            gpu_milli: milli-GPU required (0-1000).
            num_gpus: Number of GPUs needed (for multi-GPU jobs).
            strategy: Placement strategy ('fgd', 'bestfit', 'worstfit', 'firstfit').
            
        Returns:
            List of GPU IDs where the job was placed, or None if cannot fit.
        """
        if num_gpus == 1:
            # Single GPU job
            if strategy == 'fgd':
                gpu_id = fgd_select_gpu(self.gpu_states, gpu_milli)
            elif strategy == 'bestfit':
                gpu_id = bestfit_select_gpu(self.gpu_states, gpu_milli)
            elif strategy == 'worstfit':
                gpu_id = worstfit_select_gpu(self.gpu_states, gpu_milli)
            elif strategy == 'firstfit':
                gpu_id = firstfit_select_gpu(self.gpu_states, gpu_milli)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if gpu_id is None:
                return None
            
            # Update state
            self.gpu_states[gpu_id].used_milli += gpu_milli
            self.gpu_states[gpu_id].job_assignments.append((job_id, gpu_milli))
            return [gpu_id]
        else:
            # Multi-GPU job
            gpu_ids = fgd_select_gpus_multi(self.gpu_states, gpu_milli, num_gpus)
            if gpu_ids is None:
                return None
            
            for gpu_id in gpu_ids:
                self.gpu_states[gpu_id].used_milli += gpu_milli
                self.gpu_states[gpu_id].job_assignments.append((job_id, gpu_milli))
            return gpu_ids
    
    def remove_job(self, job_id: int) -> None:
        """Remove a job from the cluster, freeing its GPU resources."""
        for gpu in self.gpu_states:
            to_remove = [(jid, milli) for jid, milli in gpu.job_assignments if jid == job_id]
            for jid, milli in to_remove:
                gpu.used_milli -= milli
                gpu.job_assignments.remove((jid, milli))
    
    def get_fragmentation(self) -> float:
        """Get current cluster fragmentation score."""
        return compute_gpu_fragmentation(self.gpu_states)
    
    def get_utilization(self) -> float:
        """Get current GPU utilization (0.0-1.0)."""
        total_used = sum(gpu.used_milli for gpu in self.gpu_states)
        total_capacity = self.total_gpus * 1000
        return total_used / total_capacity
    
    def print_state(self) -> None:
        """Print current cluster state for debugging."""
        print("\n=== GPU Cluster State ===")
        for server_id in range(self.num_servers):
            server_gpus = [g for g in self.gpu_states if g.server_id == server_id]
            gpu_strs = []
            for g in server_gpus:
                if g.is_empty:
                    gpu_strs.append(f"[____]")
                elif g.is_full:
                    gpu_strs.append(f"[FULL]")
                else:
                    gpu_strs.append(f"[{g.used_fraction:.1f} ]")
            print(f"Server {server_id}: {' '.join(gpu_strs)}")
        print(f"Fragmentation: {self.get_fragmentation():.3f}")
        print(f"Utilization: {self.get_utilization():.1%}")


# =============================================================================
# Legacy Support: Whole-GPU Allocation (Original Gavel Integration)
# =============================================================================

def compute_fragmentation(server_states: List[ServerState]) -> int:
    """Compute fragmentation for whole-GPU allocation."""
    return sum(1 for s in server_states if s.is_partial)


def compute_fragmentation_increment(
    server_states: List[ServerState],
    server_id: int,
    gpus_to_allocate: int
) -> float:
    """Compute fragmentation increment for whole-GPU allocation."""
    server = server_states[server_id]
    
    if server.available_gpus < gpus_to_allocate:
        return float('inf')
    
    was_empty = server.is_empty
    was_partial = server.is_partial
    
    new_available = server.available_gpus - gpus_to_allocate
    new_allocated = server.allocated_gpus + gpus_to_allocate
    will_be_full = (new_available == 0)
    will_be_partial = (new_allocated > 0 and new_available > 0)
    
    delta = 0.0
    if was_empty and will_be_partial:
        delta += 1.0
    if was_partial and will_be_full:
        delta -= 1.0
    
    return delta


def fgd_select_server(
    server_states: List[ServerState],
    gpus_needed: int,
    use_weighted: bool = False
) -> Optional[int]:
    """Select best server for whole-GPU allocation using FGD."""
    best_server = None
    best_delta = float('inf')
    best_available = float('inf')
    
    for server in server_states:
        if server.available_gpus < gpus_needed:
            continue
        
        delta = compute_fragmentation_increment(server_states, server.server_id, gpus_needed)
        
        if (delta < best_delta or 
            (delta == best_delta and server.available_gpus < best_available)):
            best_delta = delta
            best_server = server.server_id
            best_available = server.available_gpus
    
    return best_server


def fgd_select_servers_multi_gpu(
    server_states: List[ServerState],
    gpus_needed: int,
    gpus_per_server: int,
    use_weighted: bool = False
) -> Optional[List[int]]:
    """Select multiple servers for a multi-GPU job using FGD."""
    if gpus_needed <= gpus_per_server:
        server_id = fgd_select_server(server_states, gpus_needed, use_weighted)
        return [server_id] if server_id is not None else None
    
    simulated_states = [
        ServerState(s.server_id, s.total_gpus, s.allocated_gpus, s.available_gpus)
        for s in server_states
    ]
    
    selected_servers = []
    remaining_gpus = gpus_needed
    
    while remaining_gpus > 0:
        gpus_from_this_server = min(remaining_gpus, gpus_per_server)
        server_id = fgd_select_server(simulated_states, gpus_from_this_server, use_weighted)
        
        if server_id is None:
            return None
        
        selected_servers.append(server_id)
        s = simulated_states[server_id]
        simulated_states[server_id] = ServerState(
            s.server_id, s.total_gpus,
            s.allocated_gpus + gpus_from_this_server,
            s.available_gpus - gpus_from_this_server
        )
        remaining_gpus -= gpus_from_this_server
    
    return selected_servers


# ============================================================================
# Integration with Gavel Scheduler
# ============================================================================

def build_server_states_from_gavel(
    worker_ids: List[List[int]],
    assigned_worker_ids: Set[int],
    gpus_per_server: int
) -> List[ServerState]:
    """
    Build ServerState objects from Gavel's worker state representation.
    
    In Gavel:
    - worker_ids is a list of lists, where worker_ids[server_id] contains
      the GPU/worker IDs for that server (gets modified during assignment)
    - assigned_worker_ids is a set of already-assigned worker IDs
    
    Args:
        worker_ids: Gavel's worker_ids structure (list of lists by server).
        assigned_worker_ids: Set of already assigned worker IDs.
        gpus_per_server: Number of GPUs per server (total capacity).
        
    Returns:
        List of ServerState objects.
    """
    server_states = []
    
    for server_id, server_workers in enumerate(worker_ids):
        # Count how many GPUs are still in the list and not assigned
        available = sum(1 for w in server_workers if w not in assigned_worker_ids)
        # Total GPUs allocated = gpus_per_server - current available
        # Note: some workers may have been popped from the list already
        total_in_list = len(server_workers)
        already_popped = gpus_per_server - total_in_list
        allocated = already_popped + (total_in_list - available)
        
        server_states.append(ServerState(
            server_id=server_id,
            total_gpus=gpus_per_server,
            allocated_gpus=allocated,
            available_gpus=available
        ))
    
    return server_states


def fgd_assign_workers_to_job(
    job_id,
    scale_factor: int,
    worker_type: str,
    worker_state: Dict,
    worker_assignments: Dict,
    gpus_per_server: int = 4,
    use_weighted: bool = False
) -> None:
    """
    FGD-based worker assignment for Gavel.
    
    This function is a drop-in replacement for Gavel's _assign_workers_to_job().
    It uses FGD to select the best placement that minimizes fragmentation.
    
    Args:
        job_id: The job (combination) ID to schedule.
        scale_factor: The number of GPUs requested.
        worker_type: The worker type to allocate.
        worker_state: Gavel's worker state dict containing:
            - worker_ids: Worker IDs organized into servers.
            - assigned_worker_ids: The set of worker IDs assigned so far.
            - server_id_ptr: The server to assign workers from (not used by FGD).
        worker_assignments: A map from job_id to assigned worker_ids tuple.
        gpus_per_server: Number of GPUs per server (default 4).
        use_weighted: Use weighted fragmentation metric if True.
    """
    worker_ids = worker_state['worker_ids']
    assigned_worker_ids = worker_state['assigned_worker_ids']
    
    # Check if job already has some workers assigned
    worker_ids_for_job = []
    if job_id in worker_assignments:
        worker_ids_for_job = list(worker_assignments[job_id])
    
    # Calculate how many more workers we need
    workers_needed = scale_factor - len(worker_ids_for_job)
    
    if workers_needed <= 0:
        # Job already has all workers it needs
        worker_assignments[job_id] = tuple(worker_ids_for_job)
        return
    
    # Build server states for FGD
    server_states = build_server_states_from_gavel(
        worker_ids, assigned_worker_ids, gpus_per_server
    )
    
    # Use FGD to select best server(s)
    if workers_needed <= gpus_per_server:
        # Single-server placement
        server_id = fgd_select_server(server_states, workers_needed, use_weighted)
        if server_id is None:
            raise RuntimeError(f'FGD: Could not find server for job {job_id}')
        selected_servers = [server_id]
        gpus_from_servers = [workers_needed]
    else:
        # Multi-server placement
        selected_servers = fgd_select_servers_multi_gpu(
            server_states, workers_needed, gpus_per_server, use_weighted
        )
        if selected_servers is None:
            raise RuntimeError(f'FGD: Could not find servers for job {job_id}')
        gpus_from_servers = [min(gpus_per_server, workers_needed - i * gpus_per_server)
                            for i in range(len(selected_servers))]
    
    # Collect worker IDs from selected servers
    for server_id, gpus_needed in zip(selected_servers, gpus_from_servers):
        server_workers = worker_ids[server_id]
        gpus_collected = 0
        for worker_id in server_workers:
            if worker_id not in assigned_worker_ids and gpus_collected < gpus_needed:
                worker_ids_for_job.append(worker_id)
                assigned_worker_ids.add(worker_id)
                gpus_collected += 1
    
    if len(worker_ids_for_job) != scale_factor:
        raise RuntimeError(
            f'FGD: Could not assign {scale_factor} workers to job {job_id}! '
            f'Only got {len(worker_ids_for_job)}'
        )
    
    worker_assignments[job_id] = tuple(worker_ids_for_job)


