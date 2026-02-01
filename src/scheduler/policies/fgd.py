"""
FGD (Fragmentation Gradient Descent) Placement Algorithm
=========================================================

Based on: "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with
          Fragmentation Gradient Descent" (ATC 2023)

IMPORTANT: FGD is a PLACEMENT algorithm, NOT an allocation policy.
- Allocation (Policy class): Decides time-share fractions X_mj for each job on each worker type
- Placement (FGD): Decides WHICH specific GPUs/workers a job runs on to minimize fragmentation

The existing Gavel policies (FIFO, MaxMinFairness, etc.) handle allocation.
FGD should be used in the scheduler when assigning workers to jobs, as an
alternative to the default strided assignment in _assign_workers_to_job().

Fragmentation Definition (from paper):
--------------------------------------
Fragmentation occurs when GPU resources become unusable due to partial allocation.
A "fragment" is created when a server has some but not all GPUs allocated,
leaving the remaining GPUs potentially unusable for jobs requiring multiple GPUs.

For example:
- Server with 4 GPUs, job needs 2 GPUs
- If GPUs are allocated poorly, remaining 2 GPUs might be on different servers
- This creates fragmentation: those GPUs can't serve another 2-GPU job together

FGD Approach:
-------------
When placing a job, FGD computes the "fragmentation gradient" (Δfrag) for each
candidate placement and chooses the one that minimizes the fragmentation increase.

Integration with Gavel:
-----------------------
To use FGD placement in Gavel, modify scheduler.py:
1. In _schedule_jobs_on_workers_helper(), replace the call to _assign_workers_to_job()
   with a call to fgd_assign_workers() when FGD is enabled.
2. Add a placement_strategy parameter to the Scheduler class ('strided' or 'fgd').
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ServerState:
    """Represents the state of a single server."""
    server_id: int
    total_gpus: int
    allocated_gpus: int
    available_gpus: int
    
    @property
    def is_empty(self) -> bool:
        """Server has no allocated GPUs."""
        return self.allocated_gpus == 0
    
    @property
    def is_full(self) -> bool:
        """Server has all GPUs allocated."""
        return self.available_gpus == 0
    
    @property
    def is_partial(self) -> bool:
        """Server has some but not all GPUs allocated (fragmented)."""
        return not self.is_empty and not self.is_full


def compute_fragmentation(server_states: List[ServerState]) -> int:
    """
    Compute current cluster fragmentation.
    
    Fragmentation metric: Count of servers with partial allocation.
    A server is "fragmented" if it has some but not all GPUs allocated.
    
    This is a simple metric that captures the essence of FGD's goal:
    minimize the number of partially-used servers.
    
    Args:
        server_states: List of ServerState objects representing each server.
        
    Returns:
        Number of fragmented (partially allocated) servers.
    """
    return sum(1 for s in server_states if s.is_partial)


def compute_fragmentation_weighted(server_states: List[ServerState]) -> float:
    """
    Compute weighted fragmentation metric.
    
    More sophisticated metric that considers HOW fragmented each server is.
    Fragmentation for a server = available_gpus / total_gpus (when partial)
    
    Higher values indicate worse fragmentation.
    
    Args:
        server_states: List of ServerState objects.
        
    Returns:
        Sum of fragmentation ratios across all partially-used servers.
    """
    total_frag = 0.0
    for s in server_states:
        if s.is_partial:
            # Fragmentation ratio: what fraction of GPUs are wasted/unavailable
            total_frag += s.available_gpus / s.total_gpus
    return total_frag


def compute_fragmentation_increment(
    server_states: List[ServerState],
    server_id: int,
    gpus_to_allocate: int
) -> float:
    """
    Compute the fragmentation increment (Δfrag) if we allocate GPUs on a server.
    
    This is the core of FGD: for each candidate server, compute how much
    the fragmentation would increase if we placed the job there.
    
    Args:
        server_states: Current server states.
        server_id: The server to consider for placement.
        gpus_to_allocate: Number of GPUs the job needs.
        
    Returns:
        The change in fragmentation (Δfrag) if we place the job on this server.
        Lower is better.
    """
    server = server_states[server_id]
    
    if server.available_gpus < gpus_to_allocate:
        # Cannot fit on this server
        return float('inf')
    
    # Current fragmentation state of this server
    was_empty = server.is_empty
    was_partial = server.is_partial
    
    # Compute new state after allocation
    new_available = server.available_gpus - gpus_to_allocate
    new_allocated = server.allocated_gpus + gpus_to_allocate
    will_be_full = (new_available == 0)
    will_be_partial = (new_allocated > 0 and new_available > 0)
    
    # Compute fragmentation change
    delta = 0.0
    
    # If server was empty and becomes partial: +1 fragmented server
    if was_empty and will_be_partial:
        delta += 1.0
    
    # If server was partial and becomes full: -1 fragmented server (good!)
    if was_partial and will_be_full:
        delta -= 1.0
    
    # If server was empty and becomes full: no change (ideal case)
    # If server was partial and stays partial: no change in count
    
    return delta


def compute_fragmentation_increment_weighted(
    server_states: List[ServerState],
    server_id: int,
    gpus_to_allocate: int
) -> float:
    """
    Compute weighted fragmentation increment.
    
    More nuanced metric that considers the severity of fragmentation.
    """
    server = server_states[server_id]
    
    if server.available_gpus < gpus_to_allocate:
        return float('inf')
    
    # Current weighted fragmentation contribution from this server
    if server.is_partial:
        old_frag = server.available_gpus / server.total_gpus
    else:
        old_frag = 0.0
    
    # New state
    new_available = server.available_gpus - gpus_to_allocate
    new_allocated = server.allocated_gpus + gpus_to_allocate
    
    # New weighted fragmentation contribution
    if new_allocated > 0 and new_available > 0:
        new_frag = new_available / server.total_gpus
    else:
        new_frag = 0.0
    
    return new_frag - old_frag


def fgd_select_server(
    server_states: List[ServerState],
    gpus_needed: int,
    use_weighted: bool = False
) -> Optional[int]:
    """
    Select the best server for job placement using FGD algorithm.
    
    FGD Algorithm (simplified):
    1. For each candidate server that can fit the job
    2. Compute the fragmentation increment (Δfrag) if we place the job there
    3. Select the server with the minimum Δfrag
    
    Tie-breaking: Prefer servers with fewer available GPUs (pack tightly)
    
    Args:
        server_states: Current state of all servers.
        gpus_needed: Number of GPUs the job requires.
        use_weighted: Use weighted fragmentation metric if True.
        
    Returns:
        Server ID to place the job on, or None if no server can fit it.
    """
    best_server = None
    best_delta = float('inf')
    best_available = float('inf')  # For tie-breaking
    
    compute_fn = (compute_fragmentation_increment_weighted if use_weighted
                  else compute_fragmentation_increment)
    
    for server in server_states:
        if server.available_gpus < gpus_needed:
            continue
            
        delta = compute_fn(server_states, server.server_id, gpus_needed)
        
        # Select server with minimum fragmentation increment
        # Tie-break by preferring servers with fewer available GPUs (pack tightly)
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
    """
    Select multiple servers for a distributed job using FGD.
    
    For jobs that span multiple servers (e.g., 8 GPUs across 2 servers with 4 GPUs each),
    we need to select multiple servers while minimizing total fragmentation.
    
    This is a greedy approach: repeatedly select the best single server until
    we have enough GPUs.
    
    Args:
        server_states: Current state of all servers.
        gpus_needed: Total number of GPUs the job requires.
        gpus_per_server: Number of GPUs per server (cluster configuration).
        use_weighted: Use weighted fragmentation metric if True.
        
    Returns:
        List of server IDs to place the job on, or None if cannot fit.
    """
    if gpus_needed <= gpus_per_server:
        # Single server job
        server_id = fgd_select_server(server_states, gpus_needed, use_weighted)
        return [server_id] if server_id is not None else None
    
    # Multi-server job: need to select multiple servers
    # Make a copy of server states to simulate allocations
    simulated_states = [
        ServerState(s.server_id, s.total_gpus, s.allocated_gpus, s.available_gpus)
        for s in server_states
    ]
    
    selected_servers = []
    remaining_gpus = gpus_needed
    
    while remaining_gpus > 0:
        # How many GPUs to allocate from next server
        gpus_from_this_server = min(remaining_gpus, gpus_per_server)
        
        # Find best server for this chunk
        server_id = fgd_select_server(simulated_states, gpus_from_this_server, use_weighted)
        
        if server_id is None:
            return None  # Cannot fit job
        
        selected_servers.append(server_id)
        
        # Update simulated state
        s = simulated_states[server_id]
        simulated_states[server_id] = ServerState(
            s.server_id,
            s.total_gpus,
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
      the GPU/worker IDs for that server
    - assigned_worker_ids is a set of already-assigned worker IDs
    
    Args:
        worker_ids: Gavel's worker_ids structure (list of lists by server).
        assigned_worker_ids: Set of already assigned worker IDs.
        gpus_per_server: Number of GPUs per server.
        
    Returns:
        List of ServerState objects.
    """
    server_states = []
    
    for server_id, server_workers in enumerate(worker_ids):
        # Count how many GPUs are still available (not assigned) on this server
        available = sum(1 for w in server_workers if w not in assigned_worker_ids)
        allocated = gpus_per_server - available
        
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
    
    # Build server states for FGD
    server_states = build_server_states_from_gavel(
        worker_ids, assigned_worker_ids, gpus_per_server
    )
    
    # Use FGD to select best server(s)
    if scale_factor <= gpus_per_server:
        # Single-server placement
        server_id = fgd_select_server(server_states, scale_factor, use_weighted)
        if server_id is None:
            raise RuntimeError(f'FGD: Could not find server for job {job_id}')
        selected_servers = [server_id]
        gpus_from_servers = [scale_factor]
    else:
        # Multi-server placement
        selected_servers = fgd_select_servers_multi_gpu(
            server_states, scale_factor, gpus_per_server, use_weighted
        )
        if selected_servers is None:
            raise RuntimeError(f'FGD: Could not find servers for job {job_id}')
        gpus_from_servers = [min(gpus_per_server, scale_factor - i * gpus_per_server)
                            for i in range(len(selected_servers))]
    
    # Collect worker IDs from selected servers
    worker_ids_for_job = []
    if job_id in worker_assignments:
        worker_ids_for_job = list(worker_assignments[job_id])
    
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


# ============================================================================
# Utility Functions for Testing and Debugging
# ============================================================================

def print_cluster_state(server_states: List[ServerState]) -> None:
    """Print current cluster state for debugging."""
    print("\n=== Cluster State ===")
    print(f"{'Server':<10} {'Total':<8} {'Allocated':<12} {'Available':<12} {'Status':<10}")
    print("-" * 52)
    for s in server_states:
        status = "EMPTY" if s.is_empty else ("FULL" if s.is_full else "PARTIAL")
        print(f"{s.server_id:<10} {s.total_gpus:<8} {s.allocated_gpus:<12} {s.available_gpus:<12} {status:<10}")
    print(f"\nFragmentation (count): {compute_fragmentation(server_states)}")
    print(f"Fragmentation (weighted): {compute_fragmentation_weighted(server_states):.3f}")


def simulate_fgd_vs_strided(
    num_servers: int = 8,
    gpus_per_server: int = 4,
    job_sizes: List[int] = None
) -> Tuple[int, int]:
    """
    Simulate FGD vs strided placement to compare fragmentation.
    
    Args:
        num_servers: Number of servers in the cluster.
        gpus_per_server: GPUs per server.
        job_sizes: List of GPU requirements for jobs to place.
        
    Returns:
        Tuple of (fgd_fragmentation, strided_fragmentation).
    """
    if job_sizes is None:
        job_sizes = [2, 1, 2, 3, 1, 2, 1, 2]  # Example workload
    
    # Initialize server states
    fgd_states = [
        ServerState(i, gpus_per_server, 0, gpus_per_server)
        for i in range(num_servers)
    ]
    strided_states = [
        ServerState(i, gpus_per_server, 0, gpus_per_server)
        for i in range(num_servers)
    ]
    
    strided_ptr = 0  # Server pointer for strided allocation
    
    for job_gpus in job_sizes:
        # FGD placement
        server_id = fgd_select_server(fgd_states, job_gpus)
        if server_id is not None:
            s = fgd_states[server_id]
            fgd_states[server_id] = ServerState(
                s.server_id, s.total_gpus,
                s.allocated_gpus + job_gpus,
                s.available_gpus - job_gpus
            )
        
        # Strided placement (Gavel default)
        gpus_placed = 0
        while gpus_placed < job_gpus and strided_ptr < num_servers:
            s = strided_states[strided_ptr]
            if s.available_gpus > 0:
                gpus_to_place = min(s.available_gpus, job_gpus - gpus_placed)
                strided_states[strided_ptr] = ServerState(
                    s.server_id, s.total_gpus,
                    s.allocated_gpus + gpus_to_place,
                    s.available_gpus - gpus_to_place
                )
                gpus_placed += gpus_to_place
            if strided_states[strided_ptr].available_gpus == 0:
                strided_ptr += 1
    
    fgd_frag = compute_fragmentation(fgd_states)
    strided_frag = compute_fragmentation(strided_states)
    
    return fgd_frag, strided_frag


if __name__ == "__main__":
    # Test FGD placement algorithm
    print("Testing FGD Placement Algorithm")
    print("=" * 50)
    
    # Compare FGD vs strided
    fgd_frag, strided_frag = simulate_fgd_vs_strided()
    print(f"\nJob sizes: [2, 1, 2, 3, 1, 2, 1, 2]")
    print(f"FGD fragmentation: {fgd_frag}")
    print(f"Strided fragmentation: {strided_frag}")
    print(f"Improvement: {strided_frag - fgd_frag} fewer fragmented servers")
    
    # More detailed test
    print("\n" + "=" * 50)
    print("Detailed FGD simulation")
    
    server_states = [ServerState(i, 4, 0, 4) for i in range(4)]
    
    print("\nInitial state:")
    print_cluster_state(server_states)
    
    # Place jobs one by one
    jobs = [(0, 2), (1, 2), (2, 1), (3, 2), (4, 1)]
    
    for job_id, gpus in jobs:
        server_id = fgd_select_server(server_states, gpus)
        if server_id is not None:
            s = server_states[server_id]
            server_states[server_id] = ServerState(
                s.server_id, s.total_gpus,
                s.allocated_gpus + gpus,
                s.available_gpus - gpus
            )
            print(f"\nPlaced job {job_id} ({gpus} GPUs) on server {server_id}")
            print_cluster_state(server_states)
