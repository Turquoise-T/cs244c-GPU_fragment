"""FGD placement adapter for Gavel.

Bridges FGD's Node/Task model to Gavel's worker ID topology.
Replaces Gavel's strided placement in _assign_workers_to_job() with
fragmentation-aware placement from FGD.

Integration point: Called from _schedule_jobs_on_workers() after
_schedule_jobs_on_workers_helper() decides which jobs run this round.

Data mapping:
  - Gavel's _worker_type_to_worker_id_mapping[type] is list-of-lists:
    each inner list = one server's worker IDs.
  - Each server becomes one FGD Node.
  - GPU capacity = 1.0 if worker ID is free, 0.0 if assigned.
  - When enable_gpu_sharing=True, capacity is fractional (0.0 to 1.0).
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fgd'))

from fgd import FGDScheduler, FragmentationCalculator, Node, Task, Workload


def build_fgd_workload(mode='philly'):
    """Build a Workload model for FGD fragmentation calculation.

    Args:
        mode: 'philly' for Philly trace distribution,
              'alibaba' for Alibaba cluster-trace-gpu-v2023 distribution.

    Returns:
        FGD Workload object.
    """
    workload = Workload()
    if mode == 'alibaba':
        workload.add_task_type(
            Task(id='0.25gpu', cpu_request=2, gpu_request=0.25), 0.10)
        workload.add_task_type(
            Task(id='0.5gpu', cpu_request=4, gpu_request=0.5), 0.16)
        workload.add_task_type(
            Task(id='1gpu', cpu_request=8, gpu_request=1.0), 0.63)
        workload.add_task_type(
            Task(id='2gpu', cpu_request=16, gpu_request=2.0), 0.01)
        workload.add_task_type(
            Task(id='4gpu', cpu_request=32, gpu_request=4.0), 0.02)
        workload.add_task_type(
            Task(id='8gpu', cpu_request=64, gpu_request=8.0), 0.08)
    elif mode == 'philly':
        workload.add_task_type(
            Task(id='1gpu', cpu_request=4, gpu_request=1.0), 0.70)
        workload.add_task_type(
            Task(id='2gpu', cpu_request=8, gpu_request=2.0), 0.10)
        workload.add_task_type(
            Task(id='4gpu', cpu_request=16, gpu_request=4.0), 0.15)
        workload.add_task_type(
            Task(id='8gpu', cpu_request=32, gpu_request=8.0), 0.05)
    workload.normalize_popularity()
    return workload


class GavelFGDPlacement:
    """Adapter that uses FGD to assign Gavel worker IDs to jobs.

    Each scheduling round:
    1. Build FGD Node objects from Gavel's worker topology.
    2. Mark already-assigned workers as unavailable.
    3. For each new job, use FGD to pick a node and GPU indices.
    4. Translate back to Gavel worker IDs.
    """

    def __init__(self, workload=None, placement_mode='fgd',
                 enable_gpu_sharing=False):
        """
        Args:
            workload: FGD Workload for fragmentation calculation.
            placement_mode: 'fgd', 'bestfit', or 'firstfit'.
            enable_gpu_sharing: When True, support fractional GPU placement.
        """
        if workload is None:
            workload = build_fgd_workload('philly')
        self.workload = workload
        self.placement_mode = placement_mode
        self.enable_gpu_sharing = enable_gpu_sharing
        self._round_metrics = []
        # Sub-timers for profiling FGD internals
        self._profile = {
            'node_build': 0.0,
            'fgd_init': 0.0,
            'placement': 0.0,
            'frag_calc': 0.0,
            'calls': 0,
            'total_nodes': 0,
            'total_jobs_placed': 0,
        }

    def assign_workers_for_round(
        self,
        scheduled_jobs_for_type,
        worker_ids_by_server,
        assigned_worker_ids,
        worker_assignments,
        jobs_dict,
    ):
        """Assign workers using FGD placement for one worker type.

        Args:
            scheduled_jobs_for_type: List of (job_id, scale_factor) from
                _schedule_jobs_on_workers_helper().
            worker_ids_by_server: List-of-lists of worker IDs (one list per server).
                This is a deep copy from _worker_type_to_worker_id_mapping[type].
            assigned_worker_ids: Set (or dict when gpu_sharing enabled) of
                worker IDs already assigned this round (from lease extensions).
            worker_assignments: OrderedDict being populated with job_id -> worker_id tuple.
            jobs_dict: Scheduler's _jobs dict for looking up job properties.

        Returns:
            Fragmentation metric for this worker type this round.
        """
        self._profile['calls'] += 1

        # Build FGD nodes from Gavel server topology
        _t0 = time.perf_counter()
        nodes = []
        # Map: node_id -> (server_index, list of worker_ids for that server)
        node_to_server = {}

        for server_idx, server_worker_ids in enumerate(worker_ids_by_server):
            gpus = []
            available_worker_ids = []
            for wid in server_worker_ids:
                if self.enable_gpu_sharing:
                    used = assigned_worker_ids.get(wid, 0.0)
                    gpus.append(max(0.0, 1.0 - used))
                else:
                    if wid in assigned_worker_ids:
                        gpus.append(0.0)
                    else:
                        gpus.append(1.0)
                available_worker_ids.append(wid)

            if not server_worker_ids:
                continue

            # Pre-filter: skip servers where all GPUs are fully assigned.
            if all(g < 1e-9 for g in gpus):
                continue

            node_id = f'server-{server_idx}'
            node = Node(
                id=node_id,
                total_cpu=1000.0,  # Gavel doesn't model CPU, use large default
                total_memory=1000.0,
                gpus=gpus,
                gpu_type='generic',
            )
            # Set allocated CPU/memory proportionally
            allocated_count = sum(1 for g in gpus if g < 1e-9)
            node.allocated_cpu = allocated_count * 10.0
            node.allocated_memory = allocated_count * 10.0
            nodes.append(node)
            node_to_server[node_id] = (server_idx, server_worker_ids)

        self._profile['node_build'] += time.perf_counter() - _t0
        self._profile['total_nodes'] += len(nodes)

        if not nodes:
            return 0.0

        # Create FGD scheduler or baseline placer for this round.
        # All modes enforce single-node placement (FGD paper semantics).
        _t0 = time.perf_counter()
        if self.placement_mode == 'fgd':
            fgd = FGDScheduler(nodes, self.workload)
        elif self.placement_mode == 'bestfit':
            from baselines import BestFitPlacer
            placer = BestFitPlacer()
        elif self.placement_mode == 'firstfit':
            from baselines import FirstFitPlacer
            placer = FirstFitPlacer()
        elif self.placement_mode == 'random':
            from baselines import RandomPlacer
            placer = RandomPlacer()
        else:
            raise ValueError(f"Unknown placement mode: {self.placement_mode}")

        self._profile['fgd_init'] += time.perf_counter() - _t0

        # Sort jobs by scale factor (largest first) for better packing
        jobs_to_place = []
        for (job_id, scale_factor) in scheduled_jobs_for_type:
            if job_id in worker_assignments:
                continue  # Already has workers (lease extension)
            jobs_to_place.append((job_id, scale_factor))
        jobs_to_place.sort(key=lambda x: x[1], reverse=True)

        # Place each job
        _t0 = time.perf_counter()
        for (job_id, scale_factor) in jobs_to_place:
            # Determine actual GPU demand for this job
            if self.enable_gpu_sharing:
                single_id = job_id.singletons()[0]
                job = jobs_dict.get(single_id)
                actual_gpu = (job.gpu_request
                              if (job and job.gpu_request is not None)
                              else float(scale_factor))
            else:
                actual_gpu = float(scale_factor)

            fgd_task = Task(
                id=str(job_id),
                cpu_request=actual_gpu * 10.0,
                gpu_request=actual_gpu,
            )

            # Use FGD to pick placement
            if self.placement_mode == 'fgd':
                best_node, gpu_indices = fgd.schedule_task(fgd_task)
                if best_node is not None:
                    fgd.allocate_task(fgd_task, best_node, gpu_indices)
            else:
                best_node, gpu_indices = placer.place(fgd_task, nodes)
                if best_node is not None:
                    # Manually allocate on node
                    best_node.allocated_cpu += fgd_task.cpu_request
                    best_node.allocated_memory += fgd_task.memory_request
                    for idx in gpu_indices:
                        best_node.gpus[idx] = max(0.0,
                                                  best_node.gpus[idx] - actual_gpu)

            # Single-node placement only: if the job can't fit on one node,
            # it's a placement failure (fragmentation). This matches the FGD
            # paper's definition where tasks must fit on a single node.
            if best_node is None:
                continue

            # Translate node + GPU indices back to Gavel worker IDs
            server_idx, server_worker_ids = node_to_server[best_node.id]
            worker_ids_for_job = []
            for gpu_idx in gpu_indices:
                wid = server_worker_ids[gpu_idx]
                worker_ids_for_job.append(wid)
                if self.enable_gpu_sharing:
                    assigned_worker_ids[wid] = (
                        assigned_worker_ids.get(wid, 0.0) + actual_gpu)
                else:
                    assigned_worker_ids.add(wid)

            if len(worker_ids_for_job) == scale_factor:
                worker_assignments[job_id] = tuple(worker_ids_for_job)

        self._profile['placement'] += time.perf_counter() - _t0
        self._profile['total_jobs_placed'] += len(jobs_to_place)

        # Compute fragmentation metric for this round
        _t0 = time.perf_counter()
        frag = FragmentationCalculator.compute_cluster_fragmentation(
            nodes, self.workload
        )
        self._profile['frag_calc'] += time.perf_counter() - _t0
        return frag
