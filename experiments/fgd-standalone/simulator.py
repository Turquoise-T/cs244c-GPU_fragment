"""Event-driven simulator for standalone FGD evaluation.

Simulates task arrivals, placement decisions, and departures on a cluster.
Used to replicate FGD paper results (Figures 7-9) independently of Gavel.

Two modes:
  1. Trace replay: Tasks arrive at times from the Alibaba trace.
  2. Inflation: Submit tasks one-by-one from a shuffled task list until
     cumulative demand reaches a target fraction of cluster capacity.
"""

import heapq
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'fgd'))

from fgd import FGDScheduler, FragmentationCalculator, Node, Task, Workload
from baselines import (
    BaselinePlacer, BestFitPlacer, FirstFitPlacer, RandomPlacer,
    DotProdPlacer, GpuPackingPlacer, GpuClusteringPlacer,
)
from alibaba_trace_parser import TraceTask


# -- Events --

ARRIVAL = 0
DEPARTURE = 1


@dataclass(order=True)
class Event:
    time: float
    event_type: int = field(compare=False)
    task_id: str = field(compare=False)


# -- Metrics snapshot --

@dataclass
class MetricsSnapshot:
    time: float
    total_gpus: int
    allocated_gpus: float
    unallocated_gpus: float
    fragmentation: float  # F_N(M)
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    gpu_utilization: float  # allocated / total


class FGDSimulator:
    """Event-driven simulator for FGD and baseline placement strategies."""

    def __init__(self, cluster_config: Optional[dict] = None,
                 placement: str = 'fgd', seed: int = 42,
                 workload: Optional[Workload] = None,
                 nodes: Optional[List[Node]] = None):
        """
        Args:
            cluster_config: Cluster spec dict (from cluster_h.json). Ignored if
                           nodes is provided.
            placement: One of 'fgd', 'bestfit', 'firstfit', 'random',
                      'dotprod', 'gpupacking', 'gpuclustering'.
            seed: Random seed for reproducibility.
            workload: FGD Workload for fragmentation calculation. If None,
                      a default Philly-like distribution is used.
            nodes: Pre-built list of Node objects. If provided, cluster_config
                   is ignored. Nodes should be fresh (unallocated).
        """
        if nodes is not None:
            self.nodes = nodes
        elif cluster_config is not None:
            self.nodes = self._build_cluster(cluster_config)
        else:
            raise ValueError("Must provide either cluster_config or nodes")

        self.total_gpus = sum(n.num_gpus for n in self.nodes)
        self.placement = placement
        self.seed = seed

        # Build workload model for fragmentation measurement
        if workload is not None:
            self.workload = workload
        else:
            self.workload = self._default_workload()

        # Placement strategy
        self._init_placer(placement, seed)

        # State
        self.event_queue: List[Event] = []
        self.running_tasks: Dict[str, Tuple[Node, List[int], Task]] = {}
        self.pending_tasks: List[TraceTask] = []
        self.completed_count = 0
        self.current_time = 0.0
        self.rejected_count = 0

        # Metrics history
        self.metrics_history: List[MetricsSnapshot] = []

    def _init_placer(self, placement: str, seed: int):
        """Initialize the placement strategy."""
        if placement == 'fgd':
            self.placer = None
            self.fgd_scheduler = FGDScheduler(self.nodes, self.workload)
        elif placement == 'bestfit':
            self.placer = BestFitPlacer()
            self.fgd_scheduler = None
        elif placement == 'firstfit':
            self.placer = FirstFitPlacer()
            self.fgd_scheduler = None
        elif placement == 'random':
            self.placer = RandomPlacer(seed=seed)
            self.fgd_scheduler = None
        elif placement == 'dotprod':
            self.placer = DotProdPlacer()
            self.fgd_scheduler = None
        elif placement == 'gpupacking':
            self.placer = GpuPackingPlacer()
            self.fgd_scheduler = None
        elif placement == 'gpuclustering':
            self.placer = GpuClusteringPlacer()
            self.fgd_scheduler = None
        else:
            raise ValueError(f"Unknown placement strategy: {placement}")

    def _build_cluster(self, config: dict) -> List[Node]:
        """Build Node objects from cluster config JSON."""
        nodes = []
        node_id = 0
        for node_type, spec in config['nodes'].items():
            for _ in range(spec['count']):
                node = Node(
                    id=f'{node_type}-{node_id}',
                    total_cpu=spec['cpu_per_node'],
                    total_memory=spec['memory_per_node'],
                    gpus=[1.0] * spec['gpus_per_node'],
                    gpu_type=config.get('gpu_type', 'generic'),
                )
                nodes.append(node)
                node_id += 1
        return nodes

    def _default_workload(self) -> Workload:
        """Philly-like workload distribution (70/10/15/5 split)."""
        workload = Workload()
        workload.add_task_type(Task(id='1gpu', cpu_request=4, gpu_request=1.0), 0.70)
        workload.add_task_type(Task(id='2gpu', cpu_request=8, gpu_request=2.0), 0.10)
        workload.add_task_type(Task(id='4gpu', cpu_request=16, gpu_request=4.0), 0.15)
        workload.add_task_type(Task(id='8gpu', cpu_request=32, gpu_request=8.0), 0.05)
        workload.normalize_popularity()
        return workload

    def _place_task(self, task: Task) -> Tuple[Optional[Node], Optional[List[int]]]:
        """Place a task using the configured strategy."""
        if self.fgd_scheduler is not None:
            return self.fgd_scheduler.schedule_task(task)
        else:
            return self.placer.place(task, self.nodes)

    def _allocate(self, task: Task, node: Node, gpu_indices: List[int]):
        """Commit a placement decision to the node."""
        if self.fgd_scheduler is not None:
            self.fgd_scheduler.allocate_task(task, node, gpu_indices)
        else:
            node.allocated_cpu += task.cpu_request
            node.allocated_memory += task.memory_request
            if task.gpu_request > 0:
                if 0 < task.gpu_request < 1:
                    node.gpus[gpu_indices[0]] -= task.gpu_request
                else:
                    for idx in gpu_indices:
                        node.gpus[idx] = 0.0

    def _deallocate(self, task: Task, node: Node, gpu_indices: List[int]):
        """Release resources when a task completes."""
        node.allocated_cpu -= task.cpu_request
        node.allocated_memory -= task.memory_request
        if task.gpu_request > 0:
            if 0 < task.gpu_request < 1:
                node.gpus[gpu_indices[0]] += task.gpu_request
            else:
                for idx in gpu_indices:
                    node.gpus[idx] = 1.0

        # Update FGD scheduler state if applicable
        if self.fgd_scheduler is not None and task.id in self.fgd_scheduler.scheduled_tasks:
            del self.fgd_scheduler.scheduled_tasks[task.id]

    def _allocated_gpus(self) -> float:
        """Count allocated GPU capacity across cluster."""
        total = 0.0
        for node in self.nodes:
            for gpu in node.gpus:
                total += (1.0 - gpu)
        return total

    def _record_metrics(self):
        """Snapshot current cluster state."""
        allocated = self._allocated_gpus()
        frag = FragmentationCalculator.compute_cluster_fragmentation(
            self.nodes, self.workload
        )
        snap = MetricsSnapshot(
            time=self.current_time,
            total_gpus=self.total_gpus,
            allocated_gpus=allocated,
            unallocated_gpus=self.total_gpus - allocated,
            fragmentation=frag,
            pending_tasks=len(self.pending_tasks),
            running_tasks=len(self.running_tasks),
            completed_tasks=self.completed_count,
            gpu_utilization=allocated / self.total_gpus if self.total_gpus > 0 else 0,
        )
        self.metrics_history.append(snap)

    def _try_schedule_pending(self):
        """Attempt to place pending tasks (FCFS order)."""
        still_pending = []
        for trace_task in self.pending_tasks:
            node, gpu_indices = self._place_task(trace_task.task)
            if node is not None:
                self._allocate(trace_task.task, node, gpu_indices)
                self.running_tasks[trace_task.task.id] = (node, gpu_indices, trace_task.task)
                # Schedule departure
                departure_time = self.current_time + trace_task.duration
                heapq.heappush(
                    self.event_queue,
                    Event(time=departure_time, event_type=DEPARTURE, task_id=trace_task.task.id)
                )
            else:
                still_pending.append(trace_task)
        self.pending_tasks = still_pending

    def run(self, trace_tasks: List[TraceTask], metrics_interval: float = 60.0):
        """Run simulation on a list of trace tasks.

        Args:
            trace_tasks: Tasks with arrival times and durations.
            metrics_interval: How often to record metrics snapshots (seconds).
        """
        # Enqueue all arrivals
        for tt in trace_tasks:
            heapq.heappush(
                self.event_queue,
                Event(time=tt.arrival_time, event_type=ARRIVAL, task_id=tt.task.id)
            )

        # Map task IDs to TraceTask for lookup
        task_map = {tt.task.id: tt for tt in trace_tasks}

        next_metrics_time = 0.0

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            # Record metrics at intervals
            while self.current_time >= next_metrics_time:
                self._record_metrics()
                next_metrics_time += metrics_interval

            if event.event_type == ARRIVAL:
                trace_task = task_map[event.task_id]
                # Try to place immediately
                node, gpu_indices = self._place_task(trace_task.task)
                if node is not None:
                    self._allocate(trace_task.task, node, gpu_indices)
                    self.running_tasks[trace_task.task.id] = (node, gpu_indices, trace_task.task)
                    departure_time = self.current_time + trace_task.duration
                    heapq.heappush(
                        self.event_queue,
                        Event(time=departure_time, event_type=DEPARTURE, task_id=trace_task.task.id)
                    )
                else:
                    self.pending_tasks.append(trace_task)

            elif event.event_type == DEPARTURE:
                if event.task_id in self.running_tasks:
                    node, gpu_indices, task = self.running_tasks.pop(event.task_id)
                    self._deallocate(task, node, gpu_indices)
                    self.completed_count += 1
                    # Try to place pending tasks now that resources freed
                    self._try_schedule_pending()

        # Final metrics
        self._record_metrics()

    def run_inflation(self, distribution, target_utilization: float = 1.0,
                      batch_size: int = 100, max_tasks: int = 50000,
                      mean_duration: float = 3600.0, seed: int = 42):
        """Monte-Carlo workload inflation experiment (legacy batch mode).

        Repeatedly submit batches of tasks sampled from the distribution
        (all arriving at time 0, infinite duration) until the cluster
        reaches the target GPU utilization.

        Args:
            distribution: List of tuples where first element is gpu_request,
                         second is cpu_request, last is popularity.
            target_utilization: Stop when allocated/total >= this.
            batch_size: Tasks per batch.
            max_tasks: Safety limit.
            mean_duration: Not used (tasks run forever in inflation mode).
            seed: Random seed.

        Returns:
            List of result dicts per batch.
        """
        import random as rng_module
        rng = rng_module.Random(seed)

        gpu_requests = [d[0] for d in distribution]
        cpu_requests = [d[1] for d in distribution]
        weights = [d[-1] for d in distribution]

        results = []
        total_demand = 0.0
        task_counter = 0

        while task_counter < max_tasks:
            # Generate a batch
            for _ in range(batch_size):
                idx = _weighted_choice(weights, rng)
                task = Task(
                    id=f'inflate-{task_counter}',
                    cpu_request=cpu_requests[idx],
                    gpu_request=gpu_requests[idx],
                )
                node, gpu_indices = self._place_task(task)
                if node is not None:
                    self._allocate(task, node, gpu_indices)
                    self.running_tasks[task.id] = (node, gpu_indices, task)
                total_demand += gpu_requests[idx]
                task_counter += 1

            # Record state
            allocated = self._allocated_gpus()
            unallocated = self.total_gpus - allocated
            frag = FragmentationCalculator.compute_cluster_fragmentation(
                self.nodes, self.workload
            )
            utilization = allocated / self.total_gpus

            results.append({
                'cumulative_gpu_demand': total_demand,
                'demand_fraction': total_demand / self.total_gpus,
                'allocated_gpus': allocated,
                'unallocated_gpus': unallocated,
                'fragmentation': frag,
                'utilization': utilization,
                'tasks_submitted': task_counter,
            })

            if utilization >= target_utilization:
                break

        return results

    def _compute_occupied_nodes(self) -> int:
        """Count nodes with at least one GPU partially or fully allocated."""
        count = 0
        for n in self.nodes:
            if any(g < 1.0 for g in n.gpus) or n.allocated_cpu > 0:
                count += 1
        return count

    def _compute_frag_breakdown(self) -> Tuple[float, float, float]:
        """Classify fragmented GPUs into non_gpu, stranded, deficient.

        For each node with free GPU capacity:
        - non_gpu: insufficient CPU or memory for ANY workload task
        - stranded: enough CPU/mem but GPU scalar too low for any task
        - deficient: individual GPU slots too small (normal fragmentation)

        Returns:
            (non_gpu_gpus, stranded_gpus, deficient_gpus) as raw GPU counts.
        """
        non_gpu = 0.0
        stranded = 0.0
        deficient = 0.0

        for node in self.nodes:
            free_gpu_capacity = sum(node.gpus)
            if free_gpu_capacity < 0.001:
                continue  # No free GPUs on this node

            # Check if node can fit ANY workload task type
            can_fit_any = False
            can_fit_cpu_mem_any = False
            for task_id, task in self.workload.tasks.items():
                has_cpu_mem = (node.available_cpu >= task.cpu_request and
                              node.available_memory >= task.memory_request)
                if has_cpu_mem:
                    can_fit_cpu_mem_any = True
                    # Check GPU type constraint
                    if task.gpu_type is not None:
                        acceptable = set(task.gpu_type.split('|'))
                        if node.gpu_type not in acceptable:
                            continue
                    if node.get_gpu_scalar() >= task.gpu_request:
                        can_fit_any = True
                        break

            if not can_fit_cpu_mem_any:
                # No task can run due to CPU/memory shortage
                non_gpu += free_gpu_capacity
            elif not can_fit_any:
                # Has CPU/mem for some task but GPU scalar too low
                stranded += free_gpu_capacity
            else:
                # Node CAN fit at least one task, but individual GPU slots
                # may still be fragmented (too small for some tasks)
                node_frag = FragmentationCalculator.compute_node_fragmentation_for_workload(
                    node, self.workload
                )
                deficient += node_frag

        return non_gpu, stranded, deficient

    def run_inflation_from_tasks(self, tasks: List[Task],
                                 target_demand_fraction: float = 1.3,
                                 record_interval: int = 1):
        """Inflation experiment matching the reference Go implementation.

        Submits tasks one at a time from a pre-shuffled list. Tasks run
        forever (no departures). Stops when cumulative GPU demand reaches
        target_demand_fraction * total_gpus.

        Args:
            tasks: Pre-shuffled list of Task objects.
            target_demand_fraction: Stop when cumulative_demand / total_gpus
                                    >= this value (default 1.3).
            record_interval: Record metrics every N tasks (default 1).

        Returns:
            List of curve points with all metrics for Figures 7a/7b/9a-9d.
        """
        curve = []
        total_demand = 0.0
        rejected = 0

        # Pending GPU-equivalents by size bucket (cumulative)
        pending_by_gpu_size = {
            'lt1_gpu': 0.0,
            '1_gpu': 0.0,
            '2_gpu': 0.0,
            '8_gpu': 0.0,
        }

        for i, task in enumerate(tasks):
            total_demand += task.gpu_request
            demand_fraction = total_demand / self.total_gpus

            node, gpu_indices = self._place_task(task)
            if node is not None:
                self._allocate(task, node, gpu_indices)
                self.running_tasks[task.id] = (node, gpu_indices, task)
            else:
                rejected += 1
                # Track rejected GPU-equivalents by size bucket
                gpu_req = task.gpu_request
                if gpu_req < 1:
                    pending_by_gpu_size['lt1_gpu'] += gpu_req
                elif gpu_req == 1:
                    pending_by_gpu_size['1_gpu'] += gpu_req
                elif gpu_req == 2:
                    pending_by_gpu_size['2_gpu'] += gpu_req
                else:  # 4 or 8
                    pending_by_gpu_size['8_gpu'] += gpu_req

            if (i + 1) % record_interval == 0 or demand_fraction >= target_demand_fraction:
                allocated = self._allocated_gpus()
                unallocated = self.total_gpus - allocated
                frag = FragmentationCalculator.compute_cluster_fragmentation(
                    self.nodes, self.workload
                )

                frag_ratio = frag / self.total_gpus if self.total_gpus > 0 else 0
                alloc_ratio = allocated / self.total_gpus if self.total_gpus > 0 else 0

                # Occupied nodes (for Fig 9b)
                occupied_nodes = self._compute_occupied_nodes()

                # Fragmentation breakdown (for Fig 9d)
                frag_non_gpu, frag_stranded, frag_deficient = self._compute_frag_breakdown()
                total_frag_breakdown = frag_non_gpu + frag_stranded + frag_deficient
                if total_frag_breakdown > 0:
                    frag_non_gpu_pct = frag_non_gpu / total_frag_breakdown * 100
                    frag_stranded_pct = frag_stranded / total_frag_breakdown * 100
                    frag_deficient_pct = frag_deficient / total_frag_breakdown * 100
                else:
                    frag_non_gpu_pct = 0.0
                    frag_stranded_pct = 0.0
                    frag_deficient_pct = 0.0

                curve.append({
                    'demand_fraction': demand_fraction,
                    'frag_ratio': frag_ratio,
                    'alloc_ratio': alloc_ratio,
                    'allocated_gpus': allocated,
                    'unallocated_gpus': unallocated,
                    'fragmentation': frag,
                    'tasks_submitted': i + 1,
                    'rejected': rejected,
                    'occupied_nodes': occupied_nodes,
                    'pending_by_gpu_size': dict(pending_by_gpu_size),
                    'frag_non_gpu_pct': frag_non_gpu_pct,
                    'frag_stranded_pct': frag_stranded_pct,
                    'frag_deficient_pct': frag_deficient_pct,
                })

            if demand_fraction >= target_demand_fraction:
                break

        self.rejected_count = rejected
        return curve


def _weighted_choice(weights, rng):
    r = rng.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1
