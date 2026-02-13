"""
GPU Cluster Simulator with Event-Driven Time-Based Simulation

Core data structures (Task, Node, Cluster, TaskDistribution) adapted from
fgd_replication/simulator.py on the clubzip/fgd-alibaba-trace-loader branch.

Key additions over the branch version:
- Node.deallocate_task(): reverse of allocate_task()
- Cluster.placement_map: tracks where each task was placed
- Cluster.unschedule_task(): deallocates a completed task
- EventDrivenSimulator: processes arrivals + departures using real timestamps

Reference: "Beware of Fragmentation: Scheduling GPU-Sharing Workloads
with Fragmentation Gradient Descent" (ATC'23)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A task/pod requesting cluster resources."""
    task_id: int
    cpu_demand: float
    gpu_demand: float
    name: str = ""
    creation_time: int = 0
    scheduled_time: int = 0
    deletion_time: int = 0
    gpu_spec: str = ""

    def is_partial_gpu(self) -> bool:
        return 0 < self.gpu_demand < 1

    def is_full_gpu(self) -> bool:
        return self.gpu_demand >= 1 and self.gpu_demand == int(self.gpu_demand)

    def is_no_gpu(self) -> bool:
        return self.gpu_demand == 0


@dataclass
class Node:
    """A cluster node with CPU and multiple GPUs.

    Each GPU tracks remaining capacity independently (0.0 to 1.0).
    GPU sharing allows multiple tasks on the same GPU.
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
            self.gpu_remaining = [1.0] * self.num_gpus

    @property
    def remaining_cpu(self) -> float:
        return self.total_cpu - self.allocated_cpu

    @property
    def fully_unallocated_gpus(self) -> int:
        """Count of GPUs with 100% capacity remaining (f in paper)."""
        return sum(1 for g in self.gpu_remaining if g == 1.0)

    @property
    def max_partial_gpu(self) -> float:
        """Maximum remaining capacity among partial GPUs (p in paper)."""
        partial = [g for g in self.gpu_remaining if 0 < g < 1.0]
        return max(partial) if partial else 0.0

    @property
    def scalar_gpu_capacity(self) -> float:
        """u = f + p (Section 3.1)."""
        return self.fully_unallocated_gpus + self.max_partial_gpu

    @property
    def total_unallocated_gpu(self) -> float:
        return sum(self.gpu_remaining)

    def can_fit_task(self, task: Task) -> bool:
        if self.remaining_cpu < task.cpu_demand:
            return False
        if task.is_no_gpu():
            return True
        if task.is_partial_gpu():
            return any(g >= task.gpu_demand for g in self.gpu_remaining)
        if task.is_full_gpu():
            return self.fully_unallocated_gpus >= int(task.gpu_demand)
        return False

    def allocate_task(self, task: Task) -> Optional[List[int]]:
        """Allocate resources to a task. Returns list of GPU indices used, or None on failure.

        For partial GPU tasks, assigns to GPU with least remaining capacity
        that can still fit (best-fit within node).
        """
        if not self.can_fit_task(task):
            return None

        self.allocated_cpu += task.cpu_demand

        if task.is_no_gpu():
            return []

        if task.is_partial_gpu():
            best_idx = -1
            best_remaining = float('inf')
            for i, g in enumerate(self.gpu_remaining):
                if g >= task.gpu_demand and g < best_remaining:
                    best_remaining = g
                    best_idx = i
            if best_idx >= 0:
                self.gpu_remaining[best_idx] -= task.gpu_demand
                return [best_idx]
            return None

        if task.is_full_gpu():
            gpus_needed = int(task.gpu_demand)
            indices = []
            for i in range(len(self.gpu_remaining)):
                if self.gpu_remaining[i] == 1.0 and len(indices) < gpus_needed:
                    self.gpu_remaining[i] = 0.0
                    indices.append(i)
            return indices if len(indices) == gpus_needed else None

        return None

    def deallocate_task(self, task: Task, gpu_indices: List[int]):
        """Release resources from a completed task."""
        self.allocated_cpu = max(0.0, self.allocated_cpu - task.cpu_demand)

        if task.is_no_gpu() or not gpu_indices:
            return

        if task.is_partial_gpu():
            for idx in gpu_indices:
                self.gpu_remaining[idx] = min(1.0, self.gpu_remaining[idx] + task.gpu_demand)
        elif task.is_full_gpu():
            for idx in gpu_indices:
                self.gpu_remaining[idx] = 1.0

    def get_fragmentation_for_task(self, task: Task) -> float:
        """F_n(m): GPU capacity on this node unusable by task m (Section 3.2)."""
        if task.is_no_gpu():
            return self.total_unallocated_gpu

        if self.remaining_cpu < task.cpu_demand:
            return self.total_unallocated_gpu

        if task.is_full_gpu():
            gpus_needed = int(task.gpu_demand)
            if self.fully_unallocated_gpus < gpus_needed:
                return self.total_unallocated_gpu
            return 0.0

        if task.is_partial_gpu():
            if not self.can_fit_task(task):
                return self.total_unallocated_gpu
            fragmented = 0.0
            for g in self.gpu_remaining:
                if g < task.gpu_demand:
                    fragmented += g
            return fragmented

        return 0.0


@dataclass
class TaskDistribution:
    """Workload distribution M: maps task types to popularity."""
    distribution: Dict[Tuple[float, float], float] = field(default_factory=dict)

    def add_task_type(self, cpu_demand: float, gpu_demand: float, popularity: float):
        self.distribution[(cpu_demand, gpu_demand)] = popularity

    def normalize(self):
        total = sum(self.distribution.values())
        if total > 0:
            for key in self.distribution:
                self.distribution[key] /= total

    def get_task_types(self) -> List[Tuple[Tuple[float, float], float]]:
        return list(self.distribution.items())


class Cluster:
    """GPU cluster with placement tracking and fragmentation computation."""

    def __init__(self):
        self.nodes: List[Node] = []
        self.task_distribution: Optional[TaskDistribution] = None
        self.placement_map: Dict[int, Tuple[int, List[int]]] = {}  # task_id -> (node_id, gpu_indices)
        self._active_tasks: Dict[int, Task] = {}  # task_id -> Task

    def add_node(self, node: Node):
        self.nodes.append(node)

    def set_task_distribution(self, distribution: TaskDistribution):
        self.task_distribution = distribution

    @property
    def total_gpu_capacity(self) -> int:
        return sum(n.num_gpus for n in self.nodes)

    @property
    def total_cpu_capacity(self) -> float:
        return sum(n.total_cpu for n in self.nodes)

    @property
    def total_unallocated_gpu(self) -> float:
        return sum(n.total_unallocated_gpu for n in self.nodes)

    @property
    def total_allocated_gpu(self) -> float:
        return self.total_gpu_capacity - self.total_unallocated_gpu

    @property
    def gpu_allocation_rate(self) -> float:
        if self.total_gpu_capacity == 0:
            return 0.0
        return (self.total_allocated_gpu / self.total_gpu_capacity) * 100

    @property
    def active_task_count(self) -> int:
        return len(self._active_tasks)

    def get_eligible_nodes(self, task: Task) -> List[Node]:
        return [n for n in self.nodes if n.can_fit_task(task)]

    def schedule_task(self, task: Task, node_id: int) -> bool:
        """Schedule a task on a specific node, tracking placement."""
        node = self.nodes[node_id]
        gpu_indices = node.allocate_task(task)
        if gpu_indices is not None:
            self.placement_map[task.task_id] = (node_id, gpu_indices)
            self._active_tasks[task.task_id] = task
            return True
        return False

    def unschedule_task(self, task_id: int) -> bool:
        """Deallocate a completed task, returning its resources."""
        if task_id not in self.placement_map:
            return False
        node_id, gpu_indices = self.placement_map[task_id]
        task = self._active_tasks[task_id]
        self.nodes[node_id].deallocate_task(task, gpu_indices)
        del self.placement_map[task_id]
        del self._active_tasks[task_id]
        return True

    def compute_node_fragmentation(self, node: Node) -> float:
        """F_n(M) = sum_m p_m * F_n(m)."""
        if self.task_distribution is None:
            return 0.0
        frag = 0.0
        for (cpu, gpu), popularity in self.task_distribution.get_task_types():
            dummy = Task(task_id=-1, cpu_demand=cpu, gpu_demand=gpu)
            frag += popularity * node.get_fragmentation_for_task(dummy)
        return frag

    def compute_cluster_fragmentation(self) -> float:
        """F_N(M): total fragmentation across all nodes."""
        return sum(self.compute_node_fragmentation(n) for n in self.nodes)

    def compute_fragmentation_rate(self) -> float:
        """f_N(M): fragmentation rate as percentage (Equation 6)."""
        total_unallocated = self.total_unallocated_gpu
        if total_unallocated == 0:
            return 100.0
        return (self.compute_cluster_fragmentation() / total_unallocated) * 100


# ---------------------------------------------------------------------------
# Event-Driven Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimulationMetrics:
    """Metrics snapshot at a point in time."""
    sim_time: float
    arrived_workload_pct: float
    active_workload_pct: float
    fragmentation_rate: float
    utilization_pct: float
    tasks_arrived: int
    tasks_scheduled: int
    tasks_completed: int
    tasks_failed: int
    tasks_active: int
    tasks_queued: int = 0
    avg_jct: float = 0.0


@dataclass
class JobRecord:
    """Per-job record for JCT tracking."""
    task_id: int
    gpu_demand: float
    creation_time: int
    trace_duration: int        # deletion_time - creation_time from trace
    placement_time: int = -1   # when actually placed (-1 = never)
    completion_time: int = -1  # when completed (-1 = never)
    frag_at_placement: float = 0.0

    @property
    def jct(self) -> float:
        """Job completion time (seconds). -1 if never completed."""
        if self.completion_time < 0:
            return -1.0
        return float(self.completion_time - self.creation_time)

    @property
    def wait_time(self) -> float:
        """Queue wait time (seconds). -1 if never placed."""
        if self.placement_time < 0:
            return -1.0
        return float(self.placement_time - self.creation_time)


class EventDrivenSimulator:
    """Time-based simulation using real trace timestamps.

    Processes job arrivals (creation_time) and departures (deletion_time)
    as discrete events. Calls the scheduler for placement on each arrival.
    Failed placements go into a FIFO queue; queued jobs are retried on
    each departure event. Tracks per-job JCT including queueing delay.
    """

    def __init__(self, cluster: Cluster, tasks: List[Task]):
        self.cluster = cluster
        self.tasks = tasks
        self.total_gpu_capacity = cluster.total_gpu_capacity
        self.job_records: List[JobRecord] = []

    def _try_place_queued(self, scheduler, queue, timestamp, active_gpu_demand):
        """Try to place queued jobs after a departure frees resources."""
        still_queued = []
        placed_demand = 0.0
        placed_count = 0

        for qtask, record in queue:
            if scheduler.schedule(qtask, self.cluster):
                record.placement_time = timestamp
                # Schedule departure for this queued job
                record.completion_time = timestamp + record.trace_duration
                placed_demand += qtask.gpu_demand
                placed_count += 1
            else:
                still_queued.append((qtask, record))

        return still_queued, placed_demand, placed_count

    def run(
        self,
        scheduler,
        sample_interval_pct: float = 5.0,
        show_progress: bool = True,
    ) -> List[SimulationMetrics]:
        """Run the event-driven simulation with job queuing.

        Jobs that cannot be placed on arrival enter a FIFO queue.
        Queued jobs are retried whenever a running job departs.
        JCT = completion_time - creation_time (includes queue wait).
        """
        # Build event queue from trace timestamps
        events: List[Tuple[int, int, str, int]] = []  # (time, priority, type, task_idx)
        task_by_id: Dict[int, Task] = {}
        record_by_id: Dict[int, JobRecord] = {}

        for idx, task in enumerate(self.tasks):
            task_by_id[task.task_id] = task
            record = JobRecord(
                task_id=task.task_id,
                gpu_demand=task.gpu_demand,
                creation_time=task.creation_time,
                trace_duration=task.deletion_time - task.creation_time,
            )
            record_by_id[task.task_id] = record
            self.job_records.append(record)

            events.append((task.creation_time, 1, 'arrive', task.task_id))

        # Departures are scheduled dynamically (placed jobs depart at
        # placement_time + trace_duration, not at trace deletion_time)
        # For immediately-placed jobs this equals deletion_time.
        # We use a separate list for dynamic departure events.

        # Sort initial events
        events.sort(key=lambda e: (e[0], e[1]))

        results: List[SimulationMetrics] = []
        queue: List[Tuple[Task, JobRecord]] = []  # FIFO queue of (task, record)
        dynamic_departures: List[Tuple[int, int]] = []  # (departure_time, task_id)

        cumulative_gpu_demand = 0.0
        active_gpu_demand = 0.0
        tasks_arrived = 0
        tasks_scheduled = 0
        tasks_completed = 0
        next_sample_pct = sample_interval_pct

        # Merge-process: arrival events from `events` + dynamic departures
        arr_idx = 0
        total_arrivals = len(events)

        def _take_snapshot(timestamp):
            frag_rate = self.cluster.compute_fragmentation_rate()
            util_pct = self.cluster.gpu_allocation_rate
            arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100
            active_pct = (active_gpu_demand / self.total_gpu_capacity) * 100
            completed_jcts = [r.jct for r in self.job_records
                              if r.completion_time >= 0 and r.completion_time <= timestamp]
            avg_jct = sum(completed_jcts) / len(completed_jcts) if completed_jcts else 0.0
            return SimulationMetrics(
                sim_time=timestamp,
                arrived_workload_pct=arrived_pct,
                active_workload_pct=active_pct,
                fragmentation_rate=frag_rate,
                utilization_pct=util_pct,
                tasks_arrived=tasks_arrived,
                tasks_scheduled=tasks_scheduled,
                tasks_completed=tasks_completed,
                tasks_failed=0,
                tasks_active=self.cluster.active_task_count,
                tasks_queued=len(queue),
                avg_jct=avg_jct,
            )

        while arr_idx < total_arrivals or dynamic_departures:
            # Determine next event: arrival or departure
            next_arr_time = events[arr_idx][0] if arr_idx < total_arrivals else float('inf')
            next_dep_time = dynamic_departures[0][0] if dynamic_departures else float('inf')

            if next_dep_time <= next_arr_time:
                # Process departure
                dep_time, dep_task_id = dynamic_departures.pop(0)

                if dep_task_id in self.cluster.placement_map:
                    task = task_by_id[dep_task_id]
                    self.cluster.unschedule_task(dep_task_id)
                    active_gpu_demand -= task.gpu_demand
                    tasks_completed += 1

                # Try to place queued jobs now that resources freed up
                if queue:
                    queue, placed_demand, placed_count = self._try_place_queued(
                        scheduler, queue, dep_time, active_gpu_demand)
                    active_gpu_demand += placed_demand
                    tasks_scheduled += placed_count
                    # Schedule departures for newly placed jobs
                    for rec in self.job_records:
                        if rec.placement_time == dep_time and rec.completion_time > 0:
                            if rec.task_id != dep_task_id:
                                # Insert sorted
                                ct = rec.completion_time
                                ins_pos = 0
                                for j, (dt, _) in enumerate(dynamic_departures):
                                    if dt > ct:
                                        break
                                    ins_pos = j + 1
                                dynamic_departures.insert(ins_pos, (ct, rec.task_id))
            else:
                # Process arrival
                _, _, _, task_id = events[arr_idx]
                arr_idx += 1
                task = task_by_id[task_id]
                record = record_by_id[task_id]

                tasks_arrived += 1
                cumulative_gpu_demand += task.gpu_demand

                if scheduler.schedule(task, self.cluster):
                    tasks_scheduled += 1
                    active_gpu_demand += task.gpu_demand
                    record.placement_time = task.creation_time
                    record.completion_time = task.creation_time + record.trace_duration
                    record.frag_at_placement = self.cluster.compute_fragmentation_rate()

                    # Schedule departure
                    ct = record.completion_time
                    ins_pos = 0
                    for j, (dt, _) in enumerate(dynamic_departures):
                        if dt > ct:
                            break
                        ins_pos = j + 1
                    dynamic_departures.insert(ins_pos, (ct, task_id))
                else:
                    queue.append((task, record))

                # Sample at workload percentage intervals
                arrived_pct = (cumulative_gpu_demand / self.total_gpu_capacity) * 100
                if arrived_pct >= next_sample_pct:
                    results.append(_take_snapshot(task.creation_time))
                    next_sample_pct = arrived_pct + sample_interval_pct

            # Progress
            processed = arr_idx + tasks_completed
            if show_progress and processed % 2000 == 0 and processed > 0:
                total_est = total_arrivals + tasks_arrived
                pct = min(99, processed * 100 // max(1, total_est))
                print(f"    [{pct:3d}%] arrived={tasks_arrived} scheduled={tasks_scheduled} "
                      f"completed={tasks_completed} queued={len(queue)}")

        # Final snapshot
        last_time = max((r.completion_time for r in self.job_records if r.completion_time > 0), default=0)
        results.append(_take_snapshot(last_time))

        return results
