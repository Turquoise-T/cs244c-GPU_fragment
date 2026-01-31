"""Parser for Alibaba cluster trace data (github.com/alibaba/clusterdata).

Reads the GPU task and node CSVs from Alibaba's cluster-trace-gpu-v2023 dataset
and converts them into FGD Task/Node objects.
"""

import csv
import copy
import os
from dataclasses import dataclass
from typing import List, Optional

from fgd import Node, Task


@dataclass
class TraceTask:
    """A task from the Alibaba trace with timing information."""
    task: Task
    arrival_time: float  # seconds from trace start
    duration: float  # seconds


def parse_alibaba_trace(csv_path: str, max_tasks: Optional[int] = None) -> List[TraceTask]:
    """Parse Alibaba GPU pod list CSV into TraceTask objects.

    Reads openb_pod_list_default.csv with columns:
        name, cpu_milli, memory_mib, num_gpu, gpu_milli, gpu_spec,
        qos, pod_phase, creation_time, deletion_time, scheduled_time

    Filters: only GPU pods (num_gpu > 0), excludes Pending pods.

    Args:
        csv_path: Path to the pod list CSV file.
        max_tasks: If set, only parse this many tasks.

    Returns:
        List of TraceTask sorted by arrival time, times normalized to 0.
    """
    tasks = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_tasks is not None and len(tasks) >= max_tasks:
                break

            try:
                num_gpu = int(row['num_gpu'])
                if num_gpu == 0:
                    continue

                phase = row.get('pod_phase', '')
                if phase == 'Pending':
                    continue

                gpu_milli = int(row['gpu_milli'])
                cpu_milli = int(row['cpu_milli'])
                memory_mib = float(row['memory_mib'])
                creation_time = float(row['creation_time'])
                deletion_time = float(row['deletion_time'])
                name = row['name']
                gpu_spec_raw = row.get('gpu_spec', '')
            except (ValueError, KeyError):
                continue

            # GPU request: fractional if single GPU with gpu_milli < 1000
            if num_gpu == 1 and gpu_milli < 1000:
                gpu_request = gpu_milli / 1000.0
            else:
                gpu_request = float(num_gpu)

            if gpu_request <= 0:
                continue

            # GPU type: None if empty/nan, else keep pipe-delimited string
            gpu_type = None
            if gpu_spec_raw and gpu_spec_raw.lower() not in ('', 'nan'):
                gpu_type = gpu_spec_raw

            # CPU: millicores to cores
            cpu_request = cpu_milli / 1000.0

            duration = max(deletion_time - creation_time, 1.0)

            task = Task(
                id=name,
                cpu_request=cpu_request,
                gpu_request=gpu_request,
                memory_request=memory_mib,
                gpu_type=gpu_type,
            )
            tasks.append(TraceTask(task=task, arrival_time=creation_time, duration=duration))

    # Sort by arrival time and normalize to start at 0
    tasks.sort(key=lambda t: t.arrival_time)
    if tasks:
        t0 = tasks[0].arrival_time
        for t in tasks:
            t.arrival_time -= t0

    return tasks


def parse_node_list(csv_path: str) -> List[Node]:
    """Parse Alibaba GPU node list CSV into Node objects.

    Reads openb_node_list_gpu_node.csv with columns:
        sn, cpu_milli, memory_mib, gpu, model

    Args:
        csv_path: Path to the node list CSV file.

    Returns:
        List of Node objects.
    """
    nodes = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sn = row['sn']
                cpu_milli = int(row['cpu_milli'])
                memory_mib = float(row['memory_mib'])
                num_gpus = int(row['gpu'])
                model = row.get('model', 'generic')
            except (ValueError, KeyError):
                continue

            if num_gpus <= 0:
                continue

            node = Node(
                id=sn,
                total_cpu=cpu_milli / 1000.0,
                total_memory=memory_mib,
                gpus=[1.0] * num_gpus,
                gpu_type=model,
            )
            nodes.append(node)

    return nodes


def derive_workload_distribution(trace_tasks: List[TraceTask]):
    """Derive a workload distribution from trace tasks for FGD's Workload model.

    Groups tasks by GPU request size and computes popularity fractions.
    Includes memory_request and gpu_type in the distribution tuples.

    Returns:
        List of (gpu_request, cpu_request, memory_request, gpu_type, popularity) tuples.
    """
    from collections import Counter, defaultdict

    gpu_buckets = Counter()
    cpu_by_bucket = defaultdict(list)
    mem_by_bucket = defaultdict(list)
    type_by_bucket = defaultdict(list)

    for tt in trace_tasks:
        gr = tt.task.gpu_request
        # Bucket GPU requests
        if gr <= 0.125:
            bucket = 0.125
        elif gr <= 0.25:
            bucket = 0.25
        elif gr <= 0.5:
            bucket = 0.5
        elif gr <= 1.0:
            bucket = 1.0
        elif gr <= 2.0:
            bucket = 2.0
        elif gr <= 4.0:
            bucket = 4.0
        else:
            bucket = 8.0

        gpu_buckets[bucket] += 1
        cpu_by_bucket[bucket].append(tt.task.cpu_request)
        mem_by_bucket[bucket].append(tt.task.memory_request)
        type_by_bucket[bucket].append(tt.task.gpu_type)

    total = sum(gpu_buckets.values())
    distribution = []
    for bucket in sorted(gpu_buckets.keys()):
        avg_cpu = sum(cpu_by_bucket[bucket]) / len(cpu_by_bucket[bucket])
        avg_mem = sum(mem_by_bucket[bucket]) / len(mem_by_bucket[bucket])
        # Most common gpu_type for this bucket (None if all None)
        type_counts = Counter(type_by_bucket[bucket])
        most_common_type = type_counts.most_common(1)[0][0]
        popularity = gpu_buckets[bucket] / total
        distribution.append((bucket, avg_cpu, avg_mem, most_common_type, popularity))

    return distribution


def generate_synthetic_trace(
    distribution: List,
    num_tasks: int,
    mean_duration: float = 3600.0,
    mean_interarrival: float = 10.0,
    seed: int = 42,
) -> List[TraceTask]:
    """Generate synthetic trace tasks from a workload distribution.

    Args:
        distribution: List of (gpu_request, cpu_request, ..., popularity) tuples.
                      Popularity is always the last element.
        num_tasks: Number of tasks to generate.
        mean_duration: Mean task duration in seconds.
        mean_interarrival: Mean inter-arrival time in seconds.
        seed: Random seed.

    Returns:
        List of TraceTask sorted by arrival time.
    """
    import random as rng_module
    rng = rng_module.Random(seed)

    gpu_requests = [d[0] for d in distribution]
    cpu_requests = [d[1] for d in distribution]
    mem_requests = [d[2] if len(d) > 3 else 0.0 for d in distribution]
    gpu_types = [d[3] if len(d) > 4 else None for d in distribution]
    weights = [d[-1] for d in distribution]

    tasks = []
    current_time = 0.0

    for i in range(num_tasks):
        idx = _weighted_choice(weights, rng)
        gpu_req = gpu_requests[idx]
        cpu_req = cpu_requests[idx]
        mem_req = mem_requests[idx]
        gpu_type = gpu_types[idx]

        duration = rng.expovariate(1.0 / mean_duration)
        duration = max(duration, 10.0)

        task = Task(
            id=f'synth-{i}',
            cpu_request=cpu_req,
            gpu_request=gpu_req,
            memory_request=mem_req,
            gpu_type=gpu_type,
        )
        tasks.append(TraceTask(task=task, arrival_time=current_time, duration=duration))
        current_time += rng.expovariate(1.0 / mean_interarrival)

    return tasks


def _weighted_choice(weights, rng):
    """Weighted random choice using cumulative distribution."""
    r = rng.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1
