"""
Alibaba GPU Cluster Trace Loader (v2023)

Parses CSV trace files from Alibaba's cluster-trace-gpu-v2023 dataset.
Adapted from fgd_replication/trace_loader.py on the clubzip/fgd-alibaba-trace-loader branch.

Reference: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
"""

import csv
import os
from typing import List, Dict
from collections import Counter

from simulator import Task, Node, Cluster, TaskDistribution


class AlibabaTraceLoader:
    """Loader for Alibaba GPU cluster trace v2023."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, 'csv')
        self.tasks: List[Task] = []
        self.nodes: List[Node] = []

    def load_nodes(self, filename: str = 'openb_node_list_gpu_node.csv') -> List[Node]:
        """Load node specifications from CSV."""
        filepath = os.path.join(self.csv_dir, filename)
        self.nodes = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                node = Node(
                    node_id=i,
                    total_cpu=int(row['cpu_milli']) / 1000.0,
                    num_gpus=int(row['gpu']),
                    name=row['sn'],
                    gpu_model=row['model'] if row['model'] else '',
                    memory_mib=int(row['memory_mib']),
                )
                self.nodes.append(node)

        return self.nodes

    def load_tasks(
        self,
        filename: str = 'openb_pod_list_default.csv',
        gpu_only: bool = True,
    ) -> List[Task]:
        """Load task/pod information from CSV.

        Args:
            filename: CSV file name.
            gpu_only: If True, filter out tasks with zero GPU demand.

        Returns:
            List of Task objects sorted by creation_time (normalized to start at 0).
        """
        filepath = os.path.join(self.csv_dir, filename)
        raw_tasks: List[Task] = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                num_gpu = int(row['num_gpu'])
                gpu_milli = int(row['gpu_milli'])

                # Compute GPU demand
                if num_gpu == 0:
                    gpu_demand = 0.0
                elif num_gpu == 1:
                    gpu_demand = gpu_milli / 1000.0
                else:
                    gpu_demand = float(num_gpu)

                # Parse timestamps (may be empty)
                creation_time = int(row['creation_time']) if row['creation_time'] else 0
                scheduled_time = int(row['scheduled_time']) if row['scheduled_time'] else 0
                deletion_time = int(row['deletion_time']) if row['deletion_time'] else 0

                task = Task(
                    task_id=i,
                    cpu_demand=int(row['cpu_milli']) / 1000.0,
                    gpu_demand=gpu_demand,
                    name=row['name'],
                    creation_time=creation_time,
                    scheduled_time=scheduled_time,
                    deletion_time=deletion_time,
                    gpu_spec=row['gpu_spec'] if row['gpu_spec'] and row['gpu_spec'] != 'nan' else '',
                )
                raw_tasks.append(task)

        # Filter: GPU tasks only (if requested)
        if gpu_only:
            raw_tasks = [t for t in raw_tasks if t.gpu_demand > 0]

        # Filter: valid timestamps (deletion > creation, both > 0)
        raw_tasks = [
            t for t in raw_tasks
            if t.creation_time > 0 and t.deletion_time > t.creation_time
        ]

        # Sort by creation_time
        raw_tasks.sort(key=lambda t: t.creation_time)

        # Normalize timestamps to start at 0
        if raw_tasks:
            t0 = raw_tasks[0].creation_time
            for t in raw_tasks:
                t.creation_time -= t0
                t.scheduled_time = max(0, t.scheduled_time - t0)
                t.deletion_time -= t0

        # Re-assign sequential task IDs
        for i, task in enumerate(raw_tasks):
            task.task_id = i

        self.tasks = raw_tasks
        return self.tasks

    def create_cluster(self, task_distribution: TaskDistribution = None) -> Cluster:
        """Create a Cluster from loaded nodes."""
        if not self.nodes:
            self.load_nodes()

        cluster = Cluster()
        for node in self.nodes:
            cluster.add_node(Node(
                node_id=node.node_id,
                total_cpu=node.total_cpu,
                num_gpus=node.num_gpus,
                name=node.name,
                gpu_model=node.gpu_model,
                memory_mib=node.memory_mib,
            ))
        if task_distribution is not None:
            cluster.set_task_distribution(task_distribution)
        return cluster

    def compute_task_distribution(self) -> TaskDistribution:
        """Compute task type distribution from loaded tasks.

        Groups tasks by (cpu_bucket, gpu_rounded) and calculates popularity.
        """
        if not self.tasks:
            self.load_tasks()

        type_counts: Counter = Counter()
        for task in self.tasks:
            gpu_rounded = round(task.gpu_demand, 2)
            cpu_bucket = round(task.cpu_demand / 4) * 4
            type_counts[(cpu_bucket, gpu_rounded)] += 1

        total = sum(type_counts.values())
        dist = TaskDistribution()
        for (cpu, gpu), count in type_counts.items():
            dist.add_task_type(cpu, gpu, count / total)
        return dist

    def scale_cluster(self, nodes: List[Node], scale_pct: float) -> List[Node]:
        """Keep scale_pct% of nodes per type. At least 1 per type."""
        import math
        from collections import defaultdict

        type_groups: Dict[tuple, List[Node]] = defaultdict(list)
        for node in nodes:
            key = (node.total_cpu, node.memory_mib, node.num_gpus, node.gpu_model)
            type_groups[key].append(node)

        scaled = []
        for key, group in sorted(type_groups.items()):
            keep = max(1, math.ceil(len(group) * scale_pct / 100.0))
            scaled.extend(group[:keep])

        # Re-assign node IDs
        for i, node in enumerate(scaled):
            node.node_id = i

        return scaled

    def get_statistics(self) -> Dict:
        """Return summary statistics about the loaded trace."""
        if not self.tasks:
            self.load_tasks()
        if not self.nodes:
            self.load_nodes()

        gpu_tasks = [t for t in self.tasks if t.gpu_demand > 0]
        gpu_sharing = [t for t in self.tasks if 0 < t.gpu_demand < 1]
        multi_gpu = [t for t in self.tasks if t.gpu_demand > 1]
        one_gpu = [t for t in self.tasks if t.gpu_demand == 1.0]

        gpu_demands = Counter()
        for t in self.tasks:
            if t.gpu_demand == 0:
                gpu_demands['0'] += 1
            elif t.gpu_demand < 1:
                gpu_demands['(0,1)'] += 1
            elif t.gpu_demand == 1:
                gpu_demands['1'] += 1
            else:
                gpu_demands[str(int(t.gpu_demand))] += 1

        total_gpus = sum(n.num_gpus for n in self.nodes)
        gpu_models = Counter(n.gpu_model for n in self.nodes)

        # Duration stats (seconds)
        durations = [t.deletion_time - t.creation_time for t in self.tasks]
        trace_span = max(t.deletion_time for t in self.tasks) if self.tasks else 0

        return {
            'num_tasks': len(self.tasks),
            'num_nodes': len(self.nodes),
            'total_gpus': total_gpus,
            'gpu_tasks': len(gpu_tasks),
            'gpu_sharing_tasks': len(gpu_sharing),
            'one_gpu_tasks': len(one_gpu),
            'multi_gpu_tasks': len(multi_gpu),
            'gpu_demand_distribution': dict(gpu_demands),
            'gpu_models': dict(gpu_models),
            'trace_span_hours': trace_span / 3600.0,
            'median_duration_sec': sorted(durations)[len(durations) // 2] if durations else 0,
            'mean_duration_sec': sum(durations) / len(durations) if durations else 0,
        }


def print_trace_statistics(loader: AlibabaTraceLoader):
    """Print formatted trace statistics."""
    stats = loader.get_statistics()

    print("=" * 55)
    print("Alibaba GPU Trace v2023 Statistics")
    print("=" * 55)

    print(f"\nCluster:")
    print(f"  Nodes:      {stats['num_nodes']}")
    print(f"  Total GPUs: {stats['total_gpus']}")

    print(f"\nTasks ({stats['num_tasks']} total):")
    print(f"  GPU tasks:      {stats['gpu_tasks']}")
    print(f"  GPU-sharing:    {stats['gpu_sharing_tasks']}")
    print(f"  1-GPU:          {stats['one_gpu_tasks']}")
    print(f"  Multi-GPU:      {stats['multi_gpu_tasks']}")

    print(f"\nGPU Demand Distribution:")
    for demand, count in sorted(stats['gpu_demand_distribution'].items()):
        pct = 100 * count / stats['num_tasks']
        print(f"  {demand:>6}: {count:5d} ({pct:5.1f}%)")

    print(f"\nTimeline:")
    print(f"  Trace span:       {stats['trace_span_hours']:.1f} hours")
    print(f"  Median duration:  {stats['median_duration_sec']:.0f} sec")
    print(f"  Mean duration:    {stats['mean_duration_sec']:.0f} sec")

    print(f"\nGPU Models:")
    for model, count in sorted(stats['gpu_models'].items(), key=lambda x: -x[1]):
        print(f"  {model:>10}: {count} nodes")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    loader = AlibabaTraceLoader(data_dir)
    loader.load_nodes()
    loader.load_tasks()
    print_trace_statistics(loader)
