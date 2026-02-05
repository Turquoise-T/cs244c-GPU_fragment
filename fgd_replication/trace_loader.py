"""
Alibaba GPU Cluster Trace Loader (v2023)

Parses the CSV trace files from Alibaba's cluster-trace-gpu-v2023 dataset.
Reference: https://github.com/alibaba/clusterdata
"""

import csv
import os
from typing import List, Dict
from collections import Counter

from simulator import Task, Node, Cluster, TaskDistribution


class AlibabaTraceLoader:
    """
    Loader for Alibaba GPU cluster trace v2023.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the trace loader.

        Args:
            data_dir: Path to the directory containing csv/ folder
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, 'csv')
        self.tasks: List[Task] = []
        self.nodes: List[Node] = []

    def load_nodes(self, filename: str = 'openb_node_list_gpu_node.csv') -> List[Node]:
        """
        Load node information from CSV file.

        Args:
            filename: Name of the node list CSV file

        Returns:
            List of Node objects
        """
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
                    memory_mib=int(row['memory_mib'])
                )
                self.nodes.append(node)

        return self.nodes

    def load_tasks(self, filename: str = 'openb_pod_list_default.csv') -> List[Task]:
        """
        Load task information from CSV file.

        Args:
            filename: Name of the pod list CSV file

        Returns:
            List of Task objects sorted by creation_time
        """
        filepath = os.path.join(self.csv_dir, filename)
        self.tasks = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                num_gpu = int(row['num_gpu'])
                gpu_milli = int(row['gpu_milli'])

                # Compute GPU demand
                if num_gpu == 0:
                    gpu_demand = 0.0
                elif num_gpu == 1:
                    # GPU-sharing task: use gpu_milli
                    gpu_demand = gpu_milli / 1000.0
                else:
                    # Multi-GPU task
                    gpu_demand = float(num_gpu)

                task = Task(
                    task_id=i,
                    cpu_demand=int(row['cpu_milli']) / 1000.0,
                    gpu_demand=gpu_demand,
                    name=row['name'],
                    creation_time=int(row['creation_time']) if row['creation_time'] else 0,
                    scheduled_time=int(row['scheduled_time']) if row['scheduled_time'] else 0,
                    deletion_time=int(row['deletion_time']) if row['deletion_time'] else 0,
                    gpu_spec=row['gpu_spec'] if row['gpu_spec'] else ''
                )
                self.tasks.append(task)

        # Sort by creation time
        self.tasks.sort(key=lambda t: t.creation_time)

        # Re-assign task_id after sorting
        for i, task in enumerate(self.tasks):
            task.task_id = i

        return self.tasks

    def create_cluster(self) -> Cluster:
        """
        Create a Cluster object from loaded nodes.

        Returns:
            Cluster object with all nodes
        """
        if not self.nodes:
            self.load_nodes()

        cluster = Cluster()
        for node in self.nodes:
            cluster.add_node(node)

        return cluster

    def compute_task_distribution(self) -> TaskDistribution:
        """
        Compute task distribution from loaded tasks.
        Groups tasks by (cpu_demand, gpu_demand) and calculates popularity.

        Returns:
            TaskDistribution object
        """
        if not self.tasks:
            self.load_tasks()

        # Count task types
        type_counts: Counter = Counter()
        for task in self.tasks:
            # Round GPU demand to 2 decimal places for grouping
            gpu_rounded = round(task.gpu_demand, 2)
            # Group CPU demands into buckets (to avoid too many types)
            cpu_bucket = self._bucket_cpu(task.cpu_demand)
            type_counts[(cpu_bucket, gpu_rounded)] += 1

        # Convert to distribution
        total = sum(type_counts.values())
        dist = TaskDistribution()
        for (cpu, gpu), count in type_counts.items():
            dist.add_task_type(cpu, gpu, count / total)

        return dist

    def _bucket_cpu(self, cpu: float) -> float:
        """Bucket CPU demand to reduce number of task types"""
        # Round to nearest 4 CPUs
        return round(cpu / 4) * 4

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded trace.

        Returns:
            Dictionary with trace statistics
        """
        if not self.tasks:
            self.load_tasks()
        if not self.nodes:
            self.load_nodes()

        # Task statistics
        gpu_tasks = [t for t in self.tasks if t.gpu_demand > 0]
        gpu_sharing_tasks = [t for t in self.tasks if 0 < t.gpu_demand < 1]
        multi_gpu_tasks = [t for t in self.tasks if t.gpu_demand > 1]
        one_gpu_tasks = [t for t in self.tasks if t.gpu_demand == 1.0]
        no_gpu_tasks = [t for t in self.tasks if t.gpu_demand == 0]

        # GPU demand distribution
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

        # Node statistics
        total_gpus = sum(n.num_gpus for n in self.nodes)
        total_cpus = sum(n.total_cpu for n in self.nodes)
        gpu_models = Counter(n.gpu_model for n in self.nodes)

        return {
            'num_tasks': len(self.tasks),
            'num_nodes': len(self.nodes),
            'total_gpus': total_gpus,
            'total_cpus': total_cpus,
            'gpu_tasks': len(gpu_tasks),
            'gpu_sharing_tasks': len(gpu_sharing_tasks),
            'one_gpu_tasks': len(one_gpu_tasks),
            'multi_gpu_tasks': len(multi_gpu_tasks),
            'no_gpu_tasks': len(no_gpu_tasks),
            'gpu_demand_distribution': dict(gpu_demands),
            'gpu_models': dict(gpu_models),
        }


def print_trace_statistics(loader: AlibabaTraceLoader):
    """Print formatted trace statistics"""
    stats = loader.get_statistics()

    print("=" * 50)
    print("Alibaba GPU Trace v2023 Statistics")
    print("=" * 50)

    print(f"\nCluster:")
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Total GPUs: {stats['total_gpus']}")
    print(f"   Total CPUs: {stats['total_cpus']:.0f}")

    print(f"\nTasks:")
    print(f"   Total: {stats['num_tasks']}")
    print(f"   GPU tasks: {stats['gpu_tasks']} ({100*stats['gpu_tasks']/stats['num_tasks']:.1f}%)")
    print(f"   GPU-sharing (partial): {stats['gpu_sharing_tasks']} ({100*stats['gpu_sharing_tasks']/stats['num_tasks']:.1f}%)")
    print(f"   One GPU: {stats['one_gpu_tasks']} ({100*stats['one_gpu_tasks']/stats['num_tasks']:.1f}%)")
    print(f"   Multi-GPU: {stats['multi_gpu_tasks']} ({100*stats['multi_gpu_tasks']/stats['num_tasks']:.1f}%)")
    print(f"   No GPU: {stats['no_gpu_tasks']} ({100*stats['no_gpu_tasks']/stats['num_tasks']:.1f}%)")

    print(f"\nGPU Demand Distribution:")
    for demand, count in sorted(stats['gpu_demand_distribution'].items()):
        pct = 100 * count / stats['num_tasks']
        print(f"   {demand}: {count} ({pct:.1f}%)")

    print(f"\nGPU Models:")
    for model, count in sorted(stats['gpu_models'].items(), key=lambda x: -x[1]):
        print(f"   {model}: {count} nodes")


if __name__ == "__main__":
    # Test the trace loader
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'alibaba_traces', 'cluster-trace-gpu-v2023')
    loader = AlibabaTraceLoader(data_dir)

    # Load data
    loader.load_nodes()
    loader.load_tasks()

    # Print statistics
    print_trace_statistics(loader)

    # Test cluster creation
    print("\n" + "=" * 50)
    print("Testing Cluster Creation")
    print("=" * 50)

    cluster = loader.create_cluster()
    dist = loader.compute_task_distribution()
    cluster.set_task_distribution(dist)

    print(f"\nCluster created: {len(cluster.nodes)} nodes, {cluster.total_gpu_capacity} GPUs")
    print(f"Task distribution has {len(dist.distribution)} task types")

    # Show top task types
    print("\nTop 10 task types by popularity:")
    sorted_types = sorted(dist.distribution.items(), key=lambda x: -x[1])
    for (cpu, gpu), pop in sorted_types[:10]:
        print(f"   CPU={cpu:.0f}, GPU={gpu:.2f}: {100*pop:.2f}%")
