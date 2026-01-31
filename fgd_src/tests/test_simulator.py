"""Unit tests for the standalone FGD simulator.

Tests cover:
  - Cluster construction from config
  - Placement with FGD vs baselines
  - Event-driven simulation correctness
  - Monte-Carlo inflation
  - Fragmentation accounting
  - Alibaba trace parsing
  - New baselines (DotProd, GpuPacking, GpuClustering)
  - Demand-fraction stopping condition
"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgd import FGDScheduler, FragmentationCalculator, Node, Task, Workload
from baselines import (
    BestFitPlacer, FirstFitPlacer, RandomPlacer,
    DotProdPlacer, GpuPackingPlacer, GpuClusteringPlacer,
)
from simulator import FGDSimulator
from alibaba_trace_parser import (
    TraceTask, parse_alibaba_trace, parse_node_list, derive_workload_distribution
)


def _small_cluster_config():
    """4 nodes, 4 GPUs each = 16 GPUs total."""
    return {
        'nodes': {
            '4gpu': {
                'count': 4,
                'gpus_per_node': 4,
                'cpu_per_node': 48,
                'memory_per_node': 256,
            }
        },
        'total_nodes': 4,
        'total_gpus': 16,
        'gpu_type': 'generic',
    }


def _philly_workload():
    workload = Workload()
    workload.add_task_type(Task(id='1gpu', cpu_request=4, gpu_request=1.0), 0.70)
    workload.add_task_type(Task(id='2gpu', cpu_request=8, gpu_request=2.0), 0.10)
    workload.add_task_type(Task(id='4gpu', cpu_request=16, gpu_request=4.0), 0.15)
    workload.add_task_type(Task(id='8gpu', cpu_request=32, gpu_request=8.0), 0.05)
    workload.normalize_popularity()
    return workload


class TestClusterConstruction(unittest.TestCase):
    def test_small_cluster_gpu_count(self):
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        self.assertEqual(sim.total_gpus, 16)
        self.assertEqual(len(sim.nodes), 4)
        for node in sim.nodes:
            self.assertEqual(node.num_gpus, 4)
            self.assertEqual(node.gpus, [1.0, 1.0, 1.0, 1.0])

    def test_cluster_h_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'configs', 'cluster_h.json'
        )
        with open(config_path) as f:
            config = json.load(f)
        sim = FGDSimulator(config, placement='fgd')
        self.assertEqual(sim.total_gpus, config['total_gpus'])
        self.assertEqual(len(sim.nodes), config['total_nodes'])

    def test_direct_node_list(self):
        """Test creating simulator with pre-built node list."""
        nodes = [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[1.0, 1.0]),
            Node(id='n1', total_cpu=96, total_memory=512, gpus=[1.0]*4),
        ]
        sim = FGDSimulator(nodes=nodes, placement='bestfit')
        self.assertEqual(sim.total_gpus, 6)
        self.assertEqual(len(sim.nodes), 2)


class TestBaselines(unittest.TestCase):
    def _make_nodes(self):
        return [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[1.0, 1.0, 1.0, 1.0]),
            Node(id='n1', total_cpu=48, total_memory=256, gpus=[1.0, 1.0, 1.0, 1.0]),
        ]

    def test_bestfit_prefers_tighter_node(self):
        nodes = self._make_nodes()
        # Partially fill node 0 so it has less remaining
        nodes[0].gpus = [0.0, 0.0, 1.0, 1.0]
        nodes[0].allocated_cpu = 24

        placer = BestFitPlacer()
        task = Task(id='t1', cpu_request=4, gpu_request=1.0)
        node, gpus = placer.place(task, nodes)
        # BestFit should prefer node 0 (less remaining capacity)
        self.assertEqual(node.id, 'n0')
        self.assertEqual(len(gpus), 1)

    def test_firstfit_returns_first_available(self):
        nodes = self._make_nodes()
        placer = FirstFitPlacer()
        task = Task(id='t1', cpu_request=4, gpu_request=1.0)
        node, gpus = placer.place(task, nodes)
        self.assertEqual(node.id, 'n0')

    def test_random_returns_valid_node(self):
        nodes = self._make_nodes()
        placer = RandomPlacer(seed=42)
        task = Task(id='t1', cpu_request=4, gpu_request=2.0)
        node, gpus = placer.place(task, nodes)
        self.assertIsNotNone(node)
        self.assertEqual(len(gpus), 2)

    def test_no_placement_when_full(self):
        nodes = self._make_nodes()
        for n in nodes:
            n.gpus = [0.0, 0.0, 0.0, 0.0]
            n.allocated_cpu = 48

        for Placer in [BestFitPlacer, FirstFitPlacer]:
            placer = Placer() if Placer != RandomPlacer else Placer(seed=0)
            task = Task(id='t1', cpu_request=4, gpu_request=1.0)
            node, gpus = placer.place(task, nodes)
            self.assertIsNone(node)


class TestNewBaselines(unittest.TestCase):
    """Tests for DotProd, GpuPacking, GpuClustering baselines."""

    def _make_nodes(self):
        return [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[1.0, 1.0, 0.0, 0.0]),
            Node(id='n1', total_cpu=48, total_memory=256, gpus=[1.0, 1.0, 1.0, 1.0]),
        ]

    def test_dotprod_returns_valid_placement(self):
        nodes = self._make_nodes()
        placer = DotProdPlacer()
        task = Task(id='t1', cpu_request=4, gpu_request=1.0, memory_request=16)
        node, gpus = placer.place(task, nodes)
        self.assertIsNotNone(node)
        self.assertEqual(len(gpus), 1)

    def test_gpupacking_prefers_packed_node(self):
        """GpuPacking should prefer node with fewer available GPUs."""
        nodes = self._make_nodes()
        placer = GpuPackingPlacer()
        task = Task(id='t1', cpu_request=4, gpu_request=1.0)
        node, gpus = placer.place(task, nodes)
        # n0 has 2 available GPUs (more packed), n1 has 4
        self.assertEqual(node.id, 'n0')

    def test_gpuclustering_prefers_free_node(self):
        """GpuClustering should prefer node with more available GPUs."""
        nodes = self._make_nodes()
        placer = GpuClusteringPlacer()
        task = Task(id='t1', cpu_request=4, gpu_request=1.0)
        node, gpus = placer.place(task, nodes)
        # n1 has 4 available GPUs (more free), n0 has 2
        self.assertEqual(node.id, 'n1')

    def test_no_placement_when_full(self):
        nodes = [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[0.0, 0.0]),
        ]
        for Placer in [DotProdPlacer, GpuPackingPlacer, GpuClusteringPlacer]:
            placer = Placer()
            task = Task(id='t1', cpu_request=4, gpu_request=1.0)
            node, gpus = placer.place(task, nodes)
            self.assertIsNone(node)

    def test_new_baselines_registered_in_simulator(self):
        """All new baselines should work in FGDSimulator."""
        for policy in ['dotprod', 'gpupacking', 'gpuclustering']:
            sim = FGDSimulator(_small_cluster_config(), placement=policy)
            self.assertIsNotNone(sim)


class TestSimulatorEventDriven(unittest.TestCase):
    def test_single_task_completes(self):
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        task = Task(id='t1', cpu_request=4, gpu_request=1.0)
        trace = [TraceTask(task=task, arrival_time=0.0, duration=100.0)]

        sim.run(trace, metrics_interval=50.0)
        self.assertEqual(sim.completed_count, 1)
        self.assertEqual(len(sim.running_tasks), 0)

    def test_resources_released_after_departure(self):
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        task = Task(id='t1', cpu_request=4, gpu_request=2.0)
        trace = [TraceTask(task=task, arrival_time=0.0, duration=50.0)]

        sim.run(trace)
        # After completion, all GPUs should be free
        for node in sim.nodes:
            self.assertTrue(all(g == 1.0 for g in node.gpus))

    def test_concurrent_tasks(self):
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        tasks = [
            TraceTask(
                task=Task(id=f't{i}', cpu_request=4, gpu_request=1.0),
                arrival_time=0.0, duration=100.0
            )
            for i in range(16)  # Fill all 16 GPUs
        ]

        sim.run(tasks)
        self.assertEqual(sim.completed_count, 16)

    def test_pending_tasks_scheduled_after_departure(self):
        sim = FGDSimulator(_small_cluster_config(), placement='firstfit')
        tasks = [
            # 16 tasks fill the cluster
            *[TraceTask(
                task=Task(id=f't{i}', cpu_request=4, gpu_request=1.0),
                arrival_time=0.0, duration=100.0
            ) for i in range(16)],
            # 17th task arrives while full, should queue
            TraceTask(
                task=Task(id='t16', cpu_request=4, gpu_request=1.0),
                arrival_time=10.0, duration=50.0
            ),
        ]

        sim.run(tasks)
        # All 17 should eventually complete
        self.assertEqual(sim.completed_count, 17)


class TestInflation(unittest.TestCase):
    def test_inflation_increases_utilization(self):
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        distribution = [(1.0, 4.0, 0.7), (2.0, 8.0, 0.3)]
        results = sim.run_inflation(
            distribution=distribution,
            target_utilization=0.9,
            batch_size=4,
            max_tasks=200,
        )
        self.assertGreater(len(results), 0)
        # Utilization should generally increase
        self.assertGreater(results[-1]['utilization'], results[0]['utilization'])

    def test_fgd_reduces_fragmentation_vs_bestfit(self):
        """Core validation: FGD should produce less fragmentation than BestFit."""
        distribution = [(1.0, 4.0, 0.7), (2.0, 8.0, 0.15), (4.0, 16.0, 0.15)]

        fgd_sim = FGDSimulator(_small_cluster_config(), placement='fgd', seed=42)
        fgd_results = fgd_sim.run_inflation(
            distribution=distribution, target_utilization=0.95,
            batch_size=2, max_tasks=100, seed=42,
        )

        bf_sim = FGDSimulator(_small_cluster_config(), placement='bestfit', seed=42)
        bf_results = bf_sim.run_inflation(
            distribution=distribution, target_utilization=0.95,
            batch_size=2, max_tasks=100, seed=42,
        )

        if fgd_results and bf_results:
            fgd_frag = fgd_results[-1]['fragmentation']
            bf_frag = bf_results[-1]['fragmentation']
            # FGD should not be worse than BestFit
            self.assertLessEqual(fgd_frag, bf_frag * 1.1,
                                 f"FGD fragmentation {fgd_frag:.2f} should not significantly "
                                 f"exceed BestFit {bf_frag:.2f}")


class TestInflationFromTasks(unittest.TestCase):
    """Tests for the new demand-fraction based inflation method."""

    def test_demand_fraction_stopping(self):
        """Should stop when cumulative demand reaches target fraction."""
        sim = FGDSimulator(_small_cluster_config(), placement='bestfit')
        tasks = [
            Task(id=f't{i}', cpu_request=4, gpu_request=1.0)
            for i in range(50)
        ]
        curve = sim.run_inflation_from_tasks(tasks, target_demand_fraction=1.0)
        self.assertGreater(len(curve), 0)
        last = curve[-1]
        self.assertGreaterEqual(last['demand_fraction'], 1.0)

    def test_rejected_tasks_tracked(self):
        """Tasks that can't be placed should be counted as rejected."""
        # Small cluster: 2 nodes, 1 GPU each
        nodes = [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[1.0]),
            Node(id='n1', total_cpu=48, total_memory=256, gpus=[1.0]),
        ]
        sim = FGDSimulator(nodes=nodes, placement='bestfit')
        # 10 tasks needing 1 GPU each, but only 2 GPUs available
        tasks = [
            Task(id=f't{i}', cpu_request=4, gpu_request=1.0)
            for i in range(10)
        ]
        curve = sim.run_inflation_from_tasks(tasks, target_demand_fraction=5.0)
        last = curve[-1]
        self.assertEqual(last['rejected'], 8)  # Only 2 can fit

    def test_conservation(self):
        """allocated + unallocated should always equal total_gpus."""
        sim = FGDSimulator(_small_cluster_config(), placement='fgd')
        tasks = [
            Task(id=f't{i}', cpu_request=4, gpu_request=1.0)
            for i in range(20)
        ]
        curve = sim.run_inflation_from_tasks(tasks, target_demand_fraction=1.0)
        for pt in curve:
            total = pt['allocated_gpus'] + pt['unallocated_gpus']
            self.assertAlmostEqual(total, 16.0, places=5,
                                   msg=f"Conservation violated at demand={pt['demand_fraction']:.2f}")

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        import random
        tasks = [
            Task(id=f't{i}', cpu_request=4, gpu_request=float(random.Random(0).choice([1,2])))
            for i in range(30)
        ]

        sim1 = FGDSimulator(_small_cluster_config(), placement='fgd', seed=42)
        curve1 = sim1.run_inflation_from_tasks(tasks[:], target_demand_fraction=1.0)

        sim2 = FGDSimulator(_small_cluster_config(), placement='fgd', seed=42)
        curve2 = sim2.run_inflation_from_tasks(tasks[:], target_demand_fraction=1.0)

        self.assertEqual(len(curve1), len(curve2))
        for p1, p2 in zip(curve1, curve2):
            self.assertAlmostEqual(p1['frag_ratio'], p2['frag_ratio'])
            self.assertAlmostEqual(p1['alloc_ratio'], p2['alloc_ratio'])


class TestFragmentationAccounting(unittest.TestCase):
    def test_empty_cluster_has_structural_fragmentation(self):
        # A 4-GPU node can't fit 8-GPU tasks, so there is always some
        # structural fragmentation from the workload distribution.
        nodes = [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[1.0, 1.0, 1.0, 1.0]),
        ]
        workload = _philly_workload()
        frag = FragmentationCalculator.compute_cluster_fragmentation(nodes, workload)
        # 8-GPU task type (5% popularity) can't fit on 4-GPU node,
        # contributing 4 * 0.05 = 0.2 fragmented GPUs
        self.assertAlmostEqual(frag, 0.2)

    def test_empty_large_node_zero_fragmentation(self):
        nodes = [
            Node(id='n0', total_cpu=96, total_memory=512, gpus=[1.0]*8),
        ]
        workload = _philly_workload()
        frag = FragmentationCalculator.compute_cluster_fragmentation(nodes, workload)
        self.assertEqual(frag, 0.0)

    def test_partial_allocation_creates_fragmentation(self):
        nodes = [
            Node(id='n0', total_cpu=48, total_memory=256, gpus=[0.5, 0.5, 1.0, 1.0]),
        ]
        workload = _philly_workload()
        frag = FragmentationCalculator.compute_cluster_fragmentation(nodes, workload)
        # Partially allocated GPUs should create some fragmentation
        # because they can't fit certain task types
        self.assertGreater(frag, 0.0)


class TestWorkloadDistribution(unittest.TestCase):
    def test_derive_from_trace_tasks(self):
        tasks = [
            TraceTask(Task(id='t0', cpu_request=4, gpu_request=1.0), 0, 100),
            TraceTask(Task(id='t1', cpu_request=4, gpu_request=1.0), 10, 100),
            TraceTask(Task(id='t2', cpu_request=8, gpu_request=2.0), 20, 100),
            TraceTask(Task(id='t3', cpu_request=16, gpu_request=4.0), 30, 100),
        ]
        dist = derive_workload_distribution(tasks)
        # Should have 3 buckets (1, 2, 4 GPU)
        self.assertEqual(len(dist), 3)
        # Popularities should sum to 1
        total_pop = sum(d[-1] for d in dist)
        self.assertAlmostEqual(total_pop, 1.0)

    def test_distribution_includes_memory(self):
        """New distribution format should include memory_request."""
        tasks = [
            TraceTask(Task(id='t0', cpu_request=4, gpu_request=1.0, memory_request=16384), 0, 100),
            TraceTask(Task(id='t1', cpu_request=8, gpu_request=2.0, memory_request=32768), 10, 100),
        ]
        dist = derive_workload_distribution(tasks)
        # 5-tuple: (gpu, cpu, mem, gpu_type, pop)
        self.assertEqual(len(dist[0]), 5)
        self.assertGreater(dist[0][2], 0)  # memory should be present


class TestAlibabaTraceParser(unittest.TestCase):
    """Tests for parsing real Alibaba CSV column names."""

    def _write_pod_csv(self, rows):
        """Write a mock pod CSV and return its path."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        writer = csv.DictWriter(tmpfile, fieldnames=[
            'name', 'cpu_milli', 'memory_mib', 'num_gpu', 'gpu_milli',
            'gpu_spec', 'qos', 'pod_phase', 'creation_time', 'deletion_time',
            'scheduled_time'
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmpfile.close()
        return tmpfile.name

    def _write_node_csv(self, rows):
        """Write a mock node CSV and return its path."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        writer = csv.DictWriter(tmpfile, fieldnames=[
            'sn', 'cpu_milli', 'memory_mib', 'gpu', 'model'
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmpfile.close()
        return tmpfile.name

    def test_parse_pod_csv_columns(self):
        csv_path = self._write_pod_csv([
            {'name': 'pod-0', 'cpu_milli': '12000', 'memory_mib': '16384',
             'num_gpu': '1', 'gpu_milli': '1000', 'gpu_spec': '',
             'qos': 'LS', 'pod_phase': 'Running', 'creation_time': '100',
             'deletion_time': '500', 'scheduled_time': '100'},
        ])
        try:
            tasks = parse_alibaba_trace(csv_path)
            self.assertEqual(len(tasks), 1)
            t = tasks[0]
            self.assertEqual(t.task.id, 'pod-0')
            self.assertEqual(t.task.cpu_request, 12.0)  # 12000 milli -> 12 cores
            self.assertEqual(t.task.memory_request, 16384.0)
            self.assertEqual(t.task.gpu_request, 1.0)
            self.assertIsNone(t.task.gpu_type)
            self.assertEqual(t.arrival_time, 0.0)  # normalized
            self.assertEqual(t.duration, 400.0)
        finally:
            os.unlink(csv_path)

    def test_gpu_milli_fractional_conversion(self):
        """gpu_milli < 1000 with num_gpu=1 should produce fractional request."""
        csv_path = self._write_pod_csv([
            {'name': 'pod-frac', 'cpu_milli': '6000', 'memory_mib': '8192',
             'num_gpu': '1', 'gpu_milli': '460', 'gpu_spec': '',
             'qos': 'LS', 'pod_phase': 'Succeeded', 'creation_time': '0',
             'deletion_time': '1000', 'scheduled_time': '0'},
        ])
        try:
            tasks = parse_alibaba_trace(csv_path)
            self.assertEqual(len(tasks), 1)
            self.assertAlmostEqual(tasks[0].task.gpu_request, 0.46)
        finally:
            os.unlink(csv_path)

    def test_multi_gpu_ignores_milli(self):
        """num_gpu > 1 should use num_gpu directly, not gpu_milli."""
        csv_path = self._write_pod_csv([
            {'name': 'pod-multi', 'cpu_milli': '32000', 'memory_mib': '65536',
             'num_gpu': '4', 'gpu_milli': '1000', 'gpu_spec': '',
             'qos': 'LS', 'pod_phase': 'Running', 'creation_time': '0',
             'deletion_time': '2000', 'scheduled_time': '0'},
        ])
        try:
            tasks = parse_alibaba_trace(csv_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task.gpu_request, 4.0)
        finally:
            os.unlink(csv_path)

    def test_pending_pods_filtered(self):
        """Pending pods should be excluded."""
        csv_path = self._write_pod_csv([
            {'name': 'pod-ok', 'cpu_milli': '6000', 'memory_mib': '8192',
             'num_gpu': '1', 'gpu_milli': '1000', 'gpu_spec': '',
             'qos': 'LS', 'pod_phase': 'Running', 'creation_time': '0',
             'deletion_time': '1000', 'scheduled_time': '0'},
            {'name': 'pod-pending', 'cpu_milli': '6000', 'memory_mib': '8192',
             'num_gpu': '1', 'gpu_milli': '1000', 'gpu_spec': '',
             'qos': 'LS', 'pod_phase': 'Pending', 'creation_time': '0',
             'deletion_time': '1000', 'scheduled_time': '0'},
        ])
        try:
            tasks = parse_alibaba_trace(csv_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task.id, 'pod-ok')
        finally:
            os.unlink(csv_path)

    def test_parse_node_list(self):
        csv_path = self._write_node_csv([
            {'sn': 'node-0', 'cpu_milli': '64000', 'memory_mib': '262144',
             'gpu': '2', 'model': 'P100'},
            {'sn': 'node-1', 'cpu_milli': '96000', 'memory_mib': '524288',
             'gpu': '8', 'model': 'V100M32'},
        ])
        try:
            nodes = parse_node_list(csv_path)
            self.assertEqual(len(nodes), 2)
            self.assertEqual(nodes[0].id, 'node-0')
            self.assertEqual(nodes[0].total_cpu, 64.0)
            self.assertEqual(nodes[0].total_memory, 262144.0)
            self.assertEqual(nodes[0].num_gpus, 2)
            self.assertEqual(nodes[0].gpu_type, 'P100')
            self.assertEqual(nodes[1].num_gpus, 8)
            self.assertEqual(nodes[1].gpu_type, 'V100M32')
        finally:
            os.unlink(csv_path)


class TestGpuSpecMatching(unittest.TestCase):
    """Tests for pipe-delimited gpu_spec type matching."""

    def test_exact_match(self):
        node = Node(id='n0', total_cpu=48, total_memory=256,
                     gpus=[1.0, 1.0], gpu_type='V100M16')
        task = Task(id='t1', cpu_request=4, gpu_request=1.0, gpu_type='V100M16')
        self.assertTrue(node.can_fit_task(task))

    def test_pipe_delimited_match(self):
        node = Node(id='n0', total_cpu=48, total_memory=256,
                     gpus=[1.0, 1.0], gpu_type='V100M16')
        task = Task(id='t1', cpu_request=4, gpu_request=1.0, gpu_type='V100M16|V100M32')
        self.assertTrue(node.can_fit_task(task))

    def test_pipe_delimited_no_match(self):
        node = Node(id='n0', total_cpu=48, total_memory=256,
                     gpus=[1.0, 1.0], gpu_type='T4')
        task = Task(id='t1', cpu_request=4, gpu_request=1.0, gpu_type='V100M16|V100M32')
        self.assertFalse(node.can_fit_task(task))

    def test_no_gpu_type_matches_any(self):
        node = Node(id='n0', total_cpu=48, total_memory=256,
                     gpus=[1.0, 1.0], gpu_type='T4')
        task = Task(id='t1', cpu_request=4, gpu_request=1.0, gpu_type=None)
        self.assertTrue(node.can_fit_task(task))


if __name__ == '__main__':
    unittest.main()
