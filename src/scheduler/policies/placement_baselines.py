"""Baseline placement policies for fragmentation comparison.

These policies implement simple placement strategies that can be compared
against FGD to demonstrate the benefit of fragmentation-aware scheduling.
"""

import os
import sys
import random
from collections import Counter

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from policy import Policy
from fgd import FGDNode, FGDJob, FGDWorkload, FragmentationCalculator


class PlacementPolicy(Policy):
    """Base class for placement-based policies with fragmentation tracking.

    All placement policies maintain a virtual per-node GPU model and convert
    placements to Gavel-format allocation fractions.
    """

    def __init__(self, node_config=None, seed=None):
        Policy.__init__(self, solver=None)
        self._node_config = node_config or {}
        self._default_gpus_per_node = 1
        self._rng = random.Random(seed)
        # Track fragmentation after each allocation
        self._last_fragmentation_rate = 0.0
        self._last_nodes = None
        self._last_workload = None

    def _build_nodes(self, cluster_spec):
        """Expand cluster_spec into a list of FGDNode objects."""
        nodes = []
        for wt in sorted(cluster_spec.keys()):
            total_gpus = cluster_spec[wt]
            gpus_per_node = self._node_config.get(wt, self._default_gpus_per_node)
            num_full_nodes = total_gpus // gpus_per_node
            remainder = total_gpus % gpus_per_node
            for i in range(num_full_nodes):
                nodes.append(FGDNode(f'{wt}_n{i}', gpus_per_node, wt))
            if remainder > 0:
                nodes.append(FGDNode(f'{wt}_n{num_full_nodes}', remainder, wt))
        return nodes

    def _build_workload(self, scale_factors, job_ids):
        """Derive workload from current active jobs."""
        wl = FGDWorkload()
        sf_counts = Counter()
        for jid in job_ids:
            sf_counts[scale_factors[jid]] += 1
        for sf, count in sf_counts.items():
            fgd_job = FGDJob(job_id=f'type_sf{sf}',
                             gpu_request=float(sf),
                             scale_factor=sf)
            wl.add_type(sf, fgd_job, float(count))
        wl.normalize()
        return wl

    def get_fragmentation_rate(self, nodes=None, workload=None):
        """Return fragmentation rate as percentage (0-100).

        If nodes/workload not provided, uses the state from last get_allocation() call.
        """
        if nodes is None:
            nodes = self._last_nodes
        if workload is None:
            workload = self._last_workload
        if nodes is None or workload is None:
            return 0.0

        total_frag = sum(
            FragmentationCalculator.node_fragmentation_for_workload(n, workload)
            for n in nodes
        )
        total_capacity = sum(sum(n.gpu_capacities) for n in nodes)
        return (total_frag / total_capacity) * 100 if total_capacity > 0 else 0.0

    def _run_placement(self, nodes, workload, job_ids, scale_factors):
        """Subclasses implement their placement strategy here."""
        raise NotImplementedError

    def _convert_to_allocation(self, placements, job_ids, scale_factors,
                                worker_types, cluster_spec):
        """Convert node placements to Gavel allocation fractions."""
        allocation = {}
        capacity_used = {wt: 0.0 for wt in worker_types}

        for jid in job_ids:
            allocation[jid] = {}
            if jid in placements:
                placed_type = placements[jid].gpu_type
                for wt in worker_types:
                    if wt == placed_type:
                        allocation[jid][wt] = 1.0
                        capacity_used[wt] += scale_factors[jid]
                    else:
                        allocation[jid][wt] = 0.0
            else:
                for wt in worker_types:
                    allocation[jid][wt] = 0.0

        # Scale down if oversubscribed
        for wt in worker_types:
            if capacity_used[wt] > cluster_spec[wt]:
                ratio = cluster_spec[wt] / capacity_used[wt]
                for jid in job_ids:
                    allocation[jid][wt] *= ratio

        return allocation

    def get_allocation(self, unflattened_throughputs, scale_factors, cluster_spec):
        """Compute allocation using placement strategy."""
        if not unflattened_throughputs:
            return None

        job_ids = sorted(unflattened_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())

        # Build virtual nodes
        nodes = self._build_nodes(cluster_spec)

        # Build workload
        workload = self._build_workload(scale_factors, job_ids)

        # Run placement
        placements = self._run_placement(nodes, workload, job_ids, scale_factors)

        # Store for fragmentation calculation
        self._last_nodes = nodes
        self._last_workload = workload
        self._last_fragmentation_rate = self.get_fragmentation_rate(nodes, workload)

        # Convert to Gavel allocation
        return self._convert_to_allocation(
            placements, job_ids, scale_factors, worker_types, cluster_spec
        )


class RandomPolicy(PlacementPolicy):
    """Random placement policy - assigns jobs to random nodes with capacity."""

    def __init__(self, node_config=None, seed=None):
        super().__init__(node_config=node_config, seed=seed)
        self._name = 'Random'

    def _run_placement(self, nodes, workload, job_ids, scale_factors):
        """Randomly place jobs on nodes with available capacity."""
        placements = {}
        # Shuffle job order for randomness
        shuffled_ids = list(job_ids)
        self._rng.shuffle(shuffled_ids)

        for jid in shuffled_ids:
            sf = scale_factors[jid]
            gpu_req = float(sf)

            # Find all nodes that can fit this job
            candidates = []
            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is not None:
                    candidates.append((node, indices))

            if candidates:
                # Random selection
                node, indices = self._rng.choice(candidates)
                node.allocate(gpu_req, indices)
                placements[jid] = node

        return placements


class BestFitPolicy(PlacementPolicy):
    """Best-fit placement policy - assigns jobs to minimize remaining capacity.

    Classic bin-packing heuristic: place each job on the node where it leaves
    the smallest remaining capacity (tightest fit).
    """

    def __init__(self, node_config=None, seed=None):
        super().__init__(node_config=node_config, seed=seed)
        self._name = 'BestFit'

    def _run_placement(self, nodes, workload, job_ids, scale_factors):
        """Place jobs using best-fit bin-packing."""
        placements = {}
        # Sort jobs largest-first (standard bin-packing)
        sorted_ids = sorted(job_ids,
                            key=lambda j: scale_factors[j], reverse=True)

        for jid in sorted_ids:
            sf = scale_factors[jid]
            gpu_req = float(sf)

            best_node = None
            best_indices = None
            best_remaining = float('inf')

            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is None:
                    continue

                # Calculate remaining capacity after placement
                remaining = node.get_gpu_scalar() - gpu_req

                # Best-fit: choose the node with smallest remaining capacity
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_node = node
                    best_indices = indices

            if best_node is not None:
                best_node.allocate(gpu_req, best_indices)
                placements[jid] = best_node

        return placements
