import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from collections import Counter

from policy import Policy


# ---------------------------------------------------------------------------
# Inlined FGD components (adapted from fgd_src/ to avoid import conflicts)
# ---------------------------------------------------------------------------

class FGDJob:
    """Lightweight job descriptor for FGD fragmentation calculations."""

    def __init__(self, job_id, gpu_request, scale_factor, gpu_type=None):
        self.job_id = job_id
        self.gpu_request = float(gpu_request)
        self.gpu_type = gpu_type
        self.scale_factor = scale_factor
        # CPU/memory set high so they never constrain placement in Gavel's
        # GPU-only simulation model.
        self.cpu_request = 0.0
        self.memory_request = 0.0


class FGDNode:
    """Per-node GPU capacity tracker for FGD placement decisions."""

    def __init__(self, node_id, num_gpus, gpu_type):
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type
        self.gpu_capacities = [1.0] * num_gpus

    def get_gpu_scalar(self):
        """u = f + p  (f = full GPUs, p = max partial GPU)."""
        full = sum(1 for c in self.gpu_capacities if c == 1.0)
        partials = [c for c in self.gpu_capacities if 0 < c < 1.0]
        return full + (max(partials) if partials else 0.0)

    def can_fit_job(self, gpu_request):
        """Check whether this node can accommodate the gpu_request."""
        return self.get_gpu_scalar() >= gpu_request

    def find_suitable_gpus(self, gpu_request):
        """Return list of GPU indices that can serve the request, or None."""
        if gpu_request == 0:
            return []
        if 0 < gpu_request < 1:
            suitable = [i for i, c in enumerate(self.gpu_capacities)
                        if c >= gpu_request]
            return suitable if suitable else None
        num_needed = int(gpu_request)
        full = [i for i, c in enumerate(self.gpu_capacities) if c == 1.0]
        return full[:num_needed] if len(full) >= num_needed else None

    def allocate(self, gpu_request, gpu_indices):
        if gpu_request <= 0:
            return
        if 0 < gpu_request < 1:
            self.gpu_capacities[gpu_indices[0]] -= gpu_request
        else:
            for idx in gpu_indices:
                self.gpu_capacities[idx] = 0.0

    def copy(self):
        n = FGDNode(self.node_id, self.num_gpus, self.gpu_type)
        n.gpu_capacities = self.gpu_capacities.copy()
        return n


class FGDWorkload:
    """Job type popularity distribution for fragmentation calculations."""

    def __init__(self):
        self.job_types = {}      # type_key -> FGDJob
        self.popularity = {}     # type_key -> float (sums to 1)

    def add_type(self, key, job, popularity):
        self.job_types[key] = job
        self.popularity[key] = popularity

    def normalize(self):
        total = sum(self.popularity.values())
        if total > 0:
            for k in self.popularity:
                self.popularity[k] /= total


class FragmentationCalculator:
    """FGD fragmentation metric (ATC'23 paper)."""

    @staticmethod
    def node_fragmentation(node, job):
        """F_n(m): GPU capacity on node n unusable by job m."""
        if job.gpu_request == 0:
            return sum(node.gpu_capacities)
        if job.gpu_request > node.get_gpu_scalar():
            return sum(node.gpu_capacities)
        frag = 0.0
        min_needed = min(job.gpu_request, 1.0)
        for cap in node.gpu_capacities:
            if cap < min_needed:
                frag += cap
        return frag

    @staticmethod
    def node_fragmentation_for_workload(node, workload):
        """F_n(M) = Σ p_m * F_n(m)."""
        total = 0.0
        for key, job in workload.job_types.items():
            p = workload.popularity.get(key, 0.0)
            total += p * FragmentationCalculator.node_fragmentation(node, job)
        return total


# ---------------------------------------------------------------------------
# FGD Policy for Gavel
# ---------------------------------------------------------------------------

class FGDPolicy(Policy):
    """Fragmentation Gradient Descent scheduling policy.

    Maintains a virtual per-node GPU model internally and uses FGD's greedy
    fragmentation-minimising placement to produce Gavel-format allocation
    fractions.

    Parameters
    ----------
    node_config : dict or None
        {worker_type: gpus_per_node}.  E.g. {'v100': 4, 'p100': 4}.
        If None, defaults to 1 GPU per node (each Gavel worker = 1 node).
    """

    def __init__(self, node_config=None):
        Policy.__init__(self, solver=None)
        self._name = 'FGD'
        self._node_config = node_config or {}
        self._default_gpus_per_node = 1
        # Track state for fragmentation metrics
        self._last_nodes = None
        self._last_workload = None
        self._last_fragmentation_rate = 0.0

    # ----- fragmentation metrics -------------------------------------------

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

    # ----- virtual cluster construction ------------------------------------

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

    # ----- workload construction -------------------------------------------

    def _build_workload(self, scale_factors, job_ids):
        """Derive FGD workload from current active jobs.

        Groups jobs by scale_factor; each group becomes a job type whose
        popularity equals its share of the active job set.
        """
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

    # ----- FGD greedy placement --------------------------------------------

    def _fragmentation_delta(self, node, fgd_job, gpu_indices, workload):
        """Δ = F_n'(M) - F_n(M) for a hypothetical placement."""
        current = FragmentationCalculator.node_fragmentation_for_workload(
            node, workload)
        hyp = node.copy()
        hyp.allocate(fgd_job.gpu_request, gpu_indices)
        after = FragmentationCalculator.node_fragmentation_for_workload(
            hyp, workload)
        return after - current

    def _run_placement(self, nodes, workload, job_ids, scale_factors):
        """Greedy FGD placement.  Returns {job_id: node}."""
        placements = {}
        # Place larger jobs first (standard bin-packing heuristic).
        sorted_ids = sorted(job_ids,
                            key=lambda j: scale_factors[j], reverse=True)
        for jid in sorted_ids:
            sf = scale_factors[jid]
            gpu_req = float(sf)
            fgd_job = FGDJob(job_id=str(jid), gpu_request=gpu_req,
                             scale_factor=sf)

            best_node = None
            best_indices = None
            best_delta = float('inf')

            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is None:
                    continue

                if 0 < gpu_req < 1:
                    # Evaluate each candidate GPU individually.
                    for idx in indices:
                        delta = self._fragmentation_delta(
                            node, fgd_job, [idx], workload)
                        if delta < best_delta:
                            best_delta = delta
                            best_node = node
                            best_indices = [idx]
                else:
                    delta = self._fragmentation_delta(
                        node, fgd_job, indices, workload)
                    if delta < best_delta:
                        best_delta = delta
                        best_node = node
                        best_indices = indices

            if best_node is not None:
                best_node.allocate(gpu_req, best_indices)
                placements[jid] = best_node

        return placements

    # ----- Gavel interface -------------------------------------------------

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        """Compute allocation using FGD placement.

        Parameters match the generic ``else`` branch in
        ``scheduler.py:_compute_allocation`` (line 2377).
        """
        if not unflattened_throughputs:
            return None

        job_ids = sorted(unflattened_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())

        # 1. Build virtual nodes
        nodes = self._build_nodes(cluster_spec)

        # 2. Build workload
        workload = self._build_workload(scale_factors, job_ids)

        # 3. Run FGD greedy placement
        placements = self._run_placement(nodes, workload, job_ids,
                                         scale_factors)

        # Store state for fragmentation metrics
        self._last_nodes = nodes
        self._last_workload = workload
        self._last_fragmentation_rate = self.get_fragmentation_rate(nodes, workload)

        # 4. Convert placements to Gavel allocation fractions
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

        # 5. Validate capacity constraints — scale down if oversubscribed
        for wt in worker_types:
            if capacity_used[wt] > cluster_spec[wt]:
                ratio = cluster_spec[wt] / capacity_used[wt]
                for jid in job_ids:
                    allocation[jid][wt] *= ratio

        return allocation
