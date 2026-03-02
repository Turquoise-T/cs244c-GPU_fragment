"""Max-min fair allocation with single GPU type assignment per job.

Each job is assigned to exactly one GPU type. This eliminates the need
for an LP solver entirely:

  1. Assignment phase: greedy load-balanced assignment using LPT
     (Longest Processing Time first) with migration stickiness.
  2. Allocation phase: analytical per-type waterfilling.

The migration penalty is handled implicitly: a job that would switch
types sees reduced effective throughput for the transition round
(proportional to migration_time / time_per_iteration). This replaces
the L1 auxiliary variables that doubled the LP variable count.

Complexity: O(m * n * log m) for assignment + O(m * n) for allocation,
where m = jobs (~1700) and n = GPU types (6).
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from policy import Policy
from proportional import ProportionalPolicy


class MaxMinFairnessSingleTypePolicy(Policy):
    """Single-type assignment with analytical max-min fair allocation."""

    def __init__(self, solver='ECOS', solver_kwargs=None):
        Policy.__init__(self, solver, solver_kwargs=solver_kwargs)
        self._name = 'MaxMinFairness_SingleType'
        self._proportional_policy = ProportionalPolicy()
        self._prev_allocation = None
        self._prev_assignment = None  # {job_id: int} GPU type index

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        priority_weights = np.multiply(
            priority_weights.reshape((m, 1)),
            1.0 / proportional_throughputs.reshape((m, 1)))

        coefficients = np.multiply(
            throughputs * priority_weights.reshape((m, 1)),
            scale_factors_array)

        num_workers = np.array(
            [cluster_spec[wt] for wt in worker_types], dtype=np.float64)

        # Mask: zero-throughput (job, type) pairs are infeasible
        mask = (throughputs > 0)

        # Build previous assignment vector
        prev_assign = np.full(m, -1, dtype=np.int64)
        if self._prev_assignment is not None:
            for i, job_id in enumerate(job_ids):
                if job_id in self._prev_assignment:
                    prev_assign[i] = self._prev_assignment[job_id]

        # Compute per-job migration fraction
        migration_frac = np.zeros(m)
        if (self._migration_times is not None
                and self._time_per_iteration is not None
                and self._time_per_iteration > 0):
            for i, job_id in enumerate(job_ids):
                mt = self._migration_times.get(job_id, 0)
                migration_frac[i] = mt / self._time_per_iteration

        # Phase 1: Assign each job to a GPU type
        assignment = _assign_types(
            coefficients, scale_factors_array, mask,
            num_workers, prev_assign, migration_frac)

        # Phase 2: Analytical allocation
        x = _allocate_single_type(
            coefficients, scale_factors_array, assignment,
            num_workers, m, n)

        # Save state for next round
        self._prev_allocation = {}
        self._prev_assignment = {}
        for i, job_id in enumerate(job_ids):
            self._prev_allocation[job_id] = {}
            self._prev_assignment[job_id] = int(assignment[i])
            for j, wt in enumerate(worker_types):
                self._prev_allocation[job_id][wt] = float(x[i, j])

        return super().unflatten(x, index)


def _assign_types(coeff, sf, mask, capacity, prev_assign, migration_frac):
    """Assign each job to exactly one GPU type.

    Uses LPT (Longest Processing Time first) load-balanced greedy.
    Jobs with the highest demand (capacity cost per unit throughput)
    are assigned first to their best-fit type, minimizing the maximum
    normalized load across GPU types.

    Migration stickiness: switching types reduces effective throughput
    by migration_frac[i] for the transition round, making the current
    type more attractive unless a new type is significantly better.

    Args:
        coeff: (m, n) coefficient matrix.
        sf: (m, n) scale factor matrix.
        mask: (m, n) boolean feasibility mask.
        capacity: (n,) GPU count per type.
        prev_assign: (m,) previous type assignment (-1 if new job).
        migration_frac: (m,) fraction of round lost to migration.

    Returns:
        assignment: (m,) GPU type index per job.
    """
    m, n = coeff.shape
    assignment = np.full(m, -1, dtype=np.int64)

    # Effective coefficients: reduced by migration cost when switching
    eff_coeff = np.zeros((m, n))
    for j in range(n):
        staying = (prev_assign == j)
        switching = ~staying
        eff_coeff[:, j] = coeff[:, j]
        # Jobs switching to type j lose migration_frac of throughput
        eff_coeff[switching, j] *= (1.0 - migration_frac[switching])
    # Zero out infeasible entries
    eff_coeff[~mask] = 0.0

    # Demand per unit throughput: d[i,j] = sf[i,j] / eff_coeff[i,j]
    # Lower demand = more efficient placement
    demand = np.full((m, n), np.inf)
    feasible = mask & (eff_coeff > 0)
    demand[feasible] = sf[feasible] / eff_coeff[feasible]

    # Sort by minimum demand (heaviest jobs first -- LPT heuristic)
    min_demand = np.min(demand, axis=1)
    order = np.argsort(-min_demand)

    # Track normalized load per type: load[j] / capacity[j]
    raw_load = np.zeros(n)

    for idx in order:
        best_j = -1
        best_normalized = np.inf

        for j in range(n):
            if demand[idx, j] >= np.inf:
                continue
            # What would the normalized load be after adding this job?
            new_load = (raw_load[j] + demand[idx, j]) / capacity[j]
            if new_load < best_normalized:
                best_normalized = new_load
                best_j = j

        if best_j >= 0:
            assignment[idx] = best_j
            raw_load[best_j] += demand[idx, best_j]

    return assignment


def _allocate_single_type(coeff, sf, assignment, capacity, m, n):
    """Compute max-min fair time allocation given single-type assignments.

    Since each job uses exactly one type, the problem decomposes into n
    independent sub-problems (one per GPU type). Within each type, a
    sort-based waterfill gives O(k log k) per type, O(m log m) total.

    For type j with sorted jobs c_1 <= c_2 <= ... <= c_k:
      - Try to give all jobs throughput T = cap_j / sum(s_i/c_i)
      - If T > c_i for some job i, that job hits time limit (y=1).
        Freeze it, reduce capacity, continue with remaining jobs.

    Args:
        coeff: (m, n) coefficient matrix.
        sf: (m, n) scale factor matrix.
        assignment: (m,) GPU type index per job.
        capacity: (n,) GPU count per type.
        m: number of jobs.
        n: number of GPU types.

    Returns:
        x: (m, n) allocation matrix (sparse: one nonzero per row).
    """
    arange_m = np.arange(m)
    valid_assign = assignment >= 0

    # Extract per-job coefficient and scale factor for assigned type
    job_coeff = np.zeros(m)
    job_sf = np.zeros(m)
    job_coeff[valid_assign] = coeff[arange_m[valid_assign],
                                    assignment[valid_assign]]
    job_sf[valid_assign] = sf[arange_m[valid_assign],
                              assignment[valid_assign]]

    # y[i] = time fraction for job i
    y = np.zeros(m, dtype=np.float64)

    # Process each GPU type independently
    for j in range(n):
        on_j = np.where(valid_assign & (assignment == j)
                        & (job_coeff > 0))[0]
        if len(on_j) == 0:
            continue
        _waterfill_type(job_coeff, job_sf, on_j, capacity[j], y)

    # Build sparse allocation matrix
    x = np.zeros((m, n), dtype=np.float64)
    placed = valid_assign & (job_coeff > 0)
    x[arange_m[placed], assignment[placed]] = y[placed]
    return x.clip(min=0.0, max=1.0)


def _waterfill_type(job_coeff, job_sf, indices, cap, y):
    """Waterfill allocation for jobs assigned to a single GPU type.

    Sorts jobs by coefficient (ascending), then greedily fills.
    Jobs with lower coefficients hit time limit (y=1) first; their
    capacity is reclaimed for remaining jobs.

    O(k log k) for k jobs.
    """
    k = len(indices)
    c = job_coeff[indices]
    s = job_sf[indices]

    # Sort by coefficient ascending (lowest throughput capacity first)
    order = np.argsort(c)
    c_sorted = c[order]
    s_sorted = s[order]

    # demand[i] = s[i] / c[i]
    d_sorted = s_sorted / c_sorted

    # Cumulative demand from the right: total_demand[i] = sum_{j>=i} d[j]
    # We process from left (lowest c) to right (highest c).
    remaining_demand = np.sum(d_sorted)
    remaining_cap = float(cap)

    for i in range(k):
        if remaining_demand <= 1e-15:
            break

        # T if we fill all remaining jobs to same throughput
        T_cap = remaining_cap / remaining_demand

        if T_cap <= c_sorted[i]:
            # Capacity is the bottleneck. All remaining jobs get T_cap.
            y[indices[order[i:]]] = T_cap / c_sorted[i:]
            break
        else:
            # Job i hits time limit: y = 1, throughput = c[i]
            y[indices[order[i]]] = 1.0
            remaining_cap -= s_sorted[i]  # Full capacity consumed
            remaining_demand -= d_sorted[i]

    # Clip to [0, 1]
    np.clip(y[indices], 0.0, 1.0, out=y[indices])
