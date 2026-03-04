"""Progressive water-filling algorithm for max-min fair allocation.

Exploits the small number of GPU types (n) to avoid solving a large LP.
Uses binary search on the throughput level T combined with a vectorized
capacity-aware allocation to find the max-min fair solution.

Complexity: O(n^2 * m * log(1/eps)) where n = GPU types (6), m = jobs (~1700).
For Alibaba scale: ~60 * 6 * 6 * 1700 = ~3.7M numpy operations.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from policy import Policy
from proportional import ProportionalPolicy


class MaxMinFairnessWaterfillPolicy(Policy):
    """Water-filling max-min fairness without migration penalty."""

    def __init__(self, solver='ECOS', solver_kwargs=None):
        Policy.__init__(self, solver, solver_kwargs=solver_kwargs)
        self._name = 'MaxMinFairness_Waterfill'
        self._proportional_policy = ProportionalPolicy()
        self._prev_allocation = None

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

        x = _waterfill(coefficients, scale_factors_array, num_workers,
                       throughputs)

        # Save for warm-start compatibility (not used by waterfill itself)
        self._prev_allocation = {}
        for i, job_id in enumerate(job_ids):
            self._prev_allocation[job_id] = {}
            for j, wt in enumerate(worker_types):
                self._prev_allocation[job_id][wt] = float(x[i, j])

        return super().unflatten(x, index)


def _waterfill(coeff, sf, capacity, throughputs):
    """Compute max-min fair allocation via progressive filling.

    Args:
        coeff: (m, n) coefficient matrix. Job i's effective throughput is
               sum_j coeff[i,j] * x[i,j].
        sf: (m, n) scale factor matrix. Capacity consumed by job i on type j
            is sf[i,j] * x[i,j].
        capacity: (n,) array of GPU counts per type.
        throughputs: (m, n) raw throughput matrix (used for zero-throughput mask).

    Returns:
        x: (m, n) allocation matrix.
    """
    m, n = coeff.shape
    x = np.zeros((m, n), dtype=np.float64)

    # Mask: where throughput is zero, allocation must be zero
    mask = (throughputs > 0)

    # active[i] = True if job i has not yet been frozen
    active = np.ones(m, dtype=bool)
    # remaining capacity per GPU type
    remaining_cap = capacity.astype(np.float64).copy()

    for iteration in range(n + m):  # at most n+m bottleneck levels
        active_idx = np.where(active)[0]
        num_active = len(active_idx)
        if num_active == 0:
            break

        # Extract active job data
        a_coeff = coeff[active_idx]    # (a, n)
        a_sf = sf[active_idx]          # (a, n)
        a_mask = mask[active_idx]      # (a, n)

        # Compute efficiency: coeff / sf (throughput per unit capacity)
        efficiency = np.zeros_like(a_coeff)
        nonzero = (a_sf > 0) & a_mask
        efficiency[nonzero] = a_coeff[nonzero] / a_sf[nonzero]

        # Sort GPU types by efficiency (descending) for each job
        sort_idx = np.argsort(-efficiency, axis=1)  # (a, n)

        # Upper bound on T: max possible single-job throughput
        max_coeff_per_job = np.max(a_coeff * a_mask, axis=1)
        T_upper = float(np.max(max_coeff_per_job)) if num_active > 0 else 0.0
        T_lower = 0.0

        if T_upper <= 1e-15:
            break

        # Binary search for max T such that capacity-aware allocation
        # is feasible for all active jobs.
        for _ in range(60):  # ~1e-18 relative precision
            T_mid = (T_lower + T_upper) / 2.0
            demand, _ = _compute_demand_capacitated(
                a_coeff, a_sf, a_mask, sort_idx, T_mid, remaining_cap)
            if np.all(demand <= remaining_cap + 1e-10):
                T_lower = T_mid
            else:
                T_upper = T_mid

        T_star = T_lower
        if T_star <= 1e-12:
            break

        # Compute allocations at T_star
        _, job_x = _compute_demand_capacitated(
            a_coeff, a_sf, a_mask, sort_idx, T_star, remaining_cap)

        # Write allocations for active jobs
        x[active_idx] = job_x

        # Check which jobs achieved their full time budget (sum_j x[i,j] = 1)
        # and which are limited by capacity
        time_used = np.sum(job_x, axis=1)  # (a,)
        achieved_T = np.sum(a_coeff * job_x, axis=1)  # (a,)

        # Identify saturated GPU types
        demand = np.sum(a_sf * job_x, axis=0)  # (n,)
        saturated = demand >= remaining_cap - 1e-8

        if not np.any(saturated):
            # No capacity bottleneck; all jobs hit their time limit.
            break

        # Freeze jobs bottlenecked by saturated types.
        # A job is bottlenecked if: (a) it uses a saturated type AND
        # (b) it hasn't used its full time budget (time_used < 1 - eps).
        # Jobs that used their full time budget are bottlenecked on time, not capacity.
        bottlenecked = np.zeros(num_active, dtype=bool)
        for j in range(n):
            if saturated[j]:
                uses_j = job_x[:, j] > 1e-10
                bottlenecked |= uses_j

        # Also freeze jobs that hit their time budget (they can't grow further)
        time_limited = time_used >= 1.0 - 1e-10
        bottlenecked |= time_limited

        if not np.any(bottlenecked):
            break

        # Freeze and update remaining capacity
        freeze_idx = active_idx[bottlenecked]
        for j in range(n):
            remaining_cap[j] -= np.sum(sf[freeze_idx, j] * x[freeze_idx, j])
        remaining_cap = np.maximum(remaining_cap, 0.0)
        active[freeze_idx] = False

    return x.clip(min=0.0).clip(max=1.0)


def _compute_demand_capacitated(coeff, sf, mask, sort_idx, T, cap):
    """Compute capacity-aware allocation for target throughput T.

    Unlike greedy per-job allocation, this respects per-type capacity limits
    during the fill. Jobs that can't get enough of their preferred type
    spill to less-efficient types.

    Fully vectorized over jobs; loops only over GPU type ranks (n=6).

    Args:
        coeff: (a, n) coefficient matrix for active jobs.
        sf: (a, n) scale factors for active jobs.
        mask: (a, n) boolean mask (True where throughput > 0).
        sort_idx: (a, n) GPU type indices sorted by efficiency (desc).
        T: target throughput level.
        cap: (n,) remaining capacity per GPU type.

    Returns:
        demand: (n,) total capacity demand per GPU type.
        job_x: (a, n) allocation matrix for active jobs.
    """
    a, n = coeff.shape
    job_x = np.zeros((a, n), dtype=np.float64)
    remaining_T = np.full(a, T, dtype=np.float64)
    remaining_time = np.ones(a, dtype=np.float64)
    used_cap = np.zeros(n, dtype=np.float64)

    arange_a = np.arange(a)

    for k in range(n):  # iterate over GPU type rank (best, 2nd best, ...)
        # For each job, get the k-th best GPU type index
        j_per_job = sort_idx[:, k]  # (a,)

        # Get coeff and sf for each job's k-th type
        c = coeff[arange_a, j_per_job]     # (a,)
        s = sf[arange_a, j_per_job]        # (a,)
        m = mask[arange_a, j_per_job]      # (a,)

        # Which jobs still need more throughput?
        needs_more = (remaining_T > 1e-12) & (remaining_time > 1e-12) & m & (c > 0)
        if not np.any(needs_more):
            break

        # How much time does each job want on this type?
        # coeff * time = remaining_T -> time = remaining_T / coeff
        safe_c = np.where(needs_more & (c > 0), c, 1.0)  # avoid div by zero
        wanted_time = np.where(needs_more, remaining_T / safe_c, 0.0)
        wanted_time = np.minimum(wanted_time, remaining_time)  # can't exceed 1

        # How much capacity does each job want?
        # capacity_used = sf * time
        wanted_cap = s * wanted_time  # (a,)

        # For each GPU type j, compute total demand and scale down if over cap
        for j in range(n):
            jobs_on_j = (j_per_job == j) & needs_more
            if not np.any(jobs_on_j):
                continue

            total_wanted = np.sum(wanted_cap[jobs_on_j])
            available = cap[j] - used_cap[j]

            if available <= 1e-12:
                # Type j is full -- these jobs get nothing on this type
                wanted_time[jobs_on_j] = 0.0
                wanted_cap[jobs_on_j] = 0.0
                continue

            if total_wanted <= available + 1e-10:
                # All fit
                used_cap[j] += total_wanted
            else:
                # Scale down proportionally
                scale = available / total_wanted
                wanted_time[jobs_on_j] *= scale
                wanted_cap[jobs_on_j] *= scale
                used_cap[j] = cap[j]

        # Apply allocation
        job_x[arange_a, j_per_job] += wanted_time
        achieved = c * wanted_time
        remaining_T -= achieved
        remaining_time -= wanted_time

    demand = np.sum(sf * job_x, axis=0)
    return demand, job_x
