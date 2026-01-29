#!/usr/bin/env python3
"""
Test script to compare buggy vs fixed behavior for V100 hardcoding.

This script:
1. Shows current buggy behavior
2. Tests a potential fix
3. Compares allocations and solver stability
"""

import sys
sys.path.append("..")

import copy
import numpy as np
import cvxpy as cp

from policies.policy import Policy
from policies.isolated import IsolatedPolicy


# ============================================================================
# CURRENT BUGGY IMPLEMENTATION (from finish_time_fairness.py)
# ============================================================================

class FinishTimeFairnessPolicyBuggy(Policy):
    """Current implementation with V100 hardcoding bug."""

    def __init__(self, solver):
        self._name = 'FinishTimeFairness_Buggy'
        self._finish_time_fairness_perf_policy = \
            FinishTimeFairnessPolicyWithPerfBase(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        # BUG: Hardcode all throughputs to V100 values
        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 # Hardcode worker_type to v100 since all other worker types
                 # have a throughput of 0 for some job.
                 new_unflattened_throughputs[job_id][worker_type] = \
                     unflattened_throughputs[job_id]['v100']

        return self._finish_time_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors,
            unflattened_priority_weights,
            times_since_start,
            num_steps_remaining, cluster_spec)


# ============================================================================
# PROPOSED FIX: Constrain zero-throughput allocations to zero
# ============================================================================

class FinishTimeFairnessPolicyFixed(Policy):
    """Fixed implementation that handles zero throughputs correctly."""

    def __init__(self, solver):
        self._name = 'FinishTimeFairness_Fixed'
        self._finish_time_fairness_perf_policy = \
            FinishTimeFairnessPolicyWithPerfFixed(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        # FIX: Pass through actual throughputs, let the optimizer handle zeros
        return self._finish_time_fairness_perf_policy.get_allocation(
            unflattened_throughputs, scale_factors,
            unflattened_priority_weights,
            times_since_start,
            num_steps_remaining, cluster_spec)


# ============================================================================
# Base FinishTimeFairness optimizer (shared logic)
# ============================================================================

class FinishTimeFairnessPolicyWithPerfBase(Policy):
    """Base implementation for finish-time fairness optimization."""

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_PerfBase'
        self._isolated_policy = IsolatedPolicy()
        self._cumulative_isolated_time = {}
        self._isolated_throughputs_prev_iteration = {}
        self._num_steps_remaining_prev_iteration = {}

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        isolated_throughputs = self._isolated_policy.get_throughputs(
            throughputs, index, scale_factors, cluster_spec)

        for job_id in job_ids:
            if job_id not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[job_id] = 0
            if job_id in self._isolated_throughputs_prev_iteration:
                if job_id in self._num_steps_remaining_prev_iteration:
                    self._cumulative_isolated_time[job_id] += \
                        (self._num_steps_remaining_prev_iteration[job_id] -
                         num_steps_remaining[job_id]) / \
                            self._isolated_throughputs_prev_iteration[job_id]

        cumulative_isolated_time = np.array(
            [self._cumulative_isolated_time[job_id] for job_id in job_ids])
        for i, job_id in enumerate(job_ids):
            self._isolated_throughputs_prev_iteration[job_id] = \
                isolated_throughputs[i][0]
            self._num_steps_remaining_prev_iteration[job_id] = \
                num_steps_remaining[job_id]

        remaining = np.array([num_steps_remaining[job_id] for job_id in job_ids])
        times_since_start_arr = np.array([times_since_start[job_id]
                                          for job_id in job_ids])

        x = cp.Variable(throughputs.shape)

        # Effective throughput for each job
        effective_throughputs = cp.sum(cp.multiply(throughputs, x), axis=1)

        # Remaining time = remaining_steps / effective_throughput
        # Use inv_pos to handle the division safely
        remaining_time = cp.multiply(remaining, cp.inv_pos(effective_throughputs))

        # Isolated remaining time
        isolated_remaining_time = remaining / isolated_throughputs.flatten()

        # Finish time ratio rho
        finish_time = times_since_start_arr + remaining_time
        isolated_finish_time = cumulative_isolated_time + isolated_remaining_time

        # Objective: minimize maximum finish time ratio
        # rho = finish_time / isolated_finish_time
        objective = cp.Minimize(cp.max(
            cp.multiply(finish_time, 1.0 / isolated_finish_time)))

        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print(f'WARNING: Allocation status: {cvxprob.status}')
            return None

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class FinishTimeFairnessPolicyWithPerfFixed(Policy):
    """Fixed implementation that explicitly constrains zero-throughput allocations."""

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_PerfFixed'
        self._isolated_policy = IsolatedPolicy()
        self._cumulative_isolated_time = {}
        self._isolated_throughputs_prev_iteration = {}
        self._num_steps_remaining_prev_iteration = {}

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        isolated_throughputs = self._isolated_policy.get_throughputs(
            throughputs, index, scale_factors, cluster_spec)

        for job_id in job_ids:
            if job_id not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[job_id] = 0
            if job_id in self._isolated_throughputs_prev_iteration:
                if job_id in self._num_steps_remaining_prev_iteration:
                    self._cumulative_isolated_time[job_id] += \
                        (self._num_steps_remaining_prev_iteration[job_id] -
                         num_steps_remaining[job_id]) / \
                            self._isolated_throughputs_prev_iteration[job_id]

        cumulative_isolated_time = np.array(
            [self._cumulative_isolated_time[job_id] for job_id in job_ids])
        for i, job_id in enumerate(job_ids):
            self._isolated_throughputs_prev_iteration[job_id] = \
                isolated_throughputs[i][0]
            self._num_steps_remaining_prev_iteration[job_id] = \
                num_steps_remaining[job_id]

        remaining = np.array([num_steps_remaining[job_id] for job_id in job_ids])
        times_since_start_arr = np.array([times_since_start[job_id]
                                          for job_id in job_ids])

        x = cp.Variable(throughputs.shape)

        # FIX: Replace zeros with small epsilon to avoid division issues
        # but constrain those allocations to be zero
        epsilon = 1e-6
        safe_throughputs = np.where(throughputs > 0, throughputs, epsilon)

        # Effective throughput for each job
        effective_throughputs = cp.sum(cp.multiply(safe_throughputs, x), axis=1)

        # Remaining time = remaining_steps / effective_throughput
        remaining_time = cp.multiply(remaining, cp.inv_pos(effective_throughputs))

        # Isolated remaining time
        isolated_remaining_time = remaining / isolated_throughputs.flatten()

        # Finish time ratio rho
        finish_time = times_since_start_arr + remaining_time
        isolated_finish_time = cumulative_isolated_time + isolated_remaining_time

        # Objective: minimize maximum finish time ratio
        objective = cp.Minimize(cp.max(
            cp.multiply(finish_time, 1.0 / isolated_finish_time)))

        constraints = self.get_base_constraints(x, scale_factors_array)

        # FIX: Add explicit constraints for zero-throughput allocations
        for i in range(m):
            for j in range(n):
                if throughputs[i, j] == 0:
                    constraints.append(x[i, j] == 0)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print(f'WARNING: Allocation status: {cvxprob.status}')
            return None

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


# ============================================================================
# Test Runner
# ============================================================================

def print_allocation(name, allocation, throughputs):
    """Print allocation with zero-throughput violations highlighted."""
    print(f"\n{name}:")
    print("-" * 60)

    violations = []
    for job_id in sorted(allocation.keys()):
        parts = []
        for gpu in ['v100', 'p100', 'k80']:
            alloc = allocation[job_id][gpu]
            tput = throughputs[job_id][gpu]
            if tput == 0 and alloc > 0.01:
                parts.append(f"{gpu}={alloc:.3f} [BUG!]")
                violations.append((job_id, gpu, alloc))
            else:
                parts.append(f"{gpu}={alloc:.3f}")
        print(f"  Job {job_id}: {', '.join(parts)}")

    if violations:
        print(f"\n  *** {len(violations)} VIOLATIONS: Jobs allocated to GPUs they cannot run on ***")
    else:
        print(f"\n  *** NO VIOLATIONS: All allocations respect throughput constraints ***")

    return violations


def compute_gpu_utilization(allocation, cluster_spec):
    """Compute total GPU utilization by type."""
    totals = {gpu: 0.0 for gpu in cluster_spec}
    for job_id, job_alloc in allocation.items():
        for gpu, alloc in job_alloc.items():
            totals[gpu] += alloc

    print("\nGPU Utilization:")
    for gpu in ['v100', 'p100', 'k80']:
        pct = totals[gpu] / cluster_spec[gpu] * 100
        print(f"  {gpu}: {totals[gpu]:.2f}/{cluster_spec[gpu]} ({pct:.1f}%)")


def main():
    print("=" * 70)
    print("V100 HARDCODING BUG: Before and After Fix Comparison")
    print("=" * 70)

    cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

    # Test case: Some jobs have zero K80 throughput
    throughputs = {
        0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},   # Can run anywhere
        1: {'v100': 8.0, 'p100': 4.0, 'k80': 0.0},    # CANNOT run on K80
        2: {'v100': 6.0, 'p100': 3.0, 'k80': 1.0},    # Can run anywhere
        3: {'v100': 4.0, 'p100': 2.0, 'k80': 0.0},    # CANNOT run on K80
    }

    scale_factors = {0: 1, 1: 1, 2: 1, 3: 1}
    priority_weights = {0: 1, 1: 1, 2: 1, 3: 1}
    times_since_start = {0: 100, 1: 200, 2: 150, 3: 50}
    num_steps_remaining = {0: 1000, 1: 800, 2: 1200, 3: 600}

    print("\nTest Case: Jobs with mixed K80 compatibility")
    print("  Jobs 0, 2: Can run on all GPU types")
    print("  Jobs 1, 3: CANNOT run on K80 (zero throughput)")

    # Test buggy implementation
    print("\n" + "=" * 70)
    print("BUGGY IMPLEMENTATION (V100 hardcoding)")
    print("=" * 70)

    try:
        policy_buggy = FinishTimeFairnessPolicyBuggy(solver='ECOS')
        allocation_buggy = policy_buggy.get_allocation(
            throughputs, scale_factors, priority_weights,
            times_since_start, num_steps_remaining, cluster_spec
        )

        if allocation_buggy:
            violations_buggy = print_allocation("Buggy Allocation", allocation_buggy, throughputs)
            compute_gpu_utilization(allocation_buggy, cluster_spec)
        else:
            print("  Solver returned None (failed)")
            violations_buggy = []
    except Exception as e:
        print(f"  SOLVER ERROR: {e}")
        violations_buggy = ["solver_error"]

    # Test fixed implementation
    print("\n" + "=" * 70)
    print("FIXED IMPLEMENTATION (explicit zero-throughput constraints)")
    print("=" * 70)

    try:
        policy_fixed = FinishTimeFairnessPolicyFixed(solver='ECOS')
        allocation_fixed = policy_fixed.get_allocation(
            throughputs, scale_factors, priority_weights,
            times_since_start, num_steps_remaining, cluster_spec
        )

        if allocation_fixed:
            violations_fixed = print_allocation("Fixed Allocation", allocation_fixed, throughputs)
            compute_gpu_utilization(allocation_fixed, cluster_spec)
        else:
            print("  Solver returned None (failed)")
            violations_fixed = ["solver_error"]
    except Exception as e:
        print(f"  SOLVER ERROR: {e}")
        violations_fixed = ["solver_error"]

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Buggy implementation: {len(violations_buggy)} violations")
    print(f"Fixed implementation: {len(violations_fixed)} violations")

    if len(violations_fixed) == 0 and len(violations_buggy) > 0:
        print("\n*** FIX SUCCESSFUL: Zero-throughput allocations are now constrained to zero ***")
    elif len(violations_fixed) > 0:
        print("\n*** FIX INCOMPLETE: Still has violations ***")

    return len(violations_buggy), len(violations_fixed)


if __name__ == "__main__":
    main()
