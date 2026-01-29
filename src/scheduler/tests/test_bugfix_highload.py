#!/usr/bin/env python3
"""
Test the bugfix under high load conditions similar to the failing experiments.
"""

import sys
sys.path.append("..")

import numpy as np
import cvxpy as cp
from policies.policy import Policy
from policies.isolated import IsolatedPolicy


class FinishTimeFairnessPolicyFixed(Policy):
    """Fixed implementation that handles zero throughputs correctly."""

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_Fixed'
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

        # Replace zeros with small epsilon for numerical stability in objective
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
            return None

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


def test_high_load(num_jobs, description):
    """Test with a specific number of concurrent jobs."""
    print(f"\n{'='*60}")
    print(f"Testing: {description} ({num_jobs} jobs)")
    print(f"{'='*60}")

    cluster_spec = {'v100': 36, 'p100': 36, 'k80': 36}

    throughputs = {}
    scale_factors = {}
    priority_weights = {}
    times_since_start = {}
    num_steps_remaining = {}

    zero_k80_jobs = []
    for i in range(num_jobs):
        # Every 5th job has zero K80 throughput (realistic scenario)
        k80_throughput = 0.0 if i % 5 == 0 else max(0.5, 2.0 - (i % 3) * 0.5)
        if k80_throughput == 0.0:
            zero_k80_jobs.append(i)

        throughputs[i] = {
            'v100': max(1.0, 10.0 - (i % 5)),
            'p100': max(0.5, 5.0 - (i % 3)),
            'k80': k80_throughput
        }
        scale_factors[i] = 1 + (i % 4)  # Mix of 1, 2, 3, 4 GPU jobs
        priority_weights[i] = 1
        times_since_start[i] = i * 100
        num_steps_remaining[i] = 1000 + i * 50

    print(f"Jobs with zero K80 throughput: {zero_k80_jobs}")

    policy = FinishTimeFairnessPolicyFixed(solver='ECOS')

    try:
        allocation = policy.get_allocation(
            throughputs, scale_factors, priority_weights,
            times_since_start, num_steps_remaining, cluster_spec
        )

        if allocation is None:
            print("RESULT: Solver returned None (infeasible or failed)")
            return False

        # Check for violations
        violations = []
        for job_id in zero_k80_jobs:
            k80_alloc = allocation[job_id].get('k80', 0)
            if k80_alloc > 0.01:
                violations.append((job_id, k80_alloc))

        if violations:
            print(f"RESULT: FAILED - {len(violations)} violations")
            for job_id, alloc in violations:
                print(f"  Job {job_id}: k80={alloc:.4f} (should be 0)")
            return False
        else:
            # Compute utilization
            totals = {gpu: 0.0 for gpu in cluster_spec}
            for job_id, job_alloc in allocation.items():
                for gpu, alloc in job_alloc.items():
                    totals[gpu] += alloc * scale_factors[job_id]

            print("RESULT: SUCCESS - No violations")
            print(f"GPU Utilization (scaled by scale_factor):")
            for gpu in ['v100', 'p100', 'k80']:
                pct = totals[gpu] / cluster_spec[gpu] * 100
                print(f"  {gpu}: {totals[gpu]:.1f}/{cluster_spec[gpu]} ({pct:.1f}%)")
            return True

    except Exception as e:
        print(f"RESULT: EXCEPTION - {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("HIGH LOAD TESTING FOR BUGFIX")
    print("=" * 60)

    results = []

    # Test cases matching the failed experiments
    test_cases = [
        (10, "Light load"),
        (20, "Medium load"),
        (30, "Heavy load"),
        (45, "Very heavy load (similar to 1.6 jph)"),
        (60, "Extreme load (similar to 2.4 jph)"),
    ]

    for num_jobs, description in test_cases:
        success = test_high_load(num_jobs, description)
        results.append((description, success))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for description, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {description}: {status}")

    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
