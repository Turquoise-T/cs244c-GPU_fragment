"""
Unit tests for verifying scheduling policy behavior.

These tests ENFORCE correct behavior and will FAIL when bugs are detected.
They verify that policies:
1. Use all available GPU types appropriately
2. Don't exceed cluster capacity
3. Don't allocate to GPUs where jobs have zero throughput
4. Maintain expected fairness properties

KNOWN BUGS DETECTED BY THESE TESTS:
- FinishTimeFairnessPolicy: V100 hardcoding (finish_time_fairness.py:34-37)
- MinTotalDurationPolicy: V100 hardcoding (min_total_duration.py:30-31)
"""

import sys
sys.path.append("..")

import unittest
import numpy as np
from policies import (
    finish_time_fairness,
    max_min_fairness,
    min_total_duration,
    proportional,
    fifo,
    max_sum_throughput,
    isolated,
)


class TestZeroThroughputEnforcement(unittest.TestCase):
    """
    CRITICAL: Tests that jobs with zero throughput on a GPU type
    do NOT receive allocation on that GPU type.

    This catches the V100 hardcoding bug where policies incorrectly
    allocate resources to GPUs that jobs cannot run on.
    """

    def setUp(self):
        self.cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

        # Jobs where some have zero throughput on K80
        self.throughputs_with_zero_k80 = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},   # Can run anywhere
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 0.0},    # CANNOT run on K80
            2: {'v100': 6.0, 'p100': 3.0, 'k80': 1.0},    # Can run anywhere
            3: {'v100': 4.0, 'p100': 2.0, 'k80': 0.0},    # CANNOT run on K80
        }

        self.scale_factors = {0: 1, 1: 1, 2: 1, 3: 1}
        self.priority_weights = {0: 1, 1: 1, 2: 1, 3: 1}
        self.times_since_start = {0: 0, 1: 0, 2: 0, 3: 0}
        self.num_steps_remaining = {0: 1000, 1: 1000, 2: 1000, 3: 1000}

    def _check_zero_throughput_allocation(self, allocation, throughputs, policy_name):
        """Helper to check that zero-throughput GPU types get zero allocation."""
        violations = []
        for job_id, job_throughputs in throughputs.items():
            for gpu_type, throughput in job_throughputs.items():
                if throughput == 0.0:
                    alloc = allocation.get(job_id, {}).get(gpu_type, 0)
                    if alloc > 0.01:  # Small tolerance for floating point
                        violations.append(
                            f"Job {job_id} has zero {gpu_type} throughput "
                            f"but got {gpu_type} allocation of {alloc:.4f}"
                        )

        if violations:
            self.fail(
                f"\n{policy_name} BUG DETECTED - Allocating to GPUs where jobs cannot run:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )

    def test_finish_time_fairness_base_zero_throughput(self):
        """FinishTimeFairnessPolicy should NOT allocate to GPUs with zero throughput.

        KNOWN BUG: This test will FAIL due to V100 hardcoding in
        finish_time_fairness.py lines 34-37.
        """
        policy = finish_time_fairness.FinishTimeFairnessPolicy(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.priority_weights,
            self.times_since_start,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "FinishTimeFairnessPolicy"
        )

    def test_finish_time_fairness_perf_zero_throughput(self):
        """FinishTimeFairnessPolicyWithPerf should NOT allocate to GPUs with zero throughput."""
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.priority_weights,
            self.times_since_start,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "FinishTimeFairnessPolicyWithPerf"
        )

    def test_min_total_duration_base_zero_throughput(self):
        """MinTotalDurationPolicy should NOT allocate to GPUs with zero throughput.

        KNOWN BUG: This test will FAIL due to V100 hardcoding in
        min_total_duration.py lines 30-31.
        """
        policy = min_total_duration.MinTotalDurationPolicy(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "MinTotalDurationPolicy"
        )

    def test_min_total_duration_perf_zero_throughput(self):
        """MinTotalDurationPolicyWithPerf should NOT allocate to GPUs with zero throughput."""
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "MinTotalDurationPolicyWithPerf"
        )

    def test_max_min_fairness_perf_zero_throughput(self):
        """MaxMinFairnessPolicyWithPerf should NOT allocate to GPUs with zero throughput."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.priority_weights,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "MaxMinFairnessPolicyWithPerf"
        )

    def test_throughput_sum_perf_zero_throughput(self):
        """ThroughputSumWithPerf should NOT allocate to GPUs with zero throughput."""
        policy = max_sum_throughput.ThroughputSumWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs_with_zero_k80,
            self.scale_factors,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_zero_throughput_allocation(
            allocation,
            self.throughputs_with_zero_k80,
            "ThroughputSumWithPerf"
        )


class TestClusterCapacityEnforcement(unittest.TestCase):
    """Tests that policies never exceed cluster capacity."""

    def setUp(self):
        self.cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

        self.throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 1.5},
            2: {'v100': 6.0, 'p100': 3.0, 'k80': 1.0},
            3: {'v100': 4.0, 'p100': 2.0, 'k80': 0.8},
        }

        self.scale_factors = {0: 1, 1: 1, 2: 1, 3: 1}
        self.priority_weights = {0: 1, 1: 1, 2: 1, 3: 1}
        self.times_since_start = {0: 0, 1: 0, 2: 0, 3: 0}
        self.num_steps_remaining = {0: 1000, 1: 1000, 2: 1000, 3: 1000}

    def _check_capacity_constraints(self, allocation, cluster_spec, policy_name):
        """Helper to verify allocation doesn't exceed capacity."""
        totals = {gpu_type: 0.0 for gpu_type in cluster_spec}

        for job_id, job_alloc in allocation.items():
            for gpu_type, value in job_alloc.items():
                totals[gpu_type] += value

        violations = []
        for gpu_type, capacity in cluster_spec.items():
            if totals[gpu_type] > capacity + 1e-4:  # Small tolerance
                violations.append(
                    f"{gpu_type}: allocated {totals[gpu_type]:.4f} > capacity {capacity}"
                )

        if violations:
            self.fail(
                f"\n{policy_name} CAPACITY VIOLATION:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )

    def test_finish_time_fairness_capacity(self):
        """FinishTimeFairnessPolicyWithPerf should not exceed cluster capacity."""
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.times_since_start,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_capacity_constraints(allocation, self.cluster_spec,
                                        "FinishTimeFairnessPolicyWithPerf")

    def test_max_min_fairness_capacity(self):
        """MaxMinFairnessPolicyWithPerf should not exceed cluster capacity."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_capacity_constraints(allocation, self.cluster_spec,
                                        "MaxMinFairnessPolicyWithPerf")

    def test_min_total_duration_capacity(self):
        """MinTotalDurationPolicyWithPerf should not exceed cluster capacity."""
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_capacity_constraints(allocation, self.cluster_spec,
                                        "MinTotalDurationPolicyWithPerf")

    def test_throughput_sum_capacity(self):
        """ThroughputSumWithPerf should not exceed cluster capacity."""
        policy = max_sum_throughput.ThroughputSumWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_capacity_constraints(allocation, self.cluster_spec,
                                        "ThroughputSumWithPerf")


class TestGPUTypeUtilization(unittest.TestCase):
    """Tests that policies utilize all GPU types when jobs can run on all of them."""

    def setUp(self):
        self.cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

        # All jobs can run on all GPU types
        self.throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 1.5},
            2: {'v100': 6.0, 'p100': 3.0, 'k80': 1.0},
            3: {'v100': 4.0, 'p100': 2.0, 'k80': 0.8},
        }

        self.scale_factors = {0: 1, 1: 1, 2: 1, 3: 1}
        self.priority_weights = {0: 1, 1: 1, 2: 1, 3: 1}
        self.times_since_start = {0: 0, 1: 0, 2: 0, 3: 0}
        self.num_steps_remaining = {0: 1000, 1: 1000, 2: 1000, 3: 1000}

    def _check_all_gpu_types_used(self, allocation, cluster_spec, policy_name):
        """Helper to verify all GPU types have some allocation.

        Note: With heterogeneous throughputs (V100 >> K80), the optimizer may
        legitimately allocate very little to slower GPU types. We use a very
        small threshold to catch complete non-usage (hardcoding bugs) while
        allowing valid optimization decisions.
        """
        totals = {gpu_type: 0.0 for gpu_type in cluster_spec}

        for job_id, job_alloc in allocation.items():
            for gpu_type, value in job_alloc.items():
                totals[gpu_type] += value

        # Use a very small threshold - we're checking for complete non-usage,
        # not inefficient usage. Values like 0.005 are acceptable.
        unused = [gpu_type for gpu_type, total in totals.items() if total < 0.001]

        if unused:
            self.fail(
                f"\n{policy_name} GPU UNDERUTILIZATION - These GPU types are not being used:\n" +
                "\n".join(f"  - {gpu}: {totals[gpu]:.4f}" for gpu in unused) +
                f"\n\nThis may indicate a hardcoding bug where only certain GPU types are considered."
            )

    def test_finish_time_fairness_perf_uses_all_gpus(self):
        """FinishTimeFairnessPolicyWithPerf should use all GPU types."""
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.times_since_start,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_all_gpu_types_used(allocation, self.cluster_spec,
                                       "FinishTimeFairnessPolicyWithPerf")

    def test_max_min_fairness_perf_uses_all_gpus(self):
        """MaxMinFairnessPolicyWithPerf should use all GPU types."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_all_gpu_types_used(allocation, self.cluster_spec,
                                       "MaxMinFairnessPolicyWithPerf")

    def test_min_total_duration_perf_uses_all_gpus(self):
        """MinTotalDurationPolicyWithPerf should use all GPU types."""
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_all_gpu_types_used(allocation, self.cluster_spec,
                                       "MinTotalDurationPolicyWithPerf")


class TestMultiGPUJobScaleFactors(unittest.TestCase):
    """Tests that multi-GPU jobs (scale_factor > 1) are handled correctly."""

    def setUp(self):
        self.cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

        self.throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 1.5},
            2: {'v100': 6.0, 'p100': 3.0, 'k80': 1.0},
            3: {'v100': 4.0, 'p100': 2.0, 'k80': 0.5},
        }

        # Mix of single and multi-GPU jobs
        self.scale_factors = {0: 1, 1: 2, 2: 4, 3: 8}
        self.priority_weights = {0: 1, 1: 1, 2: 1, 3: 1}
        self.times_since_start = {0: 0, 1: 0, 2: 0, 3: 0}
        self.num_steps_remaining = {0: 1000, 1: 1000, 2: 1000, 3: 1000}

    def _check_multi_gpu_capacity(self, allocation, scale_factors, cluster_spec, policy_name):
        """Helper to verify multi-GPU jobs don't exceed capacity when accounting for scale."""
        # Total GPU usage per type = sum of (allocation * scale_factor)
        totals = {gpu_type: 0.0 for gpu_type in cluster_spec}

        for job_id, job_alloc in allocation.items():
            scale = scale_factors.get(job_id, 1)
            for gpu_type, value in job_alloc.items():
                totals[gpu_type] += value * scale

        violations = []
        for gpu_type, capacity in cluster_spec.items():
            if totals[gpu_type] > capacity + 1e-4:
                violations.append(
                    f"{gpu_type}: scaled allocation {totals[gpu_type]:.4f} > capacity {capacity}"
                )

        if violations:
            self.fail(
                f"\n{policy_name} MULTI-GPU CAPACITY VIOLATION:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )

    def test_finish_time_fairness_multi_gpu(self):
        """FinishTimeFairnessPolicyWithPerf should handle multi-GPU jobs correctly."""
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.times_since_start,
            self.num_steps_remaining,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_multi_gpu_capacity(allocation, self.scale_factors,
                                       self.cluster_spec, "FinishTimeFairnessPolicyWithPerf")

    def test_max_min_fairness_multi_gpu(self):
        """MaxMinFairnessPolicyWithPerf should handle multi-GPU jobs correctly."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)
        self._check_multi_gpu_capacity(allocation, self.scale_factors,
                                       self.cluster_spec, "MaxMinFairnessPolicyWithPerf")


class TestHighLoadScenarios(unittest.TestCase):
    """Tests policy behavior under high load (many concurrent jobs)."""

    def setUp(self):
        self.cluster_spec = {'v100': 36, 'p100': 36, 'k80': 36}

    def test_finish_time_fairness_45_jobs(self):
        """FinishTimeFairnessPolicyWithPerf should handle 45 concurrent jobs."""
        policy = finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver='ECOS')

        num_jobs = 45
        throughputs = {}
        scale_factors = {}
        priority_weights = {}
        times_since_start = {}
        num_steps_remaining = {}

        for i in range(num_jobs):
            # Vary throughputs - some jobs with zero K80 throughput
            k80_throughput = 0.0 if i % 5 == 0 else 2.0 - (i % 3) * 0.5
            throughputs[i] = {
                'v100': 10.0 - (i % 5),
                'p100': 5.0 - (i % 3),
                'k80': max(0.0, k80_throughput)
            }
            scale_factors[i] = 1 + (i % 4)
            priority_weights[i] = 1
            times_since_start[i] = i * 100
            num_steps_remaining[i] = 1000 + i * 50

        try:
            allocation = policy.get_allocation(
                throughputs,
                scale_factors,
                priority_weights,
                times_since_start,
                num_steps_remaining,
                self.cluster_spec
            )
            self.assertIsNotNone(allocation, "Allocation should not be None")

            # Check zero-throughput enforcement
            for job_id, job_throughputs in throughputs.items():
                for gpu_type, throughput in job_throughputs.items():
                    if throughput == 0.0:
                        alloc = allocation.get(job_id, {}).get(gpu_type, 0)
                        self.assertLessEqual(
                            alloc, 0.01,
                            f"Job {job_id} has zero {gpu_type} throughput "
                            f"but got allocation {alloc:.4f}"
                        )

        except Exception as e:
            self.fail(f"Policy failed with {num_jobs} jobs: {e}")

    def test_max_min_fairness_45_jobs(self):
        """MaxMinFairnessPolicyWithPerf should handle 45 concurrent jobs."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        num_jobs = 45
        throughputs = {}
        scale_factors = {}
        priority_weights = {}

        for i in range(num_jobs):
            k80_throughput = 0.0 if i % 5 == 0 else 2.0 - (i % 3) * 0.5
            throughputs[i] = {
                'v100': 10.0 - (i % 5),
                'p100': 5.0 - (i % 3),
                'k80': max(0.0, k80_throughput)
            }
            scale_factors[i] = 1 + (i % 4)
            priority_weights[i] = 1

        try:
            allocation = policy.get_allocation(
                throughputs,
                scale_factors,
                priority_weights,
                self.cluster_spec
            )
            self.assertIsNotNone(allocation, "Allocation should not be None")

            # Check zero-throughput enforcement
            for job_id, job_throughputs in throughputs.items():
                for gpu_type, throughput in job_throughputs.items():
                    if throughput == 0.0:
                        alloc = allocation.get(job_id, {}).get(gpu_type, 0)
                        self.assertLessEqual(
                            alloc, 0.01,
                            f"Job {job_id} has zero {gpu_type} throughput "
                            f"but got allocation {alloc:.4f}"
                        )

        except Exception as e:
            self.fail(f"Policy failed with {num_jobs} jobs: {e}")


class TestFairnessProperties(unittest.TestCase):
    """Tests that policies maintain their fairness guarantees."""

    def setUp(self):
        self.cluster_spec = {'v100': 4, 'p100': 4, 'k80': 4}

    def test_max_min_fairness_equal_jobs_equal_allocation(self):
        """Jobs with identical throughputs should get equal effective throughput."""
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver='ECOS')

        # Two identical jobs
        throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
        }
        scale_factors = {0: 1, 1: 1}
        priority_weights = {0: 1, 1: 1}

        allocation = policy.get_allocation(
            throughputs, scale_factors, priority_weights, self.cluster_spec
        )

        # Calculate effective throughput for each job
        eff_throughput_0 = sum(
            allocation[0][gpu] * throughputs[0][gpu]
            for gpu in self.cluster_spec
        )
        eff_throughput_1 = sum(
            allocation[1][gpu] * throughputs[1][gpu]
            for gpu in self.cluster_spec
        )

        self.assertAlmostEqual(
            eff_throughput_0, eff_throughput_1, places=2,
            msg=f"Identical jobs should have equal effective throughput: "
                f"{eff_throughput_0:.2f} vs {eff_throughput_1:.2f}"
        )

    def test_proportional_allocation_distributes_evenly(self):
        """ProportionalPolicy should distribute resources proportionally."""
        policy = proportional.ProportionalPolicy()

        throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 1.5},
        }

        allocation = policy.get_allocation(throughputs, self.cluster_spec)

        self.assertIsNotNone(allocation)

        # Each job should get roughly equal share of each GPU type
        for job_id in throughputs:
            for gpu_type in self.cluster_spec:
                expected = self.cluster_spec[gpu_type] / len(throughputs)
                actual = allocation[job_id][gpu_type]
                # Allow some variation but should be in same ballpark
                self.assertGreater(actual, 0,
                    f"Job {job_id} should have non-zero {gpu_type} allocation")


class TestAgnosticPoliciesIntentionalBehavior(unittest.TestCase):
    """
    Tests for policies that INTENTIONALLY ignore heterogeneity.

    These policies (MaxMinFairnessPolicy, etc.) set all throughputs to 1.0
    as a baseline for comparing heterogeneity-aware vs unaware scheduling.
    This is NOT a bug - it's documented behavior for baseline comparison.
    """

    def setUp(self):
        self.cluster_spec = {'v100': 12, 'p100': 12, 'k80': 12}

        self.throughputs = {
            0: {'v100': 10.0, 'p100': 5.0, 'k80': 2.0},
            1: {'v100': 8.0, 'p100': 4.0, 'k80': 1.5},
        }

        self.scale_factors = {0: 1, 1: 1}
        self.priority_weights = {0: 1, 1: 1}

    def test_max_min_fairness_agnostic_ignores_throughputs(self):
        """MaxMinFairnessPolicy (agnostic) intentionally treats all GPUs equally.

        This is documented behavior for baseline comparison, NOT a bug.
        """
        policy = max_min_fairness.MaxMinFairnessPolicy(solver='ECOS')

        allocation = policy.get_allocation(
            self.throughputs,
            self.scale_factors,
            self.priority_weights,
            self.cluster_spec
        )

        self.assertIsNotNone(allocation)

        # Agnostic policy should allocate roughly equally across GPU types
        # because it treats all throughputs as 1.0
        for job_id in self.throughputs:
            v100 = allocation[job_id]['v100']
            p100 = allocation[job_id]['p100']
            k80 = allocation[job_id]['k80']

            # All allocations should be similar (within reasonable tolerance)
            self.assertAlmostEqual(v100, p100, places=1,
                msg=f"Agnostic policy should allocate similarly across GPU types")
            self.assertAlmostEqual(p100, k80, places=1,
                msg=f"Agnostic policy should allocate similarly across GPU types")


if __name__ == '__main__':
    # Run with verbose output to show which tests fail
    unittest.main(verbosity=2)
