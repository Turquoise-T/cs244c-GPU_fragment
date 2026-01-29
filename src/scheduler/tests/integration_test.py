"""
End-to-end integration tests for scheduler policies.

These tests verify that the simulation produces deterministic results
for fixed configurations. Any deviation indicates an unintended code change.

Run with: python -m unittest tests.integration_test -v
"""

import sys
import os
import unittest
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler
import utils


class TestIntegration(unittest.TestCase):
    """End-to-end sanity checks with known expected outputs."""

    # Expected values for 36:36:36 cluster, 50 jobs, seed=0
    # These are ground truth from a validated run - do not change unless
    # intentionally modifying the algorithm
    EXPECTED_AGNOSTIC_JCT = 73063.449596
    EXPECTED_GAVEL_JCT = 57171.408617

    # Floating-point tolerance (0.01% to account for numerical precision)
    TOLERANCE = 0.0001

    def _run_simulation(self, policy_name):
        """Run a simulation and return average JCT."""
        cluster_spec = {'v100': 36, 'p100': 36, 'k80': 36}
        policy = utils.get_policy(policy_name, solver='ECOS', seed=0)

        throughputs_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'simulation_throughputs.json'
        )

        sched = scheduler.Scheduler(
            policy,
            throughputs_file=throughputs_file,
            seed=0,
            time_per_iteration=360,
            simulate=True,
            profiling_percentage=1.0,
            num_reference_models=26
        )

        sched.simulate(
            cluster_spec=cluster_spec,
            lam=0.0,  # All jobs added at start (static workload)
            num_total_jobs=50
        )

        return sched.get_average_jct()

    def test_agnostic_policy_deterministic(self):
        """Verify agnostic policy produces expected JCT."""
        jct = self._run_simulation('max_min_fairness')

        relative_error = abs(jct - self.EXPECTED_AGNOSTIC_JCT) / self.EXPECTED_AGNOSTIC_JCT
        self.assertLess(
            relative_error,
            self.TOLERANCE,
            f"Agnostic JCT {jct:.2f} differs from expected {self.EXPECTED_AGNOSTIC_JCT:.2f} "
            f"by {relative_error:.4%} (tolerance: {self.TOLERANCE:.4%})"
        )

    def test_gavel_policy_deterministic(self):
        """Verify Gavel policy produces expected JCT."""
        jct = self._run_simulation('max_min_fairness_perf')

        relative_error = abs(jct - self.EXPECTED_GAVEL_JCT) / self.EXPECTED_GAVEL_JCT
        self.assertLess(
            relative_error,
            self.TOLERANCE,
            f"Gavel JCT {jct:.2f} differs from expected {self.EXPECTED_GAVEL_JCT:.2f} "
            f"by {relative_error:.4%} (tolerance: {self.TOLERANCE:.4%})"
        )

    def test_gavel_beats_agnostic(self):
        """Verify Gavel produces lower JCT than agnostic."""
        # This test ensures the fundamental property holds
        agnostic_jct = self._run_simulation('max_min_fairness')
        gavel_jct = self._run_simulation('max_min_fairness_perf')

        self.assertLess(
            gavel_jct,
            agnostic_jct,
            f"Gavel JCT {gavel_jct:.2f} should be less than agnostic {agnostic_jct:.2f}"
        )


if __name__ == '__main__':
    unittest.main()
