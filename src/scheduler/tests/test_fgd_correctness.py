#!/usr/bin/env python3
"""
Comprehensive correctness tests for FGD (Fragmentation Gradient Descent).

Verification strategy:
  1. Unit tests         — hand-computed expected values for every core function
  2. Invariant tests    — mathematical properties that must always hold
  3. Scenario tests     — known-optimal placement decisions
  4. Comparison tests   — FGD vs baselines on fragmentation metric
  5. Stress tests       — random workloads, invariants must not break

Run:
    python -m pytest tests/test_fgd_correctness.py -v
    # or
    python tests/test_fgd_correctness.py
"""

import os
import sys
import copy
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "policies"))

from policies.fgd import (
    GPUState,
    compute_gpu_fragmentation,
    compute_fragmentation_increment_gpu,
    fgd_select_gpu,
    fgd_select_gpus_multi,
    bestfit_select_gpu,
    worstfit_select_gpu,
    firstfit_select_gpu,
    GPUSharingCluster,
)


# =========================================================================
# Helper
# =========================================================================
def _make_gpu(gpu_id, used_milli, server_id=0):
    return GPUState(
        gpu_id=gpu_id,
        server_id=server_id,
        total_milli=1000,
        used_milli=used_milli,
        job_assignments=[],
    )


# =========================================================================
# 1. Unit Tests — GPUState properties
# =========================================================================
class TestGPUStateProperties(unittest.TestCase):
    """Verify GPUState dataclass computed properties."""

    def test_empty_gpu(self):
        g = _make_gpu(0, 0)
        self.assertTrue(g.is_empty)
        self.assertFalse(g.is_full)
        self.assertFalse(g.is_partial)
        self.assertEqual(g.free_milli, 1000)
        self.assertAlmostEqual(g.free_fraction, 1.0)
        self.assertAlmostEqual(g.used_fraction, 0.0)
        self.assertTrue(g.can_fit(1000))
        self.assertFalse(g.can_fit(1001))

    def test_full_gpu(self):
        g = _make_gpu(0, 1000)
        self.assertFalse(g.is_empty)
        self.assertTrue(g.is_full)
        self.assertFalse(g.is_partial)
        self.assertEqual(g.free_milli, 0)
        self.assertFalse(g.can_fit(1))

    def test_partial_gpu(self):
        g = _make_gpu(0, 500)
        self.assertFalse(g.is_empty)
        self.assertFalse(g.is_full)
        self.assertTrue(g.is_partial)
        self.assertEqual(g.free_milli, 500)
        self.assertAlmostEqual(g.free_fraction, 0.5)
        self.assertTrue(g.can_fit(500))
        self.assertFalse(g.can_fit(501))

    def test_boundary_values(self):
        g = _make_gpu(0, 1)
        self.assertTrue(g.is_partial)
        self.assertEqual(g.free_milli, 999)

        g = _make_gpu(0, 999)
        self.assertTrue(g.is_partial)
        self.assertEqual(g.free_milli, 1)
        self.assertTrue(g.can_fit(1))
        self.assertFalse(g.can_fit(2))


# =========================================================================
# 2. Unit Tests — compute_gpu_fragmentation
# =========================================================================
class TestComputeFragmentation(unittest.TestCase):

    def test_all_empty(self):
        gpus = [_make_gpu(i, 0) for i in range(4)]
        self.assertAlmostEqual(compute_gpu_fragmentation(gpus), 0.0)

    def test_all_full(self):
        gpus = [_make_gpu(i, 1000) for i in range(4)]
        self.assertAlmostEqual(compute_gpu_fragmentation(gpus), 0.0)

    def test_single_partial(self):
        gpus = [_make_gpu(0, 300)]
        # free_fraction = 700/1000 = 0.7
        self.assertAlmostEqual(compute_gpu_fragmentation(gpus), 0.7)

    def test_mixed(self):
        gpus = [
            _make_gpu(0, 0),     # empty  → 0
            _make_gpu(1, 500),   # partial → 0.5
            _make_gpu(2, 1000),  # full   → 0
            _make_gpu(3, 700),   # partial → 0.3
        ]
        self.assertAlmostEqual(compute_gpu_fragmentation(gpus), 0.8)

    def test_all_partial(self):
        gpus = [_make_gpu(i, 200 * (i + 1)) for i in range(4)]
        # used: 200, 400, 600, 800  → free: 800, 600, 400, 200
        # frag: 0.8 + 0.6 + 0.4 + 0.2 = 2.0
        self.assertAlmostEqual(compute_gpu_fragmentation(gpus), 2.0)


# =========================================================================
# 3. Unit Tests — compute_fragmentation_increment_gpu (Δfrag)
# =========================================================================
class TestDeltaFrag(unittest.TestCase):
    """
    Key mathematical property of Δfrag:
      - For a partial GPU allocating M milli: Δfrag = -M/1000 (always negative)
      - For an empty GPU allocating M < 1000: Δfrag = (1000-M)/1000 (positive)
      - For an empty GPU allocating M = 1000: Δfrag = 0
      - Can't fit → inf
    """

    def test_cant_fit(self):
        g = _make_gpu(0, 800)  # 200 free
        self.assertEqual(compute_fragmentation_increment_gpu(g, 300), float("inf"))

    def test_empty_to_partial(self):
        g = _make_gpu(0, 0)
        delta = compute_fragmentation_increment_gpu(g, 300)
        # old_frag=0, new_frag=700/1000=0.7 → Δ=0.7
        self.assertAlmostEqual(delta, 0.7)

    def test_empty_to_full(self):
        g = _make_gpu(0, 0)
        delta = compute_fragmentation_increment_gpu(g, 1000)
        # old_frag=0, new_frag=0 → Δ=0
        self.assertAlmostEqual(delta, 0.0)

    def test_partial_to_full(self):
        g = _make_gpu(0, 700)  # 300 free
        delta = compute_fragmentation_increment_gpu(g, 300)
        # old_frag=0.3, new_frag=0 → Δ=-0.3
        self.assertAlmostEqual(delta, -0.3)

    def test_partial_to_partial(self):
        g = _make_gpu(0, 500)  # 500 free
        delta = compute_fragmentation_increment_gpu(g, 200)
        # old_frag=0.5, new_frag=300/1000=0.3 → Δ=-0.2 = -200/1000
        self.assertAlmostEqual(delta, -0.2)

    def test_partial_delta_always_minus_m_over_1000(self):
        """For any partial GPU, Δfrag = -M/1000 regardless of GPU state."""
        for used in [100, 300, 500, 700, 900]:
            free = 1000 - used
            for m in range(100, free + 1, 100):
                g = _make_gpu(0, used)
                delta = compute_fragmentation_increment_gpu(g, m)
                expected = -m / 1000.0
                self.assertAlmostEqual(
                    delta, expected,
                    msg=f"used={used}, alloc={m}: got {delta}, expected {expected}",
                )

    def test_empty_delta_formula(self):
        """For empty GPU allocating M: Δfrag = (1000-M)/1000 if M<1000, else 0."""
        for m in range(100, 1100, 100):
            g = _make_gpu(0, 0)
            delta = compute_fragmentation_increment_gpu(g, m)
            expected = (1000 - m) / 1000.0 if m < 1000 else 0.0
            self.assertAlmostEqual(
                delta, expected,
                msg=f"empty, alloc={m}: got {delta}, expected {expected}",
            )

    def test_full_gpu_cant_fit_anything(self):
        g = _make_gpu(0, 1000)
        self.assertEqual(compute_fragmentation_increment_gpu(g, 1), float("inf"))


# =========================================================================
# 4. Invariant Tests — FGD selection properties
# =========================================================================
class TestFGDInvariants(unittest.TestCase):
    """Properties that must ALWAYS hold for fgd_select_gpu."""

    def test_never_selects_gpu_that_cant_fit(self):
        """FGD never returns a GPU without enough capacity."""
        gpus = [_make_gpu(0, 900), _make_gpu(1, 800), _make_gpu(2, 700)]
        # needs 200: only GPU 1 (200 free) and GPU 2 (300 free) can fit
        result = fgd_select_gpu(gpus, 200)
        self.assertIn(result, [1, 2])

    def test_returns_none_when_nothing_fits(self):
        gpus = [_make_gpu(i, 900) for i in range(3)]
        self.assertIsNone(fgd_select_gpu(gpus, 200))

    def test_partial_always_preferred_over_empty(self):
        """
        Core FGD property: placing on a partial GPU gives Δfrag = -M/1000 < 0,
        while placing on an empty GPU gives Δfrag = (1000-M)/1000 > 0 (for M<1000).
        So FGD must always prefer partial GPUs.
        """
        for m in [100, 200, 300, 500, 700, 900]:
            gpus = [
                _make_gpu(0, 0),     # empty
                _make_gpu(1, 500),   # partial with 500 free
            ]
            if m <= 500:
                result = fgd_select_gpu(gpus, m)
                self.assertEqual(
                    result, 1,
                    msg=f"alloc={m}: FGD should prefer partial GPU 1 over empty GPU 0",
                )

    def test_prefers_filling_gpu_completely(self):
        """When a partial GPU can be exactly filled, prefer it (Δfrag most negative)."""
        gpus = [
            _make_gpu(0, 500),   # 500 free. Allocating 300 → Δfrag=-0.3
            _make_gpu(1, 700),   # 300 free. Allocating 300 → fills it, Δfrag=-0.3
        ]
        # Δfrag is the same (-0.3), tie-break by less free → GPU 1 (300 < 500)
        result = fgd_select_gpu(gpus, 300)
        self.assertEqual(result, 1)

    def test_tiebreak_by_less_free_space(self):
        """
        When Δfrag is equal, FGD should prefer the GPU with less free space.
        For two partial GPUs with the same allocation, Δfrag = -M/1000 for both.
        """
        gpus = [
            _make_gpu(0, 300),   # 700 free
            _make_gpu(1, 600),   # 400 free
            _make_gpu(2, 800),   # 200 free
        ]
        result = fgd_select_gpu(gpus, 200)
        # All can fit, all partial → Δfrag = -0.2 for each
        # Tie-break: least free → GPU 2 (200 free)
        self.assertEqual(result, 2)

    def test_single_gpu_available(self):
        gpus = [_make_gpu(0, 0)]
        self.assertEqual(fgd_select_gpu(gpus, 500), 0)

    def test_all_full_except_one(self):
        gpus = [_make_gpu(i, 1000) for i in range(3)]
        gpus.append(_make_gpu(3, 200))
        self.assertEqual(fgd_select_gpu(gpus, 500), 3)


# =========================================================================
# 5. Scenario Tests — known-optimal placement sequences
# =========================================================================
class TestFGDScenarios(unittest.TestCase):

    def test_pack_three_jobs_on_one_gpu(self):
        """Three 300-milli jobs should pack onto one GPU, not spread."""
        cluster = GPUSharingCluster(1, 4)  # 1 server, 4 GPUs
        gpu_ids_1 = cluster.place_job(job_id=0, gpu_milli=300, strategy="fgd")
        gpu_ids_2 = cluster.place_job(job_id=1, gpu_milli=300, strategy="fgd")
        gpu_ids_3 = cluster.place_job(job_id=2, gpu_milli=300, strategy="fgd")

        # All three should be on the same GPU (tight packing)
        self.assertEqual(gpu_ids_1, gpu_ids_2)
        self.assertEqual(gpu_ids_2, gpu_ids_3)

    def test_fill_then_next(self):
        """Fill one GPU, then move to the next."""
        cluster = GPUSharingCluster(1, 4)
        ids = []
        for i in range(5):
            result = cluster.place_job(job_id=i, gpu_milli=500, strategy="fgd")
            ids.append(result[0])

        # 2 jobs per GPU: (0,1) on GPU-X, (2,3) on GPU-Y, (4) on GPU-Z
        self.assertEqual(ids[0], ids[1])       # first pair on same GPU
        self.assertEqual(ids[2], ids[3])       # second pair on same GPU
        self.assertNotEqual(ids[0], ids[2])    # different pairs on different GPUs
        self.assertNotEqual(ids[4], ids[0])    # fifth job on a third GPU
        self.assertNotEqual(ids[4], ids[2])

    def test_perfect_packing_no_fragmentation(self):
        """Jobs that perfectly fill GPUs should leave zero fragmentation."""
        cluster = GPUSharingCluster(1, 4)
        for i in range(8):
            cluster.place_job(job_id=i, gpu_milli=500, strategy="fgd")

        self.assertAlmostEqual(cluster.get_fragmentation(), 0.0)
        self.assertAlmostEqual(cluster.get_utilization(), 1.0)

    def test_worstfit_creates_more_fragmentation(self):
        """Worstfit (strided) should create more fragmentation than FGD."""
        jobs = [(i, 300) for i in range(6)]

        cluster_fgd = GPUSharingCluster(1, 4)
        cluster_wf = GPUSharingCluster(1, 4)

        for job_id, milli in jobs:
            cluster_fgd.place_job(job_id=job_id, gpu_milli=milli, strategy="fgd")
            cluster_wf.place_job(job_id=job_id, gpu_milli=milli, strategy="worstfit")

        self.assertLessEqual(
            cluster_fgd.get_fragmentation(),
            cluster_wf.get_fragmentation(),
            "FGD should produce ≤ fragmentation than worstfit",
        )

    def test_removal_and_repack(self):
        """After removing a job, FGD should reuse the freed space."""
        cluster = GPUSharingCluster(1, 2)
        cluster.place_job(0, 500, strategy="fgd")
        cluster.place_job(1, 500, strategy="fgd")  # GPU 0 now full
        cluster.place_job(2, 500, strategy="fgd")   # goes to GPU 1

        cluster.remove_job(1)  # GPU 0 now has 500 free (partial)
        result = cluster.place_job(3, 300, strategy="fgd")
        # Should pick GPU 0 (partial, 500 free) over GPU 1 (partial, 500 free)
        # Tie-break might go either way, but should NOT use a new empty GPU
        # (there is no empty GPU in a 2-GPU cluster)
        self.assertIsNotNone(result)

    def test_heterogeneous_job_sizes(self):
        """Mix of job sizes should be packed correctly."""
        cluster = GPUSharingCluster(1, 4)
        placements = {}
        job_sizes = [700, 300, 500, 500, 200, 200, 100]
        for i, milli in enumerate(job_sizes):
            result = cluster.place_job(job_id=i, gpu_milli=milli, strategy="fgd")
            if result is not None:
                placements[i] = result[0]

        # Job 0: 700 → GPU A (first available)
        # Job 1: 300 → GPU A (fills it: 700+300=1000) ← FGD prefers this
        # Job 2: 500 → GPU B
        # Job 3: 500 → GPU B (fills it)
        # Job 4: 200 → GPU C
        # Job 5: 200 → GPU C (packs with job 4)
        # Job 6: 100 → GPU C (packs further: 200+200+100=500)

        # Verify packing: jobs 0 and 1 should be on the same GPU
        self.assertEqual(placements[0], placements[1],
                         "700+300 should pack on one GPU")
        self.assertEqual(placements[2], placements[3],
                         "500+500 should pack on one GPU")
        self.assertNotEqual(placements[0], placements[2],
                            "Different size groups on different GPUs")


# =========================================================================
# 6. Comparison Tests — FGD vs all baselines
# =========================================================================
class TestFGDVsBaselines(unittest.TestCase):
    """FGD should produce ≤ fragmentation than naive baselines on known inputs."""

    def _run_placement_sequence(self, strategy, num_servers, gpus_per_server, jobs):
        cluster = GPUSharingCluster(num_servers, gpus_per_server)
        for job_id, milli in jobs:
            cluster.place_job(job_id=job_id, gpu_milli=milli, strategy=strategy)
        return cluster.get_fragmentation()

    def test_fgd_le_worstfit_varied_loads(self):
        """FGD fragmentation ≤ worstfit across varied workloads."""
        workloads = [
            [(i, 300) for i in range(10)],
            [(i, 500) for i in range(8)],
            [(i, s) for i, s in enumerate([200, 300, 500, 700, 200, 300, 500])],
            [(i, s) for i, s in enumerate([100, 100, 100, 100, 100, 800, 200])],
        ]
        for jobs in workloads:
            fgd_frag = self._run_placement_sequence("fgd", 2, 4, jobs)
            wf_frag = self._run_placement_sequence("worstfit", 2, 4, jobs)
            self.assertLessEqual(
                fgd_frag, wf_frag + 1e-9,
                f"FGD ({fgd_frag:.3f}) should be ≤ worstfit ({wf_frag:.3f})",
            )

    def test_fgd_le_firstfit(self):
        """FGD fragmentation ≤ firstfit on a pathological case."""
        jobs = [(i, s) for i, s in enumerate([200, 800, 300, 700, 500, 500])]
        fgd_frag = self._run_placement_sequence("fgd", 2, 4, jobs)
        ff_frag = self._run_placement_sequence("firstfit", 2, 4, jobs)
        self.assertLessEqual(fgd_frag, ff_frag + 1e-9)

    def test_fgd_equals_bestfit_on_uniform_jobs(self):
        """For uniform job sizes, FGD and bestfit should behave similarly."""
        jobs = [(i, 500) for i in range(6)]
        fgd_frag = self._run_placement_sequence("fgd", 1, 4, jobs)
        bf_frag = self._run_placement_sequence("bestfit", 1, 4, jobs)
        self.assertAlmostEqual(fgd_frag, bf_frag, places=5)


# =========================================================================
# 7. Multi-GPU selection tests
# =========================================================================
class TestFGDMultiGPU(unittest.TestCase):

    def test_multi_gpu_basic(self):
        gpus = [_make_gpu(i, 0) for i in range(4)]
        result = fgd_select_gpus_multi(gpus, 500, 2)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_multi_gpu_not_enough(self):
        gpus = [_make_gpu(i, 600) for i in range(4)]  # 400 free each
        result = fgd_select_gpus_multi(gpus, 500, 2)
        self.assertIsNone(result)

    def test_multi_gpu_unique_selection(self):
        """Multi-GPU should not select the same GPU twice (unless capacity allows)."""
        gpus = [_make_gpu(i, 0) for i in range(4)]
        result = fgd_select_gpus_multi(gpus, 1000, 3)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(set(result)), 3, "All selected GPUs should be unique")


# =========================================================================
# 8. GPUSharingCluster integration tests
# =========================================================================
class TestGPUSharingCluster(unittest.TestCase):

    def test_utilization_tracking(self):
        cluster = GPUSharingCluster(1, 4)
        self.assertAlmostEqual(cluster.get_utilization(), 0.0)

        cluster.place_job(0, 500, strategy="fgd")
        self.assertAlmostEqual(cluster.get_utilization(), 500 / 4000)

        cluster.place_job(1, 500, strategy="fgd")
        self.assertAlmostEqual(cluster.get_utilization(), 1000 / 4000)

    def test_remove_job_frees_capacity(self):
        cluster = GPUSharingCluster(1, 2)
        cluster.place_job(0, 1000, strategy="fgd")
        cluster.place_job(1, 1000, strategy="fgd")

        # Cluster full
        result = cluster.place_job(2, 100, strategy="fgd")
        self.assertIsNone(result)

        cluster.remove_job(0)
        result = cluster.place_job(2, 100, strategy="fgd")
        self.assertIsNotNone(result)

    def test_remove_nonexistent_job_is_safe(self):
        cluster = GPUSharingCluster(1, 2)
        cluster.remove_job(999)  # should not raise

    def test_capacity_never_exceeded(self):
        """No GPU should ever have used_milli > total_milli."""
        cluster = GPUSharingCluster(2, 4)
        for i in range(30):
            milli = random.choice([200, 300, 500, 700])
            cluster.place_job(i, milli, strategy="fgd")
        for gpu in cluster.gpu_states:
            self.assertLessEqual(
                gpu.used_milli, gpu.total_milli,
                f"GPU {gpu.gpu_id}: used={gpu.used_milli} > total={gpu.total_milli}",
            )


# =========================================================================
# 9. Stress / Property-Based Tests
# =========================================================================
class TestFGDStress(unittest.TestCase):
    """Random workloads — verify invariants hold under diverse conditions."""

    def _random_workload(self, rng, num_jobs):
        sizes = [200, 300, 500, 700, 1000]
        return [(i, rng.choice(sizes)) for i in range(num_jobs)]

    def test_fgd_never_exceeds_capacity_random(self):
        """Under random placement, no GPU capacity is ever exceeded."""
        rng = random.Random(42)
        for trial in range(20):
            cluster = GPUSharingCluster(2, 4)
            jobs = self._random_workload(rng, 50)
            for job_id, milli in jobs:
                cluster.place_job(job_id, milli, strategy="fgd")
            for gpu in cluster.gpu_states:
                self.assertLessEqual(gpu.used_milli, gpu.total_milli)

    def test_fgd_fragmentation_le_worstfit_random(self):
        """FGD fragmentation ≤ worstfit across 50 random trials."""
        rng = random.Random(123)
        fgd_wins = 0
        ties = 0
        for trial in range(50):
            jobs = self._random_workload(rng, 20)
            cluster_fgd = GPUSharingCluster(2, 4)
            cluster_wf = GPUSharingCluster(2, 4)
            for job_id, milli in jobs:
                cluster_fgd.place_job(job_id, milli, strategy="fgd")
                cluster_wf.place_job(job_id, milli, strategy="worstfit")

            fgd_frag = cluster_fgd.get_fragmentation()
            wf_frag = cluster_wf.get_fragmentation()
            self.assertLessEqual(
                fgd_frag, wf_frag + 1e-9,
                f"Trial {trial}: FGD ({fgd_frag:.4f}) > worstfit ({wf_frag:.4f})",
            )
            if fgd_frag < wf_frag - 1e-9:
                fgd_wins += 1
            else:
                ties += 1

        # FGD should be strictly better at least sometimes
        self.assertGreater(fgd_wins, 0, "FGD should beat worstfit on some trials")

    def test_fragmentation_decreases_as_gpus_fill(self):
        """
        As we add perfectly sized jobs to fill GPUs, cluster fragmentation
        should trend toward zero.
        """
        cluster = GPUSharingCluster(1, 4)
        # Fill each GPU with 500+500
        for i in range(8):
            cluster.place_job(i, 500, strategy="fgd")

        self.assertAlmostEqual(cluster.get_fragmentation(), 0.0)
        self.assertAlmostEqual(cluster.get_utilization(), 1.0)

    def test_add_remove_cycle_consistent(self):
        """Adding and removing jobs should leave cluster in consistent state."""
        rng = random.Random(77)
        cluster = GPUSharingCluster(2, 4)
        active_jobs = {}

        for step in range(100):
            if rng.random() < 0.6 or len(active_jobs) == 0:
                job_id = step
                milli = rng.choice([200, 300, 500])
                result = cluster.place_job(job_id, milli, strategy="fgd")
                if result is not None:
                    active_jobs[job_id] = milli
            else:
                job_id = rng.choice(list(active_jobs.keys()))
                cluster.remove_job(job_id)
                del active_jobs[job_id]

            # Invariant: total used across GPUs == sum of active job sizes
            total_used = sum(g.used_milli for g in cluster.gpu_states)
            expected_used = sum(active_jobs.values())
            self.assertEqual(
                total_used, expected_used,
                f"Step {step}: cluster used={total_used} ≠ active_jobs sum={expected_used}",
            )

            # Invariant: no GPU exceeds capacity
            for gpu in cluster.gpu_states:
                self.assertLessEqual(gpu.used_milli, gpu.total_milli)
                self.assertGreaterEqual(gpu.used_milli, 0)


# =========================================================================
# 10. Delta-frag drives global fragmentation down
# =========================================================================
class TestDeltaFragGlobal(unittest.TestCase):
    """
    Verify that Δfrag per-GPU correctly predicts the change in
    cluster-wide fragmentation.
    """

    def test_delta_frag_predicts_cluster_change(self):
        """
        For any single placement, the change in compute_gpu_fragmentation
        should equal the Δfrag of the selected GPU.
        """
        scenarios = [
            ([0, 0, 0, 0], 300),        # all empty
            ([500, 0, 0, 0], 300),       # one partial
            ([700, 300, 0, 0], 300),     # two partial
            ([800, 500, 200, 0], 200),   # three partial
        ]
        for used_list, milli in scenarios:
            gpus = [_make_gpu(i, u) for i, u in enumerate(used_list)]
            old_frag = compute_gpu_fragmentation(gpus)

            selected = fgd_select_gpu(gpus, milli)
            self.assertIsNotNone(selected)

            expected_delta = compute_fragmentation_increment_gpu(gpus[selected], milli)

            # Simulate the placement
            gpus[selected].used_milli += milli
            new_frag = compute_gpu_fragmentation(gpus)

            actual_delta = new_frag - old_frag
            self.assertAlmostEqual(
                actual_delta, expected_delta, places=6,
                msg=f"used={used_list}, alloc={milli}: "
                    f"actual Δ={actual_delta}, predicted Δ={expected_delta}",
            )

    def test_fgd_minimizes_delta_frag(self):
        """FGD selection matches the GPU with true minimum Δfrag."""
        gpus = [
            _make_gpu(0, 0),     # empty
            _make_gpu(1, 300),   # partial, 700 free
            _make_gpu(2, 700),   # partial, 300 free
            _make_gpu(3, 1000),  # full
        ]
        milli = 200

        deltas = {}
        for gpu in gpus:
            deltas[gpu.gpu_id] = compute_fragmentation_increment_gpu(gpu, milli)

        selected = fgd_select_gpu(gpus, milli)
        # GPU 3 can't fit (inf), GPU 0 has delta 0.8, GPU 1 has delta -0.2, GPU 2 has delta -0.2
        # Tie-break: GPU 2 (300 free < 700 free)
        self.assertEqual(selected, 2)

        # Verify it's the minimum possible delta
        min_delta = min(deltas.values())
        self.assertAlmostEqual(deltas[selected], min_delta)


# =========================================================================
# 11. Edge Cases
# =========================================================================
class TestEdgeCases(unittest.TestCase):

    def test_zero_milli_job(self):
        """A job requesting 0 milli should fit anywhere."""
        gpus = [_make_gpu(0, 1000)]  # full GPU
        # 0 milli should fit even on a full GPU
        self.assertTrue(gpus[0].can_fit(0))

    def test_exactly_1000_milli_job(self):
        gpus = [_make_gpu(0, 0), _make_gpu(1, 500)]
        result = fgd_select_gpu(gpus, 1000)
        # Only GPU 0 can fit (1000 free), GPU 1 only has 500
        self.assertEqual(result, 0)

    def test_single_server_single_gpu(self):
        cluster = GPUSharingCluster(1, 1)
        r1 = cluster.place_job(0, 500, strategy="fgd")
        self.assertEqual(r1, [0])
        r2 = cluster.place_job(1, 500, strategy="fgd")
        self.assertEqual(r2, [0])
        r3 = cluster.place_job(2, 100, strategy="fgd")
        self.assertIsNone(r3)  # full

    def test_large_cluster(self):
        """Verify FGD works on a larger cluster without errors."""
        cluster = GPUSharingCluster(10, 8)  # 80 GPUs
        for i in range(200):
            milli = [200, 300, 500][i % 3]
            cluster.place_job(i, milli, strategy="fgd")
        for gpu in cluster.gpu_states:
            self.assertLessEqual(gpu.used_milli, gpu.total_milli)


if __name__ == "__main__":
    unittest.main(verbosity=2)
