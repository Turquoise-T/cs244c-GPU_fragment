"""Tests for FGD placement integration with Gavel.

Validates:
  - Node construction from Gavel's worker topology
  - No double-assignment of workers
  - Fragmentation tracking
  - FGD vs strided placement produces valid assignments
"""

import collections
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import job_id_pair
from fgd_placement import GavelFGDPlacement, build_fgd_workload


class FakeJobId:
    """Minimal job_id that quacks like JobIdPair for testing."""
    def __init__(self, jid):
        self._id = jid

    def singletons(self):
        return [self]

    def is_pair(self):
        return False

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return isinstance(other, FakeJobId) and self._id == other._id

    def __repr__(self):
        return f'FakeJobId({self._id})'

    def __str__(self):
        return str(self._id)


class FakeJob:
    def __init__(self, scale_factor, gpu_request=None):
        self.scale_factor = scale_factor
        self._gpu_request = gpu_request


class TestGavelFGDPlacement(unittest.TestCase):
    """Test FGD placement on a 4:4:4 cluster (3 server types, 4 GPUs each)."""

    def _make_4x4x4_topology(self):
        """3 servers, 4 workers (GPUs) each = 12 workers total.
        Returns worker_ids_by_server (list of lists)."""
        return [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]

    def test_single_job_placement(self):
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        job_id = FakeJobId('job-0')
        jobs_dict = {job_id: FakeJob(scale_factor=2)}

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(job_id, 2)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        self.assertIn(job_id, assignments)
        self.assertEqual(len(assignments[job_id]), 2)
        # Workers should be from the same server
        wids = assignments[job_id]
        self.assertTrue(
            all(w in range(0, 4) for w in wids) or
            all(w in range(4, 8) for w in wids) or
            all(w in range(8, 12) for w in wids),
            f"Workers {wids} should all be from the same server"
        )

    def test_no_double_assignment(self):
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jobs_dict = {}
        scheduled = []
        for i in range(6):
            jid = FakeJobId(f'job-{i}')
            jobs_dict[jid] = FakeJob(scale_factor=2)
            scheduled.append((jid, 2))

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=scheduled,
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        # Check no worker assigned twice
        all_assigned_workers = []
        for job_id, wids in assignments.items():
            for wid in wids:
                self.assertNotIn(wid, all_assigned_workers,
                                 f"Worker {wid} assigned to multiple jobs")
                all_assigned_workers.append(wid)

    def test_respects_lease_extensions(self):
        """Jobs already in worker_assignments should not be re-placed."""
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = {0, 1}  # Workers 0,1 already taken
        assignments = collections.OrderedDict()

        # Pre-existing assignment (lease extension)
        existing_jid = FakeJobId('job-existing')
        assignments[existing_jid] = (0, 1)

        new_jid = FakeJobId('job-new')
        jobs_dict = {new_jid: FakeJob(scale_factor=2)}

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(new_jid, 2)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        self.assertIn(new_jid, assignments)
        new_wids = assignments[new_jid]
        # New job should not use workers 0, 1
        for wid in new_wids:
            self.assertNotIn(wid, [0, 1])

    def test_fragmentation_returned(self):
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jid = FakeJobId('job-0')
        jobs_dict = {jid: FakeJob(scale_factor=1)}

        frag = placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(jid, 1)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        # Fragmentation should be a non-negative number
        self.assertGreaterEqual(frag, 0.0)

    def test_fill_cluster_completely(self):
        """12 single-GPU jobs should fill all 12 workers."""
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jobs_dict = {}
        scheduled = []
        for i in range(12):
            jid = FakeJobId(f'job-{i}')
            jobs_dict[jid] = FakeJob(scale_factor=1)
            scheduled.append((jid, 1))

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=scheduled,
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        self.assertEqual(len(assignments), 12)
        all_workers = set()
        for wids in assignments.values():
            for w in wids:
                all_workers.add(w)
        self.assertEqual(len(all_workers), 12)

    def test_bestfit_mode(self):
        placement = GavelFGDPlacement(placement_mode='bestfit')
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jid = FakeJobId('job-0')
        jobs_dict = {jid: FakeJob(scale_factor=4)}

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(jid, 4)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        self.assertIn(jid, assignments)
        self.assertEqual(len(assignments[jid]), 4)

    def test_random_mode(self):
        placement = GavelFGDPlacement(placement_mode='random')
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jid = FakeJobId('job-0')
        jobs_dict = {jid: FakeJob(scale_factor=2)}

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(jid, 2)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        self.assertIn(jid, assignments)
        self.assertEqual(len(assignments[jid]), 2)

    def test_oversized_job_not_placed(self):
        """An 8-GPU job on 4-GPU servers cannot be placed (single-node semantics)."""
        placement = GavelFGDPlacement()
        worker_ids = self._make_4x4x4_topology()
        assigned = set()
        assignments = collections.OrderedDict()

        jid = FakeJobId('job-big')
        jobs_dict = {jid: FakeJob(scale_factor=8)}

        placement.assign_workers_for_round(
            scheduled_jobs_for_type=[(jid, 8)],
            worker_ids_by_server=worker_ids,
            assigned_worker_ids=assigned,
            worker_assignments=assignments,
            jobs_dict=jobs_dict,
        )

        # Job needs 8 GPUs but each server only has 4 -- placement fails
        self.assertNotIn(jid, assignments)


class TestBuildWorkload(unittest.TestCase):
    def test_philly_workload(self):
        workload = build_fgd_workload('philly')
        self.assertEqual(len(workload.tasks), 4)
        total_pop = sum(workload.popularity.values())
        self.assertAlmostEqual(total_pop, 1.0)


if __name__ == '__main__':
    unittest.main()
