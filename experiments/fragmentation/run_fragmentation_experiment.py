#!/usr/bin/env python3
"""
Fragmentation experiment to reproduce FGD paper Figure 7(a).

This experiment measures fragmentation rate as workload arrives, comparing
FGD against baseline policies (Random, BestFit).

The approach:
1. Generate a stream of job arrivals with varying GPU requests (1, 2, 4, 8 GPUs)
2. For each job arrival, compute the placement using each policy
3. After placement, measure the cluster's fragmentation rate
4. Plot fragmentation rate vs arrived workload (% of cluster capacity)
"""

import argparse
import json
import os
import sys
import random
from collections import Counter

# FGD components inlined to avoid cvxpy dependency
# (These are copied from src/scheduler/policies/fgd.py)

class FGDJob:
    """Lightweight job descriptor for FGD fragmentation calculations."""

    def __init__(self, job_id, gpu_request, scale_factor, gpu_type=None):
        self.job_id = job_id
        self.gpu_request = float(gpu_request)
        self.gpu_type = gpu_type
        self.scale_factor = scale_factor


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


def generate_job_sizes(num_jobs, rng, distribution='philly'):
    """Generate job GPU requests based on distribution.

    Philly distribution (from Microsoft Philly trace):
    - 70%: 1 GPU
    - 10%: 2 GPUs
    - 15%: 4 GPUs
    - 5%:  8 GPUs
    """
    sizes = []
    for _ in range(num_jobs):
        r = rng.random()
        if r < 0.70:
            sizes.append(1)
        elif r < 0.80:
            sizes.append(2)
        elif r < 0.95:
            sizes.append(4)
        else:
            sizes.append(8)
    return sizes


def build_nodes(cluster_spec, gpus_per_node=4):
    """Build virtual cluster nodes."""
    nodes = []
    for wt in sorted(cluster_spec.keys()):
        total_gpus = cluster_spec[wt]
        num_full_nodes = total_gpus // gpus_per_node
        remainder = total_gpus % gpus_per_node
        for i in range(num_full_nodes):
            nodes.append(FGDNode(f'{wt}_n{i}', gpus_per_node, wt))
        if remainder > 0:
            nodes.append(FGDNode(f'{wt}_n{num_full_nodes}', remainder, wt))
    return nodes


def build_workload_from_sizes(sizes):
    """Build FGD workload from list of job sizes."""
    wl = FGDWorkload()
    sf_counts = Counter(sizes)
    total = len(sizes)
    for sf, count in sf_counts.items():
        fgd_job = FGDJob(job_id=f'type_sf{sf}', gpu_request=float(sf), scale_factor=sf)
        wl.add_type(sf, fgd_job, float(count) / total)
    return wl


def compute_fragmentation_rate(nodes, workload):
    """Compute cluster-wide fragmentation rate as percentage."""
    total_frag = sum(
        FragmentationCalculator.node_fragmentation_for_workload(n, workload)
        for n in nodes
    )
    total_capacity = sum(sum(n.gpu_capacities) for n in nodes)
    return (total_frag / total_capacity) * 100 if total_capacity > 0 else 0.0


def place_job_random(nodes, gpu_req, rng):
    """Random placement: pick a random node that fits."""
    candidates = []
    for node in nodes:
        if not node.can_fit_job(gpu_req):
            continue
        indices = node.find_suitable_gpus(gpu_req)
        if indices is not None:
            candidates.append((node, indices))
    if candidates:
        node, indices = rng.choice(candidates)
        node.allocate(gpu_req, indices)
        return True
    return False


def place_job_bestfit(nodes, gpu_req):
    """Best-fit placement: pick node with smallest remaining capacity."""
    best_node = None
    best_indices = None
    best_remaining = float('inf')

    for node in nodes:
        if not node.can_fit_job(gpu_req):
            continue
        indices = node.find_suitable_gpus(gpu_req)
        if indices is None:
            continue
        remaining = node.get_gpu_scalar() - gpu_req
        if remaining < best_remaining:
            best_remaining = remaining
            best_node = node
            best_indices = indices

    if best_node is not None:
        best_node.allocate(gpu_req, best_indices)
        return True
    return False


def place_job_fgd(nodes, gpu_req, workload):
    """FGD placement: minimize fragmentation gradient."""
    fgd_job = FGDJob(job_id='temp', gpu_request=gpu_req, scale_factor=gpu_req)

    best_node = None
    best_indices = None
    best_delta = float('inf')

    for node in nodes:
        if not node.can_fit_job(gpu_req):
            continue
        indices = node.find_suitable_gpus(gpu_req)
        if indices is None:
            continue

        # Calculate fragmentation delta
        current = FragmentationCalculator.node_fragmentation_for_workload(node, workload)
        hyp = node.copy()

        if 0 < gpu_req < 1:
            # For partial GPUs, try each candidate
            for idx in indices:
                hyp_copy = node.copy()
                hyp_copy.allocate(gpu_req, [idx])
                after = FragmentationCalculator.node_fragmentation_for_workload(hyp_copy, workload)
                delta = after - current
                if delta < best_delta:
                    best_delta = delta
                    best_node = node
                    best_indices = [idx]
        else:
            hyp.allocate(gpu_req, indices)
            after = FragmentationCalculator.node_fragmentation_for_workload(hyp, workload)
            delta = after - current
            if delta < best_delta:
                best_delta = delta
                best_node = node
                best_indices = indices

    if best_node is not None:
        best_node.allocate(gpu_req, best_indices)
        return True
    return False


class RunningJob:
    """Track a running job for completion simulation."""
    def __init__(self, job_id, gpu_request, node, gpu_indices, duration):
        self.job_id = job_id
        self.gpu_request = gpu_request
        self.node = node
        self.gpu_indices = gpu_indices
        self.remaining_time = duration


def deallocate_job(job):
    """Return GPUs to the node when job completes."""
    if job.gpu_request <= 0:
        return
    if 0 < job.gpu_request < 1:
        job.node.gpu_capacities[job.gpu_indices[0]] += job.gpu_request
    else:
        for idx in job.gpu_indices:
            job.node.gpu_capacities[idx] = 1.0


def run_experiment(cluster_spec, job_sizes, gpus_per_node, policy_name, seed,
                   job_duration_range=(10, 50)):
    """Run placement experiment for one policy with job completions.

    Simulates a time-stepped system where:
    - Jobs arrive one per time step
    - Jobs have random durations and complete over time
    - This creates fragmentation as jobs leave "holes"

    Returns list of (arrived_workload_pct, fragmentation_rate) tuples.
    """
    rng = random.Random(seed)
    nodes = build_nodes(cluster_spec, gpus_per_node)
    total_capacity = sum(cluster_spec.values())

    results = []
    running_jobs = []
    cumulative_gpus = 0

    for i, gpu_req in enumerate(job_sizes):
        # Time step: decrease remaining time for all running jobs
        completed = []
        for job in running_jobs:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                deallocate_job(job)
                completed.append(job)
        for job in completed:
            running_jobs.remove(job)

        # Build workload from job size distribution (global, not just running)
        workload = build_workload_from_sizes(job_sizes)

        # Place the new job
        placed = False
        placed_node = None
        placed_indices = None

        if policy_name == 'random':
            # Find candidates
            candidates = []
            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is not None:
                    candidates.append((node, indices))
            if candidates:
                placed_node, placed_indices = rng.choice(candidates)
                placed_node.allocate(gpu_req, placed_indices)
                placed = True

        elif policy_name == 'bestfit':
            best_remaining = float('inf')
            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is None:
                    continue
                remaining = node.get_gpu_scalar() - gpu_req
                if remaining < best_remaining:
                    best_remaining = remaining
                    placed_node = node
                    placed_indices = indices
            if placed_node is not None:
                placed_node.allocate(gpu_req, placed_indices)
                placed = True

        elif policy_name == 'fgd':
            fgd_job = FGDJob(job_id='temp', gpu_request=gpu_req, scale_factor=gpu_req)
            best_delta = float('inf')
            for node in nodes:
                if not node.can_fit_job(gpu_req):
                    continue
                indices = node.find_suitable_gpus(gpu_req)
                if indices is None:
                    continue
                current = FragmentationCalculator.node_fragmentation_for_workload(node, workload)
                if 0 < gpu_req < 1:
                    for idx in indices:
                        hyp = node.copy()
                        hyp.allocate(gpu_req, [idx])
                        after = FragmentationCalculator.node_fragmentation_for_workload(hyp, workload)
                        delta = after - current
                        if delta < best_delta:
                            best_delta = delta
                            placed_node = node
                            placed_indices = [idx]
                else:
                    hyp = node.copy()
                    hyp.allocate(gpu_req, indices)
                    after = FragmentationCalculator.node_fragmentation_for_workload(hyp, workload)
                    delta = after - current
                    if delta < best_delta:
                        best_delta = delta
                        placed_node = node
                        placed_indices = indices
            if placed_node is not None:
                placed_node.allocate(gpu_req, placed_indices)
                placed = True
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Track running job if placed
        if placed:
            duration = rng.randint(job_duration_range[0], job_duration_range[1])
            running_jobs.append(RunningJob(i, gpu_req, placed_node, placed_indices, duration))

        # Calculate current utilization (GPUs in use / total)
        gpus_in_use = sum(j.gpu_request for j in running_jobs)
        utilization_pct = (gpus_in_use / total_capacity) * 100

        # Compute fragmentation rate
        frag_rate = compute_fragmentation_rate(nodes, workload)

        results.append({
            'job_index': i,
            'gpu_request': gpu_req,
            'placed': placed,
            'utilization_pct': utilization_pct,
            'arrived_workload_pct': utilization_pct,  # Use utilization as X-axis
            'fragmentation_rate': frag_rate,
            'running_jobs': len(running_jobs),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run fragmentation experiment")
    parser.add_argument("--cluster", default="36:36:36", help="Cluster spec V100:P100:K80")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per server node")
    parser.add_argument("--num-jobs", type=int, default=500, help="Number of jobs to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="fragmentation_results", help="Output directory")
    parser.add_argument("--policies", default="fgd,random,bestfit", help="Comma-separated policies")
    parser.add_argument("--job-duration-min", type=int, default=10, help="Min job duration (timesteps)")
    parser.add_argument("--job-duration-max", type=int, default=50, help="Max job duration (timesteps)")
    args = parser.parse_args()

    # Parse cluster spec
    v100s, p100s, k80s = map(int, args.cluster.split(":"))
    cluster_spec = {"v100": v100s, "p100": p100s, "k80": k80s}
    total_gpus = sum(cluster_spec.values())

    print(f"Cluster: {cluster_spec} ({total_gpus} total GPUs)")
    print(f"GPUs per node: {args.gpus_per_node}")
    print(f"Number of jobs: {args.num_jobs}")
    print(f"Seed: {args.seed}")

    # Generate job sizes (same for all policies)
    rng = random.Random(args.seed)
    job_sizes = generate_job_sizes(args.num_jobs, rng)
    total_requested = sum(job_sizes)
    print(f"Total GPU requests: {total_requested} ({100*total_requested/total_gpus:.1f}% of capacity)")
    print(f"Job size distribution: {Counter(job_sizes)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment for each policy
    policies = args.policies.split(',')
    all_results = {}

    job_duration_range = (args.job_duration_min, args.job_duration_max)
    print(f"Job duration range: {job_duration_range} timesteps")

    for policy in policies:
        print(f"\nRunning {policy}...")
        results = run_experiment(
            cluster_spec, job_sizes, args.gpus_per_node, policy, args.seed,
            job_duration_range=job_duration_range
        )
        all_results[policy] = results

        # Save individual policy results
        output_file = os.path.join(args.output_dir, f"{policy}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        final = results[-1]
        print(f"  Final arrived workload: {final['arrived_workload_pct']:.1f}%")
        print(f"  Final fragmentation rate: {final['fragmentation_rate']:.1f}%")
        placed_count = sum(1 for r in results if r['placed'])
        print(f"  Jobs placed: {placed_count}/{len(results)}")

    # Save combined results for plotting
    combined_file = os.path.join(args.output_dir, "combined_results.json")
    with open(combined_file, 'w') as f:
        json.dump({
            'config': {
                'cluster': args.cluster,
                'gpus_per_node': args.gpus_per_node,
                'num_jobs': args.num_jobs,
                'seed': args.seed,
                'total_gpus': total_gpus,
            },
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
