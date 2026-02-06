#!/usr/bin/env python3
"""
Fragmentation sweep experiment to reproduce FGD paper Figure 7(a).

This experiment measures fragmentation rate at different load levels by:
1. Generating a batch of jobs that represents X% of cluster capacity
2. Placing all jobs using each policy
3. Measuring the resulting fragmentation rate
4. Repeating for different load levels (10%, 20%, ... 100%+)

This matches the FGD paper's methodology where X-axis is "arrived workloads
as % of cluster GPU capacity".
"""

import argparse
import json
import os
import sys
import random
from collections import Counter


# FGD components inlined to avoid cvxpy dependency
class FGDJob:
    def __init__(self, job_id, gpu_request, scale_factor, gpu_type=None):
        self.job_id = job_id
        self.gpu_request = float(gpu_request)
        self.gpu_type = gpu_type
        self.scale_factor = scale_factor


class FGDNode:
    def __init__(self, node_id, num_gpus, gpu_type):
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type
        self.gpu_capacities = [1.0] * num_gpus

    def get_gpu_scalar(self):
        full = sum(1 for c in self.gpu_capacities if c == 1.0)
        partials = [c for c in self.gpu_capacities if 0 < c < 1.0]
        return full + (max(partials) if partials else 0.0)

    def can_fit_job(self, gpu_request):
        return self.get_gpu_scalar() >= gpu_request

    def find_suitable_gpus(self, gpu_request):
        if gpu_request == 0:
            return []
        if 0 < gpu_request < 1:
            suitable = [i for i, c in enumerate(self.gpu_capacities) if c >= gpu_request]
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
    def __init__(self):
        self.job_types = {}
        self.popularity = {}

    def add_type(self, key, job, popularity):
        self.job_types[key] = job
        self.popularity[key] = popularity


class FragmentationCalculator:
    @staticmethod
    def node_fragmentation(node, job):
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
        total = 0.0
        for key, job in workload.job_types.items():
            p = workload.popularity.get(key, 0.0)
            total += p * FragmentationCalculator.node_fragmentation(node, job)
        return total


def generate_job_batch(target_gpus, rng, distribution='alibaba'):
    """Generate jobs until we reach target GPU count.

    Distributions:
    - 'philly': Microsoft Philly trace (whole GPUs only: 1, 2, 4, 8)
    - 'alibaba': Alibaba GPU sharing trace (fractional GPUs: 0.25, 0.5, 1, 2, 4)

    Returns list of GPU requests.
    """
    jobs = []
    total = 0
    while total < target_gpus:
        r = rng.random()
        if distribution == 'philly':
            # Microsoft Philly: whole GPUs only
            if r < 0.70:
                size = 1
            elif r < 0.80:
                size = 2
            elif r < 0.95:
                size = 4
            else:
                size = 8
        elif distribution == 'alibaba':
            # Alibaba GPU sharing trace (approximated from FGD paper)
            # High proportion of fractional GPU jobs creates more fragmentation
            if r < 0.30:
                size = 0.25  # 30% request 1/4 GPU
            elif r < 0.55:
                size = 0.5   # 25% request 1/2 GPU
            elif r < 0.80:
                size = 1     # 25% request 1 GPU
            elif r < 0.92:
                size = 2     # 12% request 2 GPUs
            else:
                size = 4     # 8% request 4 GPUs
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        jobs.append(size)
        total += size
    return jobs


def build_nodes(cluster_spec, gpus_per_node=4):
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
    wl = FGDWorkload()
    sf_counts = Counter(sizes)
    total = len(sizes) if sizes else 1
    for sf, count in sf_counts.items():
        fgd_job = FGDJob(job_id=f'type_sf{sf}', gpu_request=float(sf), scale_factor=sf)
        wl.add_type(sf, fgd_job, float(count) / total)
    return wl


def compute_fragmentation_rate(nodes, workload):
    total_frag = sum(
        FragmentationCalculator.node_fragmentation_for_workload(n, workload)
        for n in nodes
    )
    total_capacity = sum(sum(n.gpu_capacities) for n in nodes)
    return (total_frag / total_capacity) * 100 if total_capacity > 0 else 0.0


def place_all_jobs_random(nodes, job_sizes, rng):
    """Random placement for all jobs."""
    placed_count = 0
    for gpu_req in job_sizes:
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
            placed_count += 1
    return placed_count


def place_all_jobs_bestfit(nodes, job_sizes):
    """Best-fit placement for all jobs."""
    placed_count = 0
    # Sort largest first
    sorted_sizes = sorted(enumerate(job_sizes), key=lambda x: x[1], reverse=True)
    for _, gpu_req in sorted_sizes:
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
            placed_count += 1
    return placed_count


def place_all_jobs_fgd(nodes, job_sizes, workload):
    """FGD placement for all jobs."""
    placed_count = 0
    # Sort largest first
    sorted_sizes = sorted(enumerate(job_sizes), key=lambda x: x[1], reverse=True)
    for job_id, gpu_req in sorted_sizes:
        fgd_job = FGDJob(job_id=str(job_id), gpu_request=gpu_req, scale_factor=gpu_req)
        best_node = None
        best_indices = None
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
                        best_node = node
                        best_indices = [idx]
            else:
                hyp = node.copy()
                hyp.allocate(gpu_req, indices)
                after = FragmentationCalculator.node_fragmentation_for_workload(hyp, workload)
                delta = after - current
                if delta < best_delta:
                    best_delta = delta
                    best_node = node
                    best_indices = indices
        if best_node is not None:
            best_node.allocate(gpu_req, best_indices)
            placed_count += 1
    return placed_count


def run_single_load_level(cluster_spec, gpus_per_node, target_load_pct, policy_name, seed,
                          distribution='alibaba'):
    """Run experiment at a single load level."""
    rng = random.Random(seed)
    total_capacity = sum(cluster_spec.values())
    target_gpus = int(total_capacity * target_load_pct / 100)

    # Generate jobs to reach target load
    job_sizes = generate_job_batch(target_gpus, rng, distribution=distribution)
    actual_load = sum(job_sizes)

    # Build fresh nodes
    nodes = build_nodes(cluster_spec, gpus_per_node)

    # Build workload for fragmentation calculation
    workload = build_workload_from_sizes(job_sizes)

    # Place jobs
    if policy_name == 'random':
        placed = place_all_jobs_random(nodes, job_sizes, rng)
    elif policy_name == 'bestfit':
        placed = place_all_jobs_bestfit(nodes, job_sizes)
    elif policy_name == 'fgd':
        placed = place_all_jobs_fgd(nodes, job_sizes, workload)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Compute fragmentation
    frag_rate = compute_fragmentation_rate(nodes, workload)

    # Compute actual utilization (placed GPUs / total)
    placed_gpus = sum(job_sizes[:placed]) if placed < len(job_sizes) else sum(job_sizes)
    # Actually count what's allocated
    allocated = sum(total_capacity - sum(sum(n.gpu_capacities) for n in nodes) for _ in [1])
    allocated = total_capacity - sum(sum(n.gpu_capacities) for n in nodes)

    return {
        'target_load_pct': target_load_pct,
        'actual_load_gpus': actual_load,
        'actual_load_pct': (actual_load / total_capacity) * 100,
        'jobs_generated': len(job_sizes),
        'jobs_placed': placed,
        'gpus_allocated': allocated,
        'utilization_pct': (allocated / total_capacity) * 100,
        'fragmentation_rate': frag_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Run fragmentation sweep experiment")
    parser.add_argument("--cluster", default="36:36:36", help="Cluster spec V100:P100:K80")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per server node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="fragmentation_sweep_results", help="Output directory")
    parser.add_argument("--policies", default="fgd,random,bestfit", help="Comma-separated policies")
    parser.add_argument("--load-min", type=int, default=5, help="Min load level (percent)")
    parser.add_argument("--load-max", type=int, default=120, help="Max load level (percent)")
    parser.add_argument("--load-step", type=int, default=5, help="Load level step (percent)")
    parser.add_argument("--distribution", default="alibaba", choices=["philly", "alibaba"],
                        help="Job size distribution: philly (whole GPUs) or alibaba (fractional GPUs)")
    args = parser.parse_args()

    # Parse cluster spec
    v100s, p100s, k80s = map(int, args.cluster.split(":"))
    cluster_spec = {"v100": v100s, "p100": p100s, "k80": k80s}
    total_gpus = sum(cluster_spec.values())

    print(f"Cluster: {cluster_spec} ({total_gpus} total GPUs)")
    print(f"GPUs per node: {args.gpus_per_node}")
    print(f"Seed: {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load levels to test
    load_levels = list(range(args.load_min, args.load_max + 1, args.load_step))
    print(f"Load levels: {load_levels}")

    # Run experiment for each policy
    policies = args.policies.split(',')
    all_results = {}

    print(f"Distribution: {args.distribution}")

    for policy in policies:
        print(f"\nRunning {policy}...")
        results = []
        for load_pct in load_levels:
            result = run_single_load_level(
                cluster_spec, args.gpus_per_node, load_pct, policy, args.seed,
                distribution=args.distribution
            )
            results.append(result)
            print(f"  Load {load_pct:3d}%: util={result['utilization_pct']:5.1f}%, frag={result['fragmentation_rate']:5.1f}%")
        all_results[policy] = results

        # Save individual policy results
        output_file = os.path.join(args.output_dir, f"{policy}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Save combined results for plotting
    combined_file = os.path.join(args.output_dir, "combined_results.json")
    with open(combined_file, 'w') as f:
        json.dump({
            'config': {
                'cluster': args.cluster,
                'gpus_per_node': args.gpus_per_node,
                'seed': args.seed,
                'total_gpus': total_gpus,
                'load_levels': load_levels,
                'distribution': args.distribution,
            },
            'results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
