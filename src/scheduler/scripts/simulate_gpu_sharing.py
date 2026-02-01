#!/usr/bin/env python3
"""
GPU Sharing Simulator with FGD Placement
=========================================

This simulator demonstrates the FGD (Fragmentation Gradient Descent) algorithm
for GPU sharing workloads, based on the ATC 2023 paper.

Unlike Gavel's time-slicing model, this simulator models true GPU sharing where:
- Jobs can request fractional GPUs (e.g., 0.3 GPU)
- Multiple jobs can run simultaneously on the same GPU
- Fragmentation occurs when free GPU capacity is scattered

Usage:
    python simulate_gpu_sharing.py --trace alibaba --num-jobs 100
    python simulate_gpu_sharing.py --workload synthetic --num-jobs 200
"""

import sys
import os
import argparse
import random
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies.fgd import GPUSharingCluster, GPUState


@dataclass
class SimJob:
    """A job in the GPU sharing simulation."""
    job_id: int
    arrival_time: float
    duration: float
    gpu_milli: int  # 0-1000, where 1000 = 1.0 GPU
    num_gpus: int = 1  # Number of GPUs needed
    
    # Simulation state
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    placed_gpus: List[int] = field(default_factory=list)
    
    @property
    def gpu_fraction(self) -> float:
        return self.gpu_milli / 1000.0
    
    @property
    def is_completed(self) -> bool:
        return self.end_time is not None
    
    @property
    def is_running(self) -> bool:
        return self.start_time is not None and self.end_time is None
    
    @property
    def jct(self) -> Optional[float]:
        """Job completion time (from arrival to completion)."""
        if self.end_time is not None:
            return self.end_time - self.arrival_time
        return None
    
    @property
    def waiting_time(self) -> Optional[float]:
        """Time spent waiting before execution."""
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None


class GPUSharingSimulator:
    """
    Event-driven simulator for GPU sharing workloads.
    
    Simulates job arrivals, placements, and completions with support
    for different placement strategies (FGD, best-fit, etc.).
    """
    
    def __init__(self, num_servers: int, gpus_per_server: int, strategy: str = 'fgd'):
        self.num_servers = num_servers
        self.gpus_per_server = gpus_per_server
        self.strategy = strategy
        
        # Cluster state
        self.cluster = GPUSharingCluster(num_servers, gpus_per_server)
        
        # Job tracking
        self.pending_jobs: List[SimJob] = []
        self.running_jobs: List[SimJob] = []
        self.completed_jobs: List[SimJob] = []
        
        # Metrics
        self.fragmentation_history: List[Tuple[float, float]] = []
        self.utilization_history: List[Tuple[float, float]] = []
        self.blocked_attempts: int = 0
        
        # Current simulation time
        self.current_time: float = 0.0
    
    def add_job(self, job: SimJob) -> None:
        """Add a job to the pending queue."""
        self.pending_jobs.append(job)
    
    def _try_place_job(self, job: SimJob) -> bool:
        """Try to place a job on the cluster."""
        gpu_ids = self.cluster.place_job(
            job.job_id, job.gpu_milli, job.num_gpus, self.strategy
        )
        
        if gpu_ids is not None:
            job.placed_gpus = gpu_ids
            job.start_time = self.current_time
            job.end_time = self.current_time + job.duration
            self.running_jobs.append(job)
            return True
        else:
            self.blocked_attempts += 1
            return False
    
    def _complete_job(self, job: SimJob) -> None:
        """Complete a job and free its resources."""
        self.cluster.remove_job(job.job_id)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)
    
    def _record_metrics(self) -> None:
        """Record current fragmentation and utilization."""
        self.fragmentation_history.append(
            (self.current_time, self.cluster.get_fragmentation())
        )
        self.utilization_history.append(
            (self.current_time, self.cluster.get_utilization())
        )
    
    def run(self, verbose: bool = False) -> Dict:
        """
        Run the simulation until all jobs complete.
        
        Returns:
            Dictionary of simulation results.
        """
        # Sort pending jobs by arrival time
        self.pending_jobs.sort(key=lambda j: j.arrival_time)
        
        # Process jobs in order
        job_queue = list(self.pending_jobs)
        self.pending_jobs = []
        
        events = []  # (time, event_type, job)
        
        # Add arrival events
        for job in job_queue:
            events.append((job.arrival_time, 'arrival', job))
        
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'completion' else 1))
        
        waiting_queue: List[SimJob] = []
        
        while events or waiting_queue:
            if not events and waiting_queue:
                # No events but jobs waiting - this shouldn't happen in a valid simulation
                if verbose:
                    print(f"Warning: {len(waiting_queue)} jobs stuck in queue")
                break
            
            time, event_type, job = events.pop(0)
            self.current_time = time
            
            if event_type == 'arrival':
                # Try to place the job
                if self._try_place_job(job):
                    if verbose:
                        print(f"t={time:.1f}: Job {job.job_id} placed on GPUs {job.placed_gpus} "
                              f"({job.gpu_fraction:.1f} GPU)")
                    # Schedule completion
                    events.append((job.end_time, 'completion', job))
                    events.sort(key=lambda e: (e[0], 0 if e[1] == 'completion' else 1))
                else:
                    # Add to waiting queue
                    waiting_queue.append(job)
                    if verbose:
                        print(f"t={time:.1f}: Job {job.job_id} blocked (needs {job.gpu_fraction:.1f} GPU)")
            
            elif event_type == 'completion':
                self._complete_job(job)
                if verbose:
                    print(f"t={time:.1f}: Job {job.job_id} completed (JCT={job.jct:.1f})")
                
                # Try to place waiting jobs
                still_waiting = []
                for wj in waiting_queue:
                    if self._try_place_job(wj):
                        if verbose:
                            print(f"t={time:.1f}: Waiting job {wj.job_id} now placed")
                        events.append((wj.end_time, 'completion', wj))
                    else:
                        still_waiting.append(wj)
                waiting_queue = still_waiting
                events.sort(key=lambda e: (e[0], 0 if e[1] == 'completion' else 1))
            
            self._record_metrics()
        
        # Calculate results
        jcts = [j.jct for j in self.completed_jobs if j.jct is not None]
        waiting_times = [j.waiting_time for j in self.completed_jobs if j.waiting_time is not None]
        
        return {
            'strategy': self.strategy,
            'num_jobs': len(self.completed_jobs),
            'blocked_attempts': self.blocked_attempts,
            'avg_jct': sum(jcts) / len(jcts) if jcts else 0,
            'max_jct': max(jcts) if jcts else 0,
            'avg_waiting': sum(waiting_times) / len(waiting_times) if waiting_times else 0,
            'max_waiting': max(waiting_times) if waiting_times else 0,
            'makespan': self.current_time,
            'avg_fragmentation': sum(f for _, f in self.fragmentation_history) / len(self.fragmentation_history) if self.fragmentation_history else 0,
            'avg_utilization': sum(u for _, u in self.utilization_history) / len(self.utilization_history) if self.utilization_history else 0,
        }


def load_alibaba_trace(trace_path: str, num_jobs: int = 100, seed: int = 42) -> List[SimJob]:
    """
    Load jobs from Alibaba GPU trace.
    
    Supports two formats:
    1. Pre-converted format (from convert_alibaba_trace.py): *_gpusharing.csv
       Columns: job_id,arrival_time,duration,gpu_milli,num_gpus
    
    2. Original Alibaba format: openb_pod_list_*.csv
       Columns: name,cpu_milli,memory_mib,num_gpu,gpu_milli,...,creation_time,deletion_time,...
    """
    jobs = []
    random.seed(seed)
    
    with open(trace_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Sample rows if we have more than needed
    if len(rows) > num_jobs:
        rows = random.sample(rows, num_jobs)
    
    # Detect format based on columns
    if rows:
        first_row = rows[0]
        is_converted_format = 'job_id' in first_row and 'duration' in first_row
    else:
        return jobs
    
    # Parse jobs
    for i, row in enumerate(rows):
        try:
            if is_converted_format:
                # Pre-converted format (simpler)
                gpu_milli = int(row['gpu_milli'])
                num_gpus = int(row['num_gpus'])
                arrival_time = float(row['arrival_time'])
                duration = float(row['duration'])
            else:
                # Original Alibaba format
                gpu_milli = int(row.get('gpu_milli', 1000)) if row.get('gpu_milli') else 1000
                num_gpus = int(float(row.get('num_gpu', 1)))
                creation_time = float(row.get('creation_time', 0))
                deletion_time = float(row.get('deletion_time', creation_time + 3600))
                arrival_time = creation_time
                duration = max(60, deletion_time - creation_time)
            
            if gpu_milli <= 0:
                gpu_milli = 1000
            if num_gpus <= 0:
                num_gpus = 1
            
            jobs.append(SimJob(
                job_id=i,
                arrival_time=arrival_time,
                duration=duration,
                gpu_milli=gpu_milli,
                num_gpus=num_gpus
            ))
        except (ValueError, KeyError) as e:
            continue
    
    # Normalize arrival times to start at 0
    if jobs:
        min_arrival = min(j.arrival_time for j in jobs)
        for j in jobs:
            j.arrival_time -= min_arrival
    
    return jobs


def generate_synthetic_workload(num_jobs: int = 100, seed: int = 42) -> List[SimJob]:
    """Generate synthetic GPU sharing workload."""
    random.seed(seed)
    
    jobs = []
    current_time = 0.0
    
    for i in range(num_jobs):
        # Random GPU request (bias towards fractional)
        gpu_milli = random.choice([200, 300, 400, 500, 600, 700, 800, 1000])
        
        # Random duration (1-30 minutes)
        duration = random.uniform(60, 1800)
        
        # Poisson arrival (average 10 jobs per minute)
        inter_arrival = random.expovariate(10 / 60)
        current_time += inter_arrival
        
        jobs.append(SimJob(
            job_id=i,
            arrival_time=current_time,
            duration=duration,
            gpu_milli=gpu_milli,
            num_gpus=1
        ))
    
    return jobs


def run_comparison(jobs: List[SimJob], num_servers: int, gpus_per_server: int) -> None:
    """Run simulation with different strategies and compare results."""
    strategies = ['fgd', 'bestfit', 'worstfit', 'firstfit']
    results = []
    
    print(f"\n{'='*70}")
    print(f"GPU Sharing Simulation: {len(jobs)} jobs on {num_servers}x{gpus_per_server} GPUs")
    print(f"{'='*70}")
    
    # Show workload statistics
    gpu_requests = [j.gpu_fraction for j in jobs]
    print(f"Workload: avg GPU request = {sum(gpu_requests)/len(gpu_requests):.2f}, "
          f"min = {min(gpu_requests):.2f}, max = {max(gpu_requests):.2f}")
    
    for strategy in strategies:
        # Create fresh copy of jobs for each strategy
        jobs_copy = [SimJob(
            job_id=j.job_id,
            arrival_time=j.arrival_time,
            duration=j.duration,
            gpu_milli=j.gpu_milli,
            num_gpus=j.num_gpus
        ) for j in jobs]
        
        # Run simulation
        sim = GPUSharingSimulator(num_servers, gpus_per_server, strategy)
        for job in jobs_copy:
            sim.add_job(job)
        
        result = sim.run(verbose=False)
        results.append(result)
    
    # Print comparison table
    print(f"\n{'Strategy':<12} {'Completed':<10} {'Blocked':<10} {'Avg JCT':<12} "
          f"{'Avg Wait':<12} {'Makespan':<12} {'Avg Frag':<10} {'Avg Util':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['strategy']:<12} {r['num_jobs']:<10} {r['blocked_attempts']:<10} "
              f"{r['avg_jct']:<12.1f} {r['avg_waiting']:<12.1f} {r['makespan']:<12.1f} "
              f"{r['avg_fragmentation']:<10.3f} {r['avg_utilization']:<10.1%}")
    
    # Calculate improvements
    fgd_result = results[0]
    print(f"\n{'='*70}")
    print("FGD vs Baselines:")
    print(f"{'='*70}")
    
    for r in results[1:]:
        if r['avg_jct'] > 0:
            jct_improvement = (r['avg_jct'] - fgd_result['avg_jct']) / r['avg_jct'] * 100
            frag_improvement = (r['avg_fragmentation'] - fgd_result['avg_fragmentation']) / r['avg_fragmentation'] * 100 if r['avg_fragmentation'] > 0 else 0
            print(f"vs {r['strategy']:<10}: JCT {jct_improvement:+.1f}%, Fragmentation {frag_improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='GPU Sharing Simulator with FGD')
    parser.add_argument('--workload', choices=['alibaba', 'synthetic'], default='synthetic',
                       help='Workload type')
    parser.add_argument('--trace', type=str, default=None,
                       help='Path to Alibaba trace file')
    parser.add_argument('--num-jobs', type=int, default=100,
                       help='Number of jobs to simulate')
    parser.add_argument('--num-servers', type=int, default=4,
                       help='Number of servers')
    parser.add_argument('--gpus-per-server', type=int, default=4,
                       help='GPUs per server')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load or generate workload
    if args.workload == 'alibaba' and args.trace:
        jobs = load_alibaba_trace(args.trace, args.num_jobs, args.seed)
        print(f"Loaded {len(jobs)} jobs from Alibaba trace")
    else:
        jobs = generate_synthetic_workload(args.num_jobs, args.seed)
        print(f"Generated {len(jobs)} synthetic jobs")
    
    # Run comparison
    run_comparison(jobs, args.num_servers, args.gpus_per_server)


if __name__ == '__main__':
    main()
