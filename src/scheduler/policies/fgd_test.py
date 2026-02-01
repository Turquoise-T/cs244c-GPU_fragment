#!/usr/bin/env python3
"""
Test FGD GPU sharing placement algorithm.
Reproduces the example from FGD paper Figure 8.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies.fgd import GPUSharingCluster, GPUState

def test_paper_example():
    """
    Test case from FGD paper Figure 8:
    - 2 GPUs: GPU-A has 0.5 idle, GPU-B has 1.0 idle
    - Tasks: A (0.3 GPU), B (0.5 GPU), C (0.7 GPU)
    
    FGD should make different choices than best-fit!
    """
    print('='*60)
    print('FGD vs Best-fit: Paper Figure 8 Example')
    print('='*60)
    
    def run_test(strategy):
        # Create cluster: 1 server with 2 GPUs
        cluster = GPUSharingCluster(num_servers=1, gpus_per_server=2)
        
        # Set initial state: GPU 0 has 500 milli used (0.5 GPU), GPU 1 is empty
        cluster.gpu_states[0].used_milli = 500
        cluster.gpu_states[0].job_assignments = [(-1, 500)]  # Pre-existing job
        
        print(f'\n--- Strategy: {strategy.upper()} ---')
        print('Initial: GPU-0 has 0.5 free, GPU-1 has 1.0 free')
        
        jobs = [
            ('Task A', 300),   # 0.3 GPU
            ('Task B', 500),   # 0.5 GPU  
            ('Task C', 700),   # 0.7 GPU
        ]
        
        results = []
        for job_id, (name, milli) in enumerate(jobs):
            gpu_ids = cluster.place_job(job_id, milli, strategy=strategy)
            if gpu_ids:
                gpu_id = gpu_ids[0]
                results.append((name, gpu_id))
                print(f'  {name} ({milli/1000:.1f} GPU) -> GPU-{gpu_id}')
            else:
                results.append((name, None))
                print(f'  {name} ({milli/1000:.1f} GPU) -> BLOCKED!')
        
        # Show final state
        print(f'  Final state:')
        for gpu in cluster.gpu_states:
            print(f'    GPU-{gpu.gpu_id}: {gpu.used_milli/1000:.1f} used, {gpu.free_milli/1000:.1f} free')
        print(f'  Fragmentation: {cluster.get_fragmentation():.3f}')
        
        return results, cluster.get_fragmentation()
    
    fgd_results, fgd_frag = run_test('fgd')
    bf_results, bf_frag = run_test('bestfit')
    
    print('\n' + '='*60)
    print('COMPARISON')
    print('='*60)
    print(f'FGD placements:      {fgd_results}')
    print(f'Best-fit placements: {bf_results}')
    print(f'FGD fragmentation:      {fgd_frag:.3f}')
    print(f'Best-fit fragmentation: {bf_frag:.3f}')
    
    # Check if FGD made better decisions
    if fgd_frag < bf_frag:
        print('\n✓ FGD achieved lower fragmentation!')
    elif fgd_frag == bf_frag:
        print('\n= Same fragmentation (but different placement decisions)')


def test_larger_cluster():
    """Test FGD with more jobs on a larger cluster."""
    print('\n' + '='*60)
    print('FGD vs Best-fit: Larger Workload')
    print('='*60)
    
    import random
    random.seed(42)
    
    # Generate random jobs with fractional GPU requests
    jobs = []
    for i in range(20):
        # Random GPU request between 100-800 milli (0.1-0.8 GPU)
        milli = random.choice([100, 200, 300, 400, 500, 600, 700, 800])
        jobs.append((f'Job-{i}', milli))
    
    print(f'Workload: {len(jobs)} jobs')
    print(f'GPU requests: {[j[1]/1000 for j in jobs]}')
    
    for strategy in ['fgd', 'bestfit', 'worstfit']:
        cluster = GPUSharingCluster(num_servers=2, gpus_per_server=4)
        
        placed = 0
        blocked = 0
        for job_id, (name, milli) in enumerate(jobs):
            result = cluster.place_job(job_id, milli, strategy=strategy)
            if result:
                placed += 1
            else:
                blocked += 1
        
        print(f'\n{strategy.upper():10} - Placed: {placed}, Blocked: {blocked}, '
              f'Frag: {cluster.get_fragmentation():.3f}, '
              f'Util: {cluster.get_utilization():.1%}')


def test_fgd_advantage():
    """
    Test case where FGD clearly outperforms best-fit.
    
    Scenario: Jobs arrive in specific order that causes best-fit to fragment.
    """
    print('\n' + '='*60)
    print('FGD vs Best-fit: Advantage Case')
    print('='*60)
    
    # Specific workload designed to show FGD advantage
    # Jobs that if placed greedily (best-fit) will fragment
    jobs = [
        # Phase 1: Small jobs spread across GPUs
        ('J0', 300),  # 0.3 GPU
        ('J1', 400),  # 0.4 GPU
        ('J2', 300),  # 0.3 GPU
        ('J3', 400),  # 0.4 GPU
        # Phase 2: Try to fit larger jobs
        ('J4', 700),  # 0.7 GPU - might be blocked if fragmented
        ('J5', 600),  # 0.6 GPU
        ('J6', 800),  # 0.8 GPU
    ]
    
    print(f'Jobs: {[(j[0], j[1]/1000) for j in jobs]}')
    
    results = {}
    for strategy in ['fgd', 'bestfit', 'firstfit']:
        cluster = GPUSharingCluster(num_servers=1, gpus_per_server=4)
        
        placed_jobs = []
        blocked_jobs = []
        
        for job_id, (name, milli) in enumerate(jobs):
            result = cluster.place_job(job_id, milli, strategy=strategy)
            if result:
                placed_jobs.append(name)
            else:
                blocked_jobs.append(name)
        
        results[strategy] = {
            'placed': len(placed_jobs),
            'blocked': len(blocked_jobs),
            'blocked_jobs': blocked_jobs,
            'frag': cluster.get_fragmentation(),
            'util': cluster.get_utilization()
        }
        
        print(f'\n{strategy.upper():10}:')
        print(f'  Placed: {len(placed_jobs)}, Blocked: {blocked_jobs}')
        print(f'  Fragmentation: {cluster.get_fragmentation():.3f}')
        print(f'  Utilization: {cluster.get_utilization():.1%}')
        
        # Show final GPU state
        for gpu in cluster.gpu_states:
            jobs_str = ', '.join([f'{j[1]/1000:.1f}' for j in gpu.job_assignments])
            print(f'    GPU-{gpu.gpu_id}: [{jobs_str}] = {gpu.used_fraction:.1f} used')


def test_high_load():
    """Test under high load conditions."""
    print('\n' + '='*60)
    print('High Load Test: 50 Jobs on 8 GPUs')
    print('='*60)
    
    import random
    
    for seed in [42, 123, 456]:
        random.seed(seed)
        
        # Generate 50 random jobs
        jobs = [(f'J{i}', random.choice([200, 300, 400, 500, 600])) 
                for i in range(50)]
        
        print(f'\nSeed {seed}:')
        for strategy in ['fgd', 'bestfit']:
            cluster = GPUSharingCluster(num_servers=2, gpus_per_server=4)
            
            placed = blocked = 0
            for job_id, (name, milli) in enumerate(jobs):
                if cluster.place_job(job_id, milli, strategy=strategy):
                    placed += 1
                else:
                    blocked += 1
            
            print(f'  {strategy:8}: Placed {placed:2}, Blocked {blocked:2}, '
                  f'Frag {cluster.get_fragmentation():.2f}, '
                  f'Util {cluster.get_utilization():.0%}')


if __name__ == '__main__':
    test_paper_example()
    test_larger_cluster()
    test_fgd_advantage()
    test_high_load()
