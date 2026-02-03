#!/usr/bin/env python3
"""Runner for FGD integration experiments (Phases C-F).

Usage:
  python run_fgd_experiments.py --phase c
  python run_fgd_experiments.py --phase d
  python run_fgd_experiments.py --phase e --index 0  # run single experiment
  python run_fgd_experiments.py --phase e             # run all
  python run_fgd_experiments.py --phase f
"""

import argparse
import json
import numpy as np
import os
import sys
import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add scheduler to path
SCHEDULER_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scheduler')
sys.path.insert(0, SCHEDULER_DIR)

import scheduler
import utils


def load_config(phase):
    config_path = os.path.join(
        os.path.dirname(__file__), 'configs', f'phase_{phase}.json'
    )
    with open(config_path) as f:
        return json.load(f)


def run_experiment(exp_config, common=None):
    """Run a single experiment and return results."""
    if common is None:
        common = {}

    # Merge common config with experiment-specific config
    config = {**common, **exp_config}

    name = config['name']
    policy_name = config['policy']
    seed = config.get('seed', 0)
    enable_fgd = config.get('enable_fgd', False)
    fgd_placement_mode = config.get('fgd_placement_mode', 'fgd')

    cluster_spec = config.get('cluster_spec', {'v100': 36, 'p100': 36, 'k80': 36})
    num_gpus_per_server = config.get('num_gpus_per_server', None)
    lam = config.get('lam', 0.0)
    num_total_jobs = config.get('num_total_jobs', 50)
    generate_multi_gpu_jobs = config.get('generate_multi_gpu_jobs', False)

    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"  Policy: {policy_name}, FGD: {enable_fgd} ({fgd_placement_mode})")
    print(f"  Cluster: {cluster_spec}, GPUs/server: {num_gpus_per_server}")
    print(f"  Lambda: {lam}, Jobs: {num_total_jobs}, Seed: {seed}")
    print(f"{'='*70}")

    throughputs_file = os.path.join(SCHEDULER_DIR, 'simulation_throughputs.json')

    policy = utils.get_policy(policy_name, solver='ECOS', seed=seed)

    sched = scheduler.Scheduler(
        policy,
        throughputs_file=throughputs_file,
        seed=seed,
        time_per_iteration=360,
        simulate=True,
        profiling_percentage=1.0,
        num_reference_models=26,
        enable_fgd=enable_fgd,
        fgd_placement_mode=fgd_placement_mode,
    )

    start_time = time.time()
    sched.simulate(
        cluster_spec=cluster_spec,
        lam=lam,
        num_total_jobs=num_total_jobs,
        generate_multi_gpu_jobs=generate_multi_gpu_jobs,
        num_gpus_per_server=num_gpus_per_server,
    )
    wall_time = time.time() - start_time

    avg_jct = sched.get_average_jct(verbose=False)

    # Collect fragmentation history if FGD was enabled
    frag_history = []
    if enable_fgd and hasattr(sched, '_fgd_fragmentation_history'):
        frag_history = sched._fgd_fragmentation_history

    result = {
        'name': name,
        'policy': policy_name,
        'enable_fgd': enable_fgd,
        'fgd_placement_mode': fgd_placement_mode,
        'seed': seed,
        'lam': lam,
        'avg_jct': avg_jct,
        'wall_time_seconds': wall_time,
        'num_completed_jobs': len(sched._job_completion_times),
        'fragmentation_samples': len(frag_history),
    }

    # Check expected JCT if specified
    expected_jct = config.get('expected_jct')
    if expected_jct is not None:
        tolerance = 0.01  # 1% tolerance
        relative_error = abs(avg_jct - expected_jct) / expected_jct
        result['expected_jct'] = expected_jct
        result['relative_error'] = relative_error
        result['pass'] = relative_error < tolerance
        status = 'PASS' if result['pass'] else 'FAIL'
        print(f"  Result: JCT={avg_jct:.2f} (expected {expected_jct:.2f}, error={relative_error:.4%}) [{status}]")
    else:
        print(f"  Result: JCT={avg_jct:.2f}")

    print(f"  Wall time: {wall_time:.1f}s")
    if frag_history:
        avg_frag = sum(f for _, _, f in frag_history) / len(frag_history)
        result['avg_fragmentation'] = avg_frag
        print(f"  Avg fragmentation: {avg_frag:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Run FGD integration experiments')
    parser.add_argument('--phase', required=True, choices=['c', 'd', 'e', 'f'],
                        help='Experiment phase to run')
    parser.add_argument('--index', type=int, default=None,
                        help='Run only this experiment index (0-based)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    config = load_config(args.phase)
    experiments = config['experiments']
    common = config.get('common', {})

    if args.index is not None:
        experiments = [experiments[args.index]]

    results = []
    for exp in experiments:
        result = run_experiment(exp, common)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"Phase {args.phase.upper()} Summary")
    print(f"{'='*70}")
    for r in results:
        status = ''
        if 'pass' in r:
            status = ' [PASS]' if r['pass'] else ' [FAIL]'
        frag_str = f", frag={r.get('avg_fragmentation', 'N/A')}" if 'avg_fragmentation' in r else ''
        print(f"  {r['name']}: JCT={r['avg_jct']:.2f}{frag_str}{status}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {args.output}")
    elif not args.output:
        default_output = os.path.join(
            os.path.dirname(__file__), f'results_phase_{args.phase}.json'
        )
        with open(default_output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {default_output}")


if __name__ == '__main__':
    main()
