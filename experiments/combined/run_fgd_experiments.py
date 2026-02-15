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
import logging
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
from job_id_pair import JobIdPair


def load_config(phase):
    config_path = os.path.join(
        os.path.dirname(__file__), 'configs', f'phase_{phase}.json'
    )
    with open(config_path) as f:
        return json.load(f)


def run_experiment(exp_config, common=None, log_dir=None, max_wall_time=None,
                   log_level=None):
    """Run a single experiment and return results.

    Args:
        exp_config: Experiment-specific config dict.
        common: Shared config dict merged under experiment config.
        log_dir: If set, write scheduler logs to a file in this directory.
        max_wall_time: Wall-clock timeout in seconds. If set, the simulation
            exits gracefully before this limit and saves partial results.
        log_level: Logging level for the scheduler (e.g. logging.WARNING for
            quiet mode). None means scheduler default (DEBUG).
    """
    if common is None:
        common = {}

    # Merge common config with experiment-specific config
    config = {**common, **exp_config}

    name = config['name']
    policy_name = config['policy']
    seed = config.get('seed', 0)
    enable_fgd = config.get('enable_fgd', False)
    fgd_placement_mode = config.get('fgd_placement_mode', 'fgd')
    fgd_workload_mode = config.get('fgd_workload_mode', 'philly')
    workload_mode = config.get('workload_mode', 'philly')

    cluster_spec = config.get('cluster_spec', {'v100': 36, 'p100': 36, 'k80': 36})
    num_gpus_per_server = config.get('num_gpus_per_server', None)
    # Allow shorthand: int -> per-type dict
    if isinstance(num_gpus_per_server, int):
        num_gpus_per_server = {wt: num_gpus_per_server for wt in cluster_spec}
    lam = config.get('lam', 0.0)
    num_total_jobs = config.get('num_total_jobs', 50)
    generate_multi_gpu_jobs = config.get('generate_multi_gpu_jobs', False)
    mode = config.get('mode', 'fixed_jobs')
    time_per_iteration = config.get('time_per_iteration', 600)
    enable_migration_penalty = config.get('enable_migration_penalty', False)
    solver = config.get('solver', 'ECOS')
    solver_kwargs = config.get('solver_kwargs', {})
    enable_gpu_sharing = config.get('enable_gpu_sharing', False)

    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"  Policy: {policy_name}, FGD: {enable_fgd} ({fgd_placement_mode})")
    print(f"  Cluster: {cluster_spec}, GPUs/server: {num_gpus_per_server}")
    if mode == 'steady_state':
        window_start = config['window_start']
        window_end = config['window_end']
        max_jct = config.get('max_jct', 360000)
        print(f"  Mode: steady_state, Window: [{window_start}, {window_end}), max_jct: {max_jct}")
    else:
        print(f"  Mode: fixed_jobs, Jobs: {num_total_jobs}")
    print(f"  Lambda: {lam}, Seed: {seed}, Round: {time_per_iteration}s")
    print(f"  Solver: {solver}, kwargs: {solver_kwargs}")
    if enable_gpu_sharing:
        print(f"  GPU sharing: ENABLED")
    print(f"{'='*70}")

    # Select throughputs file based on workload mode
    throughputs_filename = config.get('throughputs_file', None)
    if throughputs_filename is None:
        if workload_mode == 'alibaba':
            throughputs_filename = 'simulation_throughputs_alibaba.json'
        else:
            throughputs_filename = 'simulation_throughputs.json'
    throughputs_file = os.path.join(SCHEDULER_DIR, throughputs_filename)
    policy = utils.get_policy(policy_name, solver=solver, seed=seed,
                              solver_kwargs=solver_kwargs)

    sched = scheduler.Scheduler(
        policy,
        throughputs_file=throughputs_file,
        seed=seed,
        time_per_iteration=time_per_iteration,
        simulate=True,
        profiling_percentage=1.0,
        num_reference_models=26,
        enable_fgd=enable_fgd,
        fgd_placement_mode=fgd_placement_mode,
        fgd_workload_mode=fgd_workload_mode,
        enable_migration_penalty=enable_migration_penalty,
        enable_gpu_sharing=enable_gpu_sharing,
        log_level=log_level,
    )

    # Set up scale factor generator and reference worker type for Alibaba workload
    scale_factor_generator_func = None
    reference_worker_type = 'v100'
    if workload_mode == 'alibaba':
        scale_factor_generator_func = utils._generate_scale_factor_alibaba
        reference_worker_type = 'V100M32'

    # Optionally attach a file handler to capture scheduler logs
    file_handler = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '{asctime} {message}', style='{'))
        sched._orig_logger.addHandler(file_handler)
        print(f"  Logging to {log_path}")

    start_time = time.time()
    is_saturated = False

    if mode == 'steady_state':
        jobs_to_complete = set(
            JobIdPair(i, None) for i in range(window_start, window_end)
        )
        sched.simulate(
            cluster_spec=cluster_spec,
            lam=lam,
            jobs_to_complete=jobs_to_complete,
            generate_multi_gpu_jobs=generate_multi_gpu_jobs,
            simulate_steady_state=True,
            num_gpus_per_server=num_gpus_per_server,
            max_jct=max_jct,
            max_wall_time=max_wall_time,
            scale_factor_generator_func=scale_factor_generator_func,
            reference_worker_type=reference_worker_type,
        )
        is_saturated = sched.jct_threshold_exceeded()
        if is_saturated:
            avg_jct = float('inf')
        elif sched.saturated:
            # Wall-clock or sim timeout -- use partial JCT from completed
            # window jobs rather than losing all data
            avg_jct = sched.partial_jct if sched.partial_jct else float('inf')
            is_saturated = True
        else:
            avg_jct = sched.get_average_jct(jobs_to_complete)
    else:
        sched.simulate(
            cluster_spec=cluster_spec,
            lam=lam,
            num_total_jobs=num_total_jobs,
            generate_multi_gpu_jobs=generate_multi_gpu_jobs,
            num_gpus_per_server=num_gpus_per_server,
            scale_factor_generator_func=scale_factor_generator_func,
            reference_worker_type=reference_worker_type,
        )
        avg_jct = sched.get_average_jct(verbose=False)

    wall_time = time.time() - start_time

    # Clean up file handler
    if file_handler is not None:
        sched._orig_logger.removeHandler(file_handler)
        file_handler.close()

    # Collect fragmentation history if FGD was enabled
    frag_history = []
    if enable_fgd and hasattr(sched, '_fgd_fragmentation_history'):
        frag_history = sched._fgd_fragmentation_history

    # Count jobs with actual completion times (not None from deadlock)
    completed_count = sum(
        1 for t in sched._job_completion_times.values() if t is not None
    )
    failed_count = sum(
        1 for t in sched._job_completion_times.values() if t is None
    )

    result = {
        'name': name,
        'policy': policy_name,
        'enable_fgd': enable_fgd,
        'fgd_placement_mode': fgd_placement_mode,
        'seed': seed,
        'lam': lam,
        'avg_jct': avg_jct,
        'wall_time_seconds': wall_time,
        'num_completed_jobs': completed_count,
        'num_failed_jobs': failed_count,
        'fragmentation_samples': len(frag_history),
        'saturated': is_saturated,
        'mode': mode,
    }
    if mode == 'steady_state':
        result['generate_multi_gpu_jobs'] = generate_multi_gpu_jobs

    if is_saturated:
        print(f"  Result: SATURATED (JCT threshold exceeded)")
    else:
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
    parser.add_argument('--phase', required=True,
                        help='Experiment phase to run (e.g. c, d, e, e_test, f)')
    parser.add_argument('--index', type=int, default=None,
                        help='Run only this experiment index (0-based)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file for results')
    parser.add_argument('--save-logs', action='store_true', default=False,
                        help='Save scheduler logs to experiments/fgd/logs/')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress scheduler INFO logging')
    parser.add_argument('--max-wall-time', type=int, default=None,
                        help='Wall-clock timeout in seconds per experiment '
                             '(exits gracefully with partial results)')
    args = parser.parse_args()

    # Determine scheduler log level. Default is None (scheduler uses DEBUG).
    # With -q, suppress scheduler INFO messages by setting WARNING.
    sched_log_level = None
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        sched_log_level = logging.WARNING

    config = load_config(args.phase)
    experiments = config['experiments']
    common = config.get('common', {})

    if args.index is not None:
        experiments = [experiments[args.index]]

    log_dir = None
    if args.save_logs:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')

    results = []
    for exp in experiments:
        result = run_experiment(exp, common, log_dir=log_dir,
                                max_wall_time=args.max_wall_time,
                                log_level=sched_log_level)
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
        jct_str = 'inf' if r.get('saturated') else f"{r['avg_jct']:.2f}"
        print(f"  {r['name']}: JCT={jct_str}{frag_str}{status}")

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
