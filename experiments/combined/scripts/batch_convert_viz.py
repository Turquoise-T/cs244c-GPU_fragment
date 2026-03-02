#!/usr/bin/env python3
"""Batch convert FGD replication logs to .viz.bin files for the viz tool.

Usage:
    cd gavel
    python experiments/combined/scripts/batch_convert_viz.py \
        --logs-dir experiments/combined/logs \
        --output-dir ../gpu-scheduling-viz/data/fgd_replication \
        --config experiments/combined/configs/phase_fgd_replication.json \
        --max-rounds 1000
"""
import argparse
import json
import os
import sys
import time

# Add cs244c/ to path so `from viz.tools...` resolves via the viz -> gpu-scheduling-viz symlink
_script_dir = os.path.dirname(os.path.abspath(__file__))
cs244c_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', '..'))
sys.path.insert(0, cs244c_dir)

from viz.tools.preprocess_viz import preprocess_simulation


def build_cluster_spec_str(cluster_spec_dict):
    """Convert cluster spec dict to named format string."""
    return ','.join(f'{name}={count}' for name, count in cluster_spec_dict.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs-dir', required=True, help='Directory with .log files')
    parser.add_argument('--output-dir', required=True, help='Directory for .viz.bin output')
    parser.add_argument('--config', required=True, help='Path to experiment config JSON')
    parser.add_argument('--max-rounds', type=int, default=1000,
                        help='Max rounds per .viz.bin (0 = no limit)')
    parser.add_argument('--filter', default=None,
                        help='Only process logs matching this substring')
    parser.add_argument('--figure', default=None, help='Figure reference (e.g. fgd-fig7)')
    parser.add_argument('--trace', default=None, help='Trace (philly, alibaba)')
    parser.add_argument('--date', default=None, help='Experiment date (YYYY-MM-DD)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    common = config.get('common', {})
    cluster_spec = build_cluster_spec_str(common['cluster_spec'])
    window_start = common.get('window_start', 4000)
    window_end = common.get('window_end', 5000)

    # Build gpus_per_node mapping -- for viz we use a uniform value.
    # The mixed node sizes are handled per-type, but the viz tool expects a single int.
    # Use the most common value (8) since the majority of GPUs are in 8-GPU nodes.
    gpus_per_node = 8

    experiments = config['experiments']
    exp_by_name = {e['name']: e for e in experiments}

    log_files = sorted(f for f in os.listdir(args.logs_dir) if f.endswith('.log'))
    if args.filter:
        log_files = [f for f in log_files if args.filter in f]

    print(f"Found {len(log_files)} log files to convert")
    converted = 0
    skipped = 0
    failed = 0

    for i, log_file in enumerate(log_files):
        name = log_file.replace('.log', '')
        log_path = os.path.join(args.logs_dir, log_file)
        output_path = os.path.join(args.output_dir, f'fgd_repl_{name}.viz.bin')

        # Skip if already converted
        if os.path.exists(output_path):
            skipped += 1
            continue

        # Determine policy label from experiment config
        exp = exp_by_name.get(name, {})
        if exp.get('enable_fgd', False):
            policy = f"fgd_{exp.get('fgd_placement_mode', 'fgd')}"
        else:
            policy = 'strided'

        # Build metadata for schema validation (if --figure provided)
        metadata = None
        if args.figure:
            # Infer scheduler and placement from experiment config
            if exp.get('enable_fgd', False):
                placement = exp.get('fgd_placement_mode', 'fgd')
            else:
                placement = 'strided'

            scheduler = 'mmf'  # default
            if exp.get('policy') == 'finish_time_fairness':
                scheduler = 'fifo'

            # Extract load and seed from experiment config
            rate = exp.get('jobs_per_hr', exp.get('rate', 0))
            load = f'{int(rate)}jph' if rate == int(rate) else f'{rate}jph'
            seed = f"s{exp.get('seed', 0)}"

            metadata = {
                'date': args.date,
                'trace': args.trace,
                'figure': args.figure,
                'scheduler': scheduler,
                'placement': placement,
                'load': load,
                'seed': seed,
            }

        t0 = time.time()
        try:
            preprocess_simulation(
                log_path=log_path,
                output_path=output_path,
                cluster_spec=cluster_spec,
                measurement_window=(window_start, window_end),
                policy=policy,
                gpus_per_node=gpus_per_node,
                max_rounds=args.max_rounds,
                metadata=metadata,
            )
            elapsed = time.time() - t0
            converted += 1
            print(f"  [{i+1}/{len(log_files)}] {name} -> .viz.bin ({elapsed:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(log_files)}] {name} FAILED: {e}")

    print(f"\nDone: {converted} converted, {skipped} skipped, {failed} failed")


if __name__ == '__main__':
    main()
