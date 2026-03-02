#!/usr/bin/env python3
"""Export per-experiment JCT percentiles from .viz.bin files.

Reads .viz.bin files for all gavel replication experiments, extracts per-job
JCT from the measurement window (jobs 4000-4999), computes percentiles
(p10, p50, p75, p90, p99, p99.9, p100), and writes CSVs per figure.

Also includes paper reference data (mean only) in the same CSVs for comparison.

Usage:
    cd gavel
    python experiments/combined/scripts/export_jct_percentiles.py
"""
import csv
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add cs244c/ to path so viz.tools imports work
_script_dir = os.path.dirname(os.path.abspath(__file__))
cs244c_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', '..'))
sys.path.insert(0, cs244c_dir)

from viz.tools.binary_format import unpack_header, unpack_job_metadata, HEADER_SIZE, JOB_METADATA_SIZE

# Measurement window
WINDOW_START = 4000
WINDOW_END = 5000

PERCENTILES = [10, 50, 75, 90, 99, 99.9, 100]

# Map CSV policy names to role (baseline/gavel)
POLICY_ROLE = {
    'max_min_fairness': 'baseline',
    'max_min_fairness_perf': 'gavel',
    'finish_time_fairness': 'baseline',
    'finish_time_fairness_perf': 'gavel',
}


def extract_window_durations(viz_path):
    """Extract JCT durations for measurement window jobs from a .viz.bin file."""
    with open(viz_path, 'rb') as f:
        hdr = unpack_header(f.read(HEADER_SIZE))
        f.seek(hdr['job_metadata_offset'])

        durations = []
        for _ in range(hdr['num_jobs']):
            job = unpack_job_metadata(f.read(JOB_METADATA_SIZE))
            if WINDOW_START <= job['job_id'] < WINDOW_END and job['duration'] > 0:
                durations.append(job['duration'])

    return durations


def find_viz_file(viz_dir, exp_name):
    """Find the .viz.bin file for an experiment name."""
    # Naming convention: gavel_repl_{exp_name}.viz.bin
    candidate = viz_dir / f'gavel_repl_{exp_name}.viz.bin'
    if candidate.exists():
        return candidate
    return None


def main():
    script_dir = Path(__file__).resolve().parent
    gavel_dir = script_dir.parent.parent.parent

    # Paths
    csv_path = script_dir / '..' / 'results' / 'gavel_replication_combined.csv'
    viz_dir = gavel_dir / '..' / 'gpu-scheduling-viz' / 'data' / 'gavel_replication'
    ref_path = gavel_dir / 'experiments' / 'gavel-replication' / 'scripts' / 'paper_reference_curves.json'
    output_dir = script_dir / '..' / 'results'

    csv_path = csv_path.resolve()
    viz_dir = viz_dir.resolve()
    ref_path = ref_path.resolve()
    output_dir = output_dir.resolve()

    # Load CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        experiments = list(reader)
    print(f'Loaded {len(experiments)} experiments from CSV')

    # Load reference data
    with open(ref_path) as f:
        ref = json.load(f)
    print(f'Loaded paper reference curves')

    # Process each figure
    for fig in ['fig9', 'fig10', 'fig11']:
        fig_exps = [e for e in experiments if e['figure'] == fig]
        print(f'\n=== {fig}: {len(fig_exps)} experiments ===')

        # Group by (policy, jobs_per_hr) -> list of per-seed results
        groups = defaultdict(list)
        missing = 0

        for exp in fig_exps:
            name = exp['name']
            policy = exp['policy']
            jph = float(exp['jobs_per_hr'])
            seed = int(exp['seed'])
            saturated = exp['saturated'] == 'True'

            viz_file = find_viz_file(viz_dir, name)
            if viz_file is None:
                missing += 1
                continue

            durations = extract_window_durations(viz_file)
            if not durations:
                missing += 1
                continue

            groups[(policy, jph)].append({
                'seed': seed,
                'durations': durations,
                'saturated': saturated,
                'n_window_jobs': len(durations),
            })

        if missing:
            print(f'  Missing .viz.bin for {missing} experiments')

        # Build output rows
        rows = []
        pct_cols = [f'p{p}' for p in PERCENTILES]

        for (policy, jph), seeds in sorted(groups.items()):
            role = POLICY_ROLE.get(policy, policy)

            # Per-seed percentiles
            for s in sorted(seeds, key=lambda x: x['seed']):
                d = np.array(s['durations']) / 3600.0  # Convert to hours
                row = {
                    'figure': fig,
                    'source': 'ours',
                    'role': role,
                    'policy': policy,
                    'jobs_per_hr': jph,
                    'seed': s['seed'],
                    'saturated': s['saturated'],
                    'n_window_jobs': s['n_window_jobs'],
                    'mean': float(np.mean(d)),
                }
                for p in PERCENTILES:
                    if p == 100:
                        row[f'p{p}'] = float(np.max(d))
                    else:
                        row[f'p{p}'] = float(np.percentile(d, p))
                rows.append(row)

            # Aggregate across seeds (pool all window jobs)
            all_durations = []
            for s in seeds:
                all_durations.extend(s['durations'])
            d = np.array(all_durations) / 3600.0
            row = {
                'figure': fig,
                'source': 'ours',
                'role': role,
                'policy': policy,
                'jobs_per_hr': jph,
                'seed': 'all',
                'saturated': any(s['saturated'] for s in seeds),
                'n_window_jobs': len(all_durations),
                'mean': float(np.mean(d)),
            }
            for p in PERCENTILES:
                if p == 100:
                    row[f'p{p}'] = float(np.max(d))
                else:
                    row[f'p{p}'] = float(np.percentile(d, p))
            rows.append(row)

        # Add paper reference data (mean only, no percentiles)
        ref_data = ref.get(fig, {})
        ref_x = ref_data.get('jobs_per_hr', [])
        for role_key in ['baseline', 'gavel']:
            ref_y = ref_data.get(role_key, [])
            for x, y in zip(ref_x, ref_y):
                if y is None:
                    continue
                row = {
                    'figure': fig,
                    'source': 'paper',
                    'role': role_key,
                    'policy': '',
                    'jobs_per_hr': x,
                    'seed': '',
                    'saturated': False,
                    'n_window_jobs': '',
                    'mean': y,
                }
                for p in PERCENTILES:
                    row[f'p{p}'] = ''
                rows.append(row)

        # Write CSV
        fieldnames = [
            'figure', 'source', 'role', 'policy', 'jobs_per_hr', 'seed',
            'saturated', 'n_window_jobs', 'mean',
        ] + pct_cols
        output_path = output_dir / f'jct_percentiles_{fig}.csv'
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f'  Wrote {len(rows)} rows to {output_path.name}')

        # Print summary
        for (policy, jph), seeds in sorted(groups.items()):
            role = POLICY_ROLE.get(policy, policy)
            all_d = []
            for s in seeds:
                all_d.extend(s['durations'])
            d = np.array(all_d) / 3600.0
            sat_flag = '*' if any(s['saturated'] for s in seeds) else ' '
            print(f'  {sat_flag} {role:10s} {jph:5.1f}jph  '
                  f'mean={np.mean(d):6.1f}h  '
                  f'p50={np.median(d):6.1f}h  '
                  f'p90={np.percentile(d, 90):6.1f}h  '
                  f'p99={np.percentile(d, 99):6.1f}h  '
                  f'max={np.max(d):6.1f}h  '
                  f'({len(seeds)} seeds, {len(all_d)} jobs)')


if __name__ == '__main__':
    main()
