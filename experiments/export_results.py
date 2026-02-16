#!/usr/bin/env python3
"""Export Gavel and FGD experiment results as student-friendly CSVs.

Generates:
  - exports/gavel_results.csv  (Gavel replication: JCT vs job rate)
  - exports/fgd_results.csv    (FGD standalone: fragmentation metrics vs demand)

Each row includes our measured mean/std and the paper's reference value,
so students can plot their own results alongside both.

Usage:
    python export_results.py
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR) if SCRIPT_DIR.endswith('experiments') else SCRIPT_DIR
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'exports')


def export_gavel():
    """Export Gavel replication results."""
    results_path = os.path.join(
        SCRIPT_DIR, 'replication', 'results', 'results_combined.csv')
    ref_path = os.path.join(
        SCRIPT_DIR, 'replication', 'scripts', 'paper_reference_curves.json')

    if not os.path.exists(results_path):
        print(f"Skipping Gavel: {results_path} not found")
        return

    # Load reference
    ref = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)

    # Parse results CSV
    import pandas as pd
    df = pd.read_csv(results_path)
    df['jct_sec'] = pd.to_numeric(df['jct_sec'], errors='coerce')
    df['jct_hours'] = df['jct_sec'] / 3600
    df['figure'] = df['name'].str.extract(r'(fig\d+)')[0]

    # Policy mapping: internal name -> display name
    policy_map = {
        'max_min_fairness_perf': 'gavel',
        'max_min_fairness': 'baseline',
        'finish_time_fairness_perf': 'gavel',
        'finish_time_fairness': 'baseline',
    }

    rows = []
    for fig_key in ['fig9', 'fig10', 'fig11']:
        fig_data = df[df['figure'] == fig_key].copy()
        fig_data = fig_data[fig_data['jct_hours'].notna() & (fig_data['jct_hours'] != float('inf'))]

        ref_section = ref.get(fig_key, {})
        ref_x = ref_section.get('jobs_per_hr', [])

        for policy_internal in fig_data['policy'].unique():
            display = policy_map.get(policy_internal, policy_internal)
            policy_data = fig_data[fig_data['policy'] == policy_internal]
            stats = policy_data.groupby('jobs_per_hr')['jct_hours'].agg(['mean', 'std', 'count'])
            stats = stats.reset_index()
            stats = stats[stats['count'] >= 2]

            ref_y = ref_section.get(display, [])

            for _, row in stats.iterrows():
                rate = row['jobs_per_hr']
                # Find closest paper reference value
                paper_val = None
                if ref_x and ref_y:
                    for rx, ry in zip(ref_x, ref_y):
                        if ry is not None and abs(rx - rate) < 0.15:
                            paper_val = ry
                            break

                rows.append({
                    'figure': fig_key,
                    'jobs_per_hr': round(rate, 2),
                    'policy': display,
                    'mean_jct_hours': round(row['mean'], 2),
                    'std_jct_hours': round(row['std'], 2),
                    'n_seeds': int(row['count']),
                    'paper_jct_hours': paper_val,
                })

    out_path = os.path.join(EXPORT_DIR, 'gavel_results.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'figure', 'jobs_per_hr', 'policy',
            'mean_jct_hours', 'std_jct_hours', 'n_seeds', 'paper_jct_hours'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['figure'], r['jobs_per_hr'], r['policy'])))

    print(f"Wrote {len(rows)} rows to {out_path}")


def export_fgd():
    """Export FGD standalone results."""
    results_dir = os.path.join(
        SCRIPT_DIR, 'fgd-standalone', 'results', 'full_run')
    ref_path = os.path.join(
        SCRIPT_DIR, 'fgd-standalone', 'paper_reference_curves.json')

    if not os.path.exists(results_dir):
        print(f"Skipping FGD: {results_dir} not found")
        return

    # Load results
    all_results_path = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(all_results_path):
        with open(all_results_path) as f:
            results = json.load(f)
    else:
        results = []
        for fname in os.listdir(results_dir):
            if fname.endswith('.json') and fname != 'all_results.json':
                with open(os.path.join(results_dir, fname)) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results.extend(data)

    if not results:
        print("No FGD results found")
        return

    # Load reference
    ref = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)

    # Group by policy
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r['policy']].append(r['curve'])

    # Policy name mapping (internal -> ref key)
    policy_ref_map = {
        'random': 'random',
        'dotprod': 'dotprod',
        'gpuclustering': 'clustering',
        'gpupacking': 'packing',
        'bestfit': 'bestfit',
        'fgd': 'fgd',
    }

    # Metrics to export
    metrics = [
        ('frag_rate_pct', 'fig7a_frag_rate_pct',
         lambda pt: (pt['fragmentation'] / pt['unallocated_gpus'] * 100
                     if pt['unallocated_gpus'] > 0 else 100.0)),
        ('frag_over_total_pct', 'fig7b_frag_over_total_pct',
         lambda pt: pt['frag_ratio'] * 100),
        ('unalloc_gpu_pct', 'fig9a_unalloc_gpu_pct',
         lambda pt: (1 - pt['alloc_ratio']) * 100),
        ('occupied_nodes', 'fig9b_occupied_nodes',
         lambda pt: pt.get('occupied_nodes', 0)),
    ]

    # Interpolate onto common x-axis (demand_pct 0-120, step 5)
    x_common = np.arange(0, 125, 5)

    rows = []
    for metric_name, ref_key, y_func in metrics:
        ref_section = ref.get(ref_key, {})
        ref_x = ref_section.get('demand_pct', [])

        for policy, curves in by_policy.items():
            ref_name = policy_ref_map.get(policy, policy)
            ref_y = ref_section.get(ref_name, [])

            # Interpolate each seed's curve
            y_all = []
            for curve in curves:
                if not curve:
                    continue
                xs = np.array([pt['demand_fraction'] * 100 for pt in curve])
                ys = np.array([y_func(pt) for pt in curve])
                y_interp = np.interp(x_common, xs, ys, left=ys[0], right=ys[-1])
                y_all.append(y_interp)

            if not y_all:
                continue

            y_all = np.array(y_all)
            y_mean = np.mean(y_all, axis=0)
            y_std = np.std(y_all, axis=0)

            for i, x_val in enumerate(x_common):
                # Find closest paper reference
                paper_val = None
                if ref_x and ref_y:
                    for rx, ry in zip(ref_x, ref_y):
                        if ry is not None and abs(rx - x_val) < 3:
                            paper_val = ry
                            break

                rows.append({
                    'metric': metric_name,
                    'demand_pct': int(x_val),
                    'policy': policy,
                    'mean_value': round(float(y_mean[i]), 2),
                    'std_value': round(float(y_std[i]), 2),
                    'n_seeds': len(y_all),
                    'paper_value': paper_val,
                })

    out_path = os.path.join(EXPORT_DIR, 'fgd_results.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'metric', 'demand_pct', 'policy',
            'mean_value', 'std_value', 'n_seeds', 'paper_value'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['metric'], r['demand_pct'], r['policy'])))

    print(f"Wrote {len(rows)} rows to {out_path}")


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print("Exporting results for student comparison...\n")
    export_gavel()
    print()
    export_fgd()
    print(f"\nAll exports saved to: {EXPORT_DIR}/")


if __name__ == '__main__':
    main()
