#!/usr/bin/env python3
"""Plot FGD replication results: 4 figures + 2x2 grid.

Figures:
  - Fig 7a: Fragmentation Rate (%) vs Utilization (%)
  - Fig 7b: Frag/Total (%) vs Utilization (%)
  - Fig 9a: Unallocated GPU (%) vs Utilization (%)
  - Fig 9b: Occupied Nodes vs Utilization (%)
  - comparison_all.png: 2x2 grid

Usage:
    python plot_fgd_replication.py --results-dir results/fgd_replication --output-dir figures/fgd_replication
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLACEMENT_STYLE = {
    'strided': {'color': '#56B4E9', 'label': 'Strided (Gavel)', 'marker': 's'},
    'random':  {'color': '#888888', 'label': 'Random',           'marker': 'o'},
    'bestfit': {'color': '#D55E00', 'label': 'BestFit',          'marker': '^'},
    'fgd':     {'color': '#000000', 'label': 'FGD',              'marker': 'D'},
}

PLACEMENT_ORDER = ['random', 'strided', 'bestfit', 'fgd']

PAPER_REF_POLICIES = {
    'random': 'random',
    'bestfit': 'bestfit',
    'fgd': 'fgd',
}


def load_results(results_dir):
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def load_reference(ref_path):
    with open(ref_path) as f:
        return json.load(f)


def get_placement_label(result):
    if not result.get('enable_fgd', False):
        return 'strided'
    return result.get('fgd_placement_mode', 'fgd')


def group_results(results):
    groups = defaultdict(list)
    for r in results:
        placement = get_placement_label(r)
        lam = r['lam']
        groups[(placement, lam)].append(r)
    return groups


def compute_series(results):
    groups = group_results(results)
    series = defaultdict(lambda: {'util': [], 'frag_rate': [], 'frag_total': [],
                                   'unalloc': [], 'nodes': [],
                                   'frag_rate_std': [], 'frag_total_std': [],
                                   'unalloc_std': [], 'nodes_std': []})
    for (placement, lam), runs in groups.items():
        if not all('avg_utilization' in r for r in runs):
            continue
        s = series[placement]
        s['util'].append(np.mean([r['avg_utilization'] for r in runs]))
        s['frag_rate'].append(np.mean([r['avg_frag_rate'] for r in runs]))
        s['frag_total'].append(np.mean([r['avg_frag_total'] for r in runs]))
        s['unalloc'].append(np.mean([r['avg_unalloc_pct'] for r in runs]))
        s['nodes'].append(np.mean([r['avg_occupied_nodes'] for r in runs]))
        s['frag_rate_std'].append(np.std([r['avg_frag_rate'] for r in runs]))
        s['frag_total_std'].append(np.std([r['avg_frag_total'] for r in runs]))
        s['unalloc_std'].append(np.std([r['avg_unalloc_pct'] for r in runs]))
        s['nodes_std'].append(np.std([r['avg_occupied_nodes'] for r in runs]))

    for placement in series:
        s = series[placement]
        order = np.argsort(s['util'])
        for key in s:
            s[key] = [s[key][i] for i in order]

    return series


def plot_figure(ax, series, y_key, y_label, reference=None, ref_figure=None):
    for placement in PLACEMENT_ORDER:
        if placement not in series:
            continue
        s = series[placement]
        style = PLACEMENT_STYLE[placement]
        ax.plot(s['util'], s[y_key],
                color=style['color'], marker=style['marker'],
                label=style['label'], linewidth=2, markersize=5)
        std_key = y_key + '_std'
        if std_key in s:
            y = np.array(s[y_key])
            err = np.array(s[std_key])
            ax.fill_between(s['util'], y - err, y + err,
                            alpha=0.15, color=style['color'])

    if reference and ref_figure and ref_figure in reference:
        for ref_policy, ref_key in PAPER_REF_POLICIES.items():
            if ref_key in reference[ref_figure]:
                ref_data = reference[ref_figure][ref_key]
                x = [p[0] for p in ref_data]
                y = [p[1] for p in ref_data]
                color = PLACEMENT_STYLE.get(ref_policy, {}).get('color', '#AAAAAA')
                ax.plot(x, y, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Utilization (%)')
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference', default=None,
                        help='Path to FGD paper_reference_curves.json')
    parser.add_argument('--policy', default=None,
                        help='Filter by policy name (e.g. fifo, max_min_fairness)')
    parser.add_argument('--title-suffix', default=None,
                        help='Extra text appended to the grid title')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    if args.policy:
        results = [r for r in results if r.get('policy') == args.policy]
        print(f"Filtered to {len(results)} results for policy={args.policy}")

    reference = None
    if args.reference:
        reference = load_reference(args.reference)

    series = compute_series(results)
    print(f"Placements: {list(series.keys())}")

    figures = [
        ('fig7a_frag_rate.png',    'frag_rate',  'Fragmentation Rate (%)',  'fig_7a'),
        ('fig7b_frag_total.png',   'frag_total', 'Frag/Total (%)',         'fig_7b'),
        ('fig9a_unalloc.png',      'unalloc',    'Unallocated GPU (%)',    'fig_9a'),
        ('fig9b_nodes.png',        'nodes',      'Occupied Nodes',         'fig_9b'),
    ]

    for fname, y_key, y_label, ref_fig in figures:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_figure(ax, series, y_key, y_label, reference, ref_fig)
        ax.set_title(y_label)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (fname, y_key, y_label, ref_fig) in enumerate(figures):
        ax = axes[idx // 2][idx % 2]
        plot_figure(ax, series, y_key, y_label, reference, ref_fig)
        ax.set_title(y_label)

    title = 'FGD Replication via Gavel (Alibaba Cluster)'
    if args.title_suffix:
        title += f' -- {args.title_suffix}'
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'comparison_all.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved comparison_all.png")


if __name__ == '__main__':
    main()
