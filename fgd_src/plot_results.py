#!/usr/bin/env python3
"""Plot FGD paper replication results.

Generates:
  - Fig 7(b): Fragmentation ratio vs demand fraction
  - Fig 9(a): Allocation ratio vs demand fraction

Usage:
    python plot_results.py --results-dir results/ --output-dir figures/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Policy display names and colors matching the paper
POLICY_STYLE = {
    'random':        {'color': '#888888', 'label': 'Random'},
    'dotprod':       {'color': '#E69F00', 'label': 'DotProd'},
    'gpuclustering': {'color': '#009E73', 'label': 'GpuClustering'},
    'gpupacking':    {'color': '#0072B2', 'label': 'GpuPacking'},
    'bestfit':       {'color': '#D55E00', 'label': 'BestFit'},
    'fgd':           {'color': '#000000', 'label': 'FGD'},
}

# Order policies appear in legend (worst to best expected)
POLICY_ORDER = ['random', 'gpuclustering', 'dotprod', 'gpupacking', 'bestfit', 'fgd']


def load_results(results_dir):
    """Load all results from JSON files."""
    combined_path = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            return json.load(f)

    # Fall back to loading individual policy files
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith('.json') and fname != 'all_results.json':
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
    return results


def interpolate_curves(results, x_key, y_key, x_range=(0, 1.3), n_points=200):
    """Interpolate curves onto a common x-axis for averaging.

    Returns:
        dict: policy -> {'x': array, 'mean': array, 'std': array}
    """
    x_common = np.linspace(x_range[0], x_range[1], n_points)

    # Group by policy
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r['policy']].append(r['curve'])

    interpolated = {}
    for policy, curves in by_policy.items():
        y_all = []
        for curve in curves:
            if not curve:
                continue
            xs = np.array([pt[x_key] for pt in curve])
            ys = np.array([pt[y_key] for pt in curve])
            # Interpolate onto common x-axis
            y_interp = np.interp(x_common, xs, ys, left=ys[0], right=ys[-1])
            y_all.append(y_interp)

        if y_all:
            y_all = np.array(y_all)
            interpolated[policy] = {
                'x': x_common,
                'mean': np.mean(y_all, axis=0),
                'std': np.std(y_all, axis=0),
            }

    return interpolated


def plot_figure(interpolated, ylabel, title, output_path):
    """Plot a single figure with one line per policy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in POLICY_ORDER:
        if policy not in interpolated:
            continue
        data = interpolated[policy]
        style = POLICY_STYLE.get(policy, {'color': 'gray', 'label': policy})

        ax.plot(data['x'], data['mean'],
                color=style['color'], label=style['label'], linewidth=2)

        if len(data['std']) > 0 and np.any(data['std'] > 0):
            ax.fill_between(data['x'],
                            data['mean'] - data['std'],
                            data['mean'] + data['std'],
                            color=style['color'], alpha=0.15)

    ax.set_xlabel('Demand / Capacity', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")


def print_summary_table(results):
    """Print summary statistics at key demand fractions."""
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r['policy']].append(r['curve'])

    checkpoints = [0.5, 0.8, 1.0, 1.2, 1.3]

    print("\n--- Fragmentation Ratio at Key Demand Points ---")
    header = f"{'Policy':15s}" + "".join(f"{'d='+str(c):>10s}" for c in checkpoints)
    print(header)
    print("-" * len(header))

    for policy in POLICY_ORDER:
        if policy not in by_policy:
            continue
        row = f"{POLICY_STYLE.get(policy, {}).get('label', policy):15s}"
        for cp in checkpoints:
            vals = []
            for curve in by_policy[policy]:
                # Find closest point
                closest = min(curve, key=lambda pt: abs(pt['demand_fraction'] - cp))
                if abs(closest['demand_fraction'] - cp) < 0.05:
                    vals.append(closest['frag_ratio'])
            if vals:
                row += f"{np.mean(vals):10.4f}"
            else:
                row += f"{'N/A':>10s}"
        print(row)

    print("\n--- Allocation Ratio at Key Demand Points ---")
    print(header)
    print("-" * len(header))

    for policy in POLICY_ORDER:
        if policy not in by_policy:
            continue
        row = f"{POLICY_STYLE.get(policy, {}).get('label', policy):15s}"
        for cp in checkpoints:
            vals = []
            for curve in by_policy[policy]:
                closest = min(curve, key=lambda pt: abs(pt['demand_fraction'] - cp))
                if abs(closest['demand_fraction'] - cp) < 0.05:
                    vals.append(closest['alloc_ratio'])
            if vals:
                row += f"{np.mean(vals):10.4f}"
            else:
                row += f"{'N/A':>10s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot FGD replication results')
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default='figures/',
                        help='Output directory for figures')

    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")

    os.makedirs(args.output_dir, exist_ok=True)

    # Fig 7(b): Fragmentation ratio
    frag_data = interpolate_curves(results, 'demand_fraction', 'frag_ratio')
    plot_figure(
        frag_data,
        ylabel='Fragmentation Ratio',
        title='Fig 7(b): GPU Fragmentation vs Demand',
        output_path=os.path.join(args.output_dir, 'fig7b_frag_ratio.png'),
    )

    # Fig 9(a): Allocation ratio
    alloc_data = interpolate_curves(results, 'demand_fraction', 'alloc_ratio')
    plot_figure(
        alloc_data,
        ylabel='Allocation Ratio',
        title='Fig 9(a): GPU Allocation vs Demand',
        output_path=os.path.join(args.output_dir, 'fig9a_alloc_ratio.png'),
    )

    # Print summary table
    print_summary_table(results)


if __name__ == '__main__':
    main()
