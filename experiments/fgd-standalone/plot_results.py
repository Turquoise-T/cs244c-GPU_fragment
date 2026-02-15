#!/usr/bin/env python3
"""Plot all 6 FGD paper figures with reference overlay.

Generates:
  - Fig 7a: Frag Rate (%) vs demand
  - Fig 7b: Frag/Total (%) vs demand
  - Fig 9a: Unallocated GPU (%) vs demand
  - Fig 9b: Occupied Nodes vs demand
  - Fig 9c: Pending GPUs at 96% demand (stacked bar)
  - Fig 9d: Frag Breakdown at 96% demand (stacked bar)
  - comparison_all.png: 2x3 grid

Usage:
    python plot_results.py --results-dir results/full_run --output-dir figures/full_run
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

# Policy display names, colors, and reference-JSON key names
POLICY_STYLE = {
    'random':        {'color': '#888888', 'label': 'Random',        'ref_key': 'random'},
    'dotprod':       {'color': '#E69F00', 'label': 'DotProd',       'ref_key': 'dotprod'},
    'gpuclustering': {'color': '#009E73', 'label': 'GpuClustering', 'ref_key': 'clustering'},
    'gpupacking':    {'color': '#0072B2', 'label': 'GpuPacking',    'ref_key': 'packing'},
    'bestfit':       {'color': '#D55E00', 'label': 'BestFit',       'ref_key': 'bestfit'},
    'fgd':           {'color': '#000000', 'label': 'FGD',           'ref_key': 'fgd'},
}

POLICY_ORDER = ['random', 'gpuclustering', 'dotprod', 'gpupacking', 'bestfit', 'fgd']


def load_results(results_dir):
    """Load all results from JSON files."""
    combined_path = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            return json.load(f)
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith('.json') and fname != 'all_results.json':
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
    return results


def load_reference(ref_path):
    """Load paper reference curves JSON."""
    with open(ref_path) as f:
        return json.load(f)


def group_by_policy(results):
    """Group experiment results by policy name."""
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r['policy']].append(r['curve'])
    return by_policy


def interpolate_curves(by_policy, y_func, x_range=(0, 130), n_points=200):
    """Interpolate curves onto a common x-axis (demand_pct) for averaging.

    Args:
        by_policy: dict of policy -> list of curves
        y_func: callable(curve_point) -> y value
        x_range: (min, max) in demand_pct units (0-130)
        n_points: number of interpolation points

    Returns:
        dict: policy -> {'x': array, 'mean': array, 'std': array}
    """
    x_common = np.linspace(x_range[0], x_range[1], n_points)
    interpolated = {}

    for policy, curves in by_policy.items():
        y_all = []
        for curve in curves:
            if not curve:
                continue
            xs = np.array([pt['demand_fraction'] * 100 for pt in curve])
            ys = np.array([y_func(pt) for pt in curve])
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


def plot_line_figure(interpolated, ref_data, ylabel, title, output_path,
                     x_range=None, y_range=None):
    """Plot a line figure with reference (dashed) and our results (solid)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ref_x = ref_data.get('demand_pct', []) if ref_data else []

    for policy in POLICY_ORDER:
        style = POLICY_STYLE.get(policy, {'color': 'gray', 'label': policy})
        ref_key = style.get('ref_key', policy)

        # Reference curve (dashed)
        if ref_data and ref_key in ref_data:
            ref_y = ref_data[ref_key]
            valid = [(x, y) for x, y in zip(ref_x, ref_y) if y is not None]
            if valid:
                rx, ry = zip(*valid)
                ax.plot(rx, ry, color=style['color'], linestyle='--',
                        alpha=0.5, linewidth=1.5)

        # Our results (solid)
        if policy in interpolated:
            data = interpolated[policy]
            ax.plot(data['x'], data['mean'],
                    color=style['color'], label=style['label'], linewidth=2)
            if np.any(data['std'] > 0):
                ax.fill_between(data['x'],
                                data['mean'] - data['std'],
                                data['mean'] + data['std'],
                                color=style['color'], alpha=0.12)

    ax.set_xlabel('Arrived Workloads (% of GPU Capacity)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    if x_range:
        ax.set_xlim(*x_range)
    if y_range:
        ax.set_ylim(*y_range)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig


def plot_stacked_bar(our_data, ref_data, categories, cat_colors, ylabel,
                     title, output_path):
    """Plot grouped stacked bar chart comparing our results vs paper reference.

    Args:
        our_data: dict of policy -> dict of category -> value
        ref_data: dict of policy -> dict of category -> value
        categories: list of category keys in stacking order (bottom to top)
        cat_colors: list of colors for each category
        ylabel: y-axis label
        title: plot title
        output_path: where to save
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    policies_to_plot = [p for p in POLICY_ORDER
                        if p in our_data or POLICY_STYLE[p]['ref_key'] in (ref_data or {})]
    n = len(policies_to_plot)
    if n == 0:
        plt.close(fig)
        return

    x = np.arange(n)
    width = 0.35

    # Reference bars (left)
    if ref_data:
        bottoms = np.zeros(n)
        for ci, cat in enumerate(categories):
            vals = []
            for p in policies_to_plot:
                rk = POLICY_STYLE[p]['ref_key']
                vals.append(ref_data.get(rk, {}).get(cat, 0))
            vals = np.array(vals, dtype=float)
            label = f'{cat} (paper)' if ci == 0 else None
            ax.bar(x - width/2, vals, width, bottom=bottoms,
                   color=cat_colors[ci], alpha=0.4, edgecolor='gray',
                   hatch='//' if ci == 0 else None,
                   label=f'{cat} (paper)')
            bottoms += vals

    # Our bars (right)
    bottoms = np.zeros(n)
    for ci, cat in enumerate(categories):
        vals = []
        for p in policies_to_plot:
            vals.append(our_data.get(p, {}).get(cat, 0))
        vals = np.array(vals, dtype=float)
        ax.bar(x + width/2, vals, width, bottom=bottoms,
               color=cat_colors[ci], edgecolor='black', linewidth=0.5,
               label=f'{cat} (ours)')
        bottoms += vals

    labels = [POLICY_STYLE[p]['label'] for p in policies_to_plot]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")


def get_value_at_demand(curves, demand_pct, key_func):
    """Get mean value at a specific demand percentage across seeds."""
    vals = []
    for curve in curves:
        if not curve:
            continue
        closest = min(curve, key=lambda pt: abs(pt['demand_fraction'] * 100 - demand_pct))
        if abs(closest['demand_fraction'] * 100 - demand_pct) < 5:
            vals.append(key_func(closest))
    return np.mean(vals) if vals else None


def get_pending_at_demand(curves, demand_pct):
    """Get pending_by_gpu_size dict at a specific demand percentage."""
    all_pending = defaultdict(list)
    for curve in curves:
        if not curve:
            continue
        closest = min(curve, key=lambda pt: abs(pt['demand_fraction'] * 100 - demand_pct))
        if abs(closest['demand_fraction'] * 100 - demand_pct) < 5:
            pending = closest.get('pending_by_gpu_size', {})
            for k, v in pending.items():
                all_pending[k].append(v)
    return {k: np.mean(v) for k, v in all_pending.items()}


def get_frag_breakdown_at_demand(curves, demand_pct):
    """Get frag breakdown percentages at a specific demand percentage."""
    keys = ['frag_non_gpu_pct', 'frag_stranded_pct', 'frag_deficient_pct']
    all_vals = defaultdict(list)
    for curve in curves:
        if not curve:
            continue
        closest = min(curve, key=lambda pt: abs(pt['demand_fraction'] * 100 - demand_pct))
        if abs(closest['demand_fraction'] * 100 - demand_pct) < 5:
            for k in keys:
                all_vals[k].append(closest.get(k, 0))
    return {k: np.mean(v) for k, v in all_vals.items()}


def print_comparison_table(by_policy, ref, checkpoints=None):
    """Print numeric comparison at key demand points."""
    if checkpoints is None:
        checkpoints = [50, 80, 96, 100]

    metrics = [
        ('Frag/Total (%)', lambda pt: pt['frag_ratio'] * 100, 'fig7b_frag_over_total_pct'),
        ('Frag Rate (%)', lambda pt: (pt['fragmentation'] / pt['unallocated_gpus'] * 100
                                      if pt['unallocated_gpus'] > 0 else 100),
         'fig7a_frag_rate_pct'),
        ('Unalloc GPU (%)', lambda pt: (1 - pt['alloc_ratio']) * 100, 'fig9a_unalloc_gpu_pct'),
        ('Occupied Nodes', lambda pt: pt.get('occupied_nodes', 0), 'fig9b_occupied_nodes'),
    ]

    for metric_name, y_func, ref_key in metrics:
        ref_section = ref.get(ref_key, {})
        ref_x = ref_section.get('demand_pct', [])

        print(f"\n--- {metric_name} ---")
        header = f"{'Policy':15s}" + "".join(f"{'d=' + str(c) + '%':>18s}" for c in checkpoints)
        print(header)
        print("-" * len(header))

        for policy in POLICY_ORDER:
            if policy not in by_policy:
                continue
            label = POLICY_STYLE[policy]['label']
            rk = POLICY_STYLE[policy]['ref_key']
            row = f"{label:15s}"

            for cp in checkpoints:
                our_val = get_value_at_demand(by_policy[policy], cp, y_func)

                # Find reference value
                ref_val = None
                if rk in ref_section and ref_x:
                    ref_vals = ref_section[rk]
                    for rx, rv in zip(ref_x, ref_vals):
                        if rv is not None and abs(rx - cp) < 3:
                            ref_val = rv
                            break

                if our_val is not None and ref_val is not None:
                    row += f"  {our_val:6.1f} (ref {ref_val:5.1f})"
                elif our_val is not None:
                    row += f"  {our_val:6.1f}       ---  "
                else:
                    row += f"{'N/A':>18s}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot all 6 FGD paper figures')
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing result JSON files')
    parser.add_argument('--output-dir', default='figures/',
                        help='Output directory for figures')
    parser.add_argument('--ref-json', default=None,
                        help='Path to paper_reference_curves.json')

    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")

    # Auto-detect reference JSON location
    ref_path = args.ref_json
    if ref_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(script_dir, 'paper_reference_curves.json')
    if os.path.exists(ref_path):
        ref = load_reference(ref_path)
        print(f"Loaded reference curves from {ref_path}")
    else:
        ref = {}
        print("No reference curves found, plotting without reference overlay")

    os.makedirs(args.output_dir, exist_ok=True)
    by_policy = group_by_policy(results)

    # ---- Fig 7a: Frag Rate (%) ----
    def frag_rate_func(pt):
        if pt['unallocated_gpus'] > 0:
            return pt['fragmentation'] / pt['unallocated_gpus'] * 100
        return 100.0

    frag_rate_interp = interpolate_curves(by_policy, frag_rate_func)
    plot_line_figure(
        frag_rate_interp, ref.get('fig7a_frag_rate_pct', {}),
        ylabel='Frag Rate (%)',
        title='Fig 7a: Fragmentation Rate',
        output_path=os.path.join(args.output_dir, 'fig7a_frag_rate.png'),
        y_range=(0, 105),
    )

    # ---- Fig 7b: Frag/Total (%) ----
    frag_total_interp = interpolate_curves(
        by_policy, lambda pt: pt['frag_ratio'] * 100)
    plot_line_figure(
        frag_total_interp, ref.get('fig7b_frag_over_total_pct', {}),
        ylabel='Frag / Total (%)',
        title='Fig 7b: Fragmentation / Total GPUs',
        output_path=os.path.join(args.output_dir, 'fig7b_frag_over_total.png'),
    )

    # ---- Fig 9a: Unallocated GPU (%) ----
    unalloc_interp = interpolate_curves(
        by_policy, lambda pt: (1 - pt['alloc_ratio']) * 100,
        x_range=(0, 130))
    plot_line_figure(
        unalloc_interp, ref.get('fig9a_unalloc_gpu_pct', {}),
        ylabel='Unallocated GPU (%)',
        title='Fig 9a: Unallocated GPUs',
        output_path=os.path.join(args.output_dir, 'fig9a_unalloc_gpu.png'),
        x_range=(70, 125),
    )

    # ---- Fig 9b: Occupied Nodes ----
    occupied_interp = interpolate_curves(
        by_policy, lambda pt: pt.get('occupied_nodes', 0),
        x_range=(0, 110))
    plot_line_figure(
        occupied_interp, ref.get('fig9b_occupied_nodes', {}),
        ylabel='Occupied Nodes',
        title='Fig 9b: Occupied GPU Nodes',
        output_path=os.path.join(args.output_dir, 'fig9b_occupied_nodes.png'),
        x_range=(0, 110),
    )

    # ---- Fig 9c: Pending GPUs at 96% demand (stacked bar) ----
    our_pending = {}
    for policy in POLICY_ORDER:
        if policy in by_policy:
            our_pending[policy] = get_pending_at_demand(by_policy[policy], 96)

    ref_pending = ref.get('fig9c_pending_gpus_at_96pct', {})
    pending_cats = ['lt1_gpu', '1_gpu', '2_gpu', '8_gpu']
    pending_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

    plot_stacked_bar(
        our_pending, ref_pending, pending_cats, pending_colors,
        ylabel='Pending GPU-equivalents',
        title='Fig 9c: Pending GPUs at 96% Demand',
        output_path=os.path.join(args.output_dir, 'fig9c_pending_gpus.png'),
    )

    # ---- Fig 9d: Frag Breakdown at 96% demand (stacked bar) ----
    our_breakdown = {}
    for policy in POLICY_ORDER:
        if policy in by_policy:
            bd = get_frag_breakdown_at_demand(by_policy[policy], 96)
            our_breakdown[policy] = {
                'non_gpu': bd.get('frag_non_gpu_pct', 0),
                'stranded': bd.get('frag_stranded_pct', 0),
                'deficient': bd.get('frag_deficient_pct', 0),
            }

    ref_breakdown = ref.get('fig9d_frag_breakdown_pct', {})
    breakdown_cats = ['non_gpu', 'stranded', 'deficient']
    breakdown_colors = ['#e41a1c', '#ff7f00', '#377eb8']

    plot_stacked_bar(
        our_breakdown, ref_breakdown, breakdown_cats, breakdown_colors,
        ylabel='Fragmentation (%)',
        title='Fig 9d: Fragmentation Breakdown at 96% Demand',
        output_path=os.path.join(args.output_dir, 'fig9d_frag_breakdown.png'),
    )

    # ---- Combined 2x3 grid ----
    print("\nGenerating comparison_all.png ...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Helper to plot a line subplot
    def subplot_line(ax, interp, ref_data, ylabel, title, x_range=None, y_range=None):
        ref_x = ref_data.get('demand_pct', []) if ref_data else []
        for policy in POLICY_ORDER:
            style = POLICY_STYLE[policy]
            rk = style['ref_key']
            if ref_data and rk in ref_data:
                ref_y = ref_data[rk]
                valid = [(x, y) for x, y in zip(ref_x, ref_y) if y is not None]
                if valid:
                    rx, ry = zip(*valid)
                    ax.plot(rx, ry, color=style['color'], linestyle='--',
                            alpha=0.5, linewidth=1)
            if policy in interp:
                d = interp[policy]
                ax.plot(d['x'], d['mean'], color=style['color'],
                        label=style['label'], linewidth=1.5)
        ax.set_xlabel('Demand (%)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        if x_range:
            ax.set_xlim(*x_range)
        if y_range:
            ax.set_ylim(*y_range)

    # Helper to plot a bar subplot
    def subplot_bar(ax, our_data, ref_data, categories, cat_colors, ylabel, title):
        policies_to_plot = [p for p in POLICY_ORDER if p in our_data]
        n = len(policies_to_plot)
        if n == 0:
            return
        x = np.arange(n)
        width = 0.35
        if ref_data:
            bottoms = np.zeros(n)
            for ci, cat in enumerate(categories):
                vals = np.array([ref_data.get(POLICY_STYLE[p]['ref_key'], {}).get(cat, 0)
                                 for p in policies_to_plot], dtype=float)
                ax.bar(x - width/2, vals, width, bottom=bottoms,
                       color=cat_colors[ci], alpha=0.4, edgecolor='gray')
                bottoms += vals
        bottoms = np.zeros(n)
        for ci, cat in enumerate(categories):
            vals = np.array([our_data.get(p, {}).get(cat, 0)
                             for p in policies_to_plot], dtype=float)
            ax.bar(x + width/2, vals, width, bottom=bottoms,
                   color=cat_colors[ci], edgecolor='black', linewidth=0.5,
                   label=cat)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_STYLE[p]['label'] for p in policies_to_plot],
                           fontsize=8, rotation=15)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    subplot_line(axes[0, 0], frag_rate_interp,
                 ref.get('fig7a_frag_rate_pct', {}),
                 'Frag Rate (%)', 'Fig 7a', y_range=(0, 105))
    subplot_line(axes[0, 1], frag_total_interp,
                 ref.get('fig7b_frag_over_total_pct', {}),
                 'Frag/Total (%)', 'Fig 7b')
    subplot_line(axes[0, 2], unalloc_interp,
                 ref.get('fig9a_unalloc_gpu_pct', {}),
                 'Unalloc GPU (%)', 'Fig 9a', x_range=(70, 125))
    subplot_line(axes[1, 0], occupied_interp,
                 ref.get('fig9b_occupied_nodes', {}),
                 'Occupied Nodes', 'Fig 9b', x_range=(0, 110))
    subplot_bar(axes[1, 1], our_pending, ref_pending,
                pending_cats, pending_colors,
                'Pending GPUs', 'Fig 9c (96%)')
    subplot_bar(axes[1, 2], our_breakdown, ref_breakdown,
                breakdown_cats, breakdown_colors,
                'Frag (%)', 'Fig 9d (96%)')

    fig.suptitle('FGD Paper Replication: Solid=Ours, Dashed=Paper Reference',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'comparison_all.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {os.path.join(args.output_dir, 'comparison_all.png')}")

    # Print numeric comparison table
    print_comparison_table(by_policy, ref)


if __name__ == '__main__':
    main()
