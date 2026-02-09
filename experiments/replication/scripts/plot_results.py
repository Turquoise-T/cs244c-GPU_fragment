#!/usr/bin/env python3
"""
Generate Figures 9, 10, 11 from Gavel paper replication results.

Compares heterogeneity-aware Gavel (_perf) vs heterogeneity-agnostic baseline,
with optional overlay of digitized reference curves from the OSDI 2020 paper.

Usage:
    python plot_results.py
    python plot_results.py --results results_combined.csv --ref-json paper_reference_curves.json
"""

import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'gavel': '#2ecc71',      # Green for Gavel (heterogeneity-aware)
    'baseline': '#e74c3c',   # Red for baseline (heterogeneity-agnostic)
}
MARKERS = {
    'gavel': 'o',
    'baseline': 's',
}


def load_and_prepare_data(csv_path):
    """Load results CSV and prepare for plotting."""
    df = pd.read_csv(csv_path)
    df['jct_sec'] = pd.to_numeric(df['jct_sec'], errors='coerce')
    df['jct_hours'] = df['jct_sec'] / 3600
    df['figure'] = df['name'].str.extract(r'(fig\d+)')[0]
    df['is_gavel'] = df['policy'].str.contains('_perf')
    return df


def load_reference(ref_path):
    """Load paper reference curves JSON."""
    with open(ref_path) as f:
        return json.load(f)


def compute_stats(df, figure, gavel_policy, baseline_policy):
    """Compute mean and std for each job rate."""
    fig_data = df[df['figure'] == figure].copy()
    fig_data = fig_data[fig_data['jct_hours'].notna() & (fig_data['jct_hours'] != float('inf'))]

    results = {}
    for policy, label in [(gavel_policy, 'gavel'), (baseline_policy, 'baseline')]:
        policy_data = fig_data[fig_data['policy'] == policy]
        stats = policy_data.groupby('jobs_per_hr')['jct_hours'].agg(['mean', 'std', 'count'])
        stats = stats.reset_index()
        stats.columns = ['jobs_per_hr', 'mean', 'std', 'count']
        stats = stats[stats['count'] >= 2]
        results[label] = stats

    return results


def plot_figure(ax, stats, ref_data, title, xlabel, ylabel, xlim=None, ylim=None):
    """Plot a single figure with Gavel vs baseline, plus optional paper reference."""

    # Plot paper reference curves first (behind our data)
    if ref_data:
        ref_x = ref_data.get('jobs_per_hr', [])
        for key, display_name in [('gavel', 'Gavel (paper)'), ('baseline', 'Baseline (paper)')]:
            ref_y = ref_data.get(key, [])
            if not ref_y:
                continue
            # Filter out null values
            valid = [(x, y) for x, y in zip(ref_x, ref_y) if y is not None]
            if valid:
                rx, ry = zip(*valid)
                ax.plot(rx, ry, color=COLORS[key], linestyle='--',
                        alpha=0.45, linewidth=2.0, label=display_name, zorder=1)

    # Plot our results (solid, on top)
    for label, data in stats.items():
        if data.empty:
            continue
        display_name = f"{'Gavel' if label == 'gavel' else 'Baseline'} (ours)"
        ax.errorbar(
            data['jobs_per_hr'],
            data['mean'],
            yerr=data['std'],
            label=display_name,
            color=COLORS[label],
            marker=MARKERS[label],
            markersize=7,
            capsize=4,
            linewidth=2,
            linestyle='-',
            zorder=2,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.3)


def print_comparison_table(all_stats, ref, fig_keys):
    """Print numeric comparison at key job rates."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Ours vs Paper Reference")
    print("=" * 80)

    for fig_key, stats in zip(fig_keys, all_stats):
        ref_data = ref.get(fig_key, {})
        ref_x = ref_data.get('jobs_per_hr', [])
        ref_gavel = ref_data.get('gavel', [])
        ref_baseline = ref_data.get('baseline', [])

        print(f"\n--- {fig_key.upper()} ---")
        header = f"{'Rate':>8s}  {'Ours Gavel':>12s} {'Paper Gavel':>12s} {'Ours Base':>12s} {'Paper Base':>12s}"
        print(header)
        print("-" * len(header))

        # Gather our job rates
        our_rates = set()
        if not stats['gavel'].empty:
            our_rates |= set(stats['gavel']['jobs_per_hr'].round(1))
        if not stats['baseline'].empty:
            our_rates |= set(stats['baseline']['jobs_per_hr'].round(1))

        for rate in sorted(our_rates):
            # Our values
            our_g = stats['gavel']
            our_g_val = our_g[our_g['jobs_per_hr'].round(1) == rate]['mean'].values
            our_g_str = f"{our_g_val[0]:.1f}h" if len(our_g_val) > 0 else "---"

            our_b = stats['baseline']
            our_b_val = our_b[our_b['jobs_per_hr'].round(1) == rate]['mean'].values
            our_b_str = f"{our_b_val[0]:.1f}h" if len(our_b_val) > 0 else "---"

            # Paper values (find closest)
            ref_g_str = "---"
            ref_b_str = "---"
            if ref_x:
                for rx, rg, rb in zip(ref_x, ref_gavel, ref_baseline):
                    if abs(rx - rate) < 0.15:
                        if rg is not None:
                            ref_g_str = f"{rg:.1f}h"
                        if rb is not None:
                            ref_b_str = f"{rb:.1f}h"
                        break

            print(f"{rate:>7.1f}   {our_g_str:>12s} {ref_g_str:>12s} {our_b_str:>12s} {ref_b_str:>12s}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Gavel replication results with paper reference overlay')
    parser.add_argument('--results', default=None,
                        help='Path to results_combined.csv')
    parser.add_argument('--ref-json', default=None,
                        help='Path to paper_reference_curves.json')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent

    # Auto-detect results file
    results_path = args.results
    if results_path is None:
        candidates = [
            repo_dir / 'results' / 'results_combined.csv',
            script_dir / 'results_combined.csv',
        ]
        for c in candidates:
            if c.exists():
                results_path = c
                break
    if results_path is None or not Path(results_path).exists():
        print(f"ERROR: Cannot find results_combined.csv. Use --results to specify.")
        return

    # Auto-detect reference JSON
    ref_path = args.ref_json
    if ref_path is None:
        ref_path = script_dir / 'paper_reference_curves.json'
    if Path(ref_path).exists():
        ref = load_reference(ref_path)
        print(f"Loaded reference curves from {ref_path}")
    else:
        ref = {}
        print("No reference curves found, plotting without paper overlay")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else repo_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = load_and_prepare_data(results_path)
    print(f"Total experiments: {len(df)}")
    print(f"Experiments with finite JCT: {df['jct_hours'].notna().sum()}")

    # Figure configurations
    fig_configs = [
        {
            'key': 'fig9',
            'gavel_policy': 'max_min_fairness_perf',
            'baseline_policy': 'max_min_fairness',
            'title': 'Figure 9: Single-GPU Jobs\n(Max-Min Fairness / LAS)',
            'xlim': (0, 8),
            'ylim': (0, 100),
        },
        {
            'key': 'fig10',
            'gavel_policy': 'max_min_fairness_perf',
            'baseline_policy': 'max_min_fairness',
            'title': 'Figure 10: Multi-GPU Jobs\n(Max-Min Fairness / LAS)',
            'xlim': (0, 4.5),
            'ylim': (0, 100),
        },
        {
            'key': 'fig11',
            'gavel_policy': 'finish_time_fairness_perf',
            'baseline_policy': 'finish_time_fairness',
            'title': 'Figure 11: Multi-GPU Jobs\n(Finish-Time Fairness)',
            'xlim': (0, 4.5),
            'ylim': (0, 100),
        },
    ]

    # Compute stats for each figure
    all_stats = []
    for cfg in fig_configs:
        print(f"\nComputing stats for {cfg['key']}...")
        stats = compute_stats(df, cfg['key'], cfg['gavel_policy'], cfg['baseline_policy'])
        all_stats.append(stats)

    # --- Combined figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, (cfg, stats) in enumerate(zip(fig_configs, all_stats)):
        ref_data = ref.get(cfg['key'], {})
        plot_figure(
            axes[i], stats, ref_data,
            title=cfg['title'],
            xlabel='Job Arrival Rate (jobs/hr)',
            ylabel='Average JCT (hours)',
            xlim=cfg['xlim'],
            ylim=cfg['ylim'],
        )

    suptitle = 'Gavel Replication: Solid=Ours, Dashed=Paper Reference (OSDI 2020)'
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()

    combined_path = output_dir / 'gavel_comparison_all.png'
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved combined figure to: {combined_path}")

    # --- Individual figures ---
    for i, (cfg, stats) in enumerate(zip(fig_configs, all_stats)):
        fig_single, ax_single = plt.subplots(figsize=(7, 5.5))
        ref_data = ref.get(cfg['key'], {})
        plot_figure(
            ax_single, stats, ref_data,
            title=cfg['title'],
            xlabel='Job Arrival Rate (jobs/hr)',
            ylabel='Average JCT (hours)',
            xlim=cfg['xlim'],
            ylim=cfg['ylim'],
        )
        if ref:
            fig_single.text(0.5, -0.02, 'Solid=Ours, Dashed=Paper Reference',
                            ha='center', fontsize=9, style='italic',
                            transform=fig_single.transFigure)
        fig_single.tight_layout()

        single_path = output_dir / f'{cfg["key"]}_comparison.png'
        fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved {single_path}")

    # --- Summary statistics ---
    print("\n" + "=" * 60)
    print("OUR RESULTS SUMMARY")
    print("=" * 60)

    for cfg, stats in zip(fig_configs, all_stats):
        print(f"\n{cfg['key'].upper()}:")
        if stats['gavel'].empty or stats['baseline'].empty:
            print("  Insufficient data for comparison")
            continue

        common_rates = set(stats['gavel']['jobs_per_hr']) & set(stats['baseline']['jobs_per_hr'])
        for rate in sorted(common_rates):
            gavel_jct = stats['gavel'][stats['gavel']['jobs_per_hr'] == rate]['mean'].values[0]
            baseline_jct = stats['baseline'][stats['baseline']['jobs_per_hr'] == rate]['mean'].values[0]
            improvement = (baseline_jct - gavel_jct) / baseline_jct * 100
            print(f"  {rate:.1f} jobs/hr: Gavel={gavel_jct:.1f}h, Baseline={baseline_jct:.1f}h, Improvement={improvement:.1f}%")

    # --- Paper comparison table ---
    if ref:
        fig_keys = [cfg['key'] for cfg in fig_configs]
        print_comparison_table(all_stats, ref, fig_keys)


if __name__ == '__main__':
    main()
