#!/usr/bin/env python3
"""
Generate Figures 9, 10, 11 from Gavel paper replication results.

Compares heterogeneity-aware Gavel (_perf) vs heterogeneity-agnostic baseline.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_FILE = Path(__file__).parent / "results_combined.csv"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

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

    # Convert inf to NaN for proper handling
    df['jct_sec'] = pd.to_numeric(df['jct_sec'], errors='coerce')

    # Convert JCT from seconds to hours
    df['jct_hours'] = df['jct_sec'] / 3600

    # Extract figure number from name
    df['figure'] = df['name'].str.extract(r'(fig\d+)')[0]

    # Determine if this is the Gavel (_perf) or baseline version
    df['is_gavel'] = df['policy'].str.contains('_perf')

    return df


def compute_stats(df, figure, gavel_policy, baseline_policy):
    """Compute mean and std for each job rate."""
    fig_data = df[df['figure'] == figure].copy()

    # Filter out inf values
    fig_data = fig_data[fig_data['jct_hours'].notna() & (fig_data['jct_hours'] != float('inf'))]

    results = {}
    for policy, label in [(gavel_policy, 'gavel'), (baseline_policy, 'baseline')]:
        policy_data = fig_data[fig_data['policy'] == policy]

        # Group by job rate and compute stats
        stats = policy_data.groupby('jobs_per_hr')['jct_hours'].agg(['mean', 'std', 'count'])
        stats = stats.reset_index()
        stats.columns = ['jobs_per_hr', 'mean', 'std', 'count']

        # Only keep points with all 3 seeds (or at least 2)
        stats = stats[stats['count'] >= 2]

        results[label] = stats

    return results


def plot_figure(ax, stats, title, xlabel, ylabel, xlim=None, ylim=None):
    """Plot a single figure with Gavel vs baseline comparison."""
    for label, data in stats.items():
        if data.empty:
            continue

        ax.errorbar(
            data['jobs_per_hr'],
            data['mean'],
            yerr=data['std'],
            label=f"{'Gavel' if label == 'gavel' else 'Baseline'} ({label})",
            color=COLORS[label],
            marker=MARKERS[label],
            markersize=8,
            capsize=4,
            linewidth=2,
            linestyle='-' if label == 'gavel' else '--'
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.3)


def main():
    print("Loading results...")
    df = load_and_prepare_data(RESULTS_FILE)

    print(f"Total experiments: {len(df)}")
    print(f"Experiments with finite JCT: {df['jct_hours'].notna().sum()}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Figure 9: Single-GPU, Max-Min Fairness (LAS)
    print("\nPlotting Figure 9 (Single-GPU, Max-Min Fairness)...")
    stats_fig9 = compute_stats(df, 'fig9', 'max_min_fairness_perf', 'max_min_fairness')
    plot_figure(
        axes[0], stats_fig9,
        title='Figure 9: Single-GPU Jobs\n(Max-Min Fairness)',
        xlabel='Job Arrival Rate (jobs/hr)',
        ylabel='Average JCT (hours)',
        xlim=(0, 8),
        ylim=(0, 100),
    )

    # Figure 10: Multi-GPU, Max-Min Fairness (LAS)
    print("Plotting Figure 10 (Multi-GPU, Max-Min Fairness)...")
    stats_fig10 = compute_stats(df, 'fig10', 'max_min_fairness_perf', 'max_min_fairness')
    plot_figure(
        axes[1], stats_fig10,
        title='Figure 10: Multi-GPU Jobs\n(Max-Min Fairness)',
        xlabel='Job Arrival Rate (jobs/hr)',
        ylabel='Average JCT (hours)',
        xlim=(0, 4.5),
        ylim=(0, 100),
    )

    # Figure 11: Multi-GPU, Finish-Time Fairness (FTF)
    print("Plotting Figure 11 (Multi-GPU, Finish-Time Fairness)...")
    stats_fig11 = compute_stats(df, 'fig11', 'finish_time_fairness_perf', 'finish_time_fairness')
    plot_figure(
        axes[2], stats_fig11,
        title='Figure 11: Multi-GPU Jobs\n(Finish-Time Fairness)',
        xlabel='Job Arrival Rate (jobs/hr)',
        ylabel='Average JCT (hours)',
        xlim=(0, 4.5),
        ylim=(0, 100),
    )

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'gavel_replication_figures.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined figure to: {output_path}")

    # Also save individual figures
    for i, (fig_name, title) in enumerate([
        ('fig9', 'Figure 9: Single-GPU Max-Min Fairness'),
        ('fig10', 'Figure 10: Multi-GPU Max-Min Fairness'),
        ('fig11', 'Figure 11: Multi-GPU Finish-Time Fairness'),
    ]):
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        stats = [stats_fig9, stats_fig10, stats_fig11][i]
        xlim = [(0, 8), (0, 4.5), (0, 4.5)][i]
        plot_figure(ax_single, stats, title, 'Job Arrival Rate (jobs/hr)', 'Average JCT (hours)', xlim, ylim=(0, 100))

        single_path = OUTPUT_DIR / f'{fig_name}_replication.png'
        fig_single.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved {single_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for fig_name, stats in [('Fig 9', stats_fig9), ('Fig 10', stats_fig10), ('Fig 11', stats_fig11)]:
        print(f"\n{fig_name}:")
        if stats['gavel'].empty or stats['baseline'].empty:
            print("  Insufficient data for comparison")
            continue

        # Find common job rates
        common_rates = set(stats['gavel']['jobs_per_hr']) & set(stats['baseline']['jobs_per_hr'])

        for rate in sorted(common_rates):
            gavel_jct = stats['gavel'][stats['gavel']['jobs_per_hr'] == rate]['mean'].values[0]
            baseline_jct = stats['baseline'][stats['baseline']['jobs_per_hr'] == rate]['mean'].values[0]
            improvement = (baseline_jct - gavel_jct) / baseline_jct * 100
            print(f"  {rate:.1f} jobs/hr: Gavel={gavel_jct:.1f}h, Baseline={baseline_jct:.1f}h, Improvement={improvement:.1f}%")

    # plt.show()  # Commented out for non-interactive use


if __name__ == '__main__':
    main()
