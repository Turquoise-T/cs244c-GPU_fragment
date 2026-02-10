#!/usr/bin/env python3
"""
Plot FGD vs Strided comparison figure from experiment results CSV.

Produces a figure similar to Gavel Fig 9:
  X-axis: Job Arrival Rate (jobs/hr)
  Y-axis: Average JCT (hours)
  Two lines: FGD (green, solid) vs Strided (red, dashed)
  Error bars from multiple seeds.

Usage:
  python plot_fgd_results.py
  python plot_fgd_results.py --csv ../results/results_fgd.csv --output ../figures/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'fgd': '#2ecc71',       # Green for FGD (the "good" strategy)
    'strided': '#e74c3c',   # Red for Strided / worst-fit (baseline)
}
MARKERS = {
    'fgd': 'o',
    'strided': 's',
}
LABELS = {
    'fgd': 'FGD (fragmentation-aware)',
    'strided': 'Strided (worst-fit)',
}


def load_data(csv_path):
    """Load results CSV."""
    df = pd.read_csv(csv_path)
    df['jct_hours'] = df['jct_sec'] / 3600.0
    df['makespan_hours'] = df['makespan_sec'] / 3600.0
    return df


def compute_stats(df):
    """Compute per-strategy, per-rate mean and std."""
    stats = {}
    for strategy in df['placement_strategy'].unique():
        sdf = df[df['placement_strategy'] == strategy]
        # Filter out inf
        sdf = sdf[sdf['jct_hours'].notna() & np.isfinite(sdf['jct_hours'])]
        grouped = sdf.groupby('jobs_per_hr')['jct_hours'].agg(
            ['mean', 'std', 'count']).reset_index()
        grouped.columns = ['jobs_per_hr', 'mean', 'std', 'count']
        # Fill NaN std (single seed) with 0
        grouped['std'] = grouped['std'].fillna(0)
        stats[strategy] = grouped
    return stats


def plot_jct_comparison(stats, output_path):
    """Plot JCT comparison: FGD vs Strided."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for strategy in ['strided', 'fgd']:
        if strategy not in stats or stats[strategy].empty:
            continue
        data = stats[strategy]
        ax.errorbar(
            data['jobs_per_hr'],
            data['mean'],
            yerr=data['std'],
            label=LABELS.get(strategy, strategy),
            color=COLORS.get(strategy, 'gray'),
            marker=MARKERS.get(strategy, 'x'),
            markersize=8,
            capsize=4,
            linewidth=2,
            linestyle='-' if strategy == 'fgd' else '--',
        )

    ax.set_xlabel('Job Arrival Rate (jobs/hr)', fontsize=12)
    ax.set_ylabel('Average JCT (hours)', fontsize=12)
    ax.set_title('GPU Sharing: FGD vs Strided Placement\n'
                 '(Cluster 8×V100 + 4×P100 + 4×K80, FIFO policy)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_improvement(stats, output_path):
    """Plot percentage improvement of FGD over Strided."""
    if 'fgd' not in stats or 'strided' not in stats:
        print("Need both fgd and strided data to plot improvement.")
        return

    fgd_data = stats['fgd'].set_index('jobs_per_hr')
    strided_data = stats['strided'].set_index('jobs_per_hr')

    common_rates = sorted(set(fgd_data.index) & set(strided_data.index))
    if not common_rates:
        print("No common job rates to compare.")
        return

    rates = []
    improvements = []
    for rate in common_rates:
        s_jct = strided_data.loc[rate, 'mean']
        f_jct = fgd_data.loc[rate, 'mean']
        if s_jct > 0 and np.isfinite(s_jct) and np.isfinite(f_jct):
            pct = (s_jct - f_jct) / s_jct * 100
            rates.append(rate)
            improvements.append(pct)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rates, improvements, width=0.6, color='#2ecc71', alpha=0.85,
           edgecolor='#27ae60')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.set_xlabel('Job Arrival Rate (jobs/hr)', fontsize=12)
    ax.set_ylabel('JCT Improvement (%)', fontsize=12)
    ax.set_title('FGD Improvement over Strided (% JCT Reduction)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {output_path}")


def print_summary(stats):
    """Print summary table."""
    if 'fgd' not in stats or 'strided' not in stats:
        return

    fgd_data = stats['fgd'].set_index('jobs_per_hr')
    strided_data = stats['strided'].set_index('jobs_per_hr')
    common_rates = sorted(set(fgd_data.index) & set(strided_data.index))

    print("\n" + "=" * 65)
    print(f"{'Rate':>6s}  {'Strided JCT':>12s}  {'FGD JCT':>12s}  "
          f"{'Improvement':>12s}")
    print("-" * 65)
    for rate in common_rates:
        s = strided_data.loc[rate, 'mean']
        f = fgd_data.loc[rate, 'mean']
        if s > 0 and np.isfinite(s) and np.isfinite(f):
            pct = (s - f) / s * 100
            print(f"{rate:6.1f}  {s:10.1f} h  {f:10.1f} h  {pct:+10.1f} %")
        else:
            print(f"{rate:6.1f}  {s:10.1f} h  {f:10.1f} h       N/A")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Plot FGD vs Strided comparison")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "results",
                                             "results_fgd.csv"),
                        help="Input CSV file")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Run experiments first with run_fgd_experiment.py")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from {args.csv}")
    df = load_data(args.csv)
    print(f"  Total rows: {len(df)}")
    print(f"  Strategies: {df['placement_strategy'].unique().tolist()}")
    print(f"  Rates: {sorted(df['jobs_per_hr'].unique().tolist())}")

    stats = compute_stats(df)

    # Plot 1: JCT comparison (main figure, like Fig 9)
    plot_jct_comparison(stats, os.path.join(args.output,
                                            "fgd_vs_strided_jct.png"))

    # Plot 2: Improvement bar chart
    plot_improvement(stats, os.path.join(args.output,
                                         "fgd_improvement.png"))

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
