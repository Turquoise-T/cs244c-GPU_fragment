#!/usr/bin/env python3
"""
Plot Gavel replication results from the combined runner vs paper reference curves.

Produces a 1x3 subplot grid (Figures 9, 10, 11) comparing our simulation
results against digitized reference data from the Gavel OSDI 2020 paper.

Input:
    - CSV from combined runner (columns: name, figure, policy, jobs_per_hr,
      seed, jct_sec, saturated, wall_time_sec, num_completed, multi_gpu)
    - Paper reference JSON with digitized curves for fig9/fig10/fig11

Output:
    - Single PNG with 1x3 subplots

Usage:
    python plot_gavel_replication.py
    python plot_gavel_replication.py --csv path/to/results.csv --output path/to/output.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Styling constants
# ------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'baseline': '#e74c3c',   # Red
    'gavel':    '#2ecc71',   # Green
    'packed':   '#3498db',   # Blue
}
MARKERS = {
    'baseline': 's',
    'gavel':    'o',
    'packed':   '^',
}

# Policy -> (role, display_name) mapping per figure
FIG_POLICIES = {
    'fig9': [
        ('max_min_fairness',        'baseline', 'Baseline'),
        ('max_min_fairness_perf',   'gavel',    'Gavel'),
        ('max_min_fairness_packed', 'packed',   'Gavel+Packing'),
    ],
    'fig10': [
        ('max_min_fairness',        'baseline', 'Baseline'),
        ('max_min_fairness_perf',   'gavel',    'Gavel'),
        ('max_min_fairness_packed', 'packed',   'Gavel+Packing'),
    ],
    'fig11': [
        ('finish_time_fairness',      'baseline', 'Baseline'),
        ('finish_time_fairness_perf', 'gavel',    'Gavel'),
    ],
}

FIG_TITLES = {
    'fig9':  'Figure 9: Single-GPU Jobs\n(Max-Min Fairness / LAS)',
    'fig10': 'Figure 10: Multi-GPU Jobs\n(Max-Min Fairness / LAS)',
    'fig11': 'Figure 11: Multi-GPU Jobs\n(Finish-Time Fairness)',
}

# Reference curve keys that exist in the JSON (only gavel and baseline)
REF_ROLE_MAP = {
    'gavel':    'gavel',
    'baseline': 'baseline',
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_csv(csv_path, include_saturated=False):
    """Load combined runner CSV and add derived columns."""
    df = pd.read_csv(csv_path)
    df['jct_sec'] = pd.to_numeric(df['jct_sec'], errors='coerce')
    df['jct_hours'] = df['jct_sec'] / 3600.0
    mask_inf = ~np.isfinite(df['jct_hours'])
    df.loc[mask_inf, 'jct_hours'] = np.nan
    if not include_saturated:
        # Exclude saturated experiments: partial JCT from incomplete windows is
        # unreliable due to extreme heavy-tail (top 10% of jobs = 95% of total
        # JCT).  Missing straggler jobs can change the mean by 10-20x.
        mask_saturated = df['saturated'].astype(str).str.lower().isin(
            ['true', '1', 'yes'])
        df.loc[mask_saturated, 'jct_hours'] = np.nan
    return df


def load_reference(ref_path):
    """Load paper reference curves JSON."""
    with open(ref_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Stats computation
# ------------------------------------------------------------------

def compute_policy_stats(df, figure, policy):
    """Return a DataFrame with jobs_per_hr, mean, std for one (figure, policy)."""
    subset = df[(df['figure'] == figure) & (df['policy'] == policy)].copy()
    subset = subset.dropna(subset=['jct_hours'])
    if subset.empty:
        return pd.DataFrame(columns=['jobs_per_hr', 'mean', 'std'])
    stats = (
        subset
        .groupby('jobs_per_hr')['jct_hours']
        .agg(['mean', 'std'])
        .reset_index()
    )
    stats.columns = ['jobs_per_hr', 'mean', 'std']
    stats['std'] = stats['std'].fillna(0)
    return stats


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_subplot(ax, fig_key, df, ref):
    """Draw one subplot (our data + paper reference)."""

    policies = FIG_POLICIES[fig_key]
    ref_data = ref.get(fig_key, {})
    ref_x = ref_data.get('jobs_per_hr', [])

    # -- Paper reference curves (behind our data) --
    for _policy, role, display in policies:
        ref_key = REF_ROLE_MAP.get(role)
        if ref_key is None:
            continue
        ref_y = ref_data.get(ref_key, [])
        if not ref_y:
            continue
        valid = [(x, y) for x, y in zip(ref_x, ref_y) if y is not None]
        if not valid:
            continue
        rx, ry = zip(*valid)
        ax.plot(
            rx, ry,
            color=COLORS[role],
            linestyle='--',
            alpha=0.4,
            linewidth=2.0,
            label=f'{display} (paper)',
            zorder=1,
        )

    # -- Our simulation results (solid, on top) --
    for policy, role, display in policies:
        stats = compute_policy_stats(df, fig_key, policy)
        if stats.empty:
            continue
        ax.errorbar(
            stats['jobs_per_hr'],
            stats['mean'],
            yerr=stats['std'],
            label=f'{display} (ours)',
            color=COLORS[role],
            marker=MARKERS[role],
            markersize=7,
            capsize=4,
            linewidth=2,
            linestyle='-',
            zorder=2,
        )

    # -- Axis formatting --
    ax.set_xlabel('Input Job Rate (jobs/hr)', fontsize=12)
    ax.set_ylabel('Average JCT (hours)', fontsize=12)
    ax.set_title(FIG_TITLES[fig_key], fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0, top=120)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


def make_figure(df, ref, output_path):
    """Create the 1x3 combined figure and save to disk."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, fig_key in enumerate(['fig9', 'fig10', 'fig11']):
        plot_subplot(axes[i], fig_key, df, ref)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {output_path}')


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Gavel replication results from combined runner')
    parser.add_argument(
        '--csv',
        default=None,
        help='Path to gavel_replication_combined.csv '
             '(default: ../results/gavel_replication_combined.csv)')
    parser.add_argument(
        '--ref-json',
        default=None,
        help='Path to paper_reference_curves.json '
             '(default: ../../gavel-replication/scripts/paper_reference_curves.json)')
    parser.add_argument(
        '--output',
        default=None,
        help='Output PNG path '
             '(default: ../figures/gavel_replication_combined.png)')
    parser.add_argument(
        '--include-saturated',
        action='store_true',
        help='Include saturated experiments (partial JCT from nearly-complete '
             'windows).  Use when window completion is 99%%+.')
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # -- Resolve CSV path --
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        csv_path = script_dir / '..' / 'results' / 'gavel_replication_combined.csv'
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        print(f'ERROR: CSV not found at {csv_path}', file=sys.stderr)
        print('Use --csv to specify the path.', file=sys.stderr)
        sys.exit(1)

    # -- Resolve reference JSON path --
    if args.ref_json is not None:
        ref_path = Path(args.ref_json)
    else:
        ref_path = (
            script_dir / '..' / '..' / 'gavel-replication'
            / 'scripts' / 'paper_reference_curves.json'
        )
    ref_path = ref_path.resolve()
    if ref_path.exists():
        ref = load_reference(ref_path)
        print(f'Loaded reference curves from {ref_path}')
    else:
        ref = {}
        print(f'Warning: reference JSON not found at {ref_path}, '
              'plotting without paper overlay')

    # -- Resolve output path --
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = script_dir / '..' / 'figures' / 'gavel_replication_combined.png'
    output_path = output_path.resolve()

    # -- Load and plot --
    print(f'Loading CSV from {csv_path} ...')
    df = load_csv(csv_path, include_saturated=args.include_saturated)
    print(f'  Total rows: {len(df)}')
    print(f'  Rows with valid JCT: {df["jct_hours"].notna().sum()}')

    make_figure(df, ref, output_path)


if __name__ == '__main__':
    main()
