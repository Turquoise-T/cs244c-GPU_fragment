#!/usr/bin/env python3
"""Parse raw results and generate Figure 8 and 9 plots."""

import csv
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_raw_results(raw_file):
    """Parse the raw CSV output from shell extraction."""
    results = []

    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: ./fig9/v100=36.p100=36.k80=36/gandiva/seed=1/lambda=7200.000000.log,62354.153,0.141
            parts = line.split(',')
            if len(parts) != 3:
                continue

            path, jct_str, util_str = parts

            # Parse path
            m = re.match(r'\./(\w+)/v100=(\d+)\.p100=(\d+)\.k80=(\d+)/(\w+)/seed=(\d+)/lambda=([\d.]+)\.log', path)
            if not m:
                print(f"Could not parse: {path}")
                continue

            figure, v100, p100, k80, policy, seed, lam = m.groups()

            results.append({
                'figure': figure,
                'policy': policy,
                'seed': int(seed),
                'lambda': float(lam),
                'input_job_rate': 3600.0 / float(lam),  # jobs per hour
                'avg_jct': float(jct_str),
                'avg_jct_hours': float(jct_str) / 3600.0,
                'utilization': float(util_str),
            })

    return results

def plot_figure_9(results, output_file='figure9.png'):
    """Plot Figure 9: JCT vs Input Job Rate for multi-GPU jobs."""
    fig9_results = [r for r in results if r['figure'] == 'fig9']

    if not fig9_results:
        print("No Figure 9 results found")
        return

    # Group by policy, then average across seeds
    by_policy = defaultdict(lambda: defaultdict(list))
    for r in fig9_results:
        by_policy[r['policy']][r['input_job_rate']].append(r['avg_jct_hours'])

    # Policy display names and colors (matching paper style)
    policy_names = {
        'gandiva': 'Gandiva',
        'fifo': 'FIFO',
        'max_min_fairness': 'Max-Min Fairness',
        'max_min_fairness_perf': 'Max-Min Fairness (Perf)',
        'finish_time_fairness': 'Finish-Time Fairness',
    }

    policy_colors = {
        'gandiva': '#1f77b4',
        'fifo': '#ff7f0e',
        'max_min_fairness': '#2ca02c',
        'max_min_fairness_perf': '#d62728',
        'finish_time_fairness': '#9467bd',
    }

    plt.figure(figsize=(10, 6))

    for policy in sorted(by_policy.keys()):
        rates = sorted(by_policy[policy].keys())
        means = [np.mean(by_policy[policy][r]) for r in rates]
        stds = [np.std(by_policy[policy][r]) for r in rates]

        label = policy_names.get(policy, policy)
        color = policy_colors.get(policy, None)

        plt.errorbar(rates, means, yerr=stds, marker='o', label=label,
                    color=color, capsize=3, linewidth=2, markersize=6)

    plt.xlabel('Input Job Rate (jobs/hour)', fontsize=12)
    plt.ylabel('Average JCT (hours)', fontsize=12)
    plt.title('Figure 9: JCT vs Load (Multi-GPU Jobs)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")

def plot_figure_8(results, output_file='figure8.png'):
    """Plot Figure 8: JCT vs Input Job Rate for single-GPU jobs."""
    fig8_results = [r for r in results if r['figure'] == 'fig8']

    if not fig8_results:
        print("No Figure 8 results found")
        return

    # Group by policy, then average across seeds
    by_policy = defaultdict(lambda: defaultdict(list))
    for r in fig8_results:
        by_policy[r['policy']][r['input_job_rate']].append(r['avg_jct_hours'])

    policy_names = {
        'gandiva': 'Gandiva',
        'fifo': 'FIFO',
        'max_min_fairness': 'Max-Min Fairness',
        'max_min_fairness_perf': 'Max-Min Fairness (Perf)',
        'finish_time_fairness': 'Finish-Time Fairness',
    }

    policy_colors = {
        'gandiva': '#1f77b4',
        'fifo': '#ff7f0e',
        'max_min_fairness': '#2ca02c',
        'max_min_fairness_perf': '#d62728',
        'finish_time_fairness': '#9467bd',
    }

    plt.figure(figsize=(10, 6))

    for policy in sorted(by_policy.keys()):
        rates = sorted(by_policy[policy].keys())
        means = [np.mean(by_policy[policy][r]) for r in rates]
        stds = [np.std(by_policy[policy][r]) for r in rates]

        label = policy_names.get(policy, policy)
        color = policy_colors.get(policy, None)

        plt.errorbar(rates, means, yerr=stds, marker='o', label=label,
                    color=color, capsize=3, linewidth=2, markersize=6)

    plt.xlabel('Input Job Rate (jobs/hour)', fontsize=12)
    plt.ylabel('Average JCT (hours)', fontsize=12)
    plt.title('Figure 8: JCT vs Load (Single-GPU Jobs)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")

def print_summary(results):
    """Print a summary of results."""
    by_figure = defaultdict(list)
    for r in results:
        by_figure[r['figure']].append(r)

    print("\n=== Results Summary ===")
    for figure in sorted(by_figure.keys()):
        fig_results = by_figure[figure]
        policies = set(r['policy'] for r in fig_results)
        seeds = set(r['seed'] for r in fig_results)
        rates = set(r['input_job_rate'] for r in fig_results)

        print(f"\n{figure}:")
        print(f"  Total results: {len(fig_results)}")
        print(f"  Policies: {', '.join(sorted(policies))}")
        print(f"  Seeds: {sorted(seeds)}")
        print(f"  Job rates: {sorted(rates)}")

        # Show sample result per policy
        for policy in sorted(policies):
            policy_results = [r for r in fig_results if r['policy'] == policy]
            avg_jct = np.mean([r['avg_jct_hours'] for r in policy_results])
            print(f"    {policy}: avg JCT = {avg_jct:.2f} hours ({len(policy_results)} runs)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/tmp/gavel_results_raw.csv', help='Raw results CSV')
    parser.add_argument('--output-dir', default='.', help='Output directory for figures')
    args = parser.parse_args()

    print(f"Parsing results from {args.input}...")
    results = parse_raw_results(args.input)
    print(f"Loaded {len(results)} results")

    print_summary(results)

    # Generate figures
    import os
    plot_figure_8(results, os.path.join(args.output_dir, 'figure8.png'))
    plot_figure_9(results, os.path.join(args.output_dir, 'figure9.png'))

if __name__ == '__main__':
    main()
