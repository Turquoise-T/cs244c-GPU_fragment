#!/usr/bin/env python3
"""Plot FGD improvement experiment results."""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = os.path.dirname(__file__)


def load_results(subdir):
    """Load all JSON results from a subdirectory."""
    results = []
    pattern = os.path.join(RESULTS_DIR, 'results', subdir, 'exp_*.json')
    for path in glob.glob(pattern):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def extract_method_and_load(name):
    """Extract method name and load level from experiment name."""
    # e.g., "gavel_60jph_s0" -> ("gavel", 60)
    # e.g., "gavelfgd_improved_110jph_s1" -> ("gavelfgd_improved", 110)
    parts = name.rsplit('_', 2)  # Split from right: [method, Xjph, sY]
    if len(parts) >= 2:
        load_part = parts[-2]  # e.g., "60jph"
        if 'jph' in load_part:
            load = int(load_part.replace('jph', ''))
            method = '_'.join(parts[:-2])
            return method, load
    return name, 0


def aggregate_by_method_and_load(results, include_saturated=True):
    """Aggregate results by method and load level."""
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        name = r.get('name', '')
        method, load = extract_method_and_load(name)
        jct = r.get('avg_jct', float('inf'))
        saturated = r.get('saturated', False)
        # Include saturated results since they still have valid JCT values
        if not include_saturated and saturated:
            jct = float('inf')
        data[method][load].append(jct)
    return data


def compute_stats(data):
    """Compute mean and std for each method and load."""
    stats = {}
    for method, loads in data.items():
        stats[method] = {'loads': [], 'mean': [], 'std': []}
        for load in sorted(loads.keys()):
            jcts = [j for j in loads[load] if j < float('inf')]
            if jcts:
                stats[method]['loads'].append(load)
                stats[method]['mean'].append(np.mean(jcts))
                stats[method]['std'].append(np.std(jcts))
    return stats


def plot_gavelfgd_improved():
    """Plot gavelfgd_improved comparison."""
    results = load_results('gavelfgd_improved')
    if not results:
        print("No results found for gavelfgd_improved")
        return

    data = aggregate_by_method_and_load(results)
    stats = compute_stats(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'gavel': 'blue', 'gavelfgd': 'orange', 'gavelfgd_improved': 'green'}
    labels = {'gavel': 'Gavel (strided)', 'gavelfgd': 'Gavel+FGD (vanilla)',
              'gavelfgd_improved': 'Gavel+FGD (paper+buddy)'}

    for method in ['gavel', 'gavelfgd', 'gavelfgd_improved']:
        if method in stats:
            s = stats[method]
            ax.errorbar(s['loads'], s['mean'], yerr=s['std'],
                       label=labels.get(method, method),
                       color=colors.get(method, None),
                       marker='o', capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel('Job Arrival Rate (jobs/hour)', fontsize=12)
    ax.set_ylabel('Average JCT (seconds)', fontsize=12)
    ax.set_title('Gavel+FGD Improvements: JCT vs Load', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'results', 'gavelfgd_improved_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_gavelfgd_all_improvements():
    """Plot gavelfgd_all_improvements comparison."""
    results = load_results('gavelfgd_all_improvements')
    if not results:
        print("No results found for gavelfgd_all_improvements")
        return

    data = aggregate_by_method_and_load(results)
    stats = compute_stats(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'gavel': 'blue',
        'gavelfgd_vanilla': 'orange',
        'gavelfgd_paper': 'green',
        'gavelfgd_full': 'red'
    }
    labels = {
        'gavel': 'Gavel (baseline)',
        'gavelfgd_vanilla': 'Gavel+FGD (vanilla)',
        'gavelfgd_paper': 'Gavel+FGD (paper+buddy)',
        'gavelfgd_full': 'Gavel+FGD (full + dynamic penalty)'
    }

    for method in ['gavel', 'gavelfgd_vanilla', 'gavelfgd_paper', 'gavelfgd_full']:
        if method in stats:
            s = stats[method]
            ax.errorbar(s['loads'], s['mean'], yerr=s['std'],
                       label=labels.get(method, method),
                       color=colors.get(method, None),
                       marker='o', capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel('Job Arrival Rate (jobs/hour)', fontsize=12)
    ax.set_ylabel('Average JCT (seconds)', fontsize=12)
    ax.set_title('All FGD Improvements: JCT vs Load', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'results', 'gavelfgd_all_improvements_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_improvement_percentage():
    """Plot improvement percentage over baseline."""
    results = load_results('gavelfgd_all_improvements')
    if not results:
        print("No results found for gavelfgd_all_improvements")
        return

    data = aggregate_by_method_and_load(results)
    stats = compute_stats(data)

    if 'gavel' not in stats:
        print("No gavel baseline found")
        return

    baseline = stats['gavel']
    baseline_dict = dict(zip(baseline['loads'], baseline['mean']))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'gavelfgd_vanilla': 'orange',
        'gavelfgd_paper': 'green',
        'gavelfgd_full': 'red'
    }
    labels = {
        'gavelfgd_vanilla': 'Gavel+FGD (vanilla)',
        'gavelfgd_paper': 'Gavel+FGD (paper+buddy)',
        'gavelfgd_full': 'Gavel+FGD (full + dynamic penalty)'
    }

    for method in ['gavelfgd_vanilla', 'gavelfgd_paper', 'gavelfgd_full']:
        if method in stats:
            s = stats[method]
            improvements = []
            loads = []
            for load, mean in zip(s['loads'], s['mean']):
                if load in baseline_dict and baseline_dict[load] > 0:
                    improvement = (baseline_dict[load] - mean) / baseline_dict[load] * 100
                    improvements.append(improvement)
                    loads.append(load)

            ax.plot(loads, improvements,
                   label=labels.get(method, method),
                   color=colors.get(method, None),
                   marker='o', linewidth=2, markersize=6)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Job Arrival Rate (jobs/hour)', fontsize=12)
    ax.set_ylabel('JCT Improvement over Gavel (%)', fontsize=12)
    ax.set_title('FGD Improvements: Percentage Reduction in JCT', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'results', 'gavelfgd_improvement_percentage.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_fgd_improved():
    """Plot FGD placement algorithm comparison."""
    results = load_results('fgd_improved')
    if not results:
        print("No results found for fgd_improved")
        return

    data = aggregate_by_method_and_load(results)
    stats = compute_stats(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'strided': 'blue',
        'fgd_vanilla': 'orange',
        'fgd_paper': 'green',
        'fgd_buddy': 'red',
        'fgd_cluster': 'purple'
    }
    labels = {
        'strided': 'Strided (baseline)',
        'fgd_vanilla': 'FGD (vanilla)',
        'fgd_paper': 'FGD (paper scoring + 60%)',
        'fgd_buddy': 'FGD (+ buddy tiebreak)',
        'fgd_cluster': 'FGD (+ per-cluster)'
    }

    for method in ['strided', 'fgd_vanilla', 'fgd_paper', 'fgd_buddy', 'fgd_cluster']:
        if method in stats:
            s = stats[method]
            ax.errorbar(s['loads'], s['mean'], yerr=s['std'],
                       label=labels.get(method, method),
                       color=colors.get(method, None),
                       marker='o', capsize=3, linewidth=2, markersize=6)

    ax.set_xlabel('Job Arrival Rate (jobs/hour)', fontsize=12)
    ax.set_ylabel('Average JCT (seconds)', fontsize=12)
    ax.set_title('FGD Placement Improvements: JCT vs Load', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'results', 'fgd_improved_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def print_summary():
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    for subdir in ['gavelfgd_improved', 'gavelfgd_all_improvements', 'fgd_improved']:
        results = load_results(subdir)
        if not results:
            continue

        print(f"\n{subdir}:")
        print("-" * 60)

        data = aggregate_by_method_and_load(results)
        stats = compute_stats(data)

        # Find baseline
        baseline_method = 'gavel' if 'gavel' in stats else 'strided'
        baseline = stats.get(baseline_method, {})
        baseline_dict = dict(zip(baseline.get('loads', []), baseline.get('mean', [])))

        for method, s in sorted(stats.items()):
            print(f"\n  {method}:")
            for load, mean, std in zip(s['loads'], s['mean'], s['std']):
                improvement = ""
                if method != baseline_method and load in baseline_dict and baseline_dict[load] > 0:
                    imp_pct = (baseline_dict[load] - mean) / baseline_dict[load] * 100
                    improvement = f" ({imp_pct:+.1f}%)"
                print(f"    {load:3d} jph: {mean:8.1f}s ± {std:6.1f}s{improvement}")


if __name__ == '__main__':
    print("Generating plots...")
    plot_gavelfgd_improved()
    plot_gavelfgd_all_improvements()
    plot_improvement_percentage()
    plot_fgd_improved()
    print_summary()
    print("\nDone!")
