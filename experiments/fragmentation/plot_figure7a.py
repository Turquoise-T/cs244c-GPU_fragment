#!/usr/bin/env python3
"""
Plot Figure 7(a): Fragmentation Rate vs Arrived Workloads

Generates a plot matching the style of FGD paper Figure 7(a).
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


# Style configuration to match paper
POLICY_STYLES = {
    'random': {
        'color': '#808080',  # gray
        'marker': 'o',
        'linestyle': '-.',
        'label': 'Random',
    },
    'bestfit': {
        'color': '#9400D3',  # purple
        'marker': 'v',
        'linestyle': '--',
        'label': 'BestFit',
    },
    'fgd': {
        'color': '#DC143C',  # red
        'marker': 'x',
        'linestyle': '-',
        'label': 'FGD',
    },
    'dotprod': {
        'color': '#0000FF',  # blue
        'marker': 's',
        'linestyle': '-.',
        'label': 'DotProd',
    },
    'packing': {
        'color': '#FFD700',  # gold
        'marker': 'D',
        'linestyle': '-',
        'label': 'Packing',
    },
    'clustering': {
        'color': '#228B22',  # green
        'marker': '^',
        'linestyle': ':',
        'label': 'Clustering',
    },
}


def downsample_results(results, num_points=20):
    """Downsample results to get evenly spaced points for clearer plotting."""
    if len(results) <= num_points:
        return results

    indices = np.linspace(0, len(results) - 1, num_points, dtype=int)
    return [results[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Plot fragmentation results")
    parser.add_argument("--input", default="sweep_results/combined_results.json",
                        help="Input JSON file from run_fragmentation_sweep.py")
    parser.add_argument("--output", default="sweep_results/figure7a.png",
                        help="Output plot file")
    parser.add_argument("--title", default="Figure 7(a): Fragmentation Rate vs Arrived Workloads",
                        help="Plot title")
    parser.add_argument("--downsample", type=int, default=0,
                        help="Number of points to show per policy (0 for all)")
    parser.add_argument("--x-axis", default="actual_load_pct",
                        choices=["actual_load_pct", "utilization_pct", "target_load_pct"],
                        help="Which field to use for X-axis")
    args = parser.parse_args()

    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each policy
    for policy_name, policy_results in results.items():
        style = POLICY_STYLES.get(policy_name, {
            'color': 'black',
            'marker': '.',
            'linestyle': '-',
            'label': policy_name,
        })

        # Extract x and y values
        if args.downsample > 0:
            policy_results = downsample_results(policy_results, args.downsample)

        # Support both sweep and time-series data formats
        if args.x_axis in policy_results[0]:
            x = [r[args.x_axis] for r in policy_results]
        elif 'arrived_workload_pct' in policy_results[0]:
            x = [r['arrived_workload_pct'] for r in policy_results]
        else:
            x = [r['target_load_pct'] for r in policy_results]

        y = [r['fragmentation_rate'] for r in policy_results]

        ax.plot(x, y,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                label=style['label'],
                markersize=8,
                linewidth=2)

    # Configure axes
    ax.set_xlabel('Arrived workloads (in % of cluster GPU capacity)', fontsize=12)
    ax.set_ylabel('Frag Rate (%)', fontsize=12)
    ax.set_title(args.title, fontsize=14)

    # Set axis limits
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, linestyle='-', alpha=0.3)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

    # Also save as PDF for paper quality
    pdf_output = args.output.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches='tight')
    print(f"PDF saved to {pdf_output}")

    plt.show()


if __name__ == "__main__":
    main()
