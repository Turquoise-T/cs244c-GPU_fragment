#!/usr/bin/env python3
"""
Plot FGD evaluation results from Alibaba trace experiments.

Generates two plots:
1. Fragmentation rate vs arrived workload (Figure 7a style)
2. Fragmentation rate vs simulation time (time-based insight)

Usage:
    python plot_results.py --results-dir alibaba_results/
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


POLICY_STYLES = {
    'Random':     {'color': '#808080', 'marker': 'o', 'linestyle': '-.'},
    'BestFit':    {'color': '#9400D3', 'marker': 'v', 'linestyle': '--'},
    'DotProd':    {'color': '#1E90FF', 'marker': 's', 'linestyle': '-.'},
    'Packing':    {'color': '#FF8C00', 'marker': 'D', 'linestyle': '-'},
    'Clustering': {'color': '#228B22', 'marker': '^', 'linestyle': ':'},
    'FGD':        {'color': '#DC143C', 'marker': 'x', 'linestyle': '-'},
    'FGD-Full':   {'color': '#DC143C', 'marker': 'x', 'linestyle': '-'},
}


def get_style(name):
    """Get plot style for a scheduler, with fallback for unknown names."""
    if name in POLICY_STYLES:
        return POLICY_STYLES[name]
    if name.startswith('W-FGD'):
        return {'color': '#006400', 'marker': '*', 'linestyle': '-'}
    if name.startswith('FGD'):
        return {'color': '#B22222', 'marker': 'P', 'linestyle': '--'}
    return {'color': 'black', 'marker': '.', 'linestyle': '-'}


def load_curves(csv_path):
    """Load fragmentation curves from CSV."""
    data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['scheduler']].append({
                'sim_time': float(row['sim_time']),
                'arrived_workload_pct': float(row['arrived_workload_pct']),
                'active_workload_pct': float(row['active_workload_pct']),
                'fragmentation_rate': float(row['fragmentation_rate']),
                'utilization_pct': float(row['utilization_pct']),
                'tasks_active': int(row['tasks_active']),
                'avg_jct': float(row.get('avg_jct', 0)),
            })
    return data


def load_job_records(csv_path):
    """Load per-job records from CSV."""
    data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['scheduler']].append({
                'task_id': int(row['task_id']),
                'gpu_demand': float(row['gpu_demand']),
                'creation_time': int(row['creation_time']),
                'trace_duration': int(row['trace_duration']),
                'placement_time': int(row['placement_time']),
                'completion_time': int(row['completion_time']),
                'wait_time': float(row['wait_time']),
                'jct': float(row['jct']),
                'frag_at_placement': float(row['frag_at_placement']),
            })
    return data


def plot_frag_vs_workload(data, output_path):
    """Plot fragmentation rate vs arrived workload (Figure 7a style)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, points in data.items():
        style = get_style(name)
        x = [p['arrived_workload_pct'] for p in points]
        y = [p['fragmentation_rate'] for p in points]

        # Downsample if too many points
        markevery = max(1, len(x) // 30)

        ax.plot(x, y, label=name,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=5,
                markevery=markevery,
                linewidth=1.5)

    ax.set_xlabel('Arrived workloads (% of cluster GPU capacity)', fontsize=12)
    ax.set_ylabel('Fragmentation Rate (%)', fontsize=12)
    ax.set_title('Fragmentation Rate vs Arrived Workloads (Alibaba Trace)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_frag_vs_time(data, output_path):
    """Plot fragmentation rate vs simulation time."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, points in data.items():
        style = get_style(name)
        x = [p['sim_time'] / 3600.0 for p in points]  # Convert to hours
        y = [p['fragmentation_rate'] for p in points]

        markevery = max(1, len(x) // 30)

        ax.plot(x, y, label=name,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=5,
                markevery=markevery,
                linewidth=1.5)

    ax.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax.set_ylabel('Fragmentation Rate (%)', fontsize=12)
    ax.set_title('Fragmentation Rate Over Time (Alibaba Trace)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_util_vs_time(data, output_path):
    """Plot utilization and active tasks over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for name, points in data.items():
        style = get_style(name)
        x = [p['sim_time'] / 3600.0 for p in points]
        markevery = max(1, len(x) // 30)

        # Utilization
        y_util = [p['utilization_pct'] for p in points]
        ax1.plot(x, y_util, label=name,
                 color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markersize=4, markevery=markevery,
                 linewidth=1.5)

        # Active tasks
        y_active = [p['tasks_active'] for p in points]
        ax2.plot(x, y_active, label=name,
                 color=style['color'], linestyle=style['linestyle'],
                 marker=style['marker'], markersize=4, markevery=markevery,
                 linewidth=1.5)

    ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax1.set_title('Cluster Utilization Over Time (Alibaba Trace)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax2.set_ylabel('Active Tasks', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_jct_vs_frag(job_data, output_path, num_bins=20):
    """Plot fragmentation rate vs JCT.

    Each completed job has a fragmentation_rate recorded when it was placed.
    We bin jobs by JCT (hours) and plot the average fragmentation per bin.
    X-axis: JCT (hours), Y-axis: Fragmentation rate (%).
    Jobs beyond the 95th percentile JCT are excluded to avoid outlier distortion.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compute a global JCT cap at the 95th percentile across all schedulers
    all_jcts_h = []
    for records in job_data.values():
        for r in records:
            if r['jct'] > 0:
                all_jcts_h.append(r['jct'] / 3600.0)
    all_jcts_h.sort()
    jct_cap = all_jcts_h[int(len(all_jcts_h) * 0.95)] if all_jcts_h else 100.0

    for name, records in job_data.items():
        # Filter to completed jobs with valid frag data, clip to p95
        completed = [r for r in records
                     if r['jct'] > 0 and r['frag_at_placement'] >= 0
                     and r['jct'] / 3600.0 <= jct_cap]
        if not completed:
            continue

        style = get_style(name)

        # Bin by JCT (in hours)
        jct_vals = [r['jct'] / 3600.0 for r in completed]
        jct_min, jct_max = min(jct_vals), max(jct_vals)

        if jct_max - jct_min < 0.01:
            avg_frag = sum(r['frag_at_placement'] for r in completed) / len(completed)
            ax.scatter([jct_min], [avg_frag], label=name,
                       color=style['color'], marker=style['marker'], s=80)
            continue

        bin_width = (jct_max - jct_min) / num_bins
        bins = defaultdict(list)
        for r in completed:
            jct_h = r['jct'] / 3600.0
            b = int((jct_h - jct_min) / bin_width)
            b = min(b, num_bins - 1)
            bins[b].append(r['frag_at_placement'])

        x_vals = []
        y_vals = []
        for b in sorted(bins.keys()):
            x_vals.append(jct_min + (b + 0.5) * bin_width)
            y_vals.append(sum(bins[b]) / len(bins[b]))

        markevery = max(1, len(x_vals) // 20)
        ax.plot(x_vals, y_vals, label=name,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=6,
                markevery=markevery,
                linewidth=1.5)

    ax.set_xlabel('JCT (hours)', fontsize=12)
    ax.set_ylabel('Fragmentation Rate at Placement (%)', fontsize=12)
    ax.set_title('Fragmentation Rate vs Job Completion Time (Alibaba Trace)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_jct_vs_arrival_rate(job_data, output_path, window_hours=5.0,
                             num_bins=15):
    """Plot average JCT vs job arrival rate.

    Divides the simulation timeline into fixed-size windows, computes the
    arrival rate (jobs/hour) in each window, then bins windows by arrival
    rate and plots the average JCT of jobs arriving during those windows.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Determine global time range from all jobs (same arrivals for all schedulers)
    first_sched = next(iter(job_data.values()))
    all_times = [r['creation_time'] for r in first_sched]
    t_min = min(all_times)
    t_max = max(all_times)
    window_s = window_hours * 3600

    for name, records in job_data.items():
        completed = [r for r in records if r['jct'] > 0]
        if not completed:
            continue

        style = get_style(name)

        # Group completed jobs by time window
        window_jobs = defaultdict(list)
        for r in completed:
            w = int((r['creation_time'] - t_min) / window_s)
            window_jobs[w].append(r['jct'] / 3600.0)

        # Compute arrival rate and avg JCT per window
        window_rates = []
        window_jcts = []
        for w, jcts in window_jobs.items():
            # Count ALL arrivals in this window (including uncompleted)
            w_start = t_min + w * window_s
            w_end = w_start + window_s
            n_arrivals = sum(1 for r in records
                            if w_start <= r['creation_time'] < w_end)
            rate = n_arrivals / window_hours
            avg_jct = sum(jcts) / len(jcts)
            window_rates.append(rate)
            window_jcts.append(avg_jct)

        # Bin by arrival rate
        rate_max = 4.5
        bin_width = rate_max / num_bins
        bins = defaultdict(list)
        for rate, jct in zip(window_rates, window_jcts):
            if rate > rate_max:
                continue  # Clip to x-axis range
            b = int(rate / bin_width)
            b = min(b, num_bins - 1)
            bins[b].append(jct)

        x_vals = []
        y_vals = []
        for b in sorted(bins.keys()):
            x_vals.append((b + 0.5) * bin_width)
            y_vals.append(sum(bins[b]) / len(bins[b]))

        markevery = max(1, len(x_vals) // 20)
        ax.plot(x_vals, y_vals, label=name,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=6,
                markevery=markevery,
                linewidth=1.5)

    ax.set_xlabel('Job Arrival Rate (jobs/hour)', fontsize=12)
    ax.set_ylabel('Average JCT (hours)', fontsize=12)
    ax.set_title('Average JCT vs Job Arrival Rate (Alibaba Trace)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4.5)
    ax.set_xticks([i * 0.5 for i in range(10)])
    ax.set_ylim(0, 100)
    ax.set_yticks([i * 20 for i in range(6)])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Alibaba trace experiment results")
    parser.add_argument("--results-dir",
                        default=os.path.join(os.path.dirname(__file__), "alibaba_results"),
                        help="Directory containing results CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: same as results-dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    csv_path = os.path.join(args.results_dir, "fragmentation_curves.csv")
    jobs_csv_path = os.path.join(args.results_dir, "job_records.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run the experiment first.")
        return 1

    print(f"Loading curves from {csv_path}...")
    data = load_curves(csv_path)
    print(f"  Loaded {len(data)} schedulers: {list(data.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating fragmentation plots...")
    plot_frag_vs_workload(data, os.path.join(output_dir, "frag_vs_workload.png"))
    plot_frag_vs_time(data, os.path.join(output_dir, "frag_vs_time.png"))
    plot_util_vs_time(data, os.path.join(output_dir, "util_vs_time.png"))

    # Also save as PDF
    plot_frag_vs_workload(data, os.path.join(output_dir, "frag_vs_workload.pdf"))
    plot_frag_vs_time(data, os.path.join(output_dir, "frag_vs_time.pdf"))

    # JCT plots (require job_records.csv)
    if os.path.exists(jobs_csv_path):
        print(f"\nLoading job records from {jobs_csv_path}...")
        job_data = load_job_records(jobs_csv_path)
        print(f"  Loaded {sum(len(v) for v in job_data.values())} records "
              f"across {len(job_data)} schedulers")

        print("\nGenerating JCT plots...")
        plot_jct_vs_frag(job_data, os.path.join(output_dir, "jct_vs_frag.png"))
        plot_jct_vs_frag(job_data, os.path.join(output_dir, "jct_vs_frag.pdf"))
        plot_jct_vs_arrival_rate(job_data, os.path.join(output_dir, "jct_vs_arrival_rate.png"))
        plot_jct_vs_arrival_rate(job_data, os.path.join(output_dir, "jct_vs_arrival_rate.pdf"))
    else:
        print(f"\nSkipping JCT plots ({jobs_csv_path} not found). "
              "Re-run experiment to generate job records.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
