#!/usr/bin/env python3
"""
Convert official Microsoft Philly traces to Gavel trace format.

This script takes the cluster_job_log JSON from msr-fiddle/philly-traces
and converts it to Gavel's 10-field TSV format for use with
simulate_scheduler_with_trace.py.

Since Philly traces don't include workload type information, we use
heuristics to assign Gavel workload types based on job characteristics.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Add scheduler to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'scheduler'))

from job_table import JobTable
from job_template import JobTemplate


def parse_philly_job(job_entry):
    """Extract relevant fields from a Philly job log entry."""
    if not job_entry.get('attempts'):
        return None

    # Use the last attempt (most recent execution)
    attempt = job_entry['attempts'][-1]

    try:
        # Parse timestamps
        submitted = datetime.strptime(job_entry['submitted_time'], '%Y-%m-%d %H:%M:%S')
        start = datetime.strptime(attempt['start_time'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(attempt['end_time'], '%Y-%m-%d %H:%M:%S')

        # Calculate duration in seconds
        duration_seconds = (end - start).total_seconds()
        if duration_seconds <= 0:
            return None

        # Count total GPUs
        num_gpus = sum(len(d['gpus']) for d in attempt['detail'])
        if num_gpus == 0:
            return None

        return {
            'status': job_entry['status'],
            'vc': job_entry['vc'],
            'jobid': job_entry['jobid'],
            'user': job_entry['user'],
            'submitted_time': submitted,
            'start_time': start,
            'end_time': end,
            'duration_seconds': duration_seconds,
            'num_gpus': num_gpus,
        }
    except (KeyError, ValueError) as e:
        return None


def map_gpu_count_to_scale_factor(num_gpus):
    """Map Philly GPU count to Gavel scale factor (1, 2, 4, or 8)."""
    if num_gpus <= 1:
        return 1
    elif num_gpus <= 2:
        return 2
    elif num_gpus <= 4:
        return 4
    else:
        return 8


def select_workload_type(duration_seconds, num_gpus, rng, job_table):
    """
    Select a Gavel workload type based on job characteristics.

    Heuristics:
    - Short jobs (<30 min): Prefer smaller batch sizes, recommendation tasks
    - Medium jobs (30 min - 4 hours): Balanced distribution
    - Long jobs (>4 hours): Prefer larger models like ResNet-50, LM
    - Multi-GPU jobs: Only select distributed workloads
    """
    scale_factor = map_gpu_count_to_scale_factor(num_gpus)

    # Filter to workloads that support the required scale factor
    if scale_factor > 1:
        eligible = [jt for jt in job_table if jt.distributed]
    else:
        eligible = job_table

    duration_hours = duration_seconds / 3600

    # Weight workloads based on duration
    weights = []
    for jt in eligible:
        model = jt.model
        weight = 1.0

        if duration_hours < 0.5:  # Short jobs
            # Prefer recommendation, small batch transformers
            if 'Recommendation' in model or 'batch size 16' in model:
                weight = 2.0
            elif 'ResNet-50' in model or 'LM' in model:
                weight = 0.5
        elif duration_hours > 4:  # Long jobs
            # Prefer larger models
            if 'ResNet-50' in model or 'LM' in model:
                weight = 2.0
            elif 'Recommendation' in model:
                weight = 0.5

        weights.append(weight)

    # Weighted random selection
    total_weight = sum(weights)
    r = rng.random() * total_weight
    cumulative = 0
    for jt, w in zip(eligible, weights):
        cumulative += w
        if r <= cumulative:
            return jt

    return eligible[-1]  # Fallback


def load_throughputs(throughputs_file):
    """Load throughput profiles for calculating iterations."""
    with open(throughputs_file, 'r') as f:
        return json.load(f)


def calculate_total_steps(job_template, scale_factor, duration_seconds, throughputs,
                          reference_worker='v100'):
    """Calculate total iterations based on duration and throughput profile."""
    key = str((job_template.model, scale_factor))

    if reference_worker not in throughputs:
        reference_worker = list(throughputs.keys())[0]

    if key not in throughputs[reference_worker]:
        # Fallback: use average throughput
        return int(duration_seconds * 10)  # ~10 iter/sec default

    isolated_throughput = throughputs[reference_worker][key].get('null', 10.0)
    if isolated_throughput <= 0:
        isolated_throughput = 10.0

    return int(duration_seconds * isolated_throughput)


def format_gavel_trace_line(job_template, scale_factor, total_steps, arrival_time,
                            priority_weight=1, slo=-1):
    """
    Format a single line in Gavel's 10-field TSV format.

    Fields:
    1. job_type
    2. command
    3. working_directory
    4. num_steps_arg
    5. needs_data_dir (0 or 1)
    6. total_steps
    7. scale_factor
    8. priority_weight
    9. SLO
    10. arrival_time
    """
    needs_data_dir = 1 if job_template.needs_data_dir else 0

    return '\t'.join([
        job_template.model,
        job_template.command,
        job_template.working_directory,
        job_template.num_steps_arg,
        str(needs_data_dir),
        str(total_steps),
        str(scale_factor),
        str(priority_weight),
        f'{slo:.6f}',
        f'{arrival_time:.6f}'
    ])


def convert_philly_to_gavel(input_file, output_file, throughputs_file,
                            seed=42, status_filter='Pass', max_jobs=None,
                            vc_filter=None, time_window_hours=None):
    """
    Main conversion function.

    Args:
        input_file: Path to cluster_job_log JSON
        output_file: Path to output .trace file
        throughputs_file: Path to simulation_throughputs.json
        seed: Random seed for workload assignment
        status_filter: Only include jobs with this status (Pass/Killed/Failed/all)
        max_jobs: Limit number of jobs (for testing)
        vc_filter: Only include jobs from these virtual clusters
        time_window_hours: Only include jobs within this duration window
    """
    rng = random.Random(seed)

    print(f"Loading Philly traces from {input_file}...")
    with open(input_file, 'r') as f:
        philly_jobs = json.load(f)
    print(f"Loaded {len(philly_jobs)} job entries")

    print(f"Loading throughputs from {throughputs_file}...")
    throughputs = load_throughputs(throughputs_file)

    # Parse and filter jobs
    parsed_jobs = []
    for entry in philly_jobs:
        job = parse_philly_job(entry)
        if job is None:
            continue

        # Apply filters
        if status_filter != 'all' and job['status'] != status_filter:
            continue
        if vc_filter and job['vc'] not in vc_filter:
            continue
        if time_window_hours:
            max_duration = time_window_hours * 3600
            if job['duration_seconds'] > max_duration:
                continue

        parsed_jobs.append(job)

    print(f"Filtered to {len(parsed_jobs)} jobs")

    if max_jobs:
        parsed_jobs = parsed_jobs[:max_jobs]
        print(f"Limited to {len(parsed_jobs)} jobs")

    # Sort by submission time
    parsed_jobs.sort(key=lambda j: j['submitted_time'])

    # Calculate relative arrival times (seconds from first job)
    if parsed_jobs:
        base_time = parsed_jobs[0]['submitted_time']

    # Convert to Gavel format
    print("Converting to Gavel format...")
    output_lines = []
    stats = defaultdict(int)

    for job in parsed_jobs:
        # Calculate arrival time relative to first job
        arrival_time = (job['submitted_time'] - base_time).total_seconds()

        # Map GPU count to scale factor
        scale_factor = map_gpu_count_to_scale_factor(job['num_gpus'])
        stats[f'scale_factor_{scale_factor}'] += 1

        # Select workload type
        job_template = select_workload_type(
            job['duration_seconds'],
            job['num_gpus'],
            rng,
            JobTable
        )
        stats[job_template.model] += 1

        # Calculate total steps
        total_steps = calculate_total_steps(
            job_template,
            scale_factor,
            job['duration_seconds'],
            throughputs
        )

        # Format output line
        line = format_gavel_trace_line(
            job_template,
            scale_factor,
            total_steps,
            arrival_time
        )
        output_lines.append(line)

    # Write output
    print(f"Writing {len(output_lines)} jobs to {output_file}...")
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    # Print statistics
    print("\n=== Conversion Statistics ===")
    print(f"Total jobs converted: {len(output_lines)}")
    print("\nScale factor distribution:")
    for sf in [1, 2, 4, 8]:
        key = f'scale_factor_{sf}'
        count = stats.get(key, 0)
        pct = 100 * count / len(output_lines) if output_lines else 0
        print(f"  {sf} GPUs: {count} ({pct:.1f}%)")

    print("\nWorkload type distribution:")
    workload_stats = [(k, v) for k, v in stats.items() if not k.startswith('scale_factor')]
    workload_stats.sort(key=lambda x: -x[1])
    for model, count in workload_stats[:10]:
        pct = 100 * count / len(output_lines) if output_lines else 0
        print(f"  {model}: {count} ({pct:.1f}%)")

    print(f"\nOutput written to: {output_file}")
    return len(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Microsoft Philly traces to Gavel format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all successful jobs
  python convert_philly_to_gavel.py -i cluster_job_log -o philly_converted.trace

  # Convert with specific virtual cluster
  python convert_philly_to_gavel.py -i cluster_job_log -o vc_6214e9.trace --vc 6214e9

  # Convert first 1000 jobs for testing
  python convert_philly_to_gavel.py -i cluster_job_log -o test.trace --max-jobs 1000
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Path to cluster_job_log JSON file')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to output .trace file')
    parser.add_argument('--throughputs',
                        default='src/scheduler/simulation_throughputs.json',
                        help='Path to simulation_throughputs.json')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for workload assignment')
    parser.add_argument('--status', choices=['Pass', 'Killed', 'Failed', 'all'],
                        default='Pass',
                        help='Filter by job status')
    parser.add_argument('--max-jobs', type=int, default=None,
                        help='Maximum number of jobs to convert')
    parser.add_argument('--vc', nargs='+', default=None,
                        help='Filter by virtual cluster IDs')
    parser.add_argument('--max-duration-hours', type=float, default=None,
                        help='Filter out jobs longer than this duration')

    args = parser.parse_args()

    convert_philly_to_gavel(
        input_file=args.input,
        output_file=args.output,
        throughputs_file=args.throughputs,
        seed=args.seed,
        status_filter=args.status,
        max_jobs=args.max_jobs,
        vc_filter=args.vc,
        time_window_hours=args.max_duration_hours
    )


if __name__ == '__main__':
    main()
