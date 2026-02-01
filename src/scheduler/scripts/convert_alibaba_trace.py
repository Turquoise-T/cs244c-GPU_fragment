#!/usr/bin/env python3
"""
Convert Alibaba GPU trace to MSR/Gavel format.

Alibaba format (CSV):
    name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time

MSR/Gavel format (TSV):
    job_type\tcommand\tnum_steps_arg\tscale_factor\ttotal_steps\tarrival_time\tpriority_weight

Usage:
    python convert_alibaba_trace.py --input traces/cluster-trace-gpu-v2023/csv/openb_pod_list_default.csv \
                                    --output traces/alibaba_converted/default.trace
"""

import os
import sys
import csv
import argparse
import random

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Known Gavel job types that have throughput data
GAVEL_JOB_TYPES = {
    1: [  # Single GPU jobs
        'ResNet-18 (batch size 64)',
        'ResNet-18 (batch size 128)',
        'ResNet-50 (batch size 64)',
        'ResNet-50 (batch size 128)',
        'Transformer (batch size 64)',
        'Transformer (batch size 128)',
        'LM (batch size 20)',
        'LM (batch size 40)',
        'Recommendation (batch size 4096)',
        'Recommendation (batch size 8192)',
        'CycleGAN',
    ],
    2: [  # 2 GPU jobs
        'ResNet-18 (batch size 64)',
        'ResNet-50 (batch size 64)',
        'Transformer (batch size 64)',
    ],
    4: [  # 4 GPU jobs
        'ResNet-18 (batch size 64)',
        'ResNet-50 (batch size 64)',
        'Transformer (batch size 64)',
    ],
}

# Default commands for job types
JOB_COMMANDS = {
    'ResNet-18': 'cd %s/workloads/pytorch/image_classification/cifar10 && python3 main.py --data_dir=%s/data/cifar10',
    'ResNet-50': 'cd %s/workloads/pytorch/image_classification/imagenet && python3 main.py -j 4 -a resnet50',
    'Transformer': 'cd %s/workloads/pytorch/translation && python3 train.py -data %s/data/translation/multi30k.atok.low.pt',
    'LM': 'cd %s/workloads/pytorch/language_modeling && python main.py --cuda --data %s/data/wikitext2',
    'Recommendation': 'cd %s/workloads/pytorch/recommendation/scripts/ml-20m && python3 train.py --data_dir %s/data/ml-20m/pro_sg/',
    'CycleGAN': 'cd %s/workloads/pytorch/cyclegan && python3 cyclegan.py --dataset_path %s/data/monet2photo',
}

# QoS to priority mapping
QOS_PRIORITY = {
    'LS': 2.0,          # Latency Sensitive
    'Burstable': 1.5,
    'BE': 1.0,          # Best Effort
    'Guaranteed': 2.5,
}


def get_job_type(scale_factor: int, rng: random.Random) -> str:
    """Get a random Gavel job type that supports the given scale factor."""
    if scale_factor in GAVEL_JOB_TYPES:
        return rng.choice(GAVEL_JOB_TYPES[scale_factor])
    elif scale_factor > 4:
        # For large jobs, use multi-GPU capable types
        return rng.choice(GAVEL_JOB_TYPES[4])
    else:
        return rng.choice(GAVEL_JOB_TYPES[1])


def get_command(job_type: str) -> str:
    """Get command template for job type."""
    for prefix, cmd in JOB_COMMANDS.items():
        if job_type.startswith(prefix):
            return cmd
    return 'echo "placeholder"'


def get_num_steps_arg(job_type: str) -> str:
    """Get the argument name for number of steps."""
    if 'ResNet' in job_type:
        return '--num_steps'
    elif 'Transformer' in job_type:
        return '-step'
    elif 'LM' in job_type:
        return '--steps'
    elif 'Recommendation' in job_type:
        return '-n'
    elif 'CycleGAN' in job_type:
        return '--n_steps'
    return '--steps'


def convert_trace(input_file: str, output_file: str, 
                  gpu_only: bool = True, 
                  normalize_time: bool = True,
                  throughput_per_gpu: int = 100,
                  seed: int = 42,
                  max_jobs: int = None) -> dict:
    """
    Convert Alibaba trace to MSR format.
    
    Args:
        input_file: Path to Alibaba CSV trace
        output_file: Path to output MSR trace
        gpu_only: Only include GPU jobs
        normalize_time: Normalize arrival times to start from 0
        throughput_per_gpu: Assumed throughput for duration->steps conversion
        seed: Random seed for job type assignment
        max_jobs: Maximum number of jobs to convert (None = all)
    
    Returns:
        Statistics dictionary
    """
    rng = random.Random(seed)
    
    # Read input
    jobs = []
    min_time = float('inf')
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num_gpu = int(row.get('num_gpu', 0))
                gpu_milli = int(row.get('gpu_milli', 0)) if row.get('gpu_milli') else 0
                
                # Skip non-GPU jobs if gpu_only
                if gpu_only and num_gpu == 0 and gpu_milli == 0:
                    continue
                
                # Skip failed/pending jobs
                pod_phase = row.get('pod_phase', 'Running')
                if pod_phase in ['Failed', 'Pending']:
                    continue
                
                creation_time = float(row.get('creation_time', 0))
                deletion_time = float(row.get('deletion_time', creation_time + 3600))
                qos = row.get('qos', 'BE')
                
                # Calculate scale_factor
                if num_gpu > 0:
                    scale_factor = num_gpu
                elif gpu_milli > 0:
                    scale_factor = 1  # GPU sharing job treated as 1 GPU
                else:
                    scale_factor = 1
                
                # Cap scale_factor to reasonable values
                scale_factor = min(scale_factor, 8)
                
                # Calculate duration and steps
                duration = max(deletion_time - creation_time, 1)
                total_steps = int(duration * throughput_per_gpu)
                
                # Get priority
                priority = QOS_PRIORITY.get(qos, 1.0)
                
                jobs.append({
                    'creation_time': creation_time,
                    'scale_factor': scale_factor,
                    'total_steps': total_steps,
                    'priority': priority,
                    'gpu_milli': gpu_milli,
                    'duration': duration,
                })
                
                if creation_time < min_time:
                    min_time = creation_time
                    
            except (ValueError, KeyError) as e:
                continue
    
    # Sort by arrival time
    jobs.sort(key=lambda j: j['creation_time'])
    
    # Limit jobs if specified
    if max_jobs and len(jobs) > max_jobs:
        jobs = jobs[:max_jobs]
    
    # Normalize times
    if normalize_time and jobs:
        for job in jobs:
            job['creation_time'] -= min_time
    
    # Write output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for job in jobs:
            job_type = get_job_type(job['scale_factor'], rng)
            command = get_command(job_type)
            num_steps_arg = get_num_steps_arg(job_type)
            
            # MSR format: job_type, command, num_steps_arg, scale_factor, total_steps, arrival_time, priority
            line = f"{job_type}\t{command}\t{num_steps_arg}\t{job['scale_factor']}\t{job['total_steps']}\t{job['creation_time']:.6f}\t{job['priority']}\n"
            f.write(line)
    
    # Also save GPU sharing format (preserves gpu_milli)
    sharing_file = output_file.replace('.trace', '_gpusharing.csv')
    with open(sharing_file, 'w') as f:
        f.write('job_id,arrival_time,duration,gpu_milli,num_gpus\n')
        for i, job in enumerate(jobs):
            f.write(f"{i},{job['creation_time']:.2f},{job['duration']:.2f},{job['gpu_milli']},{job['scale_factor']}\n")
    
    # Statistics
    stats = {
        'total_jobs': len(jobs),
        'gpu_sharing_jobs': sum(1 for j in jobs if 0 < j['gpu_milli'] < 1000),
        'full_gpu_jobs': sum(1 for j in jobs if j['gpu_milli'] >= 1000 or j['gpu_milli'] == 0),
        'multi_gpu_jobs': sum(1 for j in jobs if j['scale_factor'] > 1),
        'avg_duration': sum(j['duration'] for j in jobs) / len(jobs) if jobs else 0,
        'time_span': jobs[-1]['creation_time'] - jobs[0]['creation_time'] if len(jobs) > 1 else 0,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert Alibaba trace to MSR format')
    parser.add_argument('--input', '-i', required=True, help='Input Alibaba CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output MSR trace file')
    parser.add_argument('--gpu-only', action='store_true', default=True, help='Only GPU jobs')
    parser.add_argument('--max-jobs', type=int, default=None, help='Max jobs to convert')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} -> {args.output}")
    stats = convert_trace(
        args.input, args.output,
        gpu_only=args.gpu_only,
        max_jobs=args.max_jobs,
        seed=args.seed
    )
    
    print(f"\nConversion complete!")
    print(f"  Total jobs: {stats['total_jobs']}")
    print(f"  GPU sharing jobs (0 < gpu_milli < 1000): {stats['gpu_sharing_jobs']}")
    print(f"  Full GPU jobs: {stats['full_gpu_jobs']}")
    print(f"  Multi-GPU jobs: {stats['multi_gpu_jobs']}")
    print(f"  Avg duration: {stats['avg_duration']:.1f}s")
    print(f"  Time span: {stats['time_span']:.1f}s")
    print(f"\nOutput files:")
    print(f"  MSR format: {args.output}")
    print(f"  GPU sharing format: {args.output.replace('.trace', '_gpusharing.csv')}")


if __name__ == '__main__':
    main()
