#!/usr/bin/env python3
"""Check progress of figure reproduction sweeps."""

import os
import re
import sys
from collections import defaultdict

def check_progress(log_dir):
    """Check sweep progress by parsing sweep.log and counting completed experiments."""
    sweep_log = os.path.join(log_dir, "sweep.log")
    raw_logs_dir = os.path.join(log_dir, "raw_logs")

    if not os.path.exists(sweep_log):
        print(f"No sweep.log found in {log_dir}")
        return

    # Parse sweep.log for total experiments and results
    total_experiments = 0
    completed = []

    with open(sweep_log, 'r') as f:
        for line in f:
            m = re.match(r'.*Running (\d+) total experiment.*', line)
            if m:
                total_experiments = int(m.group(1))

            m = re.match(r'.*Experiment ID:\s*(\d+).*Results: average JCT=(\d+\.\d+), utilization=(\d+\.\d+)', line)
            if m:
                completed.append({
                    'id': int(m.group(1)),
                    'jct': float(m.group(2)),
                    'util': float(m.group(3))
                })

    print(f"\n{'='*60}")
    print(f"Sweep: {log_dir}")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {len(completed)}")
    print(f"Progress: {len(completed)*100/total_experiments:.1f}%" if total_experiments > 0 else "N/A")

    # Count raw log files by policy
    if os.path.exists(raw_logs_dir):
        policy_counts = defaultdict(int)
        for root, _, files in os.walk(raw_logs_dir):
            for f in files:
                if f.endswith('.log'):
                    # Extract policy from path
                    path_parts = root.split(os.sep)
                    for i, part in enumerate(path_parts):
                        if part.startswith('v100='):
                            if i+1 < len(path_parts):
                                policy_counts[path_parts[i+1]] += 1

        print("\nLogs by policy:")
        for policy, count in sorted(policy_counts.items()):
            print(f"  {policy}: {count} log files")

    # Show some completed results
    if completed:
        print(f"\nRecent completed experiments:")
        for exp in completed[-5:]:
            print(f"  Exp {exp['id']}: JCT={exp['jct']/3600:.2f}hrs, util={exp['util']:.2%}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")

    if len(sys.argv) > 1:
        dirs = [os.path.join(logs_dir, d) for d in sys.argv[1:]]
    else:
        dirs = [os.path.join(logs_dir, d) for d in ['fig8', 'fig9']]

    for d in dirs:
        if os.path.exists(d):
            check_progress(d)
