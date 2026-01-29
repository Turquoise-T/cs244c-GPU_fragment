#!/usr/bin/env python3
"""Extract telemetry and completion data from simulation logs for visualization."""

import json
import re
import sys
import gzip
from pathlib import Path

def extract_telemetry(log_path, output_dir):
    """Extract TELEMETRY and job completion events from simulation log."""

    name = log_path.parent.name
    telemetry_out = output_dir / f"{name}_telemetry.json"
    completions_out = output_dir / f"{name}_completions.json"

    telemetry_entries = []
    completion_events = []

    # Handle both .log and .log.gz
    if str(log_path).endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(log_path, 'rt') as f:
        for line in f:
            # Extract TELEMETRY entries
            if 'TELEMETRY' in line:
                match = re.search(r'TELEMETRY ({.*})', line)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        telemetry_entries.append(data)
                    except json.JSONDecodeError:
                        pass

            # Extract job completion events (handles both job_complete and job_completion)
            elif 'EVENT' in line and ('job_complete' in line or 'job_completion' in line):
                match = re.search(r'EVENT ({.*})', line)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        if data.get('event') in ('job_complete', 'job_completion'):
                            completion_events.append({
                                'job_id': int(data['job_id']),
                                'sim_time': data['sim_time'],
                                'duration': data.get('duration', 0)
                            })
                    except json.JSONDecodeError:
                        pass

    # Write telemetry (one JSON per line for streaming)
    with open(telemetry_out, 'w') as f:
        for entry in telemetry_entries:
            f.write(json.dumps(entry) + '\n')

    # Write completions as array
    with open(completions_out, 'w') as f:
        json.dump(completion_events, f)

    print(f"Extracted {len(telemetry_entries)} telemetry entries -> {telemetry_out}")
    print(f"Extracted {len(completion_events)} completions -> {completions_out}")

    return len(telemetry_entries), len(completion_events)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_telemetry.py <results_dir> [output_dir]")
        print("Example: python extract_telemetry.py results_full/fig11_finish_time_fairness_0.4jph_multi_s0")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("telemetry_data")
    output_dir.mkdir(exist_ok=True)

    # Find simulation log
    log_path = results_dir / "simulation.log"
    if not log_path.exists():
        log_path = results_dir / "simulation.log.gz"

    if not log_path.exists():
        print(f"Error: No simulation.log found in {results_dir}")
        sys.exit(1)

    extract_telemetry(log_path, output_dir)

if __name__ == "__main__":
    main()
