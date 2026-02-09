#!/usr/bin/env python3
"""Generate throughput matrix for Alibaba cluster GPU types.

Takes Gavel's measured V100/P100 throughputs and produces a new JSON file
with 6 GPU types matching the Alibaba cluster-trace-gpu-v2023 node inventory.

GPU type mapping:
  V100M16 -> V100 data (1.0x, same chip 16GB)
  V100M32 -> V100 data (1.0x, same chip 32GB)
  P100    -> P100 data  (1.0x, exact match)
  T4      -> V100 * 0.35 (Dell MLPerf: V100 is 2.2-3.6x faster)
  G2      -> V100 * 0.80 (anonymized; assumed high-end given 8-GPU nodes)
  G3      -> V100 * 0.90 (anonymized; assumed slightly higher tier than G2)

Usage:
  python scripts/generate_alibaba_throughputs.py
"""

import json
import os
import sys


# Scaling factors relative to V100 throughputs
SCALING = {
    'V100M16': ('v100', 1.0),
    'V100M32': ('v100', 1.0),
    'P100':    ('p100', 1.0),
    'T4':      ('v100', 0.35),
    'G2':      ('v100', 0.80),
    'G3':      ('v100', 0.90),
}


def scale_value(val, factor):
    """Scale a throughput value (float or list of floats)."""
    if isinstance(val, list):
        return [v * factor for v in val]
    elif isinstance(val, (int, float)):
        return val * factor
    else:
        return val


def scale_gpu_type(source_data, factor):
    """Scale all throughput entries for a GPU type."""
    scaled = {}
    for model_key, entry in source_data.items():
        if isinstance(entry, dict):
            # Entry is a dict with 'null' and colocated pair keys
            scaled[model_key] = {}
            for sub_key, sub_val in entry.items():
                scaled[model_key][sub_key] = scale_value(sub_val, factor)
        else:
            # Direct numeric value
            scaled[model_key] = scale_value(entry, factor)
    return scaled


def main():
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'scheduler')
    input_path = os.path.join(src_dir, 'simulation_throughputs.json')
    output_path = os.path.join(src_dir, 'simulation_throughputs_alibaba.json')

    with open(input_path) as f:
        original = json.load(f)

    result = {}

    for new_type, (source_type, factor) in SCALING.items():
        # Consolidated throughputs
        source_key = source_type
        if source_key not in original:
            print(f'ERROR: source type {source_key} not in throughputs file')
            sys.exit(1)
        result[new_type] = scale_gpu_type(original[source_key], factor)

        # Unconsolidated throughputs (for space-sharing)
        source_uncons = f'{source_type}_unconsolidated'
        if source_uncons in original:
            result[f'{new_type}_unconsolidated'] = scale_gpu_type(
                original[source_uncons], factor)

    # Summary
    print('Generated Alibaba throughput matrix:')
    for gpu_type in sorted(result.keys()):
        if 'unconsolidated' not in gpu_type:
            n_entries = len(result[gpu_type])
            source, factor = SCALING[gpu_type]
            print(f'  {gpu_type}: {n_entries} entries (from {source} x {factor})')

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nWritten to {output_path}')
    print(f'File size: {os.path.getsize(output_path) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
