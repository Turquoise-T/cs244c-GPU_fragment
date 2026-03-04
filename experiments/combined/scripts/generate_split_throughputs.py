#!/usr/bin/env python3
"""Generate simulation_throughputs_alibaba_split.json.

Duplicates throughput entries from the 6-type Alibaba throughput file
into 12 sub-types matching the exact (model, node_size) split.
"""
import json
import os

SCHEDULER_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'scheduler')
INPUT = os.path.join(SCHEDULER_DIR, 'simulation_throughputs_alibaba.json')
OUTPUT = os.path.join(SCHEDULER_DIR, 'simulation_throughputs_alibaba_split.json')

SPLIT_MAP = {
    'G2': ['G2_8'],
    'T4': ['T4_2', 'T4_4'],
    'G3': ['G3_8'],
    'P100': ['P100_1', 'P100_2'],
    'V100M32': ['V100M32_4', 'V100M32_8'],
    'V100M16': ['V100M16_1', 'V100M16_4', 'V100M16_8'],
}
A10_PROXY = 'P100'

def main():
    with open(INPUT) as f:
        orig = json.load(f)

    result = {}
    for parent, sub_types in SPLIT_MAP.items():
        if parent not in orig:
            raise KeyError(f"Parent model '{parent}' not found in {INPUT}")
        for sub in sub_types:
            result[sub] = orig[parent]

    result['A10_1'] = orig[A10_PROXY]

    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUTPUT} with {len(result)} GPU types: {sorted(result.keys())}")

if __name__ == '__main__':
    main()
