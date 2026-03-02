# FGD Replication via Gavel -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replicate FGD paper (ATC'23) fragmentation results (Figs 7a, 7b, 9a, 9b) using the integrated Gavel+FGD scheduler with exact Alibaba cluster topology (12 sub-types, 6,212 GPUs).

**Architecture:** Split the real Alibaba cluster's 12 (model, node_size) combinations into separate Gavel GPU types, preserving per-server GPU counts. Run 180 experiments (4 placements x 15 arrival rates x 3 seeds) to sweep utilization levels and measure fragmentation metrics each scheduling round. Plot fragmentation rate, frag/total, unallocated GPU %, and occupied nodes vs utilization with paper reference overlay.

**Tech Stack:** Python 3, cvxpy (ECOS solver), matplotlib, SLURM (FarmShare), existing Gavel scheduler + FGD integration

---

## Task 1: Create Split Throughput Data

**Files:**
- Read: `src/scheduler/simulation_throughputs_alibaba.json`
- Create: `src/scheduler/simulation_throughputs_alibaba_split.json`

**Step 1: Write a script to generate the split throughputs file**

Create a one-time generation script. The existing `simulation_throughputs_alibaba.json` has 6 GPU types: `V100M16`, `V100M32`, `G2`, `G3`, `T4`, `P100`. We need 12 sub-types where sub-types of the same parent model get identical throughput data.

The mapping is:
- `G2` -> `G2_8`
- `T4` -> `T4_2`, `T4_4`
- `G3` -> `G3_8`
- `P100` -> `P100_1`, `P100_2`
- `V100M32` -> `V100M32_4`, `V100M32_8`
- `V100M16` -> `V100M16_1`, `V100M16_4`, `V100M16_8`
- `A10` -> `A10_1` (use `P100` as proxy -- only 2 GPUs, negligible impact)

Write this in `experiments/combined/scripts/generate_split_throughputs.py`:

```python
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

# parent_model -> list of sub-type names
SPLIT_MAP = {
    'G2': ['G2_8'],
    'T4': ['T4_2', 'T4_4'],
    'G3': ['G3_8'],
    'P100': ['P100_1', 'P100_2'],
    'V100M32': ['V100M32_4', 'V100M32_8'],
    'V100M16': ['V100M16_1', 'V100M16_4', 'V100M16_8'],
}
# A10 has no throughput data; use P100 as proxy (only 2 GPUs total)
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

    # A10_1 proxied from P100
    result['A10_1'] = orig[A10_PROXY]

    # The throughput JSON is a nested dict: gpu_type -> model_key -> ...
    # Each model_key's value contains cross-type throughput pairs.
    # We also need to update the *inner* references: when a model entry
    # references throughput on OTHER gpu types (for co-location), those
    # keys must also be renamed to sub-types.
    #
    # However, in practice Gavel only reads the top-level key matching
    # the worker type and uses `null` (isolated throughput) for scheduling.
    # Cross-type references are for job packing, which we don't use.
    # So a simple copy of the parent's data is sufficient.

    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUTPUT} with {len(result)} GPU types: {sorted(result.keys())}")

if __name__ == '__main__':
    main()
```

**Step 2: Run the script**

Run: `cd /path/to/gavel && python experiments/combined/scripts/generate_split_throughputs.py`
Expected: File created with 12 keys: `A10_1, G2_8, G3_8, P100_1, P100_2, T4_2, T4_4, V100M16_1, V100M16_4, V100M16_8, V100M32_4, V100M32_8`

**Step 3: Validate the output**

Quick sanity check:
```bash
python3 -c "
import json
with open('src/scheduler/simulation_throughputs_alibaba_split.json') as f:
    d = json.load(f)
print(f'{len(d)} types: {sorted(d.keys())}')
# Verify sub-types share parent data
assert d['T4_2'] == d['T4_4'], 'T4 sub-types should be identical'
assert d['V100M16_1'] == d['V100M16_4'] == d['V100M16_8'], 'V100M16 sub-types should be identical'
assert d['V100M32_4'] == d['V100M32_8'], 'V100M32 sub-types should be identical'
assert d['P100_1'] == d['P100_2'], 'P100 sub-types should be identical'
assert d['A10_1'] == d['P100_1'], 'A10 proxied from P100'
print('All assertions passed')
"
```

**Step 4: Commit**

```bash
git add experiments/combined/scripts/generate_split_throughputs.py src/scheduler/simulation_throughputs_alibaba_split.json
git commit -m "feat: add split throughput data for 12 Alibaba sub-types"
```

---

## Task 2: Add Per-Round Metrics Recording to Scheduler

**Files:**
- Modify: `src/scheduler/scheduler.py`
  - Lines 81-93 (FGD init section -- add workload/frag calc init for non-FGD runs too)
  - After line 1982 (end of round profiling -- insert metrics recording call)
  - New method `_record_round_metrics()` (add near line 700, next to `_get_current_utilization`)

**Step 1: Add `_round_metrics_history` initialization and fragmentation calculator for non-FGD runs**

In `scheduler.py` `__init__`, after line 87 (`self._fgd_fragmentation_history = []`), add:

```python
        self._round_metrics_history = []
```

After line 93 (end of the `if enable_fgd:` block), add the fallback fragmentation calculator for non-FGD runs:

```python
        # For metrics recording: always have a workload + frag calculator,
        # even when FGD placement is disabled (strided mode).
        self._metrics_workload = None
        if enable_fgd:
            self._metrics_workload = self._fgd_placement.workload
        else:
            # Import only when needed for metrics
            from fgd_placement import build_fgd_workload
            self._metrics_workload = build_fgd_workload(fgd_workload_mode)
```

**Step 2: Add `_record_round_metrics()` method**

Add this method after `_get_current_utilization()` (around line 727):

```python
    def _record_round_metrics(self, cluster_spec, num_gpus_per_server):
        """Record fragmentation and utilization metrics for the current round.

        Computes:
        - utilization: allocated GPUs / total GPUs
        - frag_rate: F_N(M) / unallocated_gpus * 100
        - frag_total: F_N(M) / total_gpus * 100
        - unalloc_pct: unallocated / total * 100
        - occupied_nodes: servers with >= 1 GPU allocated

        Requires self._metrics_workload to be set.
        """
        if self._metrics_workload is None:
            return

        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fgd'))
        from fgd import FragmentationCalculator, Node

        total_gpus = sum(cluster_spec.values())
        allocated_gpus = 0
        occupied_nodes = 0
        all_nodes = []  # For fragmentation calculation

        # Build set of assigned worker IDs from current assignments
        assigned_wids = set()
        for job_id, worker_ids in self._current_worker_assignments.items():
            for wid in worker_ids:
                assigned_wids.add(wid)

        for worker_type, servers in self._worker_type_to_worker_id_mapping.items():
            for server_idx, server_wids in enumerate(servers):
                gpus = []
                server_allocated = 0
                for wid in server_wids:
                    if wid in assigned_wids:
                        gpus.append(0.0)
                        server_allocated += 1
                    else:
                        gpus.append(1.0)

                allocated_gpus += server_allocated
                if server_allocated > 0:
                    occupied_nodes += 1

                node = Node(
                    id=f'{worker_type}-{server_idx}',
                    total_cpu=1000.0,
                    total_memory=1000.0,
                    gpus=gpus,
                    gpu_type=worker_type,
                )
                node.allocated_cpu = server_allocated * 10.0
                node.allocated_memory = server_allocated * 10.0
                all_nodes.append(node)

        unallocated_gpus = total_gpus - allocated_gpus
        utilization = allocated_gpus / total_gpus * 100.0 if total_gpus > 0 else 0.0

        # Compute F_N(M)
        frag_value = FragmentationCalculator.compute_cluster_fragmentation(
            all_nodes, self._metrics_workload
        )

        frag_rate = (frag_value / unallocated_gpus * 100.0
                     if unallocated_gpus > 0 else 0.0)
        frag_total = frag_value / total_gpus * 100.0 if total_gpus > 0 else 0.0

        self._round_metrics_history.append({
            'simulated_time': self._current_timestamp,
            'utilization': utilization,
            'frag_value': frag_value,
            'frag_rate': frag_rate,
            'frag_total': frag_total,
            'unalloc_pct': unallocated_gpus / total_gpus * 100.0 if total_gpus > 0 else 0.0,
            'occupied_nodes': occupied_nodes,
            'allocated_gpus': allocated_gpus,
            'total_gpus': total_gpus,
        })
```

**Step 3: Call `_record_round_metrics()` at end of each scheduling round**

In `simulate()`, after line 1982 (`_profile['round_total'] += ...`), add:

```python
            # Record per-round fragmentation/utilization metrics
            if hasattr(self, '_round_metrics_history') and num_gpus_per_server is not None:
                self._record_round_metrics(cluster_spec, num_gpus_per_server)
```

The `num_gpus_per_server is not None` guard ensures we only record metrics when server topology is defined (fragmentation requires multi-GPU nodes).

**Step 4: Run integration tests to verify no regression**

Run: `cd src/scheduler && python -m unittest tests.integration_test -v`
Expected: Both tests PASS with exact JCT values (73063.45 and 57171.41). The metrics recording code path is not triggered because integration tests use `num_gpus_per_server=None`.

**Step 5: Commit**

```bash
git add src/scheduler/scheduler.py
git commit -m "feat: add per-round fragmentation metrics recording to scheduler"
```

---

## Task 3: Update Experiment Runner to Extract Metrics

**Files:**
- Modify: `experiments/combined/run_fgd_experiments.py`
  - Lines 205-232 (result extraction section)

**Step 1: Add metrics extraction after simulation completes**

In `run_fgd_experiments.py`, after line 208 (frag_history extraction), add code to extract `_round_metrics_history` and compute window-averaged metrics:

```python
    # Extract per-round metrics if available
    round_metrics = []
    if hasattr(sched, '_round_metrics_history'):
        round_metrics = sched._round_metrics_history

    # Compute measurement-window averages for fragmentation metrics
    frag_metrics = {}
    if round_metrics and mode == 'steady_state':
        # Filter to rounds within the measurement window
        # Use simulated_time: the window covers jobs window_start to window_end,
        # but we want metrics during the period when those jobs are active.
        # Use all metrics after the first window job arrives.
        window_metrics = round_metrics  # Use all metrics (window filtering via job IDs)

        if window_metrics:
            import numpy as np
            frag_metrics = {
                'avg_utilization': float(np.mean([m['utilization'] for m in window_metrics])),
                'avg_frag_rate': float(np.mean([m['frag_rate'] for m in window_metrics])),
                'avg_frag_total': float(np.mean([m['frag_total'] for m in window_metrics])),
                'avg_unalloc_pct': float(np.mean([m['unalloc_pct'] for m in window_metrics])),
                'avg_occupied_nodes': float(np.mean([m['occupied_nodes'] for m in window_metrics])),
                'std_utilization': float(np.std([m['utilization'] for m in window_metrics])),
                'std_frag_rate': float(np.std([m['frag_rate'] for m in window_metrics])),
                'std_frag_total': float(np.std([m['frag_total'] for m in window_metrics])),
                'std_unalloc_pct': float(np.std([m['unalloc_pct'] for m in window_metrics])),
                'std_occupied_nodes': float(np.std([m['occupied_nodes'] for m in window_metrics])),
                'num_metric_samples': len(window_metrics),
            }
```

**Step 2: Include metrics in the result dict**

After the existing result dict construction (line 232), merge in frag_metrics:

```python
    result.update(frag_metrics)
```

And add a print for the new metrics:

```python
    if frag_metrics:
        print(f"  Utilization: {frag_metrics['avg_utilization']:.1f}% +/- {frag_metrics['std_utilization']:.1f}")
        print(f"  Frag rate: {frag_metrics['avg_frag_rate']:.1f}% +/- {frag_metrics['std_frag_rate']:.1f}")
        print(f"  Frag/total: {frag_metrics['avg_frag_total']:.1f}% +/- {frag_metrics['std_frag_total']:.1f}")
        print(f"  Unalloc: {frag_metrics['avg_unalloc_pct']:.1f}%, Occupied nodes: {frag_metrics['avg_occupied_nodes']:.0f}")
```

**Step 3: Run integration tests**

Run: `cd src/scheduler && python -m unittest tests.integration_test -v`
Expected: PASS (run_fgd_experiments changes don't affect scheduler core)

**Step 4: Commit**

```bash
git add experiments/combined/run_fgd_experiments.py
git commit -m "feat: extract per-round fragmentation metrics in experiment runner"
```

---

## Task 4: Local Smoke Test

**Files:**
- No new files created; uses existing code

**Step 1: Run a minimal experiment locally to verify the full pipeline**

Create a small test config inline and run it to verify metrics are computed correctly:

```bash
cd /path/to/gavel
python3 -c "
import json, sys, os
sys.path.insert(0, 'src/scheduler')
import scheduler, utils

# Tiny cluster: 2 sub-types, 4 GPUs each, 2 GPUs/server
cluster_spec = {'G2_8': 8, 'T4_2': 4}
num_gpus_per_server = {'G2_8': 4, 'T4_2': 2}

throughputs_file = 'src/scheduler/simulation_throughputs_alibaba_split.json'
policy = utils.get_policy('max_min_fairness', solver='ECOS', seed=0)

sched = scheduler.Scheduler(
    policy,
    throughputs_file=throughputs_file,
    seed=0,
    time_per_iteration=360,
    simulate=True,
    profiling_percentage=1.0,
    num_reference_models=26,
    enable_fgd=True,
    fgd_placement_mode='fgd',
    fgd_workload_mode='alibaba',
    log_level=30,  # WARNING
)

from job_id_pair import JobIdPair
sched.simulate(
    cluster_spec=cluster_spec,
    lam=360.0,
    jobs_to_complete=set(JobIdPair(i, None) for i in range(10, 15)),
    generate_multi_gpu_jobs=True,
    simulate_steady_state=True,
    num_gpus_per_server=num_gpus_per_server,
)

print(f'Metrics recorded: {len(sched._round_metrics_history)}')
if sched._round_metrics_history:
    m = sched._round_metrics_history[-1]
    print(f'Last round: util={m[\"utilization\"]:.1f}%, frag_rate={m[\"frag_rate\"]:.1f}%, '
          f'occupied_nodes={m[\"occupied_nodes\"]}')
print('Smoke test PASSED')
"
```

Expected: Prints metrics and "Smoke test PASSED" without errors.

**Step 2: Also verify strided (non-FGD) mode records metrics**

```bash
python3 -c "
import json, sys, os
sys.path.insert(0, 'src/scheduler')
import scheduler, utils

cluster_spec = {'G2_8': 8, 'T4_2': 4}
num_gpus_per_server = {'G2_8': 4, 'T4_2': 2}

throughputs_file = 'src/scheduler/simulation_throughputs_alibaba_split.json'
policy = utils.get_policy('max_min_fairness', solver='ECOS', seed=0)

sched = scheduler.Scheduler(
    policy,
    throughputs_file=throughputs_file,
    seed=0, time_per_iteration=360, simulate=True,
    profiling_percentage=1.0, num_reference_models=26,
    enable_fgd=False,  # Strided mode
    fgd_workload_mode='alibaba',
    log_level=30,
)

from job_id_pair import JobIdPair
sched.simulate(
    cluster_spec=cluster_spec,
    lam=360.0,
    jobs_to_complete=set(JobIdPair(i, None) for i in range(10, 15)),
    generate_multi_gpu_jobs=True,
    simulate_steady_state=True,
    num_gpus_per_server=num_gpus_per_server,
)

print(f'Strided metrics: {len(sched._round_metrics_history)} samples')
if sched._round_metrics_history:
    m = sched._round_metrics_history[-1]
    print(f'Last: util={m[\"utilization\"]:.1f}%, frag_rate={m[\"frag_rate\"]:.1f}%')
print('Strided smoke test PASSED')
"
```

Expected: Prints metrics and "Strided smoke test PASSED"

---

## Task 5: Generate Experiment Config (180 experiments)

**Files:**
- Create: `experiments/combined/scripts/generate_fgd_replication_config.py`
- Create: `experiments/combined/configs/phase_fgd_replication.json`

**Step 1: Write the config generator**

```python
#!/usr/bin/env python3
"""Generate phase_fgd_replication.json -- 180 experiments.

4 placements x 15 arrival rates x 3 seeds.
"""
import json
import os

PLACEMENTS = [
    {'label': 'strided',  'enable_fgd': False, 'fgd_placement_mode': 'fgd'},
    {'label': 'random',   'enable_fgd': True,  'fgd_placement_mode': 'random'},
    {'label': 'bestfit',  'enable_fgd': True,  'fgd_placement_mode': 'bestfit'},
    {'label': 'fgd',      'enable_fgd': True,  'fgd_placement_mode': 'fgd'},
]

# Jobs/hour -> lambda (inter-arrival time in seconds)
RATES = [5, 10, 20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 360]

SEEDS = [0, 1, 2]

COMMON = {
    "policy": "max_min_fairness",
    "mode": "steady_state",
    "window_start": 4000,
    "window_end": 5000,
    "time_per_iteration": 600,
    "generate_multi_gpu_jobs": True,
    "workload_mode": "alibaba",
    "fgd_workload_mode": "alibaba",
    "enable_migration_penalty": False,
    "enable_gpu_sharing": False,
    "solver": "ECOS",
    "completion_rate_threshold": 0.1,
    "throughputs_file": "simulation_throughputs_alibaba_split.json",
    "cluster_spec": {
        "G2_8": 4392, "T4_2": 774, "G3_8": 312, "P100_2": 262,
        "V100M32_8": 168, "V100M16_4": 112, "T4_4": 68, "V100M16_8": 64,
        "V100M32_4": 36, "V100M16_1": 19, "P100_1": 3, "A10_1": 2
    },
    "num_gpus_per_server": {
        "G2_8": 8, "T4_2": 2, "G3_8": 8, "P100_2": 2,
        "V100M32_8": 8, "V100M16_4": 4, "T4_4": 4, "V100M16_8": 8,
        "V100M32_4": 4, "V100M16_1": 1, "P100_1": 1, "A10_1": 1
    }
}

def main():
    experiments = []
    for placement in PLACEMENTS:
        for rate in RATES:
            lam = 3600.0 / rate
            for seed in SEEDS:
                name = f"fgd_{placement['label']}_{rate}jph_s{seed}"
                exp = {
                    "name": name,
                    "seed": seed,
                    "lam": lam,
                    "enable_fgd": placement["enable_fgd"],
                    "fgd_placement_mode": placement["fgd_placement_mode"],
                }
                experiments.append(exp)

    config = {
        "description": (
            f"FGD replication via Gavel: {len(experiments)} experiments. "
            f"4 placements x {len(RATES)} rates x {len(SEEDS)} seeds. "
            f"Alibaba split cluster (12 sub-types, 6212 GPUs)."
        ),
        "common": COMMON,
        "experiments": experiments,
    }

    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'phase_fgd_replication.json'
    )
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {output_path}: {len(experiments)} experiments")

if __name__ == '__main__':
    main()
```

**Step 2: Run the generator**

Run: `python experiments/combined/scripts/generate_fgd_replication_config.py`
Expected: `Wrote .../phase_fgd_replication.json: 180 experiments`

**Step 3: Validate the config**

```bash
python3 -c "
import json
with open('experiments/combined/configs/phase_fgd_replication.json') as f:
    c = json.load(f)
exps = c['experiments']
print(f'{len(exps)} experiments')
# Check naming
names = [e['name'] for e in exps]
assert len(set(names)) == 180, f'Expected 180 unique names, got {len(set(names))}'
# Check placements
placements = set(e['fgd_placement_mode'] for e in exps)
print(f'Placements: {placements}')
# Check that strided experiments have enable_fgd=False
strided = [e for e in exps if 'strided' in e['name']]
assert all(not e['enable_fgd'] for e in strided), 'Strided should have enable_fgd=False'
print('Config validation PASSED')
"
```

**Step 4: Commit**

```bash
git add experiments/combined/scripts/generate_fgd_replication_config.py experiments/combined/configs/phase_fgd_replication.json
git commit -m "feat: add 180-experiment config for FGD replication"
```

---

## Task 6: Create SLURM Job Script

**Files:**
- Create: `experiments/combined/slurm/submit_fgd_replication.sbatch`

**Step 1: Write the SLURM array job script**

```bash
#!/bin/bash
#SBATCH --job-name=fgd-repl
#SBATCH --output=experiments/combined/logs/fgd_repl_%a.out
#SBATCH --error=experiments/combined/logs/fgd_repl_%a.err
#SBATCH --array=0-179
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1

cd ~/gavel

mkdir -p experiments/combined/logs
mkdir -p experiments/combined/results/fgd_replication

python3 experiments/combined/run_fgd_experiments.py \
    --phase fgd_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "experiments/combined/results/fgd_replication/result_${SLURM_ARRAY_TASK_ID}.json" \
    --quiet \
    --max-wall-time 13500
```

The `--max-wall-time 13500` (3h45m) gives a 15-minute buffer before the 4-hour SLURM limit.

**Step 2: Commit**

```bash
git add experiments/combined/slurm/submit_fgd_replication.sbatch
git commit -m "feat: add SLURM script for FGD replication (180 experiments)"
```

---

## Task 7: Create Plotting Script

**Files:**
- Create: `experiments/combined/scripts/plot_fgd_replication.py`
- Read: `src/fgd/data/paper_reference_curves.json` (for reference overlays)

**Step 1: Write the plotting script**

```python
#!/usr/bin/env python3
"""Plot FGD replication results: 4 figures + 2x2 grid.

Figures:
  - Fig 7a: Fragmentation Rate (%) vs Utilization (%)
  - Fig 7b: Frag/Total (%) vs Utilization (%)
  - Fig 9a: Unallocated GPU (%) vs Utilization (%)
  - Fig 9b: Occupied Nodes vs Utilization (%)
  - comparison_all.png: 2x2 grid

Usage:
    python plot_fgd_replication.py --results-dir results/fgd_replication --output-dir figures/fgd_replication
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Placement styles: label, color, marker
PLACEMENT_STYLE = {
    'strided': {'color': '#56B4E9', 'label': 'Strided (Gavel)', 'marker': 's'},
    'random':  {'color': '#888888', 'label': 'Random',           'marker': 'o'},
    'bestfit': {'color': '#D55E00', 'label': 'BestFit',          'marker': '^'},
    'fgd':     {'color': '#000000', 'label': 'FGD',              'marker': 'D'},
}

PLACEMENT_ORDER = ['random', 'strided', 'bestfit', 'fgd']

# Paper reference policy mapping (for dashed overlay)
PAPER_REF_POLICIES = {
    'random': 'random',
    'bestfit': 'bestfit',
    'fgd': 'fgd',
}


def load_results(results_dir):
    """Load all result JSONs from directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
    return results


def load_reference(ref_path):
    """Load FGD paper reference curves."""
    with open(ref_path) as f:
        return json.load(f)


def get_placement_label(result):
    """Extract placement label from result dict."""
    if not result.get('enable_fgd', False):
        return 'strided'
    return result.get('fgd_placement_mode', 'fgd')


def group_results(results):
    """Group by (placement, rate) and average over seeds."""
    groups = defaultdict(list)
    for r in results:
        placement = get_placement_label(r)
        lam = r['lam']
        groups[(placement, lam)].append(r)
    return groups


def compute_series(results):
    """Compute per-placement series: sorted by utilization, with error bars."""
    groups = group_results(results)
    series = defaultdict(lambda: {'util': [], 'frag_rate': [], 'frag_total': [],
                                   'unalloc': [], 'nodes': [],
                                   'frag_rate_std': [], 'frag_total_std': [],
                                   'unalloc_std': [], 'nodes_std': []})
    for (placement, lam), runs in groups.items():
        if not all('avg_utilization' in r for r in runs):
            continue
        s = series[placement]
        s['util'].append(np.mean([r['avg_utilization'] for r in runs]))
        s['frag_rate'].append(np.mean([r['avg_frag_rate'] for r in runs]))
        s['frag_total'].append(np.mean([r['avg_frag_total'] for r in runs]))
        s['unalloc'].append(np.mean([r['avg_unalloc_pct'] for r in runs]))
        s['nodes'].append(np.mean([r['avg_occupied_nodes'] for r in runs]))
        # Std across seeds
        s['frag_rate_std'].append(np.std([r['avg_frag_rate'] for r in runs]))
        s['frag_total_std'].append(np.std([r['avg_frag_total'] for r in runs]))
        s['unalloc_std'].append(np.std([r['avg_unalloc_pct'] for r in runs]))
        s['nodes_std'].append(np.std([r['avg_occupied_nodes'] for r in runs]))

    # Sort each series by utilization
    for placement in series:
        s = series[placement]
        order = np.argsort(s['util'])
        for key in s:
            s[key] = [s[key][i] for i in order]

    return series


def plot_figure(ax, series, y_key, y_label, reference=None, ref_figure=None):
    """Plot one figure panel."""
    for placement in PLACEMENT_ORDER:
        if placement not in series:
            continue
        s = series[placement]
        style = PLACEMENT_STYLE[placement]
        std_key = y_key + '_std' if y_key + '_std' in s else None
        ax.plot(s['util'], s[y_key],
                color=style['color'], marker=style['marker'],
                label=style['label'], linewidth=2, markersize=5)
        if std_key and std_key in s:
            y = np.array(s[y_key])
            err = np.array(s[std_key])
            ax.fill_between(s['util'], y - err, y + err,
                            alpha=0.15, color=style['color'])

    # Paper reference overlay (dashed)
    if reference and ref_figure and ref_figure in reference:
        for ref_policy, ref_key in PAPER_REF_POLICIES.items():
            if ref_key in reference[ref_figure]:
                ref_data = reference[ref_figure][ref_key]
                x = [p[0] for p in ref_data]
                y = [p[1] for p in ref_data]
                color = PLACEMENT_STYLE.get(ref_policy, {}).get('color', '#AAAAAA')
                ax.plot(x, y, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Utilization (%)')
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference', default=None,
                        help='Path to FGD paper_reference_curves.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")

    reference = None
    if args.reference:
        reference = load_reference(args.reference)

    series = compute_series(results)
    print(f"Placements: {list(series.keys())}")

    # Individual figures
    figures = [
        ('fig7a_frag_rate.png',    'frag_rate',  'Fragmentation Rate (%)',  'fig_7a'),
        ('fig7b_frag_total.png',   'frag_total', 'Frag/Total (%)',         'fig_7b'),
        ('fig9a_unalloc.png',      'unalloc',    'Unallocated GPU (%)',    'fig_9a'),
        ('fig9b_nodes.png',        'nodes',      'Occupied Nodes',         'fig_9b'),
    ]

    for fname, y_key, y_label, ref_fig in figures:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_figure(ax, series, y_key, y_label, reference, ref_fig)
        ax.set_title(y_label)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (fname, y_key, y_label, ref_fig) in enumerate(figures):
        ax = axes[idx // 2][idx % 2]
        plot_figure(ax, series, y_key, y_label, reference, ref_fig)
        ax.set_title(y_label)

    fig.suptitle('FGD Replication via Gavel (Alibaba Cluster)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'comparison_all.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved comparison_all.png")


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add experiments/combined/scripts/plot_fgd_replication.py
git commit -m "feat: add plotting script for FGD replication figures"
```

---

## Task 8: Run Integration Tests (Final Regression Check)

**Files:**
- No changes

**Step 1: Run unit tests**

Run: `cd src/scheduler && python -m unittest tests.policies_tests -v`
Expected: All tests PASS

**Step 2: Run integration tests**

Run: `cd src/scheduler && python -m unittest tests.integration_test -v`
Expected: Both tests PASS with exact expected JCT values

**Step 3: Commit any remaining changes**

If all tests pass, no further commits needed.

---

## Task 9: Sync to FarmShare and Run Calibration Batch

**Files:**
- No new files

**Step 1: Sync code to FarmShare**

```bash
rsync -avz src/scheduler/ farmshare:~/gavel/src/scheduler/
rsync -avz experiments/combined/ farmshare:~/gavel/experiments/combined/
rsync -avz src/fgd/ farmshare:~/gavel/src/fgd/
```

**Step 2: Create output directories on FarmShare**

```bash
ssh farmshare "mkdir -p ~/gavel/experiments/combined/logs ~/gavel/experiments/combined/results/fgd_replication"
```

**Step 3: Run calibration batch (20 experiments: 4 placements x 5 rates x 1 seed)**

Run indices 0, 3, 6, 9, 12 (seed=0 for rates 5, 30, 60, 130, 360 jph) for all 4 placements.
The config is ordered as: for each placement, for each rate, for each seed. So indices for seed=0 at key rates:

- Strided: 0 (5jph), 9 (30jph), 18 (60jph), 27 (130jph), 42 (360jph)
- Random: 45+same = 45, 54, 63, 72, 87
- BestFit: 90+same = 90, 99, 108, 117, 132
- FGD: 135+same = 135, 144, 153, 162, 177

Submit as: `sbatch --array=0,9,18,27,42,45,54,63,72,87,90,99,108,117,132,135,144,153,162,177 experiments/combined/slurm/submit_fgd_replication.sbatch`

**Step 4: Check calibration results**

Once jobs complete, download results and verify:
- Metrics are populated (avg_utilization, avg_frag_rate, etc.)
- Utilization increases with rate
- FGD has lower fragmentation than Random

**Step 5: If calibration looks good, run the full 180-experiment sweep**

```bash
ssh farmshare "cd ~/gavel && sbatch experiments/combined/slurm/submit_fgd_replication.sbatch"
```

---

## Task 10: Generate Plots

**Files:**
- No new code changes

**Step 1: Download results from FarmShare**

```bash
rsync -avz farmshare:~/gavel/experiments/combined/results/fgd_replication/ experiments/combined/results/fgd_replication/
```

**Step 2: Generate plots**

```bash
python experiments/combined/scripts/plot_fgd_replication.py \
    --results-dir experiments/combined/results/fgd_replication \
    --output-dir experiments/combined/figures/fgd_replication \
    --reference src/fgd/data/paper_reference_curves.json
```

Expected: 5 PNG files in `figures/fgd_replication/`

**Step 3: Review and iterate**

Open `comparison_all.png` and check:
- Policy ordering matches FGD paper: FGD < BestFit < Random
- Curves have expected shape (fragmentation rises with utilization)
- Error bands are reasonable (3 seeds)
- Paper reference dashed lines are in the right ballpark

If rates need adjustment (e.g., gap in utilization coverage), add/remove rates from the config and re-run.
