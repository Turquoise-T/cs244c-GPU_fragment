# Gavel Replication via Combined Runner -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run 417 experiments replicating Gavel Figs 9/10/11 on Philly traces, using `run_fgd_experiments.py` instead of the original runner.

**Architecture:** A Python generator script produces `phase_gavel_replication.json` with all 417 experiments. A SLURM array job runs them on FarmShare. A plot script compares results against paper reference curves.

**Tech Stack:** Python 3, json, matplotlib, SLURM

---

### Task 1: Write the config generator script

**Files:**
- Create: `experiments/combined/scripts/generate_gavel_replication_config.py`
- Create: `experiments/combined/configs/phase_gavel_replication.json` (output)

**Step 1: Create the scripts directory**

```bash
mkdir -p experiments/combined/scripts
```

**Step 2: Write the generator script**

Create `experiments/combined/scripts/generate_gavel_replication_config.py`:

```python
#!/usr/bin/env python3
"""Generate phase_gavel_replication.json for Gavel OSDI'20 Figs 9/10/11.

Produces 417 experiments matching the paper's Philly trace setup,
configured for the combined run_fgd_experiments.py runner.

Usage:
    python generate_gavel_replication_config.py
    # Output: ../configs/phase_gavel_replication.json
"""

import json
import os

FIGURES = {
    'fig9': {
        'policies': ['max_min_fairness', 'max_min_fairness_perf',
                      'max_min_fairness_packed'],
        'rates': [round(0.4 * i, 1) for i in range(1, 21)],  # 0.4--8.0
        'multi_gpu': False,
        'suffix': 'single',
    },
    'fig10': {
        'policies': ['max_min_fairness', 'max_min_fairness_perf',
                      'max_min_fairness_packed'],
        'rates': [round(0.2 * i, 1) for i in range(1, 16)],  # 0.2--3.0
        'multi_gpu': True,
        'suffix': 'multi',
    },
    'fig11': {
        'policies': ['finish_time_fairness', 'finish_time_fairness_perf'],
        'rates': [round(0.2 * i, 1) for i in range(1, 18)],  # 0.2--3.4
        'multi_gpu': True,
        'suffix': 'multi',
    },
}

SEEDS = [0, 1, 2]


def main():
    experiments = []
    for fig_name, fig in FIGURES.items():
        for policy in fig['policies']:
            for rate in fig['rates']:
                for seed in SEEDS:
                    lam = 3600.0 / rate
                    name = (f"{fig_name}_{policy}_{rate}jph"
                            f"_{fig['suffix']}_s{seed}")
                    experiments.append({
                        'name': name,
                        'policy': policy,
                        'lam': lam,
                        'seed': seed,
                        'generate_multi_gpu_jobs': fig['multi_gpu'],
                    })

    config = {
        'description': (
            'Gavel OSDI 2020 Figs 9/10/11 replication using combined runner. '
            '417 experiments: 3 figures x multiple policies x rate sweeps x 3 seeds. '
            'Matches paper setup exactly: 36:36:36 cluster, 360s rounds, '
            'window 4000-5000, FGD disabled.'
        ),
        'common': {
            'cluster_spec': {'v100': 36, 'p100': 36, 'k80': 36},
            'num_gpus_per_server': None,
            'mode': 'steady_state',
            'window_start': 4000,
            'window_end': 5000,
            'max_jct': 360000,
            'time_per_iteration': 360,
            'enable_fgd': False,
            'enable_migration_penalty': False,
            'enable_gpu_sharing': False,
            'solver': 'ECOS',
        },
        'experiments': experiments,
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'configs', 'phase_gavel_replication.json')
    out_path = os.path.abspath(out_path)
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'Wrote {len(experiments)} experiments to {out_path}')

    # Summary
    from collections import Counter
    by_fig = Counter()
    for e in experiments:
        fig = e['name'].split('_')[0]
        by_fig[fig] += 1
    for fig in sorted(by_fig):
        print(f'  {fig}: {by_fig[fig]}')


if __name__ == '__main__':
    main()
```

**Step 3: Run the generator and verify output**

```bash
cd experiments/combined/scripts
python generate_gavel_replication_config.py
```

Expected output:
```
Wrote 417 experiments to .../configs/phase_gavel_replication.json
  fig10: 135
  fig11: 102
  fig9: 180
```

**Step 4: Validate the generated config**

```bash
python3 -c "
import json
with open('../configs/phase_gavel_replication.json') as f:
    cfg = json.load(f)
exps = cfg['experiments']
assert len(exps) == 417, f'Expected 417, got {len(exps)}'
# Check common params
c = cfg['common']
assert c['time_per_iteration'] == 360
assert c['enable_fgd'] == False
assert c['cluster_spec'] == {'v100': 36, 'p100': 36, 'k80': 36}
assert c['num_gpus_per_server'] is None
assert c['window_start'] == 4000
assert c['window_end'] == 5000
# Check lambda conversion
for e in exps:
    rate = float(e['name'].split('jph')[0].split('_')[-1])
    expected_lam = 3600.0 / rate
    assert abs(e['lam'] - expected_lam) < 0.01, f'{e[\"name\"]}: lam mismatch'
# Check unique names
names = [e['name'] for e in exps]
assert len(names) == len(set(names)), 'Duplicate names'
print('All validations passed')
"
```

**Step 5: Commit**

```bash
git add experiments/combined/scripts/generate_gavel_replication_config.py \
       experiments/combined/configs/phase_gavel_replication.json
git commit -m "Add Gavel replication config for combined runner (417 experiments)"
```

---

### Task 2: Verify combined runner handles the config

**Purpose:** Smoke-test that `run_fgd_experiments.py` can parse the new config and start a simulation without errors.

**Step 1: Dry-run a single experiment locally**

Pick experiment index 90 (fig9, max_min_fairness_perf, 4.0 jph, seed 0 -- moderate rate, should complete in ~10 min).

```bash
cd experiments/combined

# Find the right index
python3 -c "
import json
with open('configs/phase_gavel_replication.json') as f:
    exps = json.load(f)['experiments']
for i, e in enumerate(exps):
    if 'max_min_fairness_perf_4.0jph_single_s0' in e['name']:
        print(f'Index {i}: {e[\"name\"]}')
        break
"
```

**Step 2: Run the experiment**

```bash
python run_fgd_experiments.py \
  --phase gavel_replication \
  --index <INDEX_FROM_STEP_1> \
  --save-logs \
  --output results/gavel_replication_validation.json
```

Expected: completes without error, prints JCT result. Should take ~10-15 min.

**Step 3: Compare with original replication**

```bash
python3 -c "
import json, csv

# Load combined runner result
with open('results/gavel_replication_validation.json') as f:
    combined = json.load(f)[0]

# Load original result
with open('../gavel-replication/results/results_combined.csv') as f:
    for row in csv.DictReader(f):
        if 'max_min_fairness_perf_4.0jph' in row['name'] and 's0' in row['name']:
            orig_jct = float(row['jct_sec'])
            break

combined_jct = combined['avg_jct']
pct_diff = abs(combined_jct - orig_jct) / orig_jct * 100
print(f'Combined runner: {combined_jct:.2f}s ({combined_jct/3600:.2f} hrs)')
print(f'Original runner: {orig_jct:.2f}s ({orig_jct/3600:.2f} hrs)')
print(f'Difference: {pct_diff:.3f}%')
assert pct_diff < 1.0, f'JCT mismatch: {pct_diff:.3f}% > 1%'
print('PASS: results match within 1%')
"
```

**Step 4: Verify log files were created**

```bash
ls -la logs/fig9_max_min_fairness_perf_4.0jph_single_s0.log
wc -l logs/fig9_max_min_fairness_perf_4.0jph_single_s0.log
```

Expected: log file exists and has substantial content (thousands of lines with round/microtask data).

---

### Task 3: Write the SLURM submission script

**Files:**
- Create: `experiments/combined/slurm/submit_gavel_replication.sbatch`

**Step 1: Write the SLURM script**

Create `experiments/combined/slurm/submit_gavel_replication.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=gavel-repl
#SBATCH --output=slurm_logs/gavel_repl-%A_%a.out
#SBATCH --error=slurm_logs/gavel_repl-%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-416

# Gavel OSDI'20 Figs 9/10/11 Replication (417 experiments)
# Policies: max_min_fairness, max_min_fairness_perf, max_min_fairness_packed,
#           finish_time_fairness, finish_time_fairness_perf
# Cluster: 36:36:36 (V100:P100:K80), 360s rounds, window 4000-5000
# Max wall time: 8 hours per experiment
#
# Usage:
#   cd ~/gavel/experiments/combined/slurm && sbatch submit_gavel_replication.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/gavel_replication"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gavel_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 28800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
```

**Step 2: Commit**

```bash
git add experiments/combined/slurm/submit_gavel_replication.sbatch
git commit -m "Add SLURM script for Gavel replication via combined runner"
```

---

### Task 4: Write the results merge script

**Files:**
- Create: `experiments/combined/scripts/merge_gavel_replication_results.py`

**Purpose:** After SLURM array completes, merge per-task JSON files into a single CSV for plotting.

**Step 1: Write the merge script**

Create `experiments/combined/scripts/merge_gavel_replication_results.py`:

```python
#!/usr/bin/env python3
"""Merge per-experiment JSON results into a single CSV.

Usage:
    python merge_gavel_replication_results.py
    python merge_gavel_replication_results.py --results-dir ../results/gavel_replication
"""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir',
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'results',
                                             'gavel_replication'))
    parser.add_argument('--output',
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'results',
                                             'gavel_replication_combined.csv'))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []

    for json_file in sorted(results_dir.glob('exp_*.json')):
        with open(json_file) as f:
            data = json.load(f)
        # Each file contains a list with one result
        for r in data:
            # Extract figure and jobs_per_hr from name
            parts = r['name'].split('_')
            figure = parts[0]  # fig9, fig10, fig11
            # Extract rate: everything between last policy word and "jph"
            rate_str = r['name'].split('jph')[0].split('_')[-1]
            jobs_per_hr = float(rate_str)

            rows.append({
                'name': r['name'],
                'figure': figure,
                'policy': r['policy'],
                'jobs_per_hr': jobs_per_hr,
                'seed': r['seed'],
                'jct_sec': r['avg_jct'],
                'saturated': r['saturated'],
                'wall_time_sec': r['wall_time_seconds'],
                'num_completed': r['num_completed_jobs'],
                'multi_gpu': r.get('generate_multi_gpu_jobs', False),
            })

    rows.sort(key=lambda r: (r['figure'], r['policy'], r['jobs_per_hr'],
                              r['seed']))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f'Merged {len(rows)} results to {out_path}')

    # Summary
    from collections import Counter
    by_fig = Counter(r['figure'] for r in rows)
    saturated = sum(1 for r in rows if r['saturated'])
    print(f'  By figure: {dict(sorted(by_fig.items()))}')
    print(f'  Saturated: {saturated}/{len(rows)}')


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add experiments/combined/scripts/merge_gavel_replication_results.py
git commit -m "Add results merge script for Gavel replication"
```

---

### Task 5: Write the comparison plot script

**Files:**
- Create: `experiments/combined/scripts/plot_gavel_replication.py`
- Reference: `experiments/gavel-replication/scripts/plot_results.py` (existing pattern)
- Reference: `experiments/gavel-replication/scripts/paper_reference_curves.json` (digitized data)

**Step 1: Write the plot script**

Create `experiments/combined/scripts/plot_gavel_replication.py`:

```python
#!/usr/bin/env python3
"""Plot Gavel replication results (combined runner) vs paper reference curves.

Generates Figs 9, 10, 11: Average JCT (hours) vs Input Job Rate (jobs/hr).
Each figure shows baseline, Gavel, and optionally _packed policies,
overlaid with digitized reference data from the OSDI'20 paper.

Usage:
    python plot_gavel_replication.py
    python plot_gavel_replication.py --csv ../results/gavel_replication_combined.csv
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

FIGURE_DEFS = {
    'fig9': {
        'title': 'Figure 9: Single-GPU Jobs (LAS / Max-Min Fairness)',
        'policies': {
            'max_min_fairness': ('Agnostic (baseline)', '#e74c3c', 's'),
            'max_min_fairness_perf': ('Gavel', '#2ecc71', 'o'),
            'max_min_fairness_packed': ('Gavel + Packing', '#3498db', '^'),
        },
        'ref_key': 'fig9',
        'y_max': 120,
    },
    'fig10': {
        'title': 'Figure 10: Multi-GPU Jobs (LAS / Max-Min Fairness)',
        'policies': {
            'max_min_fairness': ('Agnostic (baseline)', '#e74c3c', 's'),
            'max_min_fairness_perf': ('Gavel', '#2ecc71', 'o'),
            'max_min_fairness_packed': ('Gavel + Packing', '#3498db', '^'),
        },
        'ref_key': 'fig10',
        'y_max': 120,
    },
    'fig11': {
        'title': 'Figure 11: Multi-GPU Jobs (Finish-Time Fairness)',
        'policies': {
            'finish_time_fairness': ('Agnostic (baseline)', '#e74c3c', 's'),
            'finish_time_fairness_perf': ('Gavel', '#2ecc71', 'o'),
        },
        'ref_key': 'fig11',
        'y_max': 120,
    },
}


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df['jct_sec'] = pd.to_numeric(df['jct_sec'], errors='coerce')
    df['jct_hours'] = df['jct_sec'] / 3600
    return df


def load_reference(ref_path):
    with open(ref_path) as f:
        return json.load(f)


def plot_figure(ax, df, fig_name, fig_def, ref_data):
    fig_df = df[df['figure'] == fig_name].copy()

    for policy, (label, color, marker) in fig_def['policies'].items():
        policy_df = fig_df[fig_df['policy'] == policy]
        # Drop saturated/inf
        policy_df = policy_df[
            policy_df['jct_hours'].notna() &
            (policy_df['jct_hours'] != float('inf')) &
            (~policy_df['saturated'].astype(bool))
        ]
        if policy_df.empty:
            continue

        stats = (policy_df.groupby('jobs_per_hr')['jct_hours']
                 .agg(['mean', 'std', 'count'])
                 .reset_index())

        ax.errorbar(stats['jobs_per_hr'], stats['mean'],
                     yerr=stats['std'].fillna(0),
                     label=f'{label} (ours)',
                     color=color, marker=marker, markersize=5,
                     linewidth=1.5, capsize=3)

    # Overlay paper reference curves
    ref_key = fig_def['ref_key']
    if ref_data and ref_key in ref_data:
        ref = ref_data[ref_key]
        rates = ref['jobs_per_hr']

        if 'gavel' in ref:
            gavel_pts = [(r, v) for r, v in zip(rates, ref['gavel'])
                         if v is not None]
            if gavel_pts:
                rx, ry = zip(*gavel_pts)
                ax.plot(rx, ry, '--', color='#2ecc71', alpha=0.4,
                        linewidth=1, label='Gavel (paper)')

        if 'baseline' in ref:
            base_pts = [(r, v) for r, v in zip(rates, ref['baseline'])
                        if v is not None]
            if base_pts:
                rx, ry = zip(*base_pts)
                ax.plot(rx, ry, '--', color='#e74c3c', alpha=0.4,
                        linewidth=1, label='Baseline (paper)')

    ax.set_title(fig_def['title'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Input Job Rate (jobs/hr)')
    ax.set_ylabel('Average JCT (hours)')
    ax.set_ylim(0, fig_def['y_max'])
    ax.legend(fontsize=8, loc='upper left')


def main():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        default=str(script_dir / '..' / 'results' /
                                    'gavel_replication_combined.csv'))
    parser.add_argument('--ref-json',
                        default=str(script_dir / '..' / '..' /
                                    'gavel-replication' / 'scripts' /
                                    'paper_reference_curves.json'))
    parser.add_argument('--output',
                        default=str(script_dir / '..' / 'figures' /
                                    'gavel_replication_combined.png'))
    args = parser.parse_args()

    df = load_results(args.csv)
    ref_data = None
    if os.path.exists(args.ref_json):
        ref_data = load_reference(args.ref_json)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (fig_name, fig_def) in zip(axes, FIGURE_DEFS.items()):
        plot_figure(ax, df, fig_name, fig_def, ref_data)

    fig.suptitle('Gavel Replication (Combined Runner) vs Paper',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {out_path}')


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add experiments/combined/scripts/plot_gavel_replication.py
git commit -m "Add plot script for Gavel replication comparison"
```

---

### Task 6: Final commit with design doc

**Step 1: Add the design doc**

```bash
git add docs/plans/2026-02-16-gavel-replication-combined-design.md
git commit -m "Add Gavel replication experiment design doc"
```

---

## Post-Implementation: Running the Experiments

### Local validation

```bash
cd experiments/combined
# Run 2-3 experiments at moderate rates to verify correctness
python run_fgd_experiments.py --phase gavel_replication --index <IDX> --save-logs \
  --output results/gavel_replication/exp_<IDX>.json
```

### FarmShare submission

```bash
# Sync code to FarmShare
rsync -avz experiments/combined/ farmshare:~/gavel/experiments/combined/

# Submit
ssh farmshare "cd ~/gavel/experiments/combined/slurm && sbatch submit_gavel_replication.sbatch"
```

### After completion

```bash
# Sync results back
rsync -avz farmshare:~/gavel/experiments/combined/results/gavel_replication/ \
  experiments/combined/results/gavel_replication/

# Merge and plot
cd experiments/combined/scripts
python merge_gavel_replication_results.py
python plot_gavel_replication.py
```
