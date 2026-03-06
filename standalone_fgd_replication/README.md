# FGD Replication — Experiment Scripts

Replication of *"Beware of Fragmentation: Scheduling GPU-Sharing Workloads with
Fragmentation Gradient Descent"* (USENIX ATC '23).

All scripts are run from `fgd_replication/` and read trace data from
`../alibaba_traces/cluster-trace-gpu-v2023/`.

---

## Result Directory Mapping

Every script writes results into `result/<name>/` where `<name>` encodes the
script, figure, and key parameters. The table below shows the mapping from
command to result directory and the files produced inside it.

| Command | Result directory | Files |
|---|---|---|
| `python3 exp_fig7.py --num-runs 10 --seed 42` | `result/fig7-runs10-seed42/` | `figure7_results.csv`, `figure7a.png`, `figure7b.png`, `experiment_summary.log` |
| `python3 exp_fig9.py --num-runs 10 --seed 42` | `result/fig9-runs10-seed42/` | `figure9a_unalloc.csv`, `figure9b_occupied.csv`, `figure9c_failed.csv`, `figure9d_breakdown.csv`, `figure9.png`, `experiment_summary.log` |
| `python3 exp_fig11_14.py --figures 11 --num-runs 10 --seed 42` | `result/fig11-runs10-seed42/` | `figure11_results.csv`, `figure11.png`, `experiment_summary.log` |
| `python3 exp_fig11_14.py --figures 12 --num-runs 10 --seed 42` | `result/fig12-runs10-seed42/` | `figure12_results.csv`, `figure12.png`, `experiment_summary.log` |
| `python3 exp_fig11_14.py --figures 13 --num-runs 10 --seed 42` | `result/fig13-runs10-seed42/` | `figure13_results.csv`, `figure13.png`, `experiment_summary.log` |
| `python3 exp_fig11_14.py --figures 14 --num-runs 10 --seed 42` | `result/fig14-runs10-seed42/` | `figure14_results.csv`, `figure14.png`, `experiment_summary.log` |
| `python3 exp_dist_shift.py --task-order ascending --schedulers Random,BestFit,DotProd,Packing,Clustering,FGD-Full,FGD-2000,W-FGD-2000,U-FGD` | `result/dist-shift-ascending-100/` | `experiment_summary.log` |
| `python3 exp_dist_shift.py --task-order descending --schedulers Random,BestFit,DotProd,Packing,Clustering,FGD-Full,FGD-2000,W-FGD-2000,U-FGD` | `result/dist-shift-descending-100/` | `experiment_summary.log` |
| `python3 exp_dist_shift.py --task-order phased --tier-order 0,1,2,3,4 --schedulers Random,BestFit,DotProd,Packing,Clustering,FGD-Full,FGD-2000,W-FGD-2000,U-FGD` | `result/dist-shift-phased-01234-100/` | `experiment_summary.log` |
| `python3 exp_dist_shift.py --task-order phased --tier-order 1,2,0,3,4 --schedulers Random,BestFit,DotProd,Packing,Clustering,FGD-Full,FGD-2000,W-FGD-2000,U-FGD` | `result/dist-shift-phased-12034-100/` | `experiment_summary.log` |

**Key naming rules:**
- `fig7` and `fig9` are fixed prefixes for their scripts.
- `fig11_14` uses `fig{N}` for a single figure or `fig{A}-{B}-...` when
  multiple figures are run together in one invocation.
- Changing `--num-runs` or `--seed` produces a separate directory, so results
  from different runs never overwrite each other.
- `dist-shift` uses `dist-shift-{order}-{scale}` where `order` is the
  `--task-order` value (`trace`, `ascending`, `descending`, or
  `phased-{digits}` encoding the tier sequence) and `scale` is the
  `--cluster-scale` value. For example, `--task-order phased --tier-order
  3,2,1,4,0 --cluster-scale 50` produces `dist-shift-phased-32140-50/`.

**Plot-only mode** (`--plot-csv`) reads from an existing directory and writes
the updated PNG back into the same directory. It does not create a new directory.

---

## Scripts

### `exp_fig7.py` — Figure 7(a)/(b): Fragmentation vs arrived workload

Runs Monte-Carlo workload inflation using the default trace
(`openb_pod_list_default.csv`) and plots:
- **Figure 7(a):** fragmentation rate (%) vs arrived GPU workload (%)
- **Figure 7(b):** fragmented GPUs / total resources (%) vs arrived GPU workload (%)

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--num-runs` | 3 | Runs per scheduler (paper uses 10) |
| `--seed` | 42 | Base random seed |
| `--max-workload` | 120.0 | Stop when arrived workload reaches this % of GPU capacity |
| `--sample-interval` | 5.0 | Record fragmentation every this many % of arrived workload |
| `--schedulers` | `all` | Comma-separated subset to run, e.g. `FGD,Packing` |
| `--plot-csv` | — | Path to existing result CSV; skips experiment and plots only |

**Result directory:** `result/fig7-runs{N}-seed{S}/`

**Output files:**
- `figure7_results.csv` — columns: `scheduler, arrived_workload_pct, frag_rate, frag_total_pct, run`
- `figure7a.png`
- `figure7b.png`
- `experiment_summary.log`

**Examples**
```bash
# Full run (paper settings)
python3 exp_fig7.py --num-runs 10 --seed 42

# Quick test
python3 exp_fig7.py --num-runs 1

# Run only FGD and Packing
python3 exp_fig7.py --schedulers FGD,Packing

# Plot from saved CSV
python3 exp_fig7.py --plot-csv result/fig7-runs10-seed42/figure7_results.csv
```

---

### `exp_fig9.py` — Figure 9: Multi-metric evaluation (4 sub-figures)

Monte-Carlo workload inflation producing four sub-figures:
- **(a)** Unallocated GPU % vs arrived workload (80–120% range)
- **(b)** Occupied nodes vs arrived workload (0–100% range)
- **(c)** Failed task GPU demand by category at 96% arrival (bar chart)
- **(d)** Fragmentation breakdown by cause at end of run (bar chart)

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--num-runs` | 10 | Monte-Carlo runs per scheduler |
| `--seed` | 42 | Base random seed |
| `--sample-interval` | 2.0 | Record metrics every this many % of arrived workload |
| `--max-arrival` | 120.0 | Stop when arrived workload reaches this % of GPU capacity |
| `--schedulers` | `all` | Comma-separated subset to run, e.g. `FGD,Packing` |
| `--plot-csv` | — | Path to result directory containing all 4 CSVs; plots only |

**Result directory:** `result/fig9-runs{N}-seed{S}/`

**Output files:**
- `figure9a_unalloc.csv` — columns: `scheduler, arrived_pct, unalloc_gpu_pct, run`
- `figure9b_occupied.csv` — columns: `scheduler, arrived_pct, occupied_nodes, run`
- `figure9c_failed.csv` — columns: `scheduler, gpu_category, sum_gpu_demand, run`
- `figure9d_breakdown.csv` — columns: `scheduler, cause, pct, run`
- `figure9.png`
- `experiment_summary.log`

**Examples**
```bash
# Full run
python3 exp_fig9.py --num-runs 10 --seed 42

# Quick test
python3 exp_fig9.py --num-runs 1

# Run only FGD and Packing
python3 exp_fig9.py --schedulers FGD,Packing

# Plot from saved CSVs
python3 exp_fig9.py --plot-csv result/fig9-runs10-seed42/
```

---

### `exp_fig11_14.py` — Figures 11–14: Sensitivity analysis

Evaluates schedulers as workload composition varies. Each figure loads a
different set of pre-built trace files; no synthetic sampling is performed.

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--figures` | `11,12,13,14` | Comma-separated list of figures to run |
| `--num-runs` | 10 | Monte-Carlo runs per (proportion, scheduler) combination |
| `--seed` | 42 | Base random seed |
| `--schedulers` | `all` | Comma-separated subset to run, e.g. `FGD,Packing` |
| `--plot-csv` | — | One or more CSV paths; plots only. Accepts multiple files |

**Result directory:** `result/fig{X}-runs{N}-seed{S}/`
(e.g., `fig11-12-runs10-seed42` when running figures 11 and 12 together)

**Output files** (one per figure):
- `figure{N}_results.csv` — columns: `proportion, scheduler, unalloc_gpu_pct, std`
- `figure{N}.png`
- `experiment_summary.log`

**Trace files used**

| Figure | x-axis | Trace files |
|---|---|---|
| 11 | GPU-sharing % of GPU requests | `openb_pod_list_gpushare{40,60,80,100}.csv` |
| 12 | Multi-GPU % of GPU requests | `openb_pod_list_multigpu{20,30,40,50}.csv` |
| 13 | GPU-type-constrained % of GPU requests | `openb_pod_list_gpuspec{10,20,25,33}.csv` |
| 14 | Non-GPU % of task count | `openb_pod_list_cpu{050,100,200,250}.csv` |

**Examples**
```bash
# Run all four figures
python3 exp_fig11_14.py

# Run single figure
python3 exp_fig11_14.py --figures 11

# Run subset of figures
python3 exp_fig11_14.py --figures 11,12 --num-runs 10

# Run only FGD and Packing across all figures
python3 exp_fig11_14.py --schedulers FGD,Packing

# Plot from saved CSVs
python3 exp_fig11_14.py --plot-csv result/fig11-runs10-seed42/figure11_results.csv
python3 exp_fig11_14.py --plot-csv result/fig11-runs10-seed42/figure11_results.csv \
                                   result/fig12-runs10-seed42/figure12_results.csv
```

---

### `exp_dist_shift.py` — Distribution-shift experiment

Replays the full default trace (`openb_pod_list_default.csv`) through the real
cluster in a single pass and measures final fragmentation, GPU allocation, and
throughput for each scheduler. Unlike `exp_fig7.py`, there is no random
sampling — the trace is played exactly once in the chosen order.

The key question is how well each FGD variant handles a mismatch between the
distribution it was initialised with (e.g. the first N tasks, or a uniform
grid) and the actual workload that arrives.

**Scheduler variants**

| Name | Description |
|---|---|
| `Random`, `BestFit`, `DotProd`, `Packing`, `Clustering` | Baseline schedulers |
| `FGD-Full` | FGD with oracle knowledge of the full trace distribution |
| `FGD-N` | FGD with static distribution from the first N tasks (e.g. `FGD-500`) |
| `W-FGD-M` | Windowed FGD with sliding window of size M (e.g. `W-FGD-200`) |
| `B-FGD` | Bayesian FGD; starts with a uniform prior and updates online |
| `U-FGD` | FGD with a uniform prior over the CPU × GPU grid |

For `FGD-N` and `W-FGD-M`, N and M are parsed directly from the scheduler name.

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--task-order` | `trace` | Task arrival order: `trace` (original creation-time order), `ascending`/`descending` (sorted by GPU demand), `phased` (GPU demand tiers in sequence) |
| `--cluster-scale` | `100.0` | Cluster size as % of original (e.g. `50` keeps 50% of each node type) |
| `--tier-order` | `0,1,2,3,4` | Tier sequence for `phased` mode. Tiers: 0=CPU-only, 1=small fractional (<0.5), 2=large fractional (0.5–1), 3=single full GPU, 4=multi-GPU |
| `--schedulers` | `all` | Comma-separated scheduler names to run |
| `--prior-strength` | `10.0` | B-FGD pseudo-count total (only used when B-FGD is selected) |
| `--min-gpu-tasks` | `50` | B-FGD falls back to Packing until this many GPU tasks are observed (only used when B-FGD is selected) |

**Result directory:** `result/dist-shift-{order}-{scale}/`
(e.g. `dist-shift-trace-100`, `dist-shift-phased-32140-50`)

**Output files:**
- `experiment_summary.log` — final Frag%, Alloc%, Scheduled, Failed, Time(s) per scheduler

**Examples**
```bash
# Default: trace order, full cluster, all schedulers
python3 exp_dist_shift.py

# Phased arrival (full-GPU tasks first, then fractional, then CPU-only)
python3 exp_dist_shift.py --task-order phased --tier-order 3,2,1,4,0

# Ascending GPU demand order, half-size cluster
python3 exp_dist_shift.py --task-order ascending --cluster-scale 50

# Compare FGD variants only
python3 exp_dist_shift.py --schedulers FGD-Full,FGD-500,W-FGD-200,B-FGD,U-FGD

# FGD with first-1000-task distribution vs windowed FGD with window 300
python3 exp_dist_shift.py --schedulers FGD-1000,W-FGD-300
```

---

## Configuration Assumptions

### Task type representation (all experiments)

FGD needs a discrete task-type distribution to compute fragmentation gradients.
Continuous CPU/GPU demands are bucketed as follows:

| Resource | Bucketing rule | Rationale |
|---|---|---|
| CPU | Round to nearest multiple of 4 cores | Reduces ~hundreds of unique values to ~20 types |
| GPU (fractional, < 1) | Round to 2 decimal places | Preserves GPU-sharing granularity |
| GPU (integer, ≥ 1) | Use exact integer | Multi-GPU tasks are already discrete |

This bucketing is applied in `trace_loader._bucket_cpu()` and consistently
throughout all scripts.

### Per-figure assumptions

**Figure 7(a)/(b) — `exp_fig7.py`**
- Trace: `openb_pod_list_default.csv` (8,152 tasks, 13.3% non-GPU)
- FGD distribution: computed from the full default trace (oracle knowledge)
- Fragmentation snapshot: every 5% of arrived workload
- Outputs both 7(a) and 7(b) from the same run and CSV

**Figure 9 — `exp_fig9.py`**
- Trace: `openb_pod_list_default.csv`
- FGD distribution: computed from the full default trace (oracle knowledge)
- Snapshot for 9(c) failed-task breakdown: taken at exactly 96% arrival
- Snapshot interval for 9(a)/(b) curves: every 2% of arrived workload
- Occupied node criterion: `allocated_cpu > 0` OR any GPU slot `< 1.0`
- Fragmentation breakdown (9d): computed at end of run using per-node,
  per-task-type fragmentation decomposed into *deficient*, *stranded*, *non-GPU*

**Figures 11–14 — `exp_fig11_14.py`**
- Each (figure, proportion) pair loads its own dedicated trace file; no
  synthetic reweighting is applied
- FGD distribution: computed from whichever trace file is loaded for that
  proportion — FGD has oracle knowledge of the current workload mix
- Figure 13 uses `GpuTypeAwareCluster`: tasks with a non-empty `gpu_spec`
  field can only be placed on nodes whose `gpu_model` matches
- Figure 12 trace files (`multigpu*.csv`) have a shorter column format
  (no `gpu_spec`, `qos`, or timestamp columns); the loader handles this
  gracefully with optional field access

**Distribution-shift — `exp_dist_shift.py`**
- Trace: `openb_pod_list_default.csv` (same as Fig 7a/9)
- Single deterministic pass; no Monte-Carlo sampling
- `FGD-Full` uses oracle knowledge of the full trace distribution
- `FGD-N` fixes the distribution to the first N tasks and never updates
- `W-FGD-M` pre-populates its window with the first M tasks, then slides online
- `B-FGD` starts from a uniform prior (derived from cluster node specs) and updates after every observed task
- `U-FGD` uses a static uniform distribution over the CPU × GPU grid
- Phased mode groups tasks into five GPU-demand tiers and replays them in the specified tier sequence
