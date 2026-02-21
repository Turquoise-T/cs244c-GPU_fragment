# FGD Replication via Gavel Scheduler

**Date:** 2026-02-21
**Goal:** Replicate FGD paper (ATC'23) fragmentation results using the integrated Gavel+FGD scheduler, matching experimental parameters exactly against the Alibaba trace cluster topology.

## Background

The FGD paper uses a static inflation experiment: tasks arrive one-by-one, never depart, and the cluster fills up. The x-axis is "arrived workloads (% of GPU capacity)." Gavel's scheduler is dynamic: jobs arrive at rate lambda, run, and complete.

We bridge this gap with an **arrival rate sweep**: run experiments at 15 different arrival rates to achieve different steady-state utilization levels, then measure fragmentation metrics at each level. The x-axis becomes "steady-state utilization %" -- the trends and policy ordering should match the paper.

## Cluster Configuration

### Source Data

Real Alibaba trace: `src/fgd/data/alibaba-gpu-v2023/openb_node_list_gpu_node.csv`
1,213 nodes, 6,212 GPUs, 7 GPU models, 12 unique (model, node_size) combinations.

### Gavel Mapping

Split each (model, node_size) combination into a separate Gavel "GPU type" to preserve the exact cluster topology:

| Sub-type    | GPUs/Server | Nodes | Total GPUs | Parent Model |
|-------------|-------------|-------|------------|--------------|
| G2_8        | 8           | 549   | 4,392      | G2           |
| T4_2        | 2           | 387   | 774        | T4           |
| G3_8        | 8           | 39    | 312        | G3           |
| P100_2      | 2           | 131   | 262        | P100         |
| V100M32_8   | 8           | 21    | 168        | V100M32      |
| V100M16_4   | 4           | 28    | 112        | V100M16      |
| T4_4        | 4           | 17    | 68         | T4           |
| V100M16_8   | 8           | 8     | 64         | V100M16      |
| V100M32_4   | 4           | 9     | 36         | V100M32      |
| V100M16_1   | 1           | 19    | 19         | V100M16      |
| P100_1      | 1           | 3     | 3          | P100         |
| A10_1       | 1           | 2     | 2          | A10          |
| **Total**   |             | **1,213** | **6,212** |          |

Config snippet:
```json
{
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
```

### Throughput Data

Sub-types of the same model share identical per-GPU throughput. Create `simulation_throughputs_alibaba_split.json` by duplicating entries:
- T4_2 and T4_4 both get T4's throughput values
- V100M16_1, V100M16_4, V100M16_8 all get V100M16's values
- V100M32_4 and V100M32_8 both get V100M32's values
- P100_1 and P100_2 both get P100's values
- A10_1 gets new entries (use P100 as proxy if A10 throughputs unavailable -- only 2 GPUs, negligible impact)

## Experiment Matrix

### Allocation Policy (fixed)

`max_min_fairness` (heterogeneity-agnostic LAS) for all experiments. Closest to FGD paper's FIFO scheduling. Isolates the placement effect.

### Placement Strategies (4 variants)

| Label     | Config                                          | FGD Paper Equivalent |
|-----------|-------------------------------------------------|---------------------|
| Strided   | `enable_fgd: false`                             | N/A (Gavel baseline) |
| Random    | `enable_fgd: true, fgd_placement_mode: random`  | Random              |
| BestFit   | `enable_fgd: true, fgd_placement_mode: bestfit` | BestFit             |
| FGD       | `enable_fgd: true, fgd_placement_mode: fgd`     | FGD                 |

### Arrival Rate Sweep (15 rates)

| Jobs/hour | Lambda (s) | Approx. utilization |
|-----------|-----------|---------------------|
| 5         | 720       | ~5-15%              |
| 10        | 360       | ~10-25%             |
| 20        | 180       | ~20-40%             |
| 30        | 120       | ~30-50%             |
| 40        | 90        | ~35-55%             |
| 50        | 72        | ~40-60%             |
| 60        | 60        | ~50-70%             |
| 80        | 45        | ~55-75%             |
| 100       | 36        | ~65-80%             |
| 130       | 27.7      | ~70-85%             |
| 160       | 22.5      | ~75-88%             |
| 200       | 18        | ~80-92%             |
| 250       | 14.4      | ~85-95%             |
| 300       | 12        | ~88-97%             |
| 360       | 10        | ~90%+ / saturated   |

Utilization estimates are approximate. May add or drop rates after a calibration run.

### Seeds

3 seeds: 0, 1, 2

### Total

**4 placements x 15 rates x 3 seeds = 180 experiments**

### Common Parameters

```json
{
  "mode": "steady_state",
  "window_start": 4000,
  "window_end": 5000,
  "max_jct": 360000,
  "time_per_iteration": 600,
  "generate_multi_gpu_jobs": true,
  "workload_mode": "alibaba",
  "fgd_workload_mode": "alibaba",
  "enable_migration_penalty": false,
  "enable_gpu_sharing": false,
  "solver": "ECOS",
  "completion_rate_threshold": 0.1
}
```

## Metrics & Instrumentation

### Per-Round Metrics

Add `_record_round_metrics()` to `scheduler.py` that computes after each scheduling round:

| Metric | Formula | Figure |
|--------|---------|--------|
| Frag Rate (%) | `F_N(M) / unallocated_gpus * 100` | Fig 7a |
| Frag/Total (%) | `F_N(M) / total_gpus * 100` | Fig 7b |
| Unallocated GPU (%) | `(total - allocated) / total * 100` | Fig 9a |
| Occupied Nodes | servers with >= 1 GPU allocated | Fig 9b |
| Utilization (%) | `allocated / total * 100` | x-axis |

Store in `_round_metrics_history` as a list of dicts with a `simulated_time` key.

### Fragmentation Calculation for Non-FGD Runs

When `enable_fgd=false` (strided placement), the FGD calculator is not normally imported. For metrics, import `FragmentationCalculator` from `src/fgd/fgd.py` regardless of placement mode. Build a workload model from the Alibaba trace distribution for fragmentation measurement.

### Output Per Experiment

The experiment runner extracts measurement-window metrics and outputs:

```json
{
  "name": "...",
  "policy": "max_min_fairness",
  "placement": "fgd",
  "seed": 0,
  "lam": 60.0,
  "avg_jct": 12345.6,
  "avg_utilization": 72.3,
  "avg_frag_rate": 35.2,
  "avg_frag_total": 8.1,
  "avg_unalloc_pct": 27.7,
  "avg_occupied_nodes": 987,
  "std_frag_rate": 2.1,
  "std_utilization": 1.5,
  "wall_time_seconds": 180.5
}
```

## Figures to Produce

### Fig 7a: Fragmentation Rate vs Utilization

- X-axis: average steady-state utilization (%)
- Y-axis: average frag rate (%)
- 4 lines (Strided, Random, BestFit, FGD) with error bands from 3 seeds
- Dashed reference lines from FGD paper (mapped from demand_pct to utilization via `util = min(demand_pct, 100)` approximation)

### Fig 7b: Frag/Total vs Utilization

- X-axis: average steady-state utilization (%)
- Y-axis: average frag/total (%)
- Same 4-line format

### Fig 9a: Unallocated GPU vs Utilization

- X-axis: average utilization (%)
- Y-axis: unallocated GPU (%)
- Same 4-line format, x-range focused on 70-100%

### Fig 9b: Occupied Nodes vs Utilization

- X-axis: average utilization (%)
- Y-axis: occupied nodes (count, max ~1213)
- Same 4-line format

### Combined Grid

2x2 grid (`comparison_all.png`) with all 4 figures, solid=ours, dashed=paper reference.

## Implementation Steps

### Step 1: Throughput Data

Create `simulation_throughputs_alibaba_split.json` from `simulation_throughputs_alibaba.json` by duplicating entries for each sub-type.

### Step 2: Per-Round Metrics in Scheduler

Modify `scheduler.py`:
- Import `FragmentationCalculator` and `Workload` from `src/fgd/`
- Add `_record_round_metrics()` method
- Call it at the end of each scheduling round in `simulate()`
- Store `_round_metrics_history` list

### Step 3: Experiment Runner Updates

Modify `run_fgd_experiments.py`:
- Support the new throughputs file
- Extract measurement-window metrics from `_round_metrics_history`
- Compute mean/std for each metric within the window
- Output expanded result JSON

### Step 4: Experiment Config

Create `experiments/combined/configs/phase_fgd_replication.json` with all 180 experiments.

### Step 5: SLURM Job

Create SLURM array job script for FarmShare. Budget ~30min per experiment at Alibaba scale (based on 0.64s/round profiling).
- 180 experiments as array job
- 16GB memory, 4hr wall time per experiment
- Sequential within each array task

### Step 6: Plotting Script

Create `experiments/combined/scripts/plot_fgd_replication.py`:
- Load all 180 result JSONs
- Group by placement strategy
- Plot 4 figures with error bands and paper reference overlay
- Generate 2x2 comparison grid

### Step 7: Run & Iterate

1. Run a calibration batch (1 seed, 4 placements, 5 rates = 20 experiments) to validate metrics and calibrate the rate-to-utilization mapping.
2. If needed, adjust rate range.
3. Run the full 180-experiment sweep.
4. Generate plots, compare to paper reference.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Alibaba-scale experiments OOM at 16GB | Monitor first batch; bump to 32GB if needed |
| Rate-to-utilization mapping is off | Calibration run first with 5 rates |
| 12 GPU types slow down LP solver | Warm-start caching already implemented; ECOS handles well |
| FGD placement slow at scale (~1700 jobs) | Known from profiling; 0.64s/round is acceptable |
| Strided placement doesn't work with 12 types | Test locally on small cluster first |
| Fragmentation calculator import adds overhead | Calculator is lightweight; called once per round |

## Validation Criteria

**Policy ordering** should match the FGD paper:
- FGD < BestFit < Random (lower fragmentation is better)
- Strided is our addition -- expect it between Random and BestFit

**Quantitative targets** (approximate, steady-state vs inflation will differ):
- At ~80% utilization: FGD frag/total should be ~30% lower than BestFit and ~50% lower than Random
- FGD should use fewer occupied nodes than Random at equivalent utilization
- Unallocated GPU % should be lowest for FGD at high utilization
