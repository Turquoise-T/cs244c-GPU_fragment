# Gavel Replication via Combined Runner

Date: 2026-02-16

## Goal

Replicate Gavel (OSDI'20) Figures 9, 10, and 11 on Philly traces using the combined `run_fgd_experiments.py` runner. The experimental setup matches the paper exactly; only the runner code differs.

## Motivation

The original replication used `gavel-replication/scripts/run_benchmark.py` (504 experiments, 312 in `experiments_full.json`). Re-running with the combined runner:
- Validates that the combined codebase (with FGD infrastructure) produces identical Gavel results when FGD is disabled
- Establishes a single runner for all future experiments (Gavel baseline, FGD, Gavel+FGD)
- Captures per-round logs for the gpu-scheduling-viz tool

## Experimental Setup Comparison

| | Gavel (OSDI'20) | FGD (ATC'23) | Gavel+FGD (ours, Alibaba) | **This Experiment** |
|---|---|---|---|---|
| **Cluster** | | | | |
| Total GPUs | 108 | 5,592 | ~6,200 | **108** |
| GPU types | 3 (V100, P100, K80) | 7 (G2, T4, G3, P100, V100M16, V100M32, A10) | 6 (G2, T4, G3, P100, V100M16, V100M32) | **3 (V100, P100, K80)** |
| GPUs/node | 1 (flat, no servers) | Mixed (1, 2, 4, 8) | 8 (uniform) | **1 (flat, no servers)** |
| Nodes | 108 (=GPUs) | ~1,200 | ~775 | **108 (=GPUs)** |
| Heterogeneity dimension | GPU throughput (V100 >> K80) | Node size (8-GPU vs 1-GPU nodes) | GPU throughput (7 tiers) | **GPU throughput (V100 >> K80)** |
| **Workload** | | | | |
| Job source | Philly trace distribution | Alibaba trace (real pod CSVs) | Alibaba distribution | **Philly trace distribution** |
| Fractional GPU jobs | No | Yes (21.5% of pods) | Yes | **No** |
| GPU sharing | Pair-based LP (`_packed` policy) | Real fractional (MPS/MIG) | Pair-based (fractional, no interference) | **Pair-based LP (`_packed` policy)** |
| Scale factor distribution | 70% 1-GPU, 10% 2, 15% 4, 5% 8 | 63% 1-GPU, 0.4% 2, 0.2% 4, 0.6% 8, 21.5% fractional | Same as FGD | **Fig 9: 100% 1-GPU; Fig 10/11: 70% 1, 10% 2, 15% 4, 5% 8** |
| Job models | 26 DL models (measured throughputs) | Abstract (CPU/GPU/mem requests) | 26 DL models (throughputs scaled per GPU type) | **26 DL models (measured throughputs)** |
| Job durations | Philly: bimodal log-uniform | Trace replay | Philly distribution (Gavel's generator) | **Philly: bimodal log-uniform** |
| **Scheduling** | | | | |
| Allocation policy | MaxMinFairness / FinishTimeFairness (LP) | N/A (FGD is placement only) | MaxMinFairness / FinishTimeFairness (LP) | **MaxMinFairness / FinishTimeFairness (LP)** |
| Placement strategy | Strided (greedy, largest-first) | FGD (fragmentation gradient descent) | FGD / bestfit / random / strided (compared) | **Strided (default, FGD disabled)** |
| Heterogeneity-aware | Yes (throughput-aware LP) | No (jobs are type-agnostic) | Yes (throughput-aware LP) | **Yes (throughput-aware LP)** |
| Fragmentation-aware | No | Yes | Yes | **No (FGD disabled)** |
| **Methodology** | | | | |
| Simulation | Event-driven, Poisson arrivals | Event-driven, trace replay | Gavel's simulator, Poisson arrivals | **Gavel's simulator, Poisson arrivals** |
| Round duration | 360s (6 min) | N/A | 600s (10 min) | **360s (6 min)** |
| Steady-state window | Jobs 4000-5000 | N/A (full trace replay) | Jobs 4000-5000 | **Jobs 4000-5000** |
| Metric | Average JCT | Fragmentation rate, unallocated GPUs | Average JCT | **Average JCT** |
| Seeds | 3 per config | 1 (trace deterministic) | 3 per config | **3 per config (0, 1, 2)** |
| Max wall time | Unlimited (paper) | N/A | 4hr (SLURM) | **8hr per experiment** |
| **Runner** | | | | |
| Code | `run_sweep_continuous.py` (original) | N/A | `run_fgd_experiments.py` | **`run_fgd_experiments.py` (combined)** |
| Solver | ECOS (via cvxpy) | N/A | ECOS (via cvxpy) | **ECOS (via cvxpy)** |

## Experiment Matrix

417 experiments total:

| Figure | Policies | Multi-GPU | Rate Points | Seeds | Count |
|--------|----------|-----------|-------------|-------|-------|
| Fig 9 | `max_min_fairness`, `max_min_fairness_perf`, `max_min_fairness_packed` | No | 20 (0.4--8.0 jph, step 0.4) | 0,1,2 | 180 |
| Fig 10 | `max_min_fairness`, `max_min_fairness_perf`, `max_min_fairness_packed` | Yes | 15 (0.2--3.0 jph, step 0.2) | 0,1,2 | 135 |
| Fig 11 | `finish_time_fairness`, `finish_time_fairness_perf` | Yes | 17 (0.2--3.4 jph, step 0.2) | 0,1,2 | 102 |

### Policies per figure

- **Fig 9/10 "baseline"**: `max_min_fairness` (heterogeneity-agnostic LAS)
- **Fig 9/10 "Gavel"**: `max_min_fairness_perf` (heterogeneity-aware LAS)
- **Fig 9/10 "Gavel+packing"**: `max_min_fairness_packed` (space-sharing via pair-based LP)
- **Fig 11 "baseline"**: `finish_time_fairness` (heterogeneity-agnostic FTF)
- **Fig 11 "Gavel"**: `finish_time_fairness_perf` (heterogeneity-aware FTF)

### Rate ranges

- **Fig 9** (single-GPU): 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0
- **Fig 10** (multi-GPU LAS): 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0
- **Fig 11** (multi-GPU FTF): 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4

## Common Parameters

```json
{
  "common": {
    "cluster_spec": {"v100": 36, "p100": 36, "k80": 36},
    "num_gpus_per_server": null,
    "mode": "steady_state",
    "window_start": 4000,
    "window_end": 5000,
    "max_jct": 360000,
    "time_per_iteration": 360,
    "enable_fgd": false,
    "enable_migration_penalty": false,
    "enable_gpu_sharing": false,
    "solver": "ECOS"
  }
}
```

Key parameter choices:
- `num_gpus_per_server: null` -- flat allocation (1 GPU = 1 node), matching the paper
- `time_per_iteration: 360` -- 6-minute rounds, matching the paper (not the combined runner's 600s default)
- `enable_gpu_sharing: false` -- the `_packed` policy handles space-sharing at the LP level, independent of this scheduler flag (which is for FGD fractional GPU allocation)
- All FGD features disabled for clean Gavel replication

### Per-experiment config

Each experiment entry specifies only what differs from common:

```json
{
  "name": "fig9_max_min_fairness_perf_4.0jph_single_s0",
  "policy": "max_min_fairness_perf",
  "lam": 900.0,
  "seed": 0,
  "generate_multi_gpu_jobs": false
}
```

Naming convention: `{figure}_{policy}_{rate}jph_{single|multi}_s{seed}`

Lambda conversion: `lam = 3600 / jobs_per_hr`

## GPU Sharing: Packed vs enable_gpu_sharing

These are two independent mechanisms:

**`max_min_fairness_packed` (Gavel paper's space sharing)**:
- LP-level mechanism. The policy optimizes over job pairs, using measured pair-wise interference throughputs from `simulation_throughputs.json`
- `PolicyWithPacking.flatten()` builds a 3D throughput tensor: `(single_jobs x job_combinations x worker_types)` where job_combinations includes both solo and paired entries
- The LP decides whether each job runs solo or co-located, based on interference profiles

**`enable_gpu_sharing` (FGD fractional GPU extension)**:
- Scheduler-level mechanism. Controls how `_assign_workers_to_job` handles sub-GPU demands (0.25, 0.5 GPU)
- Multiple jobs pack onto one physical GPU based on fractional `gpu_request`
- Built for Alibaba workloads; irrelevant for Philly trace (all whole-GPU jobs)

For this experiment: packed policy is active via policy selection; `enable_gpu_sharing` stays false.

## Execution Plan

### Local validation (first)

Run 3-4 experiments locally to verify the combined runner matches the original replication:

```bash
cd experiments/combined
# Fig 9: Gavel at 4.0 jph, seed 0 (should take ~10-15 min)
python run_fgd_experiments.py --phase gavel_replication --index <idx> --save-logs
# Fig 10: Gavel at 2.0 jph, seed 0
python run_fgd_experiments.py --phase gavel_replication --index <idx> --save-logs
```

Compare JCT values with original replication results (`gavel-replication/results/results_combined.csv`). Should match within <1% (same code, same seeds, same parameters).

### FarmShare (full sweep)

SLURM array job: `--array=0-416`, one experiment per task.

```bash
python run_fgd_experiments.py \
  --phase gavel_replication \
  --index $SLURM_ARRAY_TASK_ID \
  --max-wall-time 28800 \
  --save-logs
```

Note: do NOT use `-q`. The viz tool's log_parser.py needs `EVENT`, `TELEMETRY`, and `ALLOCATION` lines, all emitted at INFO level. The `-q` flag sets logging to WARNING, which would suppress all viz data. At Philly scale (108 GPUs), the DEBUG overhead is negligible.

SLURM parameters:
- Wall time: 9 hours (8hr experiment + 1hr buffer)
- Memory: 16GB
- Output: per-task JSON + log files

### Known issue: low-rate timeouts

At rates below ~1 jph, reaching job 4000 (measurement window start) requires very long simulated time (e.g., 20,000 hours at 0.2 jph). The original replication had this same problem -- most low-rate fig10/fig11 experiments returned `inf`. With 8hr wall time, more experiments will complete than the original 4hr limit, but very low rates may still timeout.

## Deliverables

1. `experiments/combined/configs/phase_gavel_replication.json` -- 417-experiment config
2. `experiments/combined/slurm/submit_gavel_replication.sbatch` -- SLURM submission script
3. `experiments/combined/scripts/plot_gavel_replication.py` -- comparison plots vs paper reference curves
4. Local validation confirming combined runner matches original replication results

## Known Simplifications (Gaps for Follow-Up)

The Gavel paper's Philly simulation uses a flat allocation model that omits several real-world concerns. These are intentional for replication fidelity but represent gaps that a follow-up experiment set should address:

1. **No node structure (flat allocation).** `num_gpus_per_server=None` means every GPU is an independent "node." Multi-GPU jobs (Fig 10/11) get GPUs scattered across the cluster with no locality constraint. Real clusters group GPUs into nodes (4-8 per node), and intra-node communication (NVLink) is far faster than cross-node (network). This simplification means **fragmentation is always zero** in the viz tool.

2. **No placement strategy.** Gavel's `_assign_workers_to_job` uses a simple strided assignment. It does not consider node co-location, bin packing, or fragmentation-aware placement. FGD's contribution is exactly this: gradient-descent placement that minimizes fragmentation.

3. **No communication overhead.** Job throughputs in `simulation_throughputs.json` are measured per (model, GPU_type, scale_factor) but assume ideal placement. A 4-GPU job spread across 4 nodes would run slower than one packed on a single node -- this penalty is not modeled.

4. **No GPU sharing interference.** The `_packed` policy models space-sharing at the LP level using pair-wise interference throughputs, but the scheduler does not model runtime interference from co-location (memory pressure, cache contention).

### Packed policy infeasible at scale (discovered 2026-02-16)

All 105 `max_min_fairness_packed` experiments (60 Fig 9 + 45 Fig 10) failed in Job 1434031. Two issues:

1. **Bug (fixed):** `MaxMinFairnessPolicyWithPacking.get_allocation()` was missing the `gpu_demands` kwarg that the scheduler passes to all `MaxMinFairness*` policies (added in our FGD extensions). Fixed by adding `gpu_demands=None` to the signature in `max_min_fairness.py:529`.

2. **Fundamental scalability issue:** Even after the fix, the packed LP is orders of magnitude slower than the non-packed policies. At 4.0 jph (where non-packed Gavel completes in 47s), the packed policy ran for 600s wall time and only reached 982 simulated hours -- nowhere near the 4.88M simulated hours needed to complete the measurement window. The root cause is `PolicyWithPacking.flatten()`, which builds a 3D throughput tensor over *job pairs* (not individual jobs). With ~108 active jobs, the combinatorial space explodes and each LP solve takes seconds instead of milliseconds.

The original Gavel replication (Job 1425384) also never ran packed experiments. The Gavel paper shows packed results but likely used a more efficient implementation or smaller-scale runs. For this replication, we proceed with the 312 successful experiments (baseline + Gavel, no packed).

**Implication for follow-up:** Before adding packed to the Philly+node-structure experiments, the packed LP needs optimization (e.g., job-type aggregation to reduce pair count, or a heuristic approximation). This is a prerequisite for Phase 2.

### Planned follow-up: Philly with node structure

After completing this replication, run a second experiment set that progressively removes these simplifications on the same Philly workload:

- **Phase 1:** Add node structure (`gpus_per_node=4`) to the 36:36:36 cluster (27 nodes). Same policies, same rates. Measure how fragmentation emerges and impacts JCT.
- **Phase 2:** Add FGD placement (strided vs bestfit vs FGD). Show fragmentation reduction.
- **Phase 3:** Add communication overhead model (cross-node penalty). Show that FGD's co-location advantage compounds.
- **Goal:** Build the systematic case that Gavel (heterogeneity-aware allocation) + FGD (fragmentation-aware placement) is the best combined model for real-world multi-GPU clusters.

## Success Criteria

- JCT values from combined runner match original replication within 1% for same (policy, rate, seed) configurations
- Figures 9/10/11 curves reproduce the paper's trends (Gavel outperforms baseline, hockey-stick saturation)
- Log files are parseable by the gpu-scheduling-viz telemetry tools
