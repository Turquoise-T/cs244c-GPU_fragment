# Gavel+FGD on Alibaba Cluster: Experimental Design

**Date:** 2026-02-04
**Goal:** Evaluate Gavel's heterogeneity-aware scheduling combined with FGD's fragmentation-aware placement on a realistic Alibaba-derived cluster with fractional GPU sharing.

## 1. Overview

We combine two systems that solve complementary problems:

- **Gavel** (OSDI'20): Decides *which GPU type* to run a job on, using throughput-aware LP allocation. Does not consider server-level packing.
- **FGD** (ATC'23): Decides *which server* to place a job on, using fragmentation gradient descent. Does not consider GPU type heterogeneity.

Our experiment tests both dimensions simultaneously on a cluster derived from Alibaba's production trace (cluster-trace-gpu-v2023), which has both heterogeneous GPU types and fractional GPU requests.

## 2. Cluster Configuration

Source: Alibaba cluster-trace-gpu-v2023 node inventory.

**Simplification:** Uniform 8 GPUs/node for all types. GPU counts rounded to multiples of 8. A10 dropped (2 GPUs, negligible).

| GPU Type | Total GPUs | Nodes | Throughput Source |
|----------|-----------|-------|-------------------|
| G2       | 4,392     | 549   | 0.80x V100 (approximated; anonymized type, 8-GPU nodes suggest high-end) |
| T4       | 840       | 105   | 0.35x V100 (Dell MLPerf: V100 is 2.2-3.6x faster, median ~2.9x) |
| G3       | 312       | 39    | 0.90x V100 (approximated; anonymized type, 8-GPU nodes) |
| P100     | 264       | 33    | Measured (Gavel's original P100 data) |
| V100M32  | 200       | 25    | Measured (= Gavel's V100 data; same chip, 32GB VRAM) |
| V100M16  | 192       | 24    | Measured (= Gavel's V100 data; same chip, 16GB VRAM) |
| **Total**| **6,200** | **775** | |

```python
cluster_spec = {
    'G2': 4392, 'T4': 840, 'G3': 312,
    'P100': 264, 'V100M32': 200, 'V100M16': 192,
}
num_gpus_per_server = 8  # uniform for all types
```

### Throughput Generation

For GPU types without measured Gavel throughputs (T4, G2, G3), we generate synthetic throughput data by scaling V100's measured throughputs:

```
throughput[new_type][(model, scale_factor)] = alpha * throughput[v100][(model, scale_factor)]
```

This applies to all values in the throughput matrix: solo ('null'), colocated pairs, and unconsolidated entries.

Scaling factors are derived from published benchmarks:
- **T4 (0.35x V100):** Dell MLPerf benchmarks show V100-PCIe is 2.2-3.6x faster than T4 for DL training.
- **G2 (0.80x V100):** Unknown hardware. Conservative estimate for high-end anonymous GPU.
- **G3 (0.90x V100):** Unknown hardware. Slightly higher tier assumption than G2.

V100M16 and V100M32 use Gavel's measured V100 data directly (same chip). P100 uses Gavel's measured P100 data directly.

### Scaling Justification

The single-multiplier approach preserves the model-specific throughput variation pattern from V100 measurements (e.g., LM workloads scale differently than CNN workloads). The absolute values are approximations, but the relative ordering across models is preserved, which is what matters for the heterogeneity-aware LP.

## 3. Workload

Source: Alibaba cluster-trace-gpu-v2023 pod list (7,255 active pods).

### GPU Request Distribution

Derived from the Alibaba trace. Fractional GPU requests (gpu_milli < 1000) are bucketed:

| GPU Request | Probability | scale_factor | gpu_request | Notes |
|-------------|------------|--------------|-------------|-------|
| 0.25        | 0.10       | 1            | 0.25        | Bucketed from gpu_milli 50-370 |
| 0.50        | 0.16       | 1            | 0.50        | Bucketed from gpu_milli 440-810 |
| 1.0         | 0.63       | 1            | None (1.0)  | Full GPU jobs |
| 2.0         | 0.01       | 2            | None        | Multi-GPU distributed |
| 4.0         | 0.02       | 4            | None        | Multi-GPU distributed |
| 8.0         | 0.08       | 8            | None        | Multi-GPU distributed |

Non-GPU jobs (14.5% of Alibaba trace) are excluded since Gavel is a GPU scheduler. Their probability is redistributed proportionally across GPU job types. The 0.125 GPU bucket (0.4%) is merged into 0.25.

### Job Models

Gavel's 26 DL models (ResNet-18/50, Transformer, LM, Recommendation, CycleGAN, A3C) with measured throughputs. Job type is sampled uniformly from models compatible with the job's scale_factor, same as Gavel's existing generator.

### Job Duration

Gavel's Philly distribution (bimodal log-uniform):
- 80% of jobs: duration ~ 10^U(1.5, 3) minutes (30 min to 16.7 hours)
- 20% of jobs: duration ~ 10^U(3, 4) minutes (16.7 hours to 6.9 days)

### Job Arrival

Poisson process with configurable rate lambda (seconds between arrivals). Arrival rates swept as part of the experiment matrix.

## 4. GPU Sharing (Fractional Jobs)

Jobs with gpu_request < 1.0 share physical GPUs via pair-based colocation, extending Gavel's existing JobIdPair space-sharing mechanism.

### Mechanism

1. Fractional GPU jobs have `scale_factor=1` and `gpu_request` in {0.25, 0.5}.
2. Solo throughput is scaled: `throughput = gpu_request * base_throughput[(model, 1)]`
3. Two fractional jobs can be colocated on one GPU if `gpu_request_A + gpu_request_B <= 1.0`
4. Colocated throughput: `[gpu_req_A * base_A, gpu_req_B * base_B]` (no interference; models isolated GPU fractions via MPS/MIG)
5. Invalid pairs (total > 1.0) get throughput [0, 0], preventing the LP from selecting them.

### Changes to scheduler.py

Three touch points:

**a) `_set_initial_throughput()` -- scale solo throughput by gpu_request:**
```python
base = oracle_throughputs[worker_type][(job_type, scale_factor)]['null']
if job.gpu_request is not None and job.gpu_request < 1.0:
    base *= job.gpu_request
self._throughputs[job_id][worker_type] = base
```

**b) `_update_throughputs()` (pair creation) -- enforce gpu_request packing constraint:**
```python
# When creating merged_job_id throughputs for fractional GPU pairs:
gpu_req_a = job.gpu_request or 1.0
gpu_req_b = other_job.gpu_request or 1.0
if gpu_req_a + gpu_req_b > 1.0:
    self._throughputs[merged_job_id][worker_type] = [0.0, 0.0]
else:
    # Each job keeps its (already scaled) solo throughput
    solo_a = self._throughputs[job_id][worker_type]
    solo_b = self._throughputs[other_job_id][worker_type]
    self._throughputs[merged_job_id][worker_type] = [solo_a, solo_b]  # sorted by job_id
```

**c) Worker assignment -- unchanged.** Pairs with scale_factor=1 already share 1 worker ID.

### Limitations

- **Max 2 jobs per GPU.** Four 0.25-GPU jobs cannot all share one GPU. A pair of 0.25 jobs uses 0.5 GPU, wasting the other 0.5. This affects ~10% of jobs (the 0.25 GPU bucket). Acceptable for now; can extend to N-way sharing later by replacing JobIdPair with a general job group structure.
- **No interference model.** Colocated fractional jobs are assumed to not interfere with each other. This matches hardware-isolated sharing (MPS/MIG) but not software-shared GPUs.

## 5. Comparison Groups

All experiments use `max_min_fairness` (baseline) or `max_min_fairness_perf` (Gavel heterogeneity-aware).

| Config Name       | Allocation Policy        | FGD Enabled | Placement | Tests |
|-------------------|--------------------------|-------------|-----------|-------|
| baseline_strided  | max_min_fairness         | No          | strided   | Non-heterogeneity-aware baseline |
| gavel_strided     | max_min_fairness_perf    | No          | strided   | Gavel only (heterogeneity-aware, no fragmentation-aware) |
| gavel_random      | max_min_fairness_perf    | Yes         | random    | Gavel + random placement (control) |
| gavel_bestfit     | max_min_fairness_perf    | Yes         | bestfit   | Gavel + bin-packing heuristic |
| gavel_fgd         | max_min_fairness_perf    | Yes         | fgd       | Gavel + FGD (our contribution) |

## 6. Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| Average JCT | Gavel | Mean job completion time over measurement window (lower is better) |
| Fragmentation rate | FGD | Fraction of GPUs that are partially allocated but cannot fit any pending job (lower is better) |
| Unallocated GPUs | FGD | Fraction of GPU capacity sitting idle due to fragmentation (lower is better) |
| GPU utilization | Gavel | Fraction of total GPU-seconds used by jobs (higher is better) |
| Saturated | Gavel | Whether JCT threshold (100h) was exceeded (system overloaded) |

## 7. Methodology

- **Simulation:** Gavel's event-driven simulator with Poisson arrivals
- **Steady-state window:** Jobs 4000-5000 (first 4000 jobs are warm-up)
- **`simulate_steady_state=True`:** Jobs keep arriving after window starts
- **`max_jct=360000`:** 100-hour threshold; if exceeded, report JCT = infinity (saturated)
- **Seeds:** 3 per configuration (0, 1, 2) for error bars
- **Arrival rates:** TBD -- need to calibrate for the 6,200-GPU cluster. Likely 10-100 jobs/hr range given the cluster is ~57x larger than Gavel's original 108-GPU setup.

## 8. Arrival Rate Calibration

Gavel's original experiments used 0.5-7.0 jobs/hr on a 108-GPU cluster. The Alibaba cluster has ~6,200 GPUs (57x larger). To achieve similar per-GPU load:

- Gavel's 1.0 jph on 108 GPUs -> ~57 jph on 6,200 GPUs
- Gavel's 4.0 jph on 108 GPUs -> ~228 jph on 6,200 GPUs

However, simulation speed scales with job count and scheduling rounds, so very high arrival rates may be slow. We should:
1. Run a calibration experiment at a few rates to find the saturation point
2. Pick 3 rates: low (well below saturation), medium (moderate load), high (near saturation)

Alternatively, scale the cluster down (e.g., 1/10th: 620 GPUs) to keep arrival rates and simulation times manageable while preserving the heterogeneity and fragmentation dynamics.

## 9. Files to Create/Modify

| File | Change |
|------|--------|
| `scripts/generate_alibaba_throughputs.py` | **New.** Generate 6-type throughput matrix from V100/P100 data with scaling factors. |
| `src/scheduler/simulation_throughputs_alibaba.json` | **New.** Output of the generation script. |
| `src/scheduler/scheduler.py` | Fractional GPU throughput scaling in `_set_initial_throughput()` and pair constraint in `_update_throughputs()`. |
| `src/scheduler/utils.py` | New `_generate_scale_factor_alibaba()` and `_generate_gpu_request_alibaba()` functions for the Alibaba workload distribution. |
| `src/scheduler/fgd_placement.py` | Update workload distribution to match Alibaba (currently hardcoded to Philly). |
| `experiments/fgd/run_fgd_experiments.py` | Support configurable throughputs file path and Alibaba job generation mode. |
| `experiments/fgd/configs/phase_e_alibaba.json` | **New.** Experiment config for Alibaba cluster. |

## 10. Implementation Order

1. **Generate throughput matrix** -- Create the 6-type JSON with scaled throughputs.
2. **Fractional GPU support in scheduler.py** -- The core change (throughput scaling + pair constraints).
3. **Alibaba job generator in utils.py** -- New scale factor + gpu_request distribution.
4. **Update FGD workload model** -- Match the placement heuristic's workload to the actual job distribution.
5. **Experiment runner + config** -- Wire up the new throughputs file and cluster spec.
6. **Calibration run** -- Find good arrival rates for the 6,200-GPU cluster.
7. **Full experiment sweep** -- Run all 5 configs x 3 rates x 3 seeds.

## 11. Verification

1. Integration test still passes (no changes to existing policies or throughput data).
2. Single-experiment smoke test: one config at one arrival rate completes without errors.
3. Fractional GPU jobs correctly get scaled throughput (check logs).
4. Pairs correctly enforce gpu_request sum <= 1.0 (check that 0.5+0.5 pairs form but 0.5+1.0 do not).
5. FGD fragmentation metrics are captured and reported.
6. Results across seeds show reasonable variance (not dominated by noise).
