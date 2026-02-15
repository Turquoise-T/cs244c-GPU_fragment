# Alibaba-Scale Simulation Profiling Results

**Date:** 2026-02-10
**Job:** SLURM 1428149 on FarmShare (wheat-01)
**Config:** `phase_e_alibaba_fgd_only`, index 0 (60 jph, seed 0)
**Wall limit:** 20 minutes (`--max-wall-time 1200`)

## Cluster Configuration

| GPU Type | Count |
|----------|-------|
| G2       | 4,392 |
| T4       | 840   |
| G3       | 312   |
| P100     | 264   |
| V100M32  | 200   |
| V100M16  | 192   |
| **Total** | **6,200** |

8 GPUs per server. FGD placement enabled. Migration penalty enabled. Round duration = 600s.

## Run Summary

- **Rounds completed:** 763
- **Simulated time:** 126.8 hours (456,600s)
- **Wall time:** 1,201.9s (20.0 minutes)
- **Jobs generated:** 11,547
- **Jobs active at exit:** 1,709 (all running, 0 queued)
- **Jobs completed (window):** 925 / 1,000
- **Utilization at exit:** 41.1% (only G2 GPUs in use: 2,516 / 4,392)
- **LP solves:** 192 (one every ~4 rounds)
- **Avg round time:** 1.58s

## Time Breakdown

### Top-Level Sections

| Section | Time (s) | % of Total | Description |
|---------|----------|-----------|-------------|
| `scheduling` | 984.0 | 81.9% | Full scheduling pipeline per round |
| `event_jump_and_completion` | 189.8 | 15.8% | Timestamp advance + `_done_callback()` |
| `job_arrivals` | 13.4 | 1.1% | Job generation + `add_job()` |
| `telemetry` | 10.0 | 0.8% | Metric computation + JSON + logging |
| `exit_checks` | 4.6 | 0.4% | Exit condition evaluation |
| **`round_total`** | **1,201.9** | **100%** | |

### Scheduling Sub-Breakdown

| Sub-section | Time (s) | % of Scheduling | % of Total | Description |
|-------------|----------|----------------|-----------|-------------|
| `lp_solve` | 576.3 | 58.6% | 48.0% | `_compute_allocation()` via cvxpy/ECOS |
| `priorities` (non-LP) | 51.7 | 5.3% | 4.3% | Priority fraction computation (priorities total minus lp_solve) |
| `worker_assignment` | 210.3 | 21.4% | 17.5% | Phase 1 lease extensions + Phase 2 FGD placement |
| `helper` | 43.6 | 4.4% | 3.6% | `_schedule_jobs_on_workers_helper()` |
| overhead | 102.1 | 10.4% | 8.5% | Scheduling code outside sub-timed sections |

### LP Solve Statistics

- **Total LP solves:** 192
- **Total LP time:** 576.3s
- **Average per solve:** 3.00s
- **LP fires every:** ~4 rounds (when `_need_to_update_allocation` is True and reset interval elapsed)

## Comparison: Philly vs Alibaba Scale

| Metric | Philly (108 GPUs, 50 jobs) | Alibaba (6,200 GPUs, ~1,700 active) |
|--------|---------------------------|--------------------------------------|
| LP solve time (avg) | 2.7ms | 3,001ms (1,100x slower) |
| LP % of runtime | 9% | 48% |
| scheduling % | 60% | 82% |
| event_jump % | 26% | 16% |
| telemetry % | 13% | 0.8% |
| Rounds | 2,070 | 763 |
| Avg round time | 0.5ms | 1,575ms |

The LP problem scales roughly as O(m * n) where m = active jobs and n = GPU types. At Philly scale, the LP matrix is ~50x3 = 150 variables. At Alibaba scale, it is ~1,700x6 = 10,200 variables -- a 68x increase in problem size that translates to a 1,100x increase in solve time (superlinear due to ECOS interior-point complexity).

## Optimization Targets (Ranked by Impact)

### 1. LP Solve -- 48% of runtime

The MaxMinFairness LP is an interior-point problem solved by ECOS through cvxpy. At 3s per solve with ~1,700 active jobs across 6 GPU types, this is the single biggest target.

Potential approaches:
- **Reduce solve frequency:** Currently fires every ~4 rounds. Could skip solves when the job set hasn't changed significantly (no arrivals/completions since last solve).
- **Warm-starting:** Already implemented (`warm_start=True` with `x.value` seeded from previous allocation), but ECOS may not benefit much from warm starts since it's an interior-point solver.
- **Problem decomposition:** The LP could potentially be decomposed per-GPU-type if the coupling constraints allow it.
- **Alternative solver:** SCS (first-order method) may be faster for this problem size at the cost of some precision. Previous tests at Philly scale showed ECOS was fastest, but that was at 150 variables -- at 10,200 variables the landscape may differ.
- **Reduce problem size:** Jobs with zero throughput on all GPU types could be excluded from the LP entirely.

### 2. Worker Assignment -- 17.5% of runtime

Phase 1 (lease extensions) iterates all scheduled jobs checking previous assignments. Phase 2 (FGD placement) calls `assign_workers_for_round()` which assigns GPUs to jobs across ~775 servers.

Potential approaches:
- **Skip unchanged assignments:** If allocation hasn't changed, most jobs keep the same placement. Could short-circuit Phase 1 when no LP solve occurred this round.
- **Batch FGD placement:** Currently called per worker_type. Could batch across types.

### 3. Event/Completion Processing -- 15.8% of runtime

The `_done_callback()` is called for every job completing in a round. With ~1,700 jobs finishing per round (since all jobs run for exactly one 600s round in steady state), this involves updating multiple dicts and sets per job.

Potential approaches:
- **Batch callback processing:** Accumulate completions and process in bulk rather than per-job.
- **Reduce per-callback work:** Profile `_done_callback()` internals to identify the costly operations.

### 4. Scheduling Helper -- 3.6% of runtime

`_schedule_jobs_on_workers_helper()` builds the per-worker-type job lists. Relatively small but could benefit from caching when the job set is stable.

## Projected Impact of LP Optimization

If LP solve time could be reduced by 4x (from 3.0s to 0.75s per solve):
- LP total: 576.3s -> 144.1s (saving 432.2s)
- Round total: 1,201.9s -> 769.7s
- **Speedup: 1.56x**
- Projected full-run time: ~90min / 1.56 = ~58min

If LP solve time could be reduced by 10x (from 3.0s to 0.3s per solve):
- LP total: 576.3s -> 57.6s (saving 518.7s)
- Round total: 1,201.9s -> 683.2s
- **Speedup: 1.76x**

To hit the 20-minute target (4.5x speedup from 90min), LP optimization alone is insufficient. Would need to also address worker_assignment and event processing, or reduce round count through larger time steps.

## Solver Benchmark (Job 1428155)

Tested ECOS vs SCS vs HiGHS on identical 60 jph config with migration penalty enabled.

| Metric | ECOS | SCS (eps=1e-3) | HiGHS (simplex) |
|--------|------|----------------|-----------------|
| LP avg solve | 3.56s | 3.81s (+7%) | 34.15s (+860%) |
| Rounds completed | 635 | 599 | 115 |
| Avg round time | 1.89s | 2.00s | 10.62s |

**Conclusion:** Neither SCS warm-start nor HiGHS simplex helps at this problem size. SCS's first-order iterations don't converge faster than ECOS's interior-point for the DPP-parametrized problem. HiGHS simplex is catastrophically slow (10x worse) -- the problem has far more variables (~20K with penalty aux vars) than constraints (~1,700), which is the worst case for simplex.

## Migration Penalty Cost (Job 1428158)

ECOS with vs without migration penalty:

| Metric | With penalty (1428149) | Without penalty (1428158_0) |
|--------|----------------------|---------------------------|
| LP avg solve | 3.00s | **0.31s** (10x faster) |
| LP count | 192 | 326 |
| LP total | 576.3s | 100.9s |
| Rounds completed | 763 | 1,299 |
| Scheduling % | 82% | 67% |

**Key finding:** The migration penalty's auxiliary variables (`t >= x - x_prev, t >= -(x - x_prev)`) double the variable count from ~10K to ~20K and change the problem structure, making ECOS 10x slower per solve. Without penalty, ECOS is already fast (0.31s). The penalty is the primary LP cost driver.

## Water-Filling Algorithm (Job 1428158_1)

Custom progressive water-filling (no LP solver, pure numpy) vs ECOS without penalty:

| Metric | ECOS (no penalty) | Waterfill |
|--------|-------------------|-----------|
| Allocation time (avg) | 0.31s | 0.28s |
| Allocation count | 326 | 37 |
| Allocation total | 100.9s | 10.5s |
| Worker assignment | 330.2s (27%) | **1,073.9s (89%)** |
| Rounds completed | 1,299 | 146 |
| Avg round time | 0.92s | 8.24s |
| Fragmentation | 43.6 | 2.4 |
| Active jobs at exit | ~1,700 | ~3,900 |

**Key finding:** The waterfill allocation itself is fast (0.28s including overhead), but the resulting allocations create a massive worker_assignment bottleneck. With ~3,900 active jobs (vs ~1,700 for ECOS), FGD placement takes 7.4s/round. The very low fragmentation (2.4) suggests the allocations concentrate jobs on fewer GPU types, but the sheer job count overwhelms the placement algorithm. The waterfill allocation is not the bottleneck -- FGD placement is.

## Single-Type Assignment (Job 1428182)

Experimental policy: assign each job to exactly one GPU type, solve allocation
analytically (no LP). Migration stickiness built into assignment heuristic.

| Metric | ECOS+penalty | Single-type | ECOS no-penalty |
|--------|-------------|-------------|-----------------|
| Rounds completed | 588 | 168 | 1,162 |
| Sim time (hrs) | 97.7 | 27.7 | 193.3 |
| Avg round time | 2.04s | 7.15s | 1.03s |
| Window jobs done | 894/1000 | 673/1000 | 962/1000 |
| Active jobs at exit | 1,657 | 1,632 | 1,726 |
| Utilization at exit | 41% | 51% | 41% |
| Avg fragmentation | 43.4 | 6.1 | 43.6 |
| LP avg time | 3.46s | 0.18s | 0.34s |
| LP count | 148 | 43 | 291 |
| **worker_assignment** | 259s (22%) | **1,058s (88%)** | 348s (29%) |

### Profile Breakdown (% of 1200s wall time)

| Section | ECOS+penalty | Single-type | ECOS no-penalty |
|---------|-------------|-------------|-----------------|
| scheduling | 81% | 94% | 68% |
| -- lp_solve | 43% | 0.6% | 8% |
| -- worker_assignment | 22% | 88% | 29% |
| -- helper | 4% | 1% | 7% |
| event/completion | 17% | 6% | 28% |
| telemetry | 0.8% | 0.1% | 2% |

### Key Findings

1. **Single-type allocation is fast (0.18s) but FGD placement dominates.**
   The sort-based waterfill eliminates the LP entirely (3.5ms locally, 0.18s
   in the profiling timer which includes overhead). But FGD placement at
   1,058s (88% of runtime) makes rounds 7.15s each -- worse than ECOS+penalty.

2. **ECOS without penalty is the fastest overall.**
   At 1,162 rounds in 20 min (1.03s/round), ECOS without penalty completes
   the most work. LP solves at 0.34s are fast enough, and the resulting
   multi-type allocations spread jobs across GPU types, reducing per-type
   placement pressure.

3. **The bottleneck shifts with each optimization.**
   - Philly scale: LP = 8%, event = 26%, scheduling = 60%
   - Alibaba + penalty: LP = 48%, worker_assignment = 22%
   - Alibaba - penalty: LP = 8%, worker_assignment = 29%, event = 28%
   - Alibaba single-type: LP = 0.6%, worker_assignment = 88%

4. **Single-type concentrates jobs on fewer GPU types** (frag = 6.1 vs 43.6),
   which increases per-type placement pressure. With ~1,600 jobs on G2 alone,
   the O(jobs * servers) FGD algorithm must search 549 G2 servers per job.

5. **Migration penalty is not worth its cost.** The penalty was designed to
   stabilize allocations, but it makes the LP 10x slower (3.46s vs 0.34s)
   and only improves window completion marginally (894 vs 962 in 20 min).
   Without penalty, the LP fires more often (291 vs 148 solves) but each
   solve is 10x cheaper, and more rounds complete.

### Conclusion

The next optimization target is **FGD placement** (worker_assignment), which
is 22-88% of runtime depending on policy. The allocation algorithm (LP vs
waterfill vs single-type) matters less than placement efficiency. The
recommended baseline is ECOS without migration penalty, since it achieves
the best throughput with acceptable placement cost.

## Summary of All Experiments

| Job | Config | LP avg | Rounds | Avg round | Key bottleneck |
|-----|--------|--------|--------|-----------|---------------|
| 1428149 | ECOS + penalty | 3.00s | 763 | 1.58s | LP (48%) |
| 1428155_0 | ECOS + penalty (rerun) | 3.56s | 635 | 1.89s | LP (43%) |
| 1428155_1 | SCS + penalty | 3.81s | 599 | 2.00s | LP (no help) |
| 1428155_2 | HiGHS + penalty | 34.15s | 115 | 10.62s | LP (10x worse) |
| 1428158_0 | ECOS, no penalty | 0.31s | 1,299 | 0.92s | worker (27%) + event (29%) |
| 1428158_1 | Waterfill, no penalty | 0.28s | 146 | 8.24s | worker (89%) |
| 1428182_0 | ECOS + penalty (rerun) | 3.46s | 588 | 2.04s | LP (43%) |
| 1428182_1 | Single-type + stickiness | 0.18s | 168 | 7.15s | worker (88%) |
| 1428182_2 | ECOS, no penalty (rerun) | 0.34s | 1,162 | 1.03s | worker (29%) |
| 1428217 | ECOS no-pen + FGD skip | 0.30s | 1,383 | 0.87s | event (29%) + worker (26%) |
| 1428255 | + node pre-filter | 0.30s | 1,404 | 0.86s | event (29%) + worker (25%) |
| 1428259 | + logging demotion | 0.30s | 1,863 | 0.64s | scheduling (68%) |

**Notes on Jobs 1428217-1428259:**
- FGD skip (frozenset comparison): never triggered at 60 jph due to ~20 job changes/round
- Node pre-filter: ~3% worker_assignment improvement (skips full servers)
- Logging demotion: **35% speedup** -- demoted `[Micro-task scheduled]` and `[Micro-task succeeded]` from INFO to DEBUG, guarded string formatting behind `isEnabledFor(DEBUG)`. Saves ~200ms/round across scheduling + event_processing for ~1,700 jobs.

## Raw PROFILE Outputs

**Job 1428149 (ECOS + penalty, original):**
```
PROFILE {"round_total": 1201.8915, "telemetry": 10.0402, "exit_checks": 4.6438,
  "event_jump_and_completion": 189.7765, "job_arrivals": 13.4129,
  "scheduling": 984.033, "lp_solve": 576.2574, "priorities": 627.9598,
  "helper": 43.5817, "worker_assignment": 210.2601, "rounds": 763,
  "lp_solve_count": 192, "lp_solve_total": 576.2574, "lp_solve_avg": 3.0013,
  "avg_round_time": 1.5752}
```

**Job 1428182_0 (ECOS + penalty, rerun):**
```
PROFILE {"round_total": 1200.606, "telemetry": 9.7569, "exit_checks": 5.0324,
  "event_jump_and_completion": 201.192, "job_arrivals": 14.6331,
  "scheduling": 970.0068, "lp_solve": 511.9071, "priorities": 564.2716,
  "helper": 43.0627, "worker_assignment": 259.4115, "rounds": 588,
  "lp_solve_count": 148, "lp_solve_total": 511.9071, "lp_solve_avg": 3.4588,
  "avg_round_time": 2.0418}
```

**Job 1428182_1 (Single-type):**
```
PROFILE {"round_total": 1201.6203, "telemetry": 1.7151, "exit_checks": 0.9733,
  "event_jump_and_completion": 71.2659, "job_arrivals": 3.896,
  "scheduling": 1123.7812, "lp_solve": 7.7496, "priorities": 19.8697,
  "helper": 13.8139, "worker_assignment": 1058.2727, "rounds": 168,
  "lp_solve_count": 43, "lp_solve_total": 7.7496, "lp_solve_avg": 0.1802,
  "avg_round_time": 7.1525}
```

**Job 1428182_2 (ECOS no penalty):**
```
PROFILE {"round_total": 1200.4588, "telemetry": 20.9479, "exit_checks": 8.7643,
  "event_jump_and_completion": 336.0297, "job_arrivals": 24.4069,
  "scheduling": 810.3288, "lp_solve": 97.9733, "priorities": 192.0392,
  "helper": 79.1078, "worker_assignment": 347.8032, "rounds": 1162,
  "lp_solve_count": 291, "lp_solve_total": 97.9733, "lp_solve_avg": 0.3367,
  "avg_round_time": 1.0331}
```

**Job 1428217 (ECOS no-penalty + FGD skip):**
```
PROFILE {"round_total": 1200.1572, "telemetry": 23.4426, "exit_checks": 8.9213,
  "event_jump_and_completion": 345.0635, "job_arrivals": 24.9652,
  "scheduling": 797.7788, "lp_solve": 104.7221, "priorities": 202.9383,
  "helper": 85.3825, "worker_assignment": 313.612, "rounds": 1383,
  "lp_solve_count": 347, "lp_solve_total": 104.7221, "lp_solve_avg": 0.3018,
  "avg_round_time": 0.8678}
```

**Job 1428255 (+ node pre-filter):**
```
PROFILE {"round_total": 1200.5074, "telemetry": 23.6709, "exit_checks": 8.9952,
  "event_jump_and_completion": 348.6262, "job_arrivals": 25.0757,
  "scheduling": 794.1587, "lp_solve": 104.7991, "priorities": 203.9359,
  "helper": 87.1842, "worker_assignment": 304.9156, "rounds": 1404,
  "lp_solve_count": 352, "lp_solve_total": 104.7991, "lp_solve_avg": 0.2977,
  "avg_round_time": 0.8551}
```

**Job 1428259 (+ logging demotion):**
```
PROFILE {"round_total": 1200.615, "telemetry": 35.9661, "exit_checks": 12.2095,
  "event_jump_and_completion": 297.6707, "job_arrivals": 33.5619,
  "scheduling": 821.2266, "lp_solve": 139.3668, "priorities": 273.5498,
  "helper": 119.9366, "worker_assignment": 371.7305, "rounds": 1863,
  "lp_solve_count": 467, "lp_solve_total": 139.3668, "lp_solve_avg": 0.2984,
  "avg_round_time": 0.6445}
```

## Methodology

Instrumentation added to `scheduler.py` using `time.perf_counter()` wrappers around each major section of the `simulate()` while loop. Sub-timers added inside `_schedule_jobs_on_workers()` and `_update_priorities()`. All integration tests pass with instrumentation active (deterministic output unchanged).
