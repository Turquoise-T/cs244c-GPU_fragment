# Algorithm Changes from Original Papers

This document tracks every modification made to the Gavel (OSDI 2020) and FGD
(ATC 2023) algorithms, with rationale and status.

## Baseline: Original Algorithms

**Gavel (OSDI 2020):**
- MaxMinFairness LP solved via cvxpy/ECOS per scheduling round
- Allocation: continuous x[i,j] in [0,1] across all GPU types
- Placement: strided assignment (round-robin across servers)
- No migration penalty: allocation can change freely each round

**FGD (ATC 2023):**
- Fragmentation-aware placement: pick server minimizing fragmentation delta
- Per-task placement: iterate all nodes, compute F(node_after) - F(node_before)
- Cluster-wide fragmentation metric from workload distribution

## Active Changes (In Use)

### 1. Profiling Instrumentation
- **File:** `scheduler.py`
- **What:** `_profile` dict with `time.perf_counter()` wrappers around 6
  sections of `simulate()` main loop + 4 scheduling sub-timers. PROFILE
  JSON line emitted at simulation end.
- **Why:** Needed to identify bottlenecks at Alibaba scale.
- **Impact on results:** None. Instrumentation is lightweight (<0.1% overhead)
  and does not change any scheduling decisions.
- **Status:** Active, used in all experiments.

### 2. OOM Fixes in `_remove_job()`
- **File:** `scheduler.py`
- **What:** Clean up `_total_steps_run`, `_per_job_start_timestamps`,
  `_per_job_latest_timestamps`, `_job_timelines` when jobs are removed.
- **Why:** Without cleanup, these dicts grow unboundedly in steady-state mode,
  causing OOM at Alibaba scale (11K+ jobs over 90 minutes).
- **Impact on results:** Prevents OOM crashes. No change to scheduling logic.
- **Status:** Active, essential for Alibaba-scale runs.

### 3. `max_wall_time` Parameter
- **File:** `scheduler.py`, `run_fgd_experiments.py`
- **What:** Graceful exit before wall-clock timeout. Simulation saves partial
  results rather than being killed by SLURM.
- **Why:** Alibaba experiments can run 90+ minutes. Need bounded profiling runs.
- **Impact on results:** None if wall time is sufficient. Partial results if
  wall time exceeded.
- **Status:** Active.

### 4. FGD Placement Integration
- **File:** `fgd_placement.py`, `scheduler.py` (lines 1057-1075)
- **What:** Replace Gavel's strided placement with FGD's fragmentation-aware
  placement in `_schedule_jobs_on_workers()`. Phase 1 preserves lease
  extensions (existing jobs keep workers). Phase 2 uses FGD for new/preempted
  jobs.
- **Difference from FGD paper:** FGD is called per-worker-type (6 times per
  round) rather than cluster-wide. Node objects are rebuilt from scratch each
  round (no persistent state). Gavel's worker ID topology is translated to
  FGD's Node/Task model.
- **Impact on results:** Changes placement decisions vs strided. Reduces
  fragmentation (measured metric). Does not change allocation decisions.
- **Status:** Active.

### 5. Solver/Policy Configurability
- **Files:** `policy.py`, `max_min_fairness.py`, `utils.py`,
  `run_fgd_experiments.py`
- **What:** `solver` and `solver_kwargs` parameters flow from experiment JSON
  through `get_policy()` to the LP solver call.
- **Why:** Needed to benchmark ECOS vs SCS vs HiGHS.
- **Impact on results:** None when using default ECOS.
- **Status:** Active, used for solver benchmarks.

### 6. Warm-Start Caching
- **File:** `max_min_fairness.py`
- **What:** `_prev_allocation` always saved. Before each LP solve, `x.value`
  is seeded with previous allocation and `warm_start=True` passed to solver.
- **Why:** ECOS may benefit from warm starting at larger problem sizes.
- **Impact on results:** ECOS is deterministic regardless of warm start (it
  ignores the hint). Integration tests confirm identical output.
- **Status:** Active but has no measurable effect with ECOS.

### 7. Logging Quieting
- **File:** `scheduler.py`, `run_fgd_experiments.py`
- **What:** Scheduler accepts `log_level` param. `-q` flag passes
  `logging.WARNING`. Demoted to DEBUG: `[Micro-task scheduled]` (both in
  `_print_schedule_summary()` and simulate loop), `[Micro-task succeeded]`
  in `_done_callback()`, and `schedule_round` EVENT. String formatting for
  these messages is guarded behind `isEnabledFor(DEBUG)` to avoid
  unnecessary work at INFO level.
- **Why:** Reduce log volume and string formatting overhead at Alibaba scale.
  With ~1,700 jobs, these messages fire 1,700x per round. Demoting saves
  ~200ms/round in combined scheduling + event processing.
- **Impact on results:** None. Only affects logging, not scheduling.
  35% speedup at Alibaba scale (0.867 -> 0.644 s/round).
- **Status:** Active. Benchmarked in Job 1428259.

### 8. Allocation-Unchanged FGD Skip
- **File:** `scheduler.py`
- **What:** Before Phase 1 + Phase 2 placement, compare the current round's
  `scheduled_jobs` (output of `_schedule_jobs_on_workers_helper()`) with the
  previous round's. If identical (same jobs, same scale factors, same types),
  skip placement entirely and reuse `_current_worker_assignments`.
  Comparison uses `frozenset` for O(total_jobs) equality check.
- **Why:** At 60 jph with ~58 rounds/minute, most rounds could potentially
  skip FGD placement entirely.
- **Impact on results:** None when allocation is truly unchanged.
- **Status:** Active but **ineffective at 60 jph**. At this arrival rate,
  ~10 jobs arrive and ~10 complete per round, so the scheduled set changes
  every round. The skip never triggered in 1,383 rounds (Job 1428217).
  Would help at lower arrival rates (e.g., 7 jph Philly experiments).

### 9. FGD Node Pre-Filtering
- **File:** `fgd_placement.py`
- **What:** Skip Node object creation for servers where all GPUs are already
  assigned. Uses `all(g == 0.0 for g in gpus)` check before Node
  construction. Reduces the node list that `FGDScheduler.schedule_task()`
  iterates.
- **Why:** At ~57% G2 utilization with tight packing, ~300 of 549 servers
  are fully packed and can be skipped.
- **Impact on results:** None. Full servers can't host any job.
  ~3% reduction in worker_assignment time (Job 1428255).
- **Status:** Active.

## Experimental Changes (Benchmarked, Not Default)


### 8. Migration Penalty (L1 Switching Cost)
- **File:** `max_min_fairness.py`, `policy.py`, `utils.py`, `job.py`
- **What:** L1 switching cost in MaxMinFairness objective:
  `maximize min_i [throughput_i - alpha_i * sum_j |x[i,j] - x_prev[i,j]|]`
  Opt-in via `enable_migration_penalty=True`. Per-job `migration_time`
  computed from model size + scale_factor. DPP problem caching for repeated
  solves at same shape.
- **Difference from Gavel paper:** Gavel has no migration penalty. This is a
  new extension.
- **Impact on results:** 10x slower LP (0.34s -> 3.46s) due to auxiliary
  variables doubling variable count from ~10K to ~20K. Marginal improvement
  in allocation stability. Not worth the cost at Alibaba scale.
- **Status:** Experimental. Disabled by default. Benchmarked in Jobs 1428149,
  1428155, 1428182_0.
- **Recommendation:** Do not enable for Alibaba-scale experiments. The FGD
  placement already provides stability through lease extensions (Phase 1).

### 9. Water-Filling Policy
- **File:** `max_min_fairness_waterfill.py`
- **What:** LP-free allocation via progressive water-filling. Binary search
  on throughput level T with capacity-aware redistribution. O(n^2 * m * log eps)
  where n=6 GPU types, m=~1,700 jobs.
- **Difference from Gavel paper:** Completely different allocation algorithm.
  No LP solver. Does not support migration penalty.
- **Impact on results:** Fast allocation (0.28s) but produces allocations that
  concentrate on fewer GPU types, leading to more active jobs (~3,900) and
  overwhelming FGD placement (89% of runtime).
- **Status:** Experimental. Benchmarked in Job 1428158_1.
- **Recommendation:** Not viable at Alibaba scale due to worker_assignment
  bottleneck.

### 10. Single-Type Assignment Policy
- **File:** `max_min_fairness_single_type.py`
- **What:** Restrict each job to exactly one GPU type. Assignment via LPT
  load-balanced greedy with migration stickiness. Allocation via sort-based
  per-type waterfill (O(m log m), no LP). Migration handled by reducing
  effective throughput when switching types.
- **Difference from Gavel paper:** Gavel allows continuous allocation across
  all GPU types. This restricts to single-type.
- **Impact on results:** Allocation is 845x faster than ECOS+penalty (3.5ms
  vs 3s). But concentrates jobs on fewer types, making FGD placement 88% of
  runtime. Net result: 7.15s/round (worse than ECOS+penalty at 2.04s).
- **Status:** Experimental. Benchmarked in Job 1428182_1.
- **Recommendation:** Not viable at Alibaba scale in current form. The
  single-type concentration exacerbates the FGD bottleneck.

## Changes NOT Made (Considered and Rejected)

### SCS Solver
- **Tested:** Job 1428155_1. SCS eps=1e-3 was 7% slower than ECOS.
- **Reason:** First-order ADMM iterations don't converge faster than ECOS
  interior-point at this problem size.

### HiGHS Simplex
- **Tested:** Job 1428155_2. 860% slower than ECOS.
- **Reason:** More variables (~20K) than constraints (~1,700) is worst case
  for simplex.

### cvxpy Problem Structure Caching (Philly Scale)
- **Tested:** Earlier experiment (documented in `gavel/.claude/CLAUDE.md`).
- **Reason:** 31-41% cache hit rate, only 3-10% speedup. Not worth complexity.

## File Inventory

Files added or modified from the original Gavel/FGD codebases:

| File | Type | Description |
|------|------|-------------|
| `scheduler.py` | Modified | Profiling, OOM fixes, wall time, FGD integration, logging |
| `policies/policy.py` | Modified | solver_kwargs, set_migration_context |
| `policies/max_min_fairness.py` | Modified | Migration penalty, DPP caching, solver_kwargs |
| `policies/max_min_fairness_waterfill.py` | New | Water-filling policy (experimental) |
| `policies/max_min_fairness_single_type.py` | New | Single-type policy (experimental) |
| `utils.py` | Modified | Policy registration, solver_kwargs, estimate_migration_time |
| `job.py` | Modified | (minor) for migration time estimation |
| `fgd_placement.py` | New | FGD-Gavel adapter |
| `run_fgd_experiments.py` | New | Experiment runner with config system |
| `docs/2026-02-10-alibaba-profile-results.md` | New | Profiling results |
| `docs/algorithm-changes-from-papers.md` | New | This document |
