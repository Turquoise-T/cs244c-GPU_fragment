# Saturation Detection for High-Load Experiments

## Problem Statement & Context

**Problem**: When replicating Gavel's Figures 9, 10, 11, high-load experiments (7+ jobs/hr) exceed SLURM's 2-hour time limit. The simulation runs indefinitely because job arrivals outpace completions - the system is saturated.

**Goal**: Detect saturation early and exit gracefully, reporting partial metrics rather than timing out with no results.

**Constraints**:
- Must not trigger on normal runs (0.5-4 jobs/hr) that would complete given enough time
- Must provide realistic JCT estimates, not premature exits
- Must work within the existing scheduler.py infrastructure

**Success criteria**:
- 7 jobs/hr experiment completes in <10 minutes wall-clock time
- 4 jobs/hr experiment completes normally (no early exit)
- JCT estimates are comparable between normal and early-exit runs

## Optimization Approaches Tried (Failed)

Before tackling saturation detection, we first tried to speed up the simulation itself.

### Approach 1: Caching cvxpy Problem Structure

Hypothesis: Reusing the LP problem structure with cvxpy Parameters would avoid rebuilding the optimization problem each round.

Results:
- Cache hit rate: 31-41% (problem structure changes frequently as jobs arrive/complete)
- Speedup: Only 3-10%
- Verdict: Not worth the complexity. Abandoned.

### Approach 2: Alternative Solvers

Tested different solvers for the allocation LP:
- Direct ECOS (bypassing cvxpy): +12% slower
- Gurobi: +2% slower
- Greedy heuristic (no LP): +45% slower

Verdict: Default cvxpy+ECOS is already optimal. Abandoned.

### Root Cause Analysis

Profiled the simulation loop and found:
- LP solver: ~8% of runtime
- Event processing (simulation loop): ~80% of runtime

The bottleneck is not the solver - it's the simulation mechanics. Optimizing the solver cannot meaningfully reduce runtime. We needed a different approach: detect when to stop early.

## Saturation Detection Approaches Tried

### Approach 1: JCT-based Convergence with Coefficient of Variation (CV)

Hypothesis: When JCT stabilizes (low CV across a window of jobs), the system has reached steady state.

Implementation: Track rolling window of JCTs, compute CV, exit when CV < threshold (e.g., 15%).

Results:
- CV stays ~190% for ALL load levels (0.5, 4, and 7 jobs/hr)
- Root cause: Job heterogeneity. Different job types have vastly different durations regardless of load.
- Verdict: CV cannot distinguish saturated from normal runs. Abandoned.

### Approach 2: Completion Rate vs Arrival Rate Ratio

Hypothesis: In a saturated system, completion rate falls far below arrival rate.

Implementation: Exit if `completion_rate / arrival_rate < 0.5`

Results:
- Worked, but triggered too early (at 95% utilization with only 3.58h JCT)
- Problem: Normal runs at 4 jobs/hr have ~95% utilization too

### Approach 3: Adding Utilization Threshold

Key insight from plotting utilization vs jobs completed:
- 0.5 jobs/hr: ~15% utilization
- 4 jobs/hr: ~60-95% utilization
- 7 jobs/hr: ~95-100% utilization

Solution: Only check completion rate when utilization > 99%. This ensures normal runs complete fully.

### Approach 4: Minimum Wall-Clock Runtime

Problem: With 99% utilization threshold, early exit happened at ~27 seconds with JCT of 10.27h - still not realistic enough.

Solution: Add 5-minute minimum runtime before checking. This allows the simulation to progress further and produce more realistic JCT estimates (15.87h vs 16.67h for normal 4 jobs/hr run).

### Approach 5: Absolute Completion Rate Threshold

Simplification: Instead of comparing to arrival rate, use absolute threshold of 0.1 jobs/hr. When completion rate drops below 0.1 jobs/hr, system is clearly saturated regardless of arrival rate.

## Final Solution

The saturation detection is implemented in `scheduler.simulate()` with these parameters:

```python
completion_rate_threshold=0.1   # Exit if completion rate < 0.1 jobs/hr
utilization_threshold=0.99      # Only check when utilization > 99%
min_simulated_time=36000        # Wait 10h simulated time (warm-up)
min_runtime=300                 # Wait 5 min wall-clock time
```

### How it works

1. Every scheduling round, calculate current cluster utilization
2. If utilization >= 99% AND runtime >= 5 min AND simulated time >= 10h AND jobs completed >= 50:
   - Calculate completion rate = jobs_completed / simulated_hours
   - If completion_rate < 0.1 jobs/hr, trigger early exit
3. Set `sched.saturated = True` and `sched.partial_jct` for reporting

### Results

| Load | Runtime | Saturated | Utilization | JCT |
|------|---------|-----------|-------------|-----|
| 4 jobs/hr | 52s | No | 94.86% | 16.67h |
| 7 jobs/hr | 6.6 min | Yes | 99.58% | 15.87h |

### Properties exposed

- `sched.saturated` - Boolean, True if early exit triggered
- `sched.partial_jct` - JCT at time of early exit

## What to Keep and Discard

### Keep (commit to repo)

1. `src/scheduler/scheduler.py` changes:
   - `_saturated` and `_partial_jct` instance variables
   - `saturated` and `partial_jct` properties
   - `_get_current_utilization()` helper method
   - Saturation detection logic in `simulate()`
   - METRICS logging format

2. `.claude/CLAUDE.md` updates:
   - Lessons learned section (keep concise)

3. `cluster/run_benchmark.py` - Experiment runner with saturation parameters

### Discard (do not commit)

1. `cluster/log_viewer.html` - Useful for exploration, not needed long-term
2. `cluster/parse_simulation_log.py` - Supporting tool for log viewer
3. `cluster/plot_*.py` - One-off plotting scripts
4. `cluster/*.png` - Generated plots
5. `cluster/*_data.json` - Intermediate data files
6. `cluster/log_data_*.json` - Parsed log data

**Rationale for discarding:** These were exploration tools. The insights are captured in this design doc. If needed again, they can be recreated.
