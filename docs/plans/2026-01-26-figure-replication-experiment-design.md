# Figure 9, 10, 11 Replication Experiment Design

CS244C Project - Gavel Validation
Date: 2026-01-26

## Overview

Replicate Figures 9, 10, and 11 from the Gavel OSDI'20 paper comparing Gavel (LAS/FTF policies) against FIFO baseline.

## Experiment Matrix

### Policies

| Figure | Gavel Policy | Baseline |
|--------|--------------|----------|
| Fig 9 | `max_min_fairness` (LAS) | `fifo` |
| Fig 10 | `max_min_fairness` (LAS) | `fifo` |
| Fig 11 | `finish_time_fairness` (FTF) | `fifo` |

### Cluster Configuration

- **V100:** 36 GPUs
- **P100:** 36 GPUs
- **K80:** 36 GPUs
- **Total:** 108 GPUs (matches paper Section 7.1)

### Data Points

| Figure | Trace Type | Range | Points/Unit | Total Points | Spacing |
|--------|-----------|-------|-------------|--------------|---------|
| Fig 9 | single-GPU | 0-8 jobs/hr | 5 per 2 units | 20 | 0.4 jobs/hr |
| Fig 10 | multi-GPU | 0-3 jobs/hr | 5 per 1 unit | 15 | 0.2 jobs/hr |
| Fig 11 | multi-GPU | 0-3.5 jobs/hr | 5 per 1 unit | 17 | 0.2 jobs/hr |

### Seeds and Repetitions

- **Seeds per data point:** 3 (paper's minimum for error bars)
- **Measurement window:** Jobs 4000-5000 (steady-state)

### Total Experiment Runs (Full Sweep)

| Figure | Policies | Rates | Seeds | Total |
|--------|----------|-------|-------|-------|
| Fig 9 | 2 | 20 | 3 | 120 |
| Fig 10 | 2 | 15 | 3 | 90 |
| Fig 11 | 2 | 17 | 3 | 102 |
| **Total** | | | | **312** |

## Two-Phase Execution Plan

### Phase 1: Pilot Run (Timing & Validation)

**Goal:** Understand runtime, validate setup, identify optimization opportunities.

**Pilot data points (3 per figure, 1 seed):**

| Figure | Low | Mid | High |
|--------|-----|-----|------|
| Fig 9 | 1.0 | 4.0 | 7.0 jobs/hr |
| Fig 10 | 0.5 | 1.5 | 2.5 jobs/hr |
| Fig 11 | 0.5 | 1.75 | 3.0 jobs/hr |

**Pilot run count:** 18 experiments (2 policies x 3 rates x 3 figures x 1 seed)

**What we learn:**
1. Runtime per simulation at each load level
2. Whether results match paper trends (Gavel JCT < FIFO JCT)
3. Memory usage and potential errors
4. Parallelization opportunities

### Phase 2: Full Sweep

Run complete 312 experiments after pilot validates setup.

## Profiling Strategy

### Method: cProfile (Command-Line)

Zero code changes required. Wrap Python script with profiler:

```bash
python -m cProfile -s cumtime \
  scripts/sweeps/run_sweep_continuous.py [args] \
  2>&1 | tee profile_output.txt
```

**Overhead:** 10-30% slowdown (acceptable for understanding bottlenecks)

**Key output columns:**
- `ncalls` - Function call count
- `tottime` - Time in function (excluding subcalls)
- `cumtime` - Time in function + subcalls (primary metric)

### Expected Bottlenecks

| Component | File | Expected Cost |
|-----------|------|---------------|
| cvxpy solver | `max_min_fairness.py:83` | HIGHEST |
| Allocation computation | `scheduler.py:2099` | HIGH |
| Job scheduling | `scheduler.py:867` | Medium |
| Job generation | `utils.py` | Low |

## FarmShare Workflow

### Prerequisites

SSH ControlMaster configured in `~/.ssh/config`:
```
Host farmshare
    HostName login.farmshare.stanford.edu
    User vramesh3
    ControlMaster auto
    ControlPath ~/.ssh/controlsocket/%C
    ControlPersist 4h
```

### Existing Infrastructure

| File | Purpose |
|------|---------|
| `cluster/setup_farmshare.sh` | One-time environment setup |
| `cluster/submit_farmshare.sbatch` | SLURM array job submission |
| `cluster/run_single_experiment.py` | Runs one experiment by index |
| `cluster/generate_experiments.py` | Creates experiments.json |
| `cluster/aggregate_results.py` | Collects results into CSV |

### Execution Steps

**Step 1: Establish master SSH connection**
```bash
# Run in terminal, authenticate with Duo, keep open
ssh farmshare
```

**Step 2: Sync code changes to FarmShare**
```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  /Users/varunr/projects/courses/stanford/cs244c/gavel/ \
  farmshare:~/gavel/
```

**Step 3: Generate pilot experiments**
```bash
ssh farmshare "cd ~/gavel/cluster && python3 generate_experiments.py --pilot"
```

**Step 4: Submit pilot with profiling**
```bash
ssh farmshare "cd ~/gavel/cluster && sbatch submit_pilot.sbatch"
```

**Step 5: Monitor progress**
```bash
ssh farmshare "squeue -u \$USER"
```

**Step 6: Download results**
```bash
rsync -avz farmshare:~/gavel/cluster/results/ ./results/
rsync -avz farmshare:~/gavel/cluster/slurm_logs/ ./slurm_logs/
```

### SLURM Job Script (Pilot with Profiling)

```bash
#!/bin/bash
#SBATCH --job-name=gavel-pilot
#SBATCH --output=slurm_logs/pilot-%A_%a.out
#SBATCH --error=slurm_logs/pilot-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-17

export PATH="$HOME/.local/bin:$PATH"
cd ~/gavel/cluster

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"

# Run with profiling
python3 -m cProfile -s cumtime \
  run_single_experiment.py \
  --index $SLURM_ARRAY_TASK_ID \
  --experiments-file experiments_pilot.json \
  --output-dir results_pilot \
  --scheduler-dir ../src/scheduler \
  2>&1 | tee results_pilot/profile_$SLURM_ARRAY_TASK_ID.txt

echo "Experiment $SLURM_ARRAY_TASK_ID completed at $(date)"
```

## Pilot Analysis Checklist

### Runtime Analysis

- [ ] Record wall-clock time for each of 18 runs
- [ ] Identify if high load takes longer than low load
- [ ] Calculate estimated total time for 312 full runs

### Result Validation

| Check | Expected | Red Flag |
|-------|----------|----------|
| Gavel JCT < FIFO JCT | Yes, at all load levels | FIFO wins or tie |
| JCT increases with load | Monotonic increase | JCT decreases at high load |
| High load shows bigger gap | 2-3x difference | <1.5x difference |
| Simulations complete | All 18 finish | Timeouts or crashes |

### Profiling Analysis

- [ ] Identify top 5 functions by cumulative time
- [ ] Calculate % time in cvxpy solver vs other code
- [ ] Identify optimization opportunities

### Output Files

```
results_pilot/
  fig9_fifo/...
  fig9_las/...
  fig10_fifo/...
  fig10_las/...
  fig11_fifo/...
  fig11_ftf/...
  profile_*.txt
slurm_logs/
  pilot-*.out
  pilot-*.err
```

## Iteration Workflow

1. **Run pilot on FarmShare** - Submit SLURM jobs
2. **Download results locally** - rsync results and logs
3. **Analyze profiling** - Identify bottlenecks
4. **Make code adjustments** - Optimize if needed
5. **Re-sync to FarmShare** - Upload changes
6. **Re-run** - Validate improvements

## Success Criteria

### Pilot Phase
- [ ] All 18 experiments complete without errors
- [ ] Gavel shows lower JCT than FIFO at all tested rates
- [ ] Profiling data collected for all runs

### Full Sweep Phase
- [ ] All 312 experiments complete
- [ ] Results show 2-3.5x improvement matching paper claims
- [ ] Error bars (std dev across seeds) are reasonable

## Notes

- Random seed controls workload generation (job arrivals, types, durations), not the scheduler algorithm
- Paper uses 3 seeds minimum for error bars
- Measurement window (jobs 4000-5000) ensures steady-state behavior
- 6-minute scheduling rounds match paper Section 5
