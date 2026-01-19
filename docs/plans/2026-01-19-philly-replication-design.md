# Philly Trace Replication Plan

CS244C Project - Gavel Validation Lead

Date: 2026-01-19

## Overview

This document outlines the plan to replicate Gavel's results on Philly traces and reproduce the OSDI 2020 paper figures. This work supports Milestone M1 (February 5): Baselines reproduced within 10% of original results.

## Execution Phases

### Phase 1: Trace Provenance Analysis

**Goal:** Understand the relationship between Gavel's included traces and the official Microsoft Philly traces.

**Tasks:**
1. Download official Microsoft Philly traces from [msr-fiddle/philly-traces](https://github.com/msr-fiddle/philly-traces)
2. Compare schema/format with Gavel's `traces/msr/` files
3. Identify preprocessing, sampling, or transformations applied
4. Document findings

**Deliverable:** Trace provenance report documenting any differences.

### Phase 2: Philly Trace Validation

**Goal:** Run Gavel on included Philly traces to establish baseline behavior.

**Tasks:**
1. Run `simulate_scheduler_with_trace.py` on `traces/msr/seed=0/philly.trace`
2. Test with multiple policies: FIFO, LAS variants, Gavel
3. Record metrics: average JCT, utilization, makespan
4. Compare across seeds (0-4)

**Command template:**
```bash
python scripts/drivers/simulate_scheduler_with_trace.py \
  -t traces/msr/seed=0/philly.trace \
  -p <policy> \
  -c 36:36:36 \
  --throughputs_file simulation_throughputs.json
```

**Deliverable:** Results table comparing policies on Philly traces.

### Phase 3: Paper Figure Reproduction

**Goal:** Reproduce Figures 8-13 from the OSDI 2020 paper.

| Figure | Experiment | Command Reference |
|--------|------------|-------------------|
| Figure 8 | LAS policy, Continuous-Single | `run_sweep_continuous.py` with single-GPU jobs |
| Figure 9 | LAS policy, Continuous-Multiple | `run_sweep_continuous.py --generate-multi-gpu-jobs` |
| Figure 10 | Finish-Time Fairness | `run_sweep_continuous.py` with FTF policies |
| Figure 11 | Multi-Level Fairness | `notebooks/hierarchical.ipynb` |
| Figure 12 | Policy runtime scaling | `sweep_policy_runtimes.py` |
| Figure 13 | Scheduling mechanism efficacy | `run_sweep_continuous.py --ideal` |

**Commands from EXPERIMENTS.md:**

Figure 8:
```bash
python -u scripts/sweeps/run_sweep_continuous.py \
  -s 4000 -e 5000 -l logs/fig8 -j 24 \
  -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed \
  --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 6.0 -n 16
```

Figure 9:
```bash
python -u scripts/sweeps/run_sweep_continuous.py \
  -s 4000 -e 5000 -l logs/fig9 -j 24 \
  -p gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed \
  --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

Figure 10:
```bash
python -u scripts/sweeps/run_sweep_continuous.py \
  -s 4000 -e 5000 -l logs/fig10 -j 24 \
  -p finish_time_fairness finish_time_fairness_perf \
  --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 3.0 -n 11 --generate-multi-gpu-jobs
```

Figure 11:
```
Run notebook: scheduler/notebooks/figures/evaluation/hierarchical.ipynb
```

Figure 12:
```bash
python scripts/microbenchmarks/sweep_policy_runtimes.py \
  -n 32 64 128 256 512 1024 2048 \
  -p max_min_fairness_perf max_min_fairness_packed max_min_fairness_water_filling max_min_fairness_water_filling_packed \
  --num_trials 3
```

Figure 13:
```bash
python -u scripts/sweeps/run_sweep_continuous.py \
  -s 4000 -e 5000 -l logs/fig13 -j 24 \
  -p allox gandiva max_min_fairness max_min_fairness_perf max_min_fairness_packed \
  --seeds 0 1 2 -c 36:36:36 -a 0.0 -b 6.0 -n 16 --ideal
```

**Deliverable:** Comparison table showing our results vs published results, with deviation percentages.

## Success Criteria

Milestone M1 (February 5):
- [ ] Trace provenance documented
- [ ] Philly trace experiments run successfully
- [ ] Figures 8-13 results within 10% of published numbers
- [ ] Deviations documented with hypotheses

## Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| Week 1 | Jan 16-22 | Setup (DONE), trace provenance analysis |
| Week 2 | Jan 23-29 | Phase 2 (Philly traces), begin Phase 3 |
| Week 3 | Jan 30 - Feb 5 | Complete Phase 3, validate results |

## Notes

- Long-running experiments should use `tmux` or run in background
- `max_min_fairness_packed` policy is slow - can be omitted for quick iteration
- Use `-j` flag to control parallelism based on available CPU cores
- Notebooks in `scheduler/notebooks/figures/evaluation/` contain plotting code
