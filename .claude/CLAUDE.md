# Gavel Scheduler

## Goal

Replicate Gavel (OSDI 2020) experiments, then extend with FGD's fragmentation-aware scheduling (ATC 2023). End goal: a policy combining heterogeneity awareness + fragmentation awareness.

## Do Not Modify

- **Core policies** - Existing algorithms in `policies/*.py` are reference implementations
- **Throughput data** - `simulation_throughputs.json` is ground truth from the paper
- **Methodology** - Measurement windows, cluster specs (36:36:36), metrics (JCT, utilization)

## Key Files

| Path | Purpose |
|------|---------|
| `src/scheduler/scheduler.py` | Simulation loop - CAN modify for infrastructure (convergence detection, etc.) |
| `src/scheduler/policies/*.py` | Scheduling policies - READ to understand, create NEW files for new policies |
| `src/scheduler/utils.py` | Policy registry - ADD new policies here |
| `src/scheduler/simulation_throughputs.json` | DO NOT MODIFY |
| `experiments/gavel-replication/configs/` | Gavel replication experiment configs |
| `experiments/gavel-replication/scripts/run_benchmark.py` | Gavel replication experiment runner |
| `src/fgd/` | FGD core algorithm (fragmentation-aware placement) |
| `experiments/combined/run_fgd_experiments.py` | FGD+Gavel integrated experiment runner |
| `experiments/fgd-standalone/` | Standalone FGD evaluation |

## Existing Documentation

- `docs/paper-to-code-mapping.md` - How paper concepts map to code
- `docs/plans/2026-01-26-figure-replication-experiment-design.md` - Experiment design

## Development Process

1. **Test locally first** - Small cluster (4:4:4), few jobs
2. **Sync to FarmShare** - Run full experiments on cluster
3. **Analyze results** - Check `benchmark_results/` and SLURM logs

## FarmShare

SSH multiplexing: User keeps `ssh farmshare` running in separate terminal. Claude reuses via control socket.

```bash
# Sync code
rsync -avz src/scheduler/ farmshare:~/gavel/src/scheduler/
rsync -avz experiments/ farmshare:~/gavel/experiments/

# Run experiment
ssh farmshare "cd ~/gavel/experiments/gavel-replication && python3 scripts/run_benchmark.py --index 0 --experiments-file configs/experiments_full.json"

# Submit batch
ssh farmshare "cd ~/gavel/experiments/gavel-replication && sbatch slurm/submit_full.sbatch"
```

## Lessons Learned (Do Not Retry)

**Caching cvxpy problem structure** - Attempted to cache LP problem and reuse with Parameters. Result: Poor cache hit rate (31-41%), only 3-10% speedup. Not worth the complexity.

**Alternative solvers** - Tested Direct ECOS (+12% slower), Gurobi (+2% slower), Greedy heuristic (+45% slower). All performed worse than baseline cvxpy+ECOS.

**Root cause (Philly scale only)** - At Philly scale (108 GPUs, 50 jobs), LP solver is only 8% of runtime. But at Alibaba scale (6200 GPUs, ~1700 active jobs), LP is **48% of runtime** (3.0s/solve). The bottleneck shifts with scale. See `docs/2026-02-10-alibaba-profile-results.md`.

**What works** - Saturation detection via completion rate in `scheduler.simulate()`:
- `utilization_threshold=0.99` - Only check when cluster utilization > 99%
- `completion_rate_threshold=0.1` - Exit if completion rate < 0.1 jobs/hr
- `min_simulated_time=36000` - Wait 10h simulated time before checking (warm-up)
- `min_runtime=300` - Wait 5 min wall-clock time before checking
- At 7 jobs/hr: triggers early, saving hours of compute time
- Normal runs (0.5-4 jobs/hr) complete fully since utilization < 99%
- Properties: `sched.saturated` (bool), `sched.partial_jct` (JCT at exit)

## Testing

### Unit Tests
```bash
cd src/scheduler/tests
python -m unittest policies_tests -v
```

### Integration Test (Run Before Committing)
```bash
cd src/scheduler/tests
python -m unittest integration_test -v
```

Verifies deterministic output for fixed config (36:36:36 cluster, 50 jobs, seed=0):
- Agnostic JCT = 73063.45s
- Gavel JCT = 57171.41s
- **Any deviation = regression**

## Performance Expectations

| Load | Runtime |
|------|---------|
| 1-2 jobs/hr | ~10-15 min |
| 4 jobs/hr | ~10-20 min |
| 7+ jobs/hr | May exceed time limits |
