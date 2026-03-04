# FGD + Gavel Integration Plan

## Goal

Integrate FGD (Fragmentation Gradient Descent, ATC'23) into Gavel's simulation, then validate through phased experiments: first independently, then combined.

## Phased Approach

### Phase A: Standalone FGD Replication

- [x] Replicate the FGD paper's event-driven simulator using Alibaba traces on Cluster H (1,200 nodes, 6,200+ GPUs)
- [x] Validate FGD implementation against published results

**New files:**
- [x] `fgd_src/simulator.py` -- event-driven simulator (tasks arrive, get placed, stay until done)
- [x] `fgd_src/alibaba_trace_parser.py` -- parse Alibaba cluster trace CSVs from `github.com/alibaba/clusterdata`
- [x] `fgd_src/baselines.py` -- BestFit, FirstFit, Random, DotProd, GpuPacking, GpuClustering placement
- [x] `fgd_src/run_evaluation.py` -- CLI entry point (replaces planned `run_standalone.py`)
- [x] `fgd_src/plot_results.py` -- generates all 6 paper figures with reference curve overlay
- [x] `fgd_src/paper_reference_curves.json` -- digitized reference curves from published paper

**Methodology:** Monte-Carlo workload inflation (repeatedly submit tasks sampled from trace until cluster fills). Key metric: % unallocated GPUs when cumulative GPU requests hit 100% cluster capacity.

**Validation:** Compare fragmentation rate curves against FGD paper Figures 7-9.
- [x] Run full experiments: 6 policies x 3 seeds on Alibaba traces (18 experiments)
- [x] Generate all 6 figures (7a, 7b, 9a, 9b, 9c, 9d) with paper reference overlay
- [x] Fig 7a/7b: qualitative match -- policy ordering correct (Random worst, FGD best)
- [ ] Fig 9a-9d: known issues to investigate/fix
  - [ ] Fig 9b: GpuClustering plateau at 617 nodes (8-GPU tier saturation effect)
  - [ ] Fig 9c: pending GPU bars dramatically smaller than paper
  - [ ] Fig 9d: fragmentation breakdown nearly 100% deficient -- `_compute_frag_breakdown()` needs rewrite to use per-(node, task-type) weighted computation

**Results:** `fgd_src/results/full_run/`, **Figures:** `fgd_src/figures/full_run/`

---

### Phase B: Integrate FGD into Gavel

FGD replaces Gavel's strided placement in `_assign_workers_to_job()` -- it controls *where* jobs land, not *how much* resource they get.

**Integration point:** `scheduler.py` `_schedule_jobs_on_workers()` (line 934). After `_schedule_jobs_on_workers_helper()` decides which jobs run this round, FGD picks which server/GPUs to place them on.

**New files:**
- [x] `src/scheduler/fgd_placement.py` -- adapter bridging FGD Node model to Gavel's topology

**Modified files:**
- [x] `src/scheduler/scheduler.py` -- add `enable_fgd` flag, branch placement logic
- [x] `src/scheduler/job.py` -- add optional `gpu_request` property (for partial GPU support)
- [x] `src/scheduler/utils.py` -- add `gpu_request` parameter to `generate_job()`

**Key design decisions:**

1. **Partial GPU support**: Extend `Job` with a `gpu_request` field that can be fractional (0.25, 0.5, etc.). For LP purposes, `scale_factor = ceil(gpu_request)` (a 0.5-GPU job needs 1 worker slot). Throughput scales linearly: `0.5 * throughput[(model, 1)]`.

2. **Node bridge**: Gavel's `_worker_type_to_worker_id_mapping[type]` is already a list-of-lists (each inner list = one server). Each server becomes an FGD `Node`. GPU capacity = 1.0 if worker ID is free, 0.0 if assigned. CPU/memory set to large defaults (Gavel doesn't model CPU constraints).

3. **Workload M**: For Philly distribution: 70% 1-GPU, 10% 2-GPU, 15% 4-GPU, 5% 8-GPU. For Alibaba traces: derive from trace statistics.

4. **FGD toggle**: `enable_fgd` boolean on `Scheduler.__init__()`. When off, original strided placement runs unchanged.

5. **Lease extensions respected**: Jobs continuing from previous round keep their worker assignments before FGD runs on remaining jobs.

**Data flow per round:**
```
_schedule_jobs_on_workers_helper()  ->  scheduled_jobs (which jobs, which type)
                                            |
                              if enable_fgd:
                                GavelFGDPlacement.assign_workers_for_round()
                                  -> for each job: FGDScheduler.schedule_task()
                                  -> translate node+gpu_indices -> worker_ids
                              else:
                                _assign_workers_to_job() (strided, original)
```

**Status:**
- [x] Code written
- [x] Unit tests passing (`test_fgd_placement.py` -- 7/7 pass)
- [x] Code reviewed and committed (`5c2249a`)

---

### Phase C: Validate Gavel Baseline (FGD off)

Run with `enable_fgd=False` through the new code path. Confirm results match previous experiments exactly.

**Acceptance:** Integration test golden values unchanged:
- Agnostic JCT = 73063.45s
- Gavel JCT = 57171.41s

**Status:**
- [x] Config written: `experiments/fgd/configs/phase_c.json`
- [x] Experiment runner: `experiments/fgd/run_fgd_experiments.py`
- [x] Results: Gavel JCT = 57171.41 (relative error ~2e-08) -- PASS
- [x] Validated again during Phase B commit (integration test in pre-commit hook)

---

### Phase D: Validate FGD Alone (FIFO + FGD)

Run FIFO allocation (no heterogeneity awareness) with FGD placement enabled.

**Config:** `policy=fifo`, `enable_fgd=True`, `num_gpus_per_server=4`, `multi_gpu=True`

**Validation:** Fragmentation metrics should improve over FIFO without FGD. Trend should be consistent with Phase A standalone results (exact match not expected due to round-based vs event-driven differences).

**Status:**
- [x] Config written: `experiments/fgd/configs/phase_d.json`
- [ ] Experiments run
- [ ] Results analyzed

---

### Phase E: Baseline Comparison

Run baselines without FGD for comparison. Add BestFit and FirstFit as alternative placement strategies in `fgd_placement.py` (placement_mode parameter).

Sweep: `{fifo, max_min_fairness_perf}` x `{strided, bestfit, fgd}` x `{0.4, 1.0, 4.0} jobs/hr` x seeds `{0, 1, 2}`

**Status:**
- [x] Config written: `experiments/fgd/configs/phase_e.json`
- [ ] Experiments run
- [ ] Results analyzed

---

### Phase F: Combined (Gavel + FGD)

Enable both Gavel's heterogeneity-aware allocation AND FGD's fragmentation-aware placement.

**Config:** `policy=max_min_fairness_perf`, `enable_fgd=True`, `num_gpus_per_server=4`

Run across the full sweep matrix. Compare JCT and fragmentation against FGD-only (Phase D) and Gavel-only (Phase E).

**Status:**
- [x] Config written: `experiments/fgd/configs/phase_f.json`
- [ ] Experiments run
- [ ] Results analyzed
- [ ] Final comparison plots generated

---

## Metrics to Track

- **Fragmentation rate**: F_N(M) / total unallocated GPUs, per round
- **Average JCT** (job completion time)
- **GPU utilization**: allocated / total
- **Unallocated GPU count** over time
- **Fragmentation breakdown**: deficient vs stranded vs non-GPU causes

---

## File Structure Summary

```
fgd_src/                          (Phase A)
    fgd.py                        (exists)
    simulator.py                  (done)
    alibaba_trace_parser.py       (done)
    baselines.py                  (done)
    run_evaluation.py             (done)
    plot_results.py               (done)
    paper_reference_curves.json   (done)

src/scheduler/                    (Phase B)
    fgd_placement.py              (written, uncommitted)
    scheduler.py                  (modified, uncommitted)
    job.py                        (modified, uncommitted)
    utils.py                      (modified, uncommitted)
    tests/test_fgd_placement.py   (written, uncommitted)

experiments/fgd/                  (Phases C-F)
    configs/phase_{c,d,e,f}.json  (written, uncommitted)
    run_fgd_experiments.py        (written, uncommitted)
    results_phase_c.json          (truncated, needs re-run)
```

## Verification

| Phase | Test | Pass Criteria | Status |
|-------|------|---------------|--------|
| A | Run on Alibaba traces (6 policies x 3 seeds) | FGD reduces unallocated GPUs vs baselines | DONE |
| B | `test_fgd_placement.py` on 4:4:4 cluster | No double-assignment, fragmentation tracked | TODO |
| C | Existing `integration_test.py` with `enable_fgd=False` | JCT = 73063.45 / 57171.41 exactly | PASS (needs clean re-run) |
| D | FIFO+FGD vs FIFO baseline | Fragmentation improvement visible | TODO |
| E | Baselines sweep | Results comparable to published Gavel Fig 9 | TODO |
| F | Combined sweep | JCT at least as good as Gavel-only | TODO |
