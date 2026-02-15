# Gavel Repo Reorganization Design

**Date:** 2026-02-15
**Goal:** Restructure the gavel repo from an organic research codebase into a paper-centric layout that clearly separates Paper 1 (Gavel), Paper 2 (FGD standalone), and the combined Paper 1+2 contribution.

## Principles

1. **Paper-centric:** Each experiment directory maps to one of the three project phases
2. **Consistent pattern:** `src/` holds core algorithms, `experiments/` holds drivers + results
3. **Strip upstream:** Remove unused deployment infrastructure (workloads, RPC, upstream scripts)
4. **Preserve all results:** Every experiment output (results, logs, telemetry) is kept and organized

## New Directory Structure

```
gavel/
  src/
    scheduler/                    # Paper 1: Gavel core
      scheduler.py                # Main scheduler (lazy runtime import)
      job.py, utils.py, ...       # Core modules
      fgd_placement.py            # Bridge: FGD integrated into Gavel
      set_queue.py                # Runtime dependency -- DO NOT REMOVE
      policies/                   # 13 scheduling policies
      traces/                     # MSR + physical cluster test traces
      [throughput JSONs, data]    # Throughput data files
    fgd/                          # Paper 2: FGD core algorithm
      fgd.py                      # FGD algorithm
      baselines.py                # Baseline placement algorithms
      alibaba_trace_parser.py     # Trace data loading
      configs/                    # Cluster definitions
      data/alibaba-gpu-v2023/     # Raw Alibaba trace CSVs
      tests/                      # Unit tests
      requirements.txt

  experiments/
    gavel-replication/            # Paper 1: Gavel figures 9/10/11
      configs/                    # 5 experiment configs
      scripts/                    # generate_*, plot_results, run_benchmark
      slurm/                      # 6 SLURM submit scripts
      results/                    # CSVs + per-experiment dirs (logs, summaries)
      figures/                    # 9 replication/comparison PNGs
      debug/                      # Telemetry viewer, extracted telemetry JSONs
      README.md, requirements.txt

    fgd-standalone/               # Paper 2: FGD standalone evaluation
      simulator.py                # Simulation driver
      run_standalone.py           # Standalone entry point
      run_evaluation.py           # Full evaluation runner
      plot_results.py             # Figure generation
      paper_reference_curves.json # Reference curves for plotting
      results/                    # full_run/, test_run/, test_run2/ JSONs
      figures/                    # paper_reference_all.png

    combined/                     # Papers 1+2: Gavel+FGD at Alibaba scale
      configs/                    # 19 phase configs
      scripts/                    # run_fgd_experiments.py, generate_viz.sh, etc.
      slurm/                      # 14 SLURM submit scripts
      results/                    # 117+ detailed_cmp/full_cmp JSONs + phase results
      logs/                       # Simulation logs
      detailed_logs/              # SLURM stderr .err files (merged from slurm/slurm_logs/)
      telemetry/                  # 45 experiment JSONL files
      bottleneck_analysis.html
      bottleneck_analysis.json

  tests/                          # Unit + integration tests (unchanged)
  scripts/                        # Cross-cutting utilities (unchanged)
  docs/                           # Design docs + plans/ (unchanged)
  .claude/, .github/, .venv/      # Config (unchanged)
  requirements-sim.txt, README.md, LICENSE
```

## Deletions (Upstream Stripping)

| Path | Files | Reason |
|---|---|---|
| `src/workloads/` | ~100+ | PyTorch/TensorFlow benchmarks, never imported by scheduler |
| `src/scheduler/runtime/` | ~20 | gRPC/protobuf RPC, only used in deployment mode |
| `src/scheduler/scripts/` | ~17 | Upstream utility scripts, not imported by anything |
| `src/scheduler/gavel_iterator.py` | 1 | RPC iterator client, deployment only |
| `src/EXPERIMENTS.md` | 1 | Upstream docs |
| `src/LICENSE` | 1 | Upstream docs (root LICENSE kept) |
| `src/README.md` | 1 | Upstream docs (root README kept) |
| `experiments/replication/gavel/legacy/` | 6 | Abandoned scripts |

## File Moves

### fgd_src/ -> src/fgd/ (core) + experiments/fgd-standalone/ (tooling)

Core algorithm:
- `fgd_src/fgd.py` -> `src/fgd/fgd.py`
- `fgd_src/baselines.py` -> `src/fgd/baselines.py`
- `fgd_src/alibaba_trace_parser.py` -> `src/fgd/alibaba_trace_parser.py`
- `fgd_src/configs/` -> `src/fgd/configs/`
- `fgd_src/data/` -> `src/fgd/data/`
- `fgd_src/tests/` -> `src/fgd/tests/`
- `fgd_src/requirements.txt` -> `src/fgd/requirements.txt`

Experiment tooling:
- `fgd_src/simulator.py` -> `experiments/fgd-standalone/simulator.py`
- `fgd_src/run_standalone.py` -> `experiments/fgd-standalone/run_standalone.py`
- `fgd_src/run_evaluation.py` -> `experiments/fgd-standalone/run_evaluation.py`
- `fgd_src/plot_results.py` -> `experiments/fgd-standalone/plot_results.py`
- `fgd_src/paper_reference_curves.json` -> `experiments/fgd-standalone/paper_reference_curves.json`
- `fgd_src/results/` -> `experiments/fgd-standalone/results/`
- `fgd_src/figures/` -> `experiments/fgd-standalone/figures/`

### Flatten experiments/replication/

- `experiments/replication/gavel/*` -> `experiments/gavel-replication/*`
- `experiments/replication/fgd/*` -> `experiments/combined/*`
- `experiments/replication/fgd/slurm/slurm_logs/*` -> `experiments/combined/detailed_logs/` (merge)

## Code Changes

### 1. Lazy runtime import in scheduler.py

Line 27: Remove top-level `from runtime.rpc import scheduler_server, scheduler_client`.

Add lazy imports in the two methods that use them (only reached when `_simulate=False`).

### 2. Import path updates

`src/scheduler/fgd_placement.py` line 21:
```python
# Before:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'fgd_src'))
# After:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fgd'))
```

`experiments/fgd-standalone/*.py`: Update sys.path to point at `../../src/fgd/`.

`experiments/combined/run_fgd_experiments.py`: Verify/update relative paths to `src/scheduler/`.

## Pattern Summary

| | Core code in src/ | Experiment drivers + results |
|---|---|---|
| Paper 1 (Gavel) | `src/scheduler/` | `experiments/gavel-replication/` |
| Paper 2 (FGD) | `src/fgd/` | `experiments/fgd-standalone/` |
| Combined | `src/scheduler/fgd_placement.py` | `experiments/combined/` |
