# FGD Fragmentation Evaluation on Alibaba Traces

Replicates Figure 9 from **"Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent"** (USENIX ATC'23) using the Alibaba GPU cluster trace v2023.

## Overview

This standalone evaluation compares FGD against five baseline scheduling policies using Monte-Carlo workload inflation on production traces from a 1,213-node / 6,212-GPU heterogeneous cluster.

**Schedulers** (Section 6.1 of the paper):

| Policy | Description |
|--------|-------------|
| FGD | Fragmentation Gradient Descent — schedules to minimize fragmentation growth |
| BestFit | Node with least remaining resources (weighted CPU+GPU) |
| DotProd | Smallest dot-product of remaining resources and task demands |
| Packing | Prioritize occupied GPUs, then idle GPUs on occupied nodes |
| Clustering | Pack tasks with the same GPU request size together |
| Random | Random eligible node |

**Figure 9 subplots:**
- **(a)** Unallocatable GPU (%) vs arrived workloads (%)
- **(b)** Occupied nodes vs arrived workloads (%)
- **(c)** GPU requests of failed tasks at 96% capacity (stacked bar)
- **(d)** Fragmentation breakdown into 3 causes (non-gpu, stranded, deficient)

## Setup

```bash
# 1. Download Alibaba trace CSVs (~600KB total)
python3 download_traces.py

# 2. Install dependencies
pip install numpy matplotlib
```

## Running the Experiments

### Figure 9 — Monte-Carlo Workload Inflation (paper replication)

Tasks are randomly sampled **with replacement** from the trace and submitted until cumulative GPU demand exceeds a threshold. No departures. Repeated across 10 trials and averaged with std dev.

```bash
# Full replication (1213 nodes, 6212 GPUs, 10 trials) — ~25 min
python3 run_figure9.py --num-trials 10 --max-workload-pct 130

# Quick test (~2 min)
python3 run_figure9.py --num-trials 2 --cluster-scale 10

# Custom schedulers
python3 run_figure9.py --schedulers fgd,bestfit,random --num-trials 5
```

Outputs to `figure9_results/`:
- `figure9.png` / `figure9.pdf` — the 4-subplot figure
- `figure9_results.json` — raw data for all schedulers and trials

### Event-Driven Trace Replay

Replays the full trace with real arrival/departure timestamps, measuring fragmentation rate, utilization, and JCT over time.

```bash
# Full trace replay
python3 run_alibaba_experiment.py

# Quick test
python3 run_alibaba_experiment.py --max-tasks 500 --cluster-scale 10
```

## File Structure

| File | Purpose |
|------|---------|
| `run_figure9.py` | Figure 9 replication (Monte-Carlo workload inflation) |
| `run_alibaba_experiment.py` | Event-driven trace replay with JCT tracking |
| `simulator.py` | Core data structures: Task, Node, Cluster, EventDrivenSimulator |
| `schedulers.py` | All 6 scheduling policies (FGD + 5 baselines) |
| `trace_loader.py` | Alibaba trace CSV parser |
| `download_traces.py` | Downloads trace CSVs from GitHub |
| `plot_results.py` | Plotting utilities for trace replay results |

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num-trials` | 10 | Monte-Carlo trials (averaged with std dev) |
| `--max-workload-pct` | 130 | Stop inflation at this % of GPU capacity |
| `--cluster-scale` | 100 | Use N% of original cluster nodes (preserves heterogeneity) |
| `--num-gpus` | 0 | Override to N homogeneous GPUs (4/node). 0 = use trace cluster |
| `--seed` | 42 | Base random seed |

## Connection to Gavel

The FGD policy implementation in `schedulers.py` mirrors the algorithm in `src/scheduler/policies/fgd.py` (the Gavel integration), extended with CPU-awareness for the Alibaba trace's multi-resource tasks. The core scheduling logic — Algorithm 1 from the paper (hypothetical per-node placement, min fragmentation delta) — is identical.
