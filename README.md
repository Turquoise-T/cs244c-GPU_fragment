# Gavel + Fragmentation Awareness

CS244C Final Project: Extending GPU cluster schedulers with fragmentation awareness.

## Background

**Gavel** (OSDI 2020) is a heterogeneity-aware cluster scheduler for deep learning workloads. It uses an *effective throughput* abstraction and round-based allocation across GPU types (V100, P100, K80).

**The gap:** Gavel was validated on whole-GPU allocations (e.g. Microsoft Philly). In production (e.g. Alibaba), jobs often share GPUs via **partial allocations**, leading to **fragmentation** — scattered free capacity that cannot fit new jobs.

**This project:**
1. Integrates **FGD (Fragmentation Gradient Descent**, ATC 2023) into Gavel as a placement strategy.
2. Adds **GPU spatial sharing** so multiple jobs can run on the same GPU simultaneously (fractional `gpu_milli`).
3. Validates that FGD reduces average JCT by ~10% vs baseline (strided/worst-fit) under GPU sharing.

## References

- [Gavel paper (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak)
- [FGD paper (ATC 2023)](https://www.usenix.org/conference/atc23/presentation/weng)
- [Original Gavel repo](https://github.com/stanford-futuredata/gavel)
- [Alibaba GPU Trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023)

---

## Key Concepts

### GPU Sharing vs Time-Slicing

| Aspect | GPU Sharing (FGD / this project) | Time-Slicing (Gavel default) |
|--------|-----------------------------------|------------------------------|
| **Model** | Multiple jobs run **simultaneously** on same GPU | Jobs take turns using **whole** GPUs |
| **GPU Request** | Fractional (e.g. 0.3, 0.5 GPU via `gpu_milli`) | Whole GPUs only (1, 2, 4...) |
| **Fragmentation** | Scattered free capacity on each GPU | Partially-used servers |
| **Placement** | FGD minimizes fragmentation; strided = worst-fit | Strided fills servers in order |

### Fragmentation Example

```
Cluster: 2 GPUs
GPU-A: 0.5 GPU free (0.5 used)
GPU-B: 1.0 GPU free (empty)

Job C needs 0.7 GPU:
  GPU-A has only 0.5 free → cannot fit
  GPU-B has 1.0 free → CAN fit ✓

Bad placement leaves 0.5 GPU on A "fragmented" — unusable for jobs needing >0.5.
FGD chooses placement to minimize this kind of leftover.
```

### FGD Algorithm (high level)

- For each candidate GPU: compute **fragmentation increment (Δfrag)** if the job is placed there.
- Select the GPU with **minimum Δfrag** (prefer tight packing so fewer GPUs have small leftovers).

---

## GPU Spatial Sharing: What We Implemented

### Result

- **True GPU spatial sharing** in Gavel: multiple jobs per GPU using `gpu_milli` (0–1000 milli = 0–1.0 GPU).
- **FGD vs Strided (worst-fit):**
  - Strided: avg_jct ≈ 350,096 s
  - FGD: avg_jct ≈ 316,554 s → **~9.6% JCT reduction**

### Why It Works Now

Originally, with GPU sharing enabled, only 4 jobs were scheduled (one per GPU) because:

1. **Policy layer** (FIFO) only understood whole-GPU allocations → at most 4 jobs got allocation on 4 GPUs.
2. Jobs without allocation had **zero priority** and were skipped in placement.

**Fix:** In GPU sharing mode we **bypass** policy allocation checks for placement: the scheduler tries to place **all** waiting jobs onto GPUs based on **actual free capacity** (`gpu_milli` and per-GPU `worker_capacity_used`). Placement is then either **FGD** (minimize fragmentation) or **strided** (worst-fit: pick GPU with most free space).

### Two-Phase Placement (GPU sharing)

Each round:

1. **Phase 1:** Place **continuing jobs** (running in the previous round) first → establishes current GPU usage.
2. **Phase 2:** Place **new/waiting jobs** in remaining capacity.

So “Phase 2” is the step that fills **remaining capacity** with jobs that were not yet placed in that round. Strided does not optimize for fragmentation; FGD does, which is why FGD can fit more jobs and reduce JCT.

### Strided vs FGD in GPU Sharing Mode

- **Strided (worst-fit):** Picks the GPU with the **most** free space. Uses `gpu_milli` and per-GPU capacity (so partial GPU is supported), but **does not** try to reduce fragmentation — often increases it.
- **FGD:** Picks the GPU that **minimizes fragmentation increment**. Explicitly addresses the fragmentation problem and achieves the ~10% JCT improvement.

### Example: 4 GPUs, 10 jobs (first round)

```
GPU 0: Job 0 (1000 milli) → 100% used
GPU 1: Job 1 (700) + Job 2 (200) → 90% used  ✓ spatial sharing
GPU 2: Job 3 (700) + Job 6 (300) → 100% used ✓ spatial sharing
GPU 3: Job 4 (500) + Job 7 (300) → 80% used  ✓ spatial sharing
→ 7 jobs running on 4 GPUs.
```

---

## Prerequisites

- macOS or Linux (tested on macOS 14 with Apple Silicon)
- Python 3.9+
- Git

## Quick Start

```bash
# Clone the repo
git clone -b vr_gavel https://github.com/Turquoise-T/cs244c-GPU_fragment.git
cd cs244c-GPU_fragment

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-sim.txt

# Generate protobuf stubs (required once)
cd src/scheduler
# If "make rpc_stubs" fails (e.g. Xcode license), run instead:
# python -m grpc_tools.protoc -Iruntime/protobuf --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/protobuf/common.proto runtime/protobuf/enums.proto runtime/protobuf/iterator_to_scheduler.proto runtime/protobuf/scheduler_to_worker.proto runtime/protobuf/worker_to_scheduler.proto
make rpc_stubs

# Verify installation
python -c "import scheduler; print('Setup OK')"

# Enable pre-commit hooks (runs tests before each commit)
git config core.hooksPath .githooks
```

---

## Run a Test Simulation

### Basic Gavel (time-slicing, whole GPU)

```bash
cd src/scheduler
mkdir -p /tmp/gavel_test

python scripts/sweeps/run_sweep_static.py \
  --throughputs-file simulation_throughputs.json \
  --cluster-spec 4:4:4 \
  --policies fifo \
  --num-total-jobs-lower-bound 10 \
  --num-total-jobs-upper-bound 10 \
  --num-data-points 1 \
  --seeds 42 \
  --log-dir /tmp/gavel_test \
  --solver SCS \
  -v
```

(On macOS without ECOS, use `--solver SCS`.)

Expected output (example):
```
Configuration: cluster_spec=v100:4|p100:4|k80:4, policy=FIFO, seed=42, num_total_jobs=10
Results: average JCT=36721.82, utilization=0.14, makespan=215530.17
```

### GPU Sharing: Strided vs FGD

```bash
cd src/scheduler
conda activate gavel   # or your env with dependencies

# Quick comparison (4 V100s, 50 jobs, fractional gpu_milli)
python test_compare.py
```

Example output:
```
RESULT strided: avg_jct=350096.03, makespan=1307244.20
RESULT fgd: avg_jct=316554.26, makespan=1307362.34
```

With sweep script and GPU sharing:

```bash
python scripts/sweeps/run_sweep_static.py \
  --throughputs-file simulation_throughputs.json \
  --cluster-spec 4:0:0 \
  --num_gpus_per_server 4:4:4 \
  --policies fifo \
  --placement-strategy fgd \
  --gpu-sharing \
  -a 50 -b 50 -n 1 --seeds 0 \
  -l /tmp/gavel_gpu_sharing -v
```

**Parameters:**

| Parameter | Meaning |
|-----------|--------|
| `--cluster-spec 4:0:0` | 4 V100s, 0 P100, 0 K80 |
| `--num_gpus_per_server 4:4:4` | 4 GPUs per server |
| `--placement-strategy` | `strided` or `fgd` |
| `--gpu-sharing` | Enable fractional GPU (gpu_milli) and spatial sharing |

---

## Gavel Architecture and FGD Integration

### Inputs

1. **Throughput oracle** (`simulation_throughputs.json`): steps/sec per (job_type, scale_factor) and GPU type.
2. **Cluster:** e.g. `cluster_spec = {'v100': 4, 'p100': 0, 'k80': 0}`, `num_gpus_per_server`.
3. **Jobs:** job_type, scale_factor (or gpu_milli in GPU sharing), total_steps, arrival_time.

### Where FGD Fits

```
Gavel Scheduler
├── Allocation policy (how much resource per job): FIFO, MaxMinFairness, ...
└── Placement strategy (which physical GPU)  ← FGD lives here
    ├── strided (default) → in GPU sharing mode: worst-fit by free space
    └── fgd               → minimize fragmentation increment
```

- **Allocation** decides *who* runs and time share; **placement** decides *which* GPU(s).
- Without GPU sharing, each GPU runs one job per round → no per-GPU fragmentation → FGD and strided behave similarly.
- With **GPU spatial sharing**, FGD’s placement reduces fragmentation and yields the ~10% JCT improvement.

### Job mix and traces

- **Default mix:** Philly-like (e.g. 70% 1-GPU, 10% 2-GPU, …).
- **GPU sharing:** Jobs get `gpu_milli` from a generator (e.g. 200/300/500/700/1000 milli).
- **Alibaba trace:** CSV with `gpu_milli`; see `scripts/simulate_gpu_sharing.py` and trace docs in `src/scheduler/`.

---

## Implementation Overview

### Main files

| File | Purpose |
|------|---------|
| `src/scheduler/scheduler.py` | Core scheduling loop; GPU sharing mode, two-phase placement, bypass policy allocation for placement |
| `src/scheduler/policies/fgd.py` | FGD algorithm: `GPUState`, `fgd_select_gpu`, fragmentation increment |
| `src/scheduler/job.py` | Job with `gpu_milli` for fractional GPU |
| `src/scheduler/utils.py` | Job generation, `_generate_gpu_milli_sharing`, trace parsing |
| `src/scheduler/scripts/sweeps/run_sweep_static.py` | Sweep script; `--gpu-sharing`, `--placement-strategy` |
| `src/scheduler/test_compare.py` | Strided vs FGD under GPU sharing |

### Core FGD pieces (`policies/fgd.py`)

- **GPUState:** `gpu_id`, `server_id`, `total_milli`, `used_milli`, `job_assignments`.
- **fgd_select_gpu(gpu_states, milli_needed):** Returns GPU id that minimizes Δfrag (and tie-break by less free space).
- **Placement strategies:** `fgd`, `bestfit`, `worstfit`, `firstfit` (for comparison).

### Trace formats

- **Gavel (MSR/Philly):** scale_factor = whole GPUs; no gpu_milli.
- **Alibaba (cluster-trace-gpu-v2023):** CSV with `gpu_milli` (0–1000), `num_gpu`, etc.

---

## Running Experiments on FarmShare

For large-scale runs (e.g. replicating Gavel figures):

1. **SSH:** Ensure access to `rice.stanford.edu`; optional SSH multiplexing in `~/.ssh/config`.
2. **Sync code:** e.g. `rsync -avz src/scheduler/ farmshare:~/gavel/src/scheduler/`.
3. **Run:** See `experiments/replication/README.md` and `experiments/replication/scripts/`, e.g.:
   - Single run: `python3 scripts/run_benchmark.py --index 0 ...`
   - Batch: `sbatch slurm/submit_full.sbatch`
4. **Results:** `rsync` results back; solver/ECOS notes in `experiments/replication/debug/` if needed.

---

## Project Structure

```
.
├── src/scheduler/              # Core scheduler
│   ├── scheduler.py            # Scheduling loop, GPU sharing, two-phase placement
│   ├── job.py                  # Job (incl. gpu_milli)
│   ├── policies/               # FIFO, FGD, etc.
│   │   └── fgd.py              # FGD placement and GPUState
│   ├── scripts/sweeps/         # run_sweep_static.py
│   ├── scripts/simulate_gpu_sharing.py
│   ├── test_compare.py         # Strided vs FGD (GPU sharing)
│   ├── traces/                 # Trace data
│   └── simulation_throughputs.json
├── experiments/replication/    # Paper replication (Figs 9, 10, 11)
├── docs/
└── requirements-sim.txt
```

---

## Limitations

1. **Policy layer:** Gavel’s policies assume whole-GPU allocation; in GPU sharing we bypass allocation checks for placement so all waiting jobs can be tried. Fairness is still driven by policy but placement is capacity-based.
2. **Throughput:** GPU sharing can affect throughput (contention); the oracle does not model this.
3. **Memory:** FGD does not model GPU memory fragmentation or contention.

## Future Work

- Policy constraints that understand fractional GPU (e.g. Σ gpu_milli per GPU ≤ 1000).
- Dynamic arrivals (λ > 0) and larger-scale GPU sharing experiments.
- Memory-aware FGD; interference models for co-located jobs.
- Real Alibaba (or other) trace pipelines end-to-end in the scheduler.

---

*GPU spatial sharing and FGD validation: 2026-01-24.*
