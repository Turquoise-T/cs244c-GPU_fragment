# FGD (Fragmentation Gradient Descent) Implementation

## Overview

This project implements the **FGD (Fragmentation Gradient Descent)** algorithm from the ATC 2023 paper: *"Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent"*.

FGD is a placement algorithm designed to minimize GPU fragmentation in GPU-sharing environments where multiple jobs can share the same physical GPU.

## Key Concepts

### GPU Sharing vs Time-Slicing

| Aspect | GPU Sharing (FGD) | Time-Slicing (Gavel) |
|--------|-------------------|---------------------|
| **Model** | Multiple jobs run simultaneously on same GPU | Jobs take turns using whole GPUs |
| **GPU Request** | Fractional (e.g., 0.3 GPU, 0.5 GPU) | Whole GPUs only (1, 2, 4...) |
| **Fragmentation** | Scattered free capacity | Partially-used servers |
| **Blocking** | Jobs may wait if no suitable GPU | Jobs wait for their time slot |

### Fragmentation Example

```
Cluster: 2 GPUs
GPU-A: 0.5 GPU free (0.5 used)
GPU-B: 1.0 GPU free (empty)

Job C needs 0.7 GPU:
- GPU-A has only 0.5 free → cannot fit
- GPU-B has 1.0 free → CAN fit ✓

Without good placement, this 0.5 GPU on A becomes "fragmented" - 
unusable for jobs needing more than 0.5 GPU.
```

### FGD Algorithm

```
For each candidate GPU:
    1. Calculate fragmentation increment (Δfrag) if job placed here
    2. Δfrag = new_fragmentation - old_fragmentation
    
Select GPU with minimum Δfrag (prefer tight packing)
```

## Gavel Architecture and FGD Integration

### Input Data Structure

Gavel requires two core inputs:

#### 1. Throughput Oracle (`simulation_throughputs.json`)

This is Gavel's **primary input**, recording throughput (steps/sec) for each job type on each GPU type:

```json
{
  "ResNet-18 (batch size 32)": {
    "1": {                          // scale_factor = 1 GPU
      "v100": 1500.3,               // 1500.3 steps/sec on V100
      "p100": 1100.2,               // 1100.2 steps/sec on P100
      "k80": 600.1                  // 600.1 steps/sec on K80
    },
    "2": { ... },                   // scale_factor = 2 GPUs
    "4": { ... }
  },
  "Transformer (batch size 128)": { ... }
}
```

**Key**: Gavel uses this oracle to decide "how much faster this job runs on V100 vs P100" and performs heterogeneity-aware scheduling.

#### 2. Job Trace File (MSR/Philly format)

```
job_type                        command              num_steps_arg  scale_factor  total_steps  arrival_time  priority
Transformer (batch size 128)    cd ... && python     -step          1             95121        0.0           1
ResNet-50 (batch size 64)       cd ... && python     --num_steps    2             50000        10.0          1
A3C                             cd ... && python     --max-steps    0             1934260      100.0         4
```

| Column | Meaning |
|--------|---------|
| `job_type` | Must match a key in the throughput oracle |
| `scale_factor` | Number of GPUs needed (integer) |
| `total_steps` | Total steps to execute |
| `arrival_time` | Arrival time (seconds) |

#### 3. Cluster Configuration

```python
cluster_spec = {'v100': 64, 'p100': 0, 'k80': 0}  # 64 V100 GPUs
num_gpus_per_server = {'v100': 4, 'p100': 4, 'k80': 4}  # 4 GPUs per server
```

---

### Code Logic Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run_sweep_static.py                                │
│                                                                              │
│  1. Parse CLI args (cluster_spec, policy, placement_strategy, trace...)     │
│  2. Load throughput oracle                                                   │
│  3. Create Scheduler instance                                               │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           scheduler.Scheduler                               │
│                                                                              │
│  __init__(policy, throughputs_file, placement_strategy='strided'|'fgd')      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ simulate() main loop:                                                │   │
│  │                                                                       │   │
│  │   while jobs remain:                                                  │   │
│  │     ┌─────────────────────────────────────────────────────────────┐  │   │
│  │     │ 1. add_job(job)                                              │  │   │
│  │     │    - Parse job_type, scale_factor                            │  │   │
│  │     │    - Query throughput oracle: (job_type, scale_factor) → tp   │  │   │
│  │     │    - If not found → KeyError!                                │  │   │
│  │     └─────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                        │   │
│  │                              ▼                                        │   │
│  │     ┌─────────────────────────────────────────────────────────────┐  │   │
│  │     │ 2. policy.get_allocation(throughputs, scale_factors, ...)   │  │   │
│  │     │    - FIFO: allocate in arrival order                        │  │   │
│  │     │    - MaxMinFairness: optimize fairness                      │  │   │
│  │     │    - Output: {job_id: {worker_type: time_fraction}}          │  │   │
│  │     └─────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                        │   │
│  │                              ▼                                        │   │
│  │     ┌─────────────────────────────────────────────────────────────┐  │   │
│  │     │ 3. _assign_workers_to_job()  ← FGD integrated here!         │  │   │
│  │     │                                                              │  │   │
│  │     │    if placement_strategy == 'fgd':                           │  │   │
│  │     │        _assign_workers_to_job_fgd()   # minimize fragmentation│  │   │
│  │     │    else:                                                     │  │   │
│  │     │        _assign_workers_to_job_strided()  # fill servers in order│  │   │
│  │     │                                                              │  │   │
│  │     │    Output: job → [worker_id_1, worker_id_2, ...]              │  │   │
│  │     └─────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                        │   │
│  │                              ▼                                        │   │
│  │     ┌─────────────────────────────────────────────────────────────┐  │   │
│  │     │ 4. Simulate execution                                       │  │   │
│  │     │    - Compute steps completed from throughput                 │  │   │
│  │     │    - Advance time by time_per_iteration (default 360s)       │  │   │
│  │     │    - Check if job completed                                  │  │   │
│  │     └─────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### FGD Integration Point

FGD is integrated into Gavel's **Placement layer** (not the Allocation layer):

```
Gavel Scheduler
     │
     ├── Allocation Policy (how much resource per job)
     │   ├── fifo              → first-come first-served
     │   ├── max_min_fairness  → fairness allocation
     │   └── ...
     │
     └── Placement Strategy (which physical GPU) ← FGD lives here
         ├── strided (default) → fill servers in order
         └── fgd               → minimize fragmentation increment
```

### Strided vs FGD Comparison

```
Assume: 3 servers, 4 GPUs each, some jobs already placed

Server 0: [■■□□]  (2 used, 2 free)
Server 1: [■□□□]  (1 used, 3 free)  
Server 2: [□□□□]  (0 used, 4 free)

New job needs 2 GPUs:

Strided strategy (in order):
  → Take 2 free GPUs from Server 0
  → Server 0: [■■■■], Server 1: [■□□□], Server 2: [□□□□]
  → Goal: minimize number of servers used

FGD strategy (minimize fragmentation):
  → Compute Δfrag for each candidate placement
  → May choose Server 1 (becomes [■■■□])
  → Or Server 2 (becomes [■■□□])
  → Goal: keep remaining free space contiguous
```

**Note**: For whole-GPU workloads, both strategies produce **identical results** (same JCT, utilization, makespan). This is because Gavel's time-slicing model assigns each GPU exclusively to one job per round — there is no "fragmented free space" for FGD to optimize. FGD's advantage only appears with **fractional GPU sharing** (multiple jobs sharing one physical GPU simultaneously).

### Why Strided and FGD produce the same results in Gavel

In Gavel's round-based time-slicing model:
1. Each GPU is assigned to **exactly one job** per round (whole-GPU, exclusive).
2. The **Allocation Policy** (FIFO, MaxMinFairness, etc.) decides which jobs run and for what fraction of time.
3. The **Placement Strategy** (strided/FGD) only decides **which physical GPU** a job gets — but since each GPU runs exactly one job, there is no fragmentation to optimize.

To see a difference between FGD and strided, you need **GPU spatial sharing** (multiple jobs running concurrently on the same GPU with fractional requests). Use the standalone GPU sharing simulator (`simulate_gpu_sharing.py`) for this.

### Job Mix Options

The `--job-mix` flag controls the scale factor distribution of generated jobs:

| Mix | Distribution | Use case |
|-----|-------------|----------|
| `default` | 70% 1-GPU, 10% 2-GPU, 15% 4-GPU, 5% 8-GPU (Philly) | Standard Gavel experiments |
| `fragmentation` | 55% 1-GPU, 30% 2-GPU, 15% 4-GPU (Alibaba-like) | More diverse sizes to stress placement |

```bash
# Print sample jobs to see the mix
cd src/scheduler
python scripts/print_sample_jobs.py -n 20 --mix both

# Run comparison with fragmentation-friendly mix
python scripts/sweeps/run_sweep_static.py \
    --cluster-spec 16:0:0 --num_gpus_per_server 4:4:4 \
    --policies fifo --seeds 0 -a 100 -b 100 -n 1 \
    --job-mix fragmentation --generate-multi-gpu-jobs \
    --placement-strategy strided -l /tmp/compare/strided -v

python scripts/sweeps/run_sweep_static.py \
    --cluster-spec 16:0:0 --num_gpus_per_server 4:4:4 \
    --policies fifo --seeds 0 -a 100 -b 100 -n 1 \
    --job-mix fragmentation --generate-multi-gpu-jobs \
    --placement-strategy fgd -l /tmp/compare/fgd -v
```

---

## Implementation

### Files Modified/Created

```
src/scheduler/
├── job.py                     # Added gpu_milli field for fractional GPU requests
├── scheduler.py               # Added gpu_sharing_mode and placement_strategy options
├── utils.py                   # Updated Alibaba trace parser to preserve gpu_milli
└── policies/
    ├── fgd.py                 # Core FGD algorithm implementation
    └── fgd_test.py            # Unit tests for FGD

src/scheduler/scripts/
└── simulate_gpu_sharing.py    # Event-driven GPU sharing simulator
```

### Core Components

#### 1. GPUState (`policies/fgd.py`)

```python
@dataclass
class GPUState:
    gpu_id: int
    server_id: int
    total_milli: int = 1000      # 1000 milli = 1.0 GPU
    used_milli: int = 0
    job_assignments: List[Tuple[int, int]]
```

#### 2. GPUSharingCluster (`policies/fgd.py`)

```python
class GPUSharingCluster:
    def place_job(self, job_id, gpu_milli, num_gpus=1, strategy='fgd'):
        """Place a job using FGD or baseline strategy."""
        
    def remove_job(self, job_id):
        """Free GPU resources when job completes."""
        
    def get_fragmentation(self):
        """Current cluster fragmentation score."""
```

#### 3. Placement Strategies

| Strategy | Description |
|----------|-------------|
| `fgd` | Minimize fragmentation increment |
| `bestfit` | Select GPU with least free space |
| `worstfit` | Select GPU with most free space |
| `firstfit` | Select first available GPU |

#### 4. Job Class Extension (`job.py`)

```python
class Job:
    def __init__(self, ..., gpu_milli=None):
        # gpu_milli: 0-1000 representing 0.0-1.0 GPU
        self._gpu_milli = gpu_milli if gpu_milli else scale_factor * 1000
    
    @property
    def gpu_fraction(self):
        return self._gpu_milli / 1000.0
```

## Usage

### 1. Gavel Simulation (Strided vs FGD Comparison)

```bash
cd src/scheduler
mkdir -p /tmp/gavel_compare

# Strided (Gavel default)
python scripts/sweeps/run_sweep_static.py \
    --cluster-spec 64:0:0 \
    --num_gpus_per_server 4:4:4 \
    --policies fifo \
    --seeds 0 \
    -a 100 -b 100 -n 1 \
    --placement-strategy strided \
    -l /tmp/gavel_compare/strided \
    -v

# FGD
python scripts/sweeps/run_sweep_static.py \
    --cluster-spec 64:0:0 \
    --num_gpus_per_server 4:4:4 \
    --policies fifo \
    --seeds 0 \
    -a 100 -b 100 -n 1 \
    --placement-strategy fgd \
    -l /tmp/gavel_compare/fgd \
    -v
```

**Parameter reference:**

| Parameter | Meaning |
|-----------|---------|
| `--cluster-spec 64:0:0` | 64 V100s, 0 P100s, 0 K80s |
| `--num_gpus_per_server 4:4:4` | 4 GPUs per server |
| `--policies fifo` | Use FIFO allocation policy |
| `-a 100 -b 100 -n 1` | Generate 100 synthetic jobs |
| `--placement-strategy` | `strided` or `fgd` |

### 2. GPU Sharing Simulator (Fractional GPU, Standalone)

```bash
cd src/scheduler

# Synthetic workload
python scripts/simulate_gpu_sharing.py \
    --workload synthetic \
    --num-jobs 100 \
    --num-servers 4 \
    --gpus-per-server 4

# Alibaba trace (real fractional GPU data)
python scripts/simulate_gpu_sharing.py \
    --workload alibaba \
    --trace traces/cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec05.csv \
    --num-jobs 200
```

### 3. FGD Unit Tests

```bash
cd src/scheduler
python policies/fgd_test.py
```

## Experimental Results

### GPU Sharing Simulation (100 jobs, 4×4 GPUs)

```
Strategy     Blocked    Avg JCT    Avg Frag   Avg Util
---------------------------------------------------------
fgd          1724       1294.7     1.036      81.4%
bestfit      1752       1292.3     1.049      81.5%
worstfit     1774       1298.3     1.848      79.5%   ← Worst
firstfit     1697       1281.3     0.933      82.4%
```

**Key Findings:**
- FGD achieves similar or lower fragmentation than best-fit
- Worst-fit creates significantly more fragmentation (+78%)
- All packing strategies (FGD, best-fit, first-fit) outperform worst-fit

### Larger Scale (300 jobs, 2×4 GPUs)

```
Strategy     Avg Frag   Improvement vs Worst-fit
------------------------------------------------
fgd          0.486      +20.8%
bestfit      0.509      +17.1%
worstfit     0.614      baseline
```

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Job Arrival                    │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │     GPUSharingCluster.place_job()       │
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │ For each GPU:                   │    │
                    │  │   Calculate Δfrag if placed     │    │
                    │  │   Δfrag = F(after) - F(before)  │    │
                    │  └─────────────────────────────────┘    │
                    │                                         │
                    │  Select GPU with minimum Δfrag          │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │                                         │
                    ▼                                         ▼
        ┌───────────────────┐                     ┌───────────────────┐
        │  Job Placed       │                     │  Job Blocked      │
        │  - Update GPU     │                     │  - Add to queue   │
        │    used_milli     │                     │  - Wait for       │
        │  - Track job      │                     │    resources      │
        └───────────────────┘                     └───────────────────┘
```

## Trace Support

### Alibaba Trace Format (cluster-trace-gpu-v2023)

```csv
name,num_gpu,gpu_milli,gpu_spec,creation_time,deletion_time,qos
pod-1,1,500,V100,1000,2000,BE
pod-2,1,300,V100,1100,1800,LS
```

- `gpu_milli`: Fractional GPU request (0-1000)
- `num_gpu`: Number of GPUs needed
- `qos`: Quality of Service (LS, BE, Burstable)

### MSR/Philly Trace Format (Gavel)

```
ResNet-18 (batch size 64)    command    1    1000    0    1.0
```

- Column 3: scale_factor (whole GPUs)
- No fractional GPU support

## Limitations

1. **Gavel Integration**: Gavel uses time-slicing (whole GPUs), not spatial sharing. Full FGD benefits require GPU sharing mode.

2. **Throughput Oracle**: GPU sharing affects job throughput in ways not captured by Gavel's oracle. Real deployments need runtime profiling.

3. **Memory Contention**: FGD doesn't model GPU memory contention when jobs share a GPU.

## References

1. [FGD Paper (ATC 2023)](https://www.usenix.org/conference/atc23/presentation/weng) - "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent"

2. [Gavel Paper (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak) - "Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads"

3. [Alibaba GPU Trace](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023) - Production GPU cluster traces with sharing workloads

## Future Work

- [ ] Implement memory-aware FGD (consider GPU memory fragmentation)
- [ ] Add interference modeling for co-located jobs
- [ ] Integrate with Kubernetes GPU sharing plugins
- [ ] Support heterogeneous GPU clusters (mixed V100/A100)
