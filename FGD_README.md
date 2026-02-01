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

### 1. GPU Sharing Simulator (Standalone)

```bash
cd src/scheduler

# Synthetic workload
python scripts/simulate_gpu_sharing.py \
    --workload synthetic \
    --num-jobs 100 \
    --num-servers 4 \
    --gpus-per-server 4

# Alibaba trace
python scripts/simulate_gpu_sharing.py \
    --workload alibaba \
    --trace traces/cluster-trace-gpu-v2023/csv/openb_pod_list_gpu0.csv \
    --num-jobs 200
```

### 2. FGD Placement Tests

```bash
cd src/scheduler
python policies/fgd_test.py
```

### 3. Gavel Integration (Whole-GPU FGD)

```bash
cd src/scheduler/scripts/sweeps
python run_sweep_static.py \
    --trace ../traces/msr/seed=0/0e4a51.trace \
    --policies fifo \
    --placement-strategy fgd \
    --cluster-spec v100:64
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
