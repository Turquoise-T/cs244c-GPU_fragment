# Gavel + Fragmentation Awareness

CS244C Final Project: Extending GPU cluster schedulers with fragmentation awareness.

## Background

**Gavel** (OSDI 2020) is a heterogeneity-aware cluster scheduler for deep learning workloads. It uses an *effective throughput* abstraction to express scheduling policies and a round-based allocation mechanism to achieve target allocations across different GPU types (V100, P100, K80).

**The gap:** Gavel was validated on Microsoft's Philly trace, which uses whole-GPU allocations. In production clusters like Alibaba's, jobs often share GPUs through partial allocations, leading to *fragmentation* - unusable GPU resources scattered across nodes.

**This project:**
1. Validates Gavel on Alibaba's GPU-sharing traces
2. Integrates fragmentation-aware placement (from FGD, ATC 2023) into Gavel's framework

## References

- [Gavel paper (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak)
- [FGD paper (ATC 2023)](https://www.usenix.org/conference/atc23/presentation/weng)
- [Original Gavel repo](https://github.com/stanford-futuredata/gavel)

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
make rpc_stubs

# Verify installation
python -c "import scheduler; print('Setup OK')"
```

## Run a Test Simulation

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
  -v
```

Expected output:
```
Configuration: cluster_spec=v100:4|p100:4|k80:4, policy=FIFO, seed=42, num_total_jobs=10
Results: average JCT=36721.82, utilization=0.14, makespan=215530.17
```

## Project Structure

```
.
├── docs/plans/              # Design documents
├── requirements-sim.txt     # macOS-compatible dependencies
├── src/
│   └── scheduler/
│       ├── scheduler.py     # Main scheduler logic
│       ├── policies/        # Scheduling policies (FIFO, LAS, Gavel, etc.)
│       ├── scripts/
│       │   └── sweeps/      # Simulation scripts
│       │       ├── run_sweep_static.py    # Static trace simulation
│       │       └── run_sweep_continuous.py # Continuous job arrival
│       ├── traces/          # Trace data (Philly, etc.)
│       └── simulation_throughputs.json    # Throughput profiles for simulation
└── osdi20-narayanan_deepak.pdf  # Gavel paper
```

### Key Files

| File | Purpose |
|------|---------|
| `scheduler.py` | Core scheduling logic and simulation loop |
| `policies/` | Policy implementations (what we'll extend) |
| `scripts/sweeps/` | Entry points for running experiments |
| `simulation_throughputs.json` | Job throughput profiles by GPU type |

## Contributing

### Branch Naming

```
<member>/<feature>
```

Examples:
- `member1/alibaba-trace-loader`
- `member2/gavel-fgd-integration`

### Commit Messages

Use clear, imperative messages:
```
Add Alibaba trace parser
Fix JCT calculation for partial GPU jobs
Update simulation to handle GPU sharing
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commits
3. Push and open a PR against `main`
4. Request review from at least one teammate
5. Squash and merge once approved
