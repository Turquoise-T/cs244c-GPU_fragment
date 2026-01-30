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

# Enable pre-commit hooks (runs tests before each commit)
git config core.hooksPath .githooks
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

## Run a Simulation with FGD
With a similar setup as the section above, from inside the `venv` within the `src/scheduler` folder, run the following commands:

```bash
# Run FGD policy
python3 scripts/drivers/simulate_scheduler_with_trace.py \
  -t traces/physical_cluster/small_test.trace \
  -p fgd \
  -c 4:4:4 \
  --num_gpus_per_server 4:4:4 \
  --throughputs_file simulation_throughputs.json \
  --seed 42

# Run baseline for comparison
python3 scripts/drivers/simulate_scheduler_with_trace.py \
  -t traces/physical_cluster/small_test.trace \
  -p max_min_fairness_perf \
  -c 4:4:4 \
  --num_gpus_per_server 4:4:4 \
  --throughputs_file simulation_throughputs.json \
  --seed 42
```


## Running Experiments on FarmShare

For large-scale experiments (e.g., replicating Gavel paper figures), use Stanford's FarmShare cluster.

### Prerequisites

1. **SSH access to FarmShare** - Ensure you can SSH to `rice.stanford.edu`
2. **SSH multiplexing (recommended)** - Keep a persistent connection for faster commands

Add to `~/.ssh/config`:
```
Host farmshare
    HostName rice.stanford.edu
    User <your-sunetid>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

Create the socket directory and connect:
```bash
mkdir -p ~/.ssh/sockets
ssh farmshare  # Keep this terminal open
```

### Initial Setup (One-Time)

```bash
# 1. Sync code to FarmShare (from local machine)
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='results*' \
    /path/to/gavel farmshare:~/

# 2. SSH to FarmShare and set up Python environment
ssh farmshare
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install cvxpy numpy
```

### Syncing Code Changes

When you modify code locally, sync to FarmShare before running experiments:

```bash
# Sync scheduler code
rsync -avz src/scheduler/ farmshare:~/gavel/src/scheduler/

# Sync experiment scripts
rsync -avz experiments/ farmshare:~/gavel/experiments/
```

### Running Experiments

See `experiments/replication/README.md` for detailed instructions on running the paper replication experiments.

#### Quick Start

```bash
# Single experiment (interactive)
ssh farmshare
cd ~/gavel/experiments/replication
source ~/.venv/bin/activate
python3 scripts/run_benchmark.py --index 0 --experiments-file configs/experiments_full.json --output-dir results/test

# Batch experiments (SLURM)
ssh farmshare "cd ~/gavel/experiments/replication && sbatch slurm/submit_full.sbatch"

# Check job status
ssh farmshare "squeue -u \$USER"
```

### Retrieving Results

```bash
# Sync results back to local machine
rsync -avz --exclude='simulation.log' \
    farmshare:~/gavel/experiments/replication/results/ \
    ./experiments/replication/results/
```

### Troubleshooting

**Solver failures (ECOS):**
- See `experiments/replication/debug/2025-01-27-ecos-solver-failures-research.md`
- The codebase includes ECOS-to-SCS fallback to handle these cases

**Out of memory:**
- Increase `--mem` in the sbatch file (default 8G, try 16G)

## Project Structure

```
.
├── src/scheduler/           # Core scheduler code
│   ├── scheduler.py         # Main scheduler logic and simulation loop
│   ├── policies/            # Scheduling policies (FIFO, LAS, Gavel, etc.)
│   ├── scripts/sweeps/      # Simulation scripts
│   ├── traces/              # Trace data (Philly, etc.)
│   └── simulation_throughputs.json  # Throughput profiles
│
├── experiments/             # Experiment-specific code and results
│   └── replication/         # Gavel paper replication (Figs 9, 10, 11)
│       ├── configs/         # Experiment configurations (JSON)
│       ├── results/         # Experiment outputs and CSVs
│       ├── figures/         # Generated plots
│       ├── scripts/         # Experiment runner, generators, plotting
│       ├── slurm/           # SLURM batch scripts for FarmShare
│       ├── debug/           # Telemetry tools and investigation notes
│       └── README.md        # Replication-specific documentation
│
├── scripts/                 # Shared utilities
│   ├── sync_results.sh      # FarmShare result sync helper
│   └── compress_completed_logs.sh  # Log compression utility
│
├── docs/                    # Documentation
│   └── plans/               # Design documents
│
└── requirements-sim.txt     # Python dependencies
```

### Key Files

| File | Purpose |
|------|---------|
| `src/scheduler/scheduler.py` | Core scheduling logic and simulation loop |
| `src/scheduler/policies/` | Policy implementations (what we'll extend) |
| `src/scheduler/simulation_throughputs.json` | Job throughput profiles by GPU type |
| `experiments/replication/` | Complete Gavel paper replication with results |

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
