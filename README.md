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

# 2. SSH to FarmShare and run setup
ssh farmshare
cd ~/gavel/cluster
chmod +x setup_farmshare.sh
./setup_farmshare.sh
```

The setup script:
- Creates a Python virtual environment
- Installs dependencies (cvxpy, numpy, etc.)
- Generates experiment configurations
- Creates log directories

### Syncing Code Changes

When you modify code locally, sync to FarmShare before running experiments:

```bash
# Sync scheduler code
rsync -avz src/scheduler/ farmshare:~/gavel/src/scheduler/

# Sync cluster scripts
rsync -avz cluster/ farmshare:~/gavel/cluster/
```

### Running Experiments

#### Single Experiment (Interactive)

```bash
ssh farmshare
cd ~/gavel/cluster
source ~/.venv/bin/activate

# Run experiment index 0
python3 run_benchmark.py \
    --index 0 \
    --experiments-file experiments_full.json \
    --output-dir results_full \
    --scheduler-dir ~/gavel/src/scheduler
```

#### Batch Experiments (SLURM)

Submit all experiments as a SLURM job array:

```bash
ssh farmshare "cd ~/gavel/cluster && sbatch submit_full.sbatch"
```

The sbatch file (`cluster/submit_full.sbatch`) configures:
- `--array=0-311` - Run experiments 0 through 311
- `--time=08:00:00` - 8-hour timeout per experiment
- `--mem=8G` - Memory allocation
- `--partition=normal` - FarmShare partition

#### Monitoring Jobs

```bash
# Check job status
ssh farmshare "squeue -u \$USER"

# Count completed experiments
ssh farmshare "ls ~/gavel/cluster/results_full/*/summary.txt | wc -l"

# View a specific job's output
ssh farmshare "cat ~/gavel/cluster/slurm_logs/full-<jobid>_<index>.out"

# Check for errors
ssh farmshare "cat ~/gavel/cluster/slurm_logs/full-<jobid>_<index>.err"
```

#### Canceling Jobs

```bash
# Cancel all your jobs
ssh farmshare "scancel -u \$USER"

# Cancel a specific job array
ssh farmshare "scancel <jobid>"
```

### Retrieving Results

Sync results back to your local machine:

```bash
# Sync summary files (excludes large simulation.log files)
./cluster/sync_results.sh results_full

# Or manually with rsync
rsync -avz --exclude='simulation.log' \
    farmshare:~/gavel/cluster/results_full/ \
    ./cluster/results_full/
```

### Managing Disk Space

Simulation logs can be large (100MB+ each). Compress completed experiments:

```bash
# Compress logs for completed experiments only
ssh farmshare "cd ~/gavel/cluster && ./compress_completed_logs.sh results_full"

# Check disk usage
ssh farmshare "du -sh ~/gavel/cluster/results_full"
```

### Experiment Configuration Files

| File | Description |
|------|-------------|
| `experiments_full.json` | Full paper replication (312 experiments) |
| `experiments_pilot.json` | Pilot runs for testing (subset) |
| `submit_full.sbatch` | SLURM batch script for full experiments |
| `submit_retry.sbatch` | Retry failed experiments |

### Generating New Experiments

```bash
cd cluster

# Generate full experiment set (Figures 9, 10, 11)
python3 generate_full_experiments.py

# Generate pilot experiments
python3 generate_pilot_experiments.py
```

### Troubleshooting

**Job stuck in pending (PD) state:**
```bash
# Check why job is pending
ssh farmshare "squeue -u \$USER -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
```

**Stale file handle errors:**
- NFS issues on FarmShare - retry the failed experiments
- Use `submit_retry.sbatch` with failed indices

**Solver failures (ECOS):**
- See `docs/2025-01-27-ecos-solver-failures-research.md` for analysis
- Mostly affects `finish_time_fairness` at high loads
- These experiments are near saturation anyway

**Out of memory:**
- Increase `--mem` in the sbatch file
- Default is 8G, try 16G for complex experiments

## Project Structure

```
.
├── cluster/                 # FarmShare experiment infrastructure
│   ├── run_benchmark.py     # Main experiment runner
│   ├── generate_*.py        # Experiment config generators
│   ├── submit_*.sbatch      # SLURM batch scripts
│   ├── sync_results.sh      # Result retrieval script
│   ├── experiments_*.json   # Experiment configurations
│   └── results_*/           # Experiment outputs
├── docs/
│   ├── plans/               # Design documents
│   └── *.md                 # Research notes and analysis
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
| `cluster/run_benchmark.py` | FarmShare experiment runner with telemetry |
| `cluster/experiments_full.json` | Full paper replication configs (312 experiments) |

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
