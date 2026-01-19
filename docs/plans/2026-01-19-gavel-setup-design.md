# Gavel Simulator Setup Strategy

CS244C Project - Gavel Validation Lead Setup Plan

Date: 2026-01-19

## Overview

This document outlines the strategy for setting up the Gavel simulator on macOS for the CS244C final project. The goal is to enable fast local development while maintaining a Docker fallback for compatibility.

## Approach: Hybrid (Native First, Docker Fallback)

Try native macOS setup first for development velocity. If blockers persist beyond ~2 hours, switch to Docker. Document what broke for potential future fixes.

## Phase 1: Getting the Source

### Step 1: Clone Gavel source
```bash
cd /Users/varunr/projects/courses/stanford/cs244c/gavel
git clone https://github.com/stanford-futuredata/gavel.git src
```

### Step 2: Create Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Phase 2: Dependency Analysis

### Core Dependencies (keep, update versions)

| Package | Purpose | macOS Status |
|---------|---------|--------------|
| cvxpy | LP solver for allocation | Works natively |
| numpy, scipy | Numerics | Works natively |
| pandas | Data handling | Works natively |
| matplotlib, seaborn | Plotting | Works natively |

### Problematic Dependencies

| Package | Issue | Solution |
|---------|-------|----------|
| torch==1.4.0 | Ancient, won't install on Python 3.9+ | Update to 2.x or stub |
| numa | Linux-only | Stub with mock |
| grpcio, grpcio-tools | Physical deployment only | Skip for simulation |
| gym[atari] | RL with native deps | Skip unless testing RL policies |

### Modernized requirements-sim.txt
```
cvxpy>=1.4
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
dill
filelock
psutil
```

## Phase 3: Setup Steps

### Step 3: Install modernized dependencies
```bash
pip install -r requirements-sim.txt
```

### Step 4: Create numa stub

Create `src/scheduler/numa_stub.py`:
```python
# Stub for macOS - returns dummy NUMA topology
def info():
    return type('obj', (object,), {'numa_nodes': 1})()

def get_node_cpus(n):
    return list(range(8))
```

### Step 5: Test import
```bash
cd src/scheduler
python -c "import scheduler; print('Scheduler imports OK')"
```

### Step 6: Run minimal simulation
```bash
python scripts/sweeps/run_sweep_static.py --help
```

## Phase 4: Docker Fallback

If native setup fails after ~2 hours, use Docker.

### Dockerfile
```dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-venv \
    git cmake g++ gcc libnuma-dev make

WORKDIR /gavel
COPY src/scheduler/requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ .
CMD ["bash"]
```

### docker-compose.yml
```yaml
services:
  gavel:
    build: .
    volumes:
      - ./src:/gavel
      - ./data:/data
      - ./results:/results
    working_dir: /gavel/scheduler
```

### Docker workflow
1. Edit code locally in editor
2. Run: `docker-compose run gavel python scripts/sweeps/run_sweep_static.py ...`
3. Results appear in `./results/`

### Switch triggers
Switch to Docker if any of these block for >30 min:
- cvxpy solver issues (ECOS/SCS compilation)
- Import errors requiring Linux-only packages
- Segfaults in native extensions

## Success Criteria

Phase 1 complete when:
```bash
python -c "from scheduler import scheduler; print('OK')"
```

Full setup complete when:
```bash
python scripts/sweeps/run_sweep_static.py --help
# Shows usage without errors
```

## Next Steps

After setup is validated:
1. Download Philly traces for baseline replication
2. Run Gavel on Philly traces
3. Validate results within 10% of paper (Milestone M1: Feb 5)
