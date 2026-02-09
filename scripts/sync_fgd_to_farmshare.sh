#!/bin/bash
# Upload FGD experiment code and configs to FarmShare
# Usage: ./sync_fgd_to_farmshare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAVEL_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE="farmshare:~/gavel"

echo "========================================"
echo "Syncing FGD code to FarmShare"
echo "Local:  $GAVEL_DIR"
echo "Remote: $REMOTE"
echo "========================================"

# Verify SSH connectivity first
if ! ssh -o ConnectTimeout=5 farmshare "echo ok" &>/dev/null; then
    echo ""
    echo "ERROR: Cannot connect to FarmShare."
    echo "Start an SSH session in another terminal: ssh farmshare"
    exit 1
fi

RSYNC_OPTS="-avz --exclude=__pycache__ --exclude=*.pyc"

# 1. Scheduler code (includes simulation_throughputs_alibaba.json)
echo ""
echo "[1/3] Syncing src/scheduler/..."
rsync $RSYNC_OPTS \
    --exclude='.venv' \
    --exclude='logs/' \
    --exclude='*.log' \
    "$GAVEL_DIR/src/scheduler/" "$REMOTE/src/scheduler/"

# 2. FGD library
echo ""
echo "[2/3] Syncing fgd_src/..."
rsync $RSYNC_OPTS \
    "$GAVEL_DIR/fgd_src/" "$REMOTE/fgd_src/"

# 3. FGD experiments (runner, configs, slurm scripts)
echo ""
echo "[3/3] Syncing experiments/fgd/..."
rsync $RSYNC_OPTS \
    --exclude='results_*' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='*.log' \
    --exclude='slurm_logs/' \
    "$GAVEL_DIR/experiments/fgd/" "$REMOTE/experiments/fgd/"

echo ""
echo "========================================"
echo "Sync complete!"
echo ""
echo "Next steps on FarmShare:"
echo "  ssh farmshare"
echo "  cd ~/gavel/experiments/fgd/slurm"
echo "  sbatch submit_alibaba.sbatch"
echo "========================================"
