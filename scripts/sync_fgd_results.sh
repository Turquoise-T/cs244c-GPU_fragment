#!/bin/bash
# Download FGD experiment results from FarmShare
# Usage: ./sync_fgd_results.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAVEL_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_FGD="$GAVEL_DIR/experiments/fgd"
REMOTE="farmshare:~/gavel/experiments/fgd"

echo "========================================"
echo "Syncing FGD results from FarmShare"
echo "========================================"

# Verify SSH connectivity first
if ! ssh -o ConnectTimeout=5 farmshare "echo ok" &>/dev/null; then
    echo ""
    echo "ERROR: Cannot connect to FarmShare."
    echo "Start an SSH session in another terminal: ssh farmshare"
    exit 1
fi

# Check SLURM job status
echo ""
echo "Checking job status..."
QUEUE_INFO=$(ssh farmshare "squeue -u \$USER -n gavel-fgd-alibaba 2>/dev/null" || true)
RUNNING=$(echo "$QUEUE_INFO" | grep -c ' R ' || true)
PENDING=$(echo "$QUEUE_INFO" | grep -c ' PD ' || true)
echo "Jobs running: $RUNNING, pending: $PENDING"

# Check completed results
COMPLETED=$(ssh farmshare "ls ~/gavel/experiments/fgd/results/result_*.json 2>/dev/null | wc -l" || echo "0")
echo "Completed experiments: $COMPLETED / 45"

if [ "$RUNNING" -gt 0 ] || [ "$PENDING" -gt 0 ]; then
    echo ""
    echo "Warning: Jobs still running/pending. Syncing partial results..."
fi

# 1. Download per-experiment result JSONs
echo ""
echo "[1/3] Syncing per-experiment results..."
mkdir -p "$LOCAL_FGD/results"
rsync -avz --progress \
    "$REMOTE/results/" "$LOCAL_FGD/results/"

# 2. Download combined results file (if runner was invoked without --output)
echo ""
echo "[2/3] Syncing combined results file..."
rsync -avz --progress \
    --include='results_phase_e_alibaba.json' \
    --exclude='*' \
    "$REMOTE/" "$LOCAL_FGD/" 2>/dev/null || true

# 3. Download SLURM logs
echo ""
echo "[3/3] Syncing SLURM logs..."
mkdir -p "$LOCAL_FGD/slurm/slurm_logs"
rsync -avz --progress \
    "$REMOTE/slurm/slurm_logs/" "$LOCAL_FGD/slurm/slurm_logs/"

echo ""
echo "========================================"
echo "Sync complete!"
echo "Results in: $LOCAL_FGD/results/"
echo "SLURM logs: $LOCAL_FGD/slurm/slurm_logs/"
echo "========================================"

# Show summary
echo ""
echo "Result summary:"
RESULT_COUNT=$(ls "$LOCAL_FGD/results"/result_*.json 2>/dev/null | wc -l)
echo "  Downloaded $RESULT_COUNT / 45 experiment results"

if [ "$RESULT_COUNT" -gt 0 ]; then
    echo ""
    echo "  Sample JCT values:"
    for f in $(ls "$LOCAL_FGD/results"/result_*.json 2>/dev/null | head -5); do
        NAME=$(python3 -c "import json; d=json.load(open('$f')); print(d[0]['name'])" 2>/dev/null || echo "?")
        JCT=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d[0]['avg_jct']:.2f}\" if not d[0].get('saturated') else 'SATURATED')" 2>/dev/null || echo "?")
        echo "    $(basename $f): $NAME -> JCT=$JCT"
    done
    echo "  ..."
fi

# Check for failures
ERRORS=$(ssh farmshare "grep -l 'Error\|Traceback' ~/gavel/experiments/fgd/slurm/slurm_logs/alibaba-*.out 2>/dev/null | wc -l" || echo "0")
if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $ERRORS experiment(s) have errors in SLURM logs!"
    echo "  Check: ls $LOCAL_FGD/slurm/slurm_logs/"
fi
