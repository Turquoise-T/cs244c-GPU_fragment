#!/bin/bash
# Sync results from FarmShare to local machine
# Usage: ./sync_results.sh [results_dir]
#   results_dir: remote directory name (default: results_full)

RESULTS_DIR="${1:-results_full}"
LOCAL_DIR="$(dirname "$0")/$RESULTS_DIR"
REMOTE_DIR="farmshare:~/gavel/cluster/$RESULTS_DIR"

echo "========================================"
echo "Syncing results from FarmShare"
echo "Remote: $REMOTE_DIR"
echo "Local:  $LOCAL_DIR"
echo "========================================"

# Check how many experiments completed
COMPLETED=$(ssh farmshare "ls ~/gavel/cluster/$RESULTS_DIR 2>/dev/null | wc -l")
echo "Completed experiments: $COMPLETED"

# Check if any jobs still running
RUNNING=$(ssh farmshare "squeue -u vramesh3 -n gavel-full 2>/dev/null | grep -c ' R '")
PENDING=$(ssh farmshare "squeue -u vramesh3 -n gavel-full 2>/dev/null | grep -c ' PD '")
echo "Jobs running: $RUNNING, pending: $PENDING"

if [ "$RUNNING" -gt 0 ] || [ "$PENDING" -gt 0 ]; then
    echo ""
    echo "Warning: Jobs still running/pending. Syncing partial results..."
fi

echo ""
echo "Syncing summary files (excluding large simulation.log files)..."
mkdir -p "$LOCAL_DIR"

# Sync everything except simulation.log files (which can be 100MB+ each)
rsync -avz --progress \
    --exclude='simulation.log' \
    "$REMOTE_DIR/" "$LOCAL_DIR/"

echo ""
echo "========================================"
echo "Sync complete!"
echo "Results in: $LOCAL_DIR"
echo "========================================"

# Show summary of results
echo ""
echo "Result summary:"
find "$LOCAL_DIR" -name "summary.txt" | head -5 | while read f; do
    echo "--- $(dirname "$f" | xargs basename) ---"
    grep -E "Average JCT|Saturated" "$f"
done
echo "... ($(find "$LOCAL_DIR" -name "summary.txt" | wc -l) total)"
