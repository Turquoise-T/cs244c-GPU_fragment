#!/bin/bash
# Compress simulation.log files for completed experiments only
# Only compresses if summary.txt exists (indicating experiment finished)

RESULTS_DIR="${1:-results_full}"

echo "Compressing completed simulation logs in $RESULTS_DIR..."

count=0
for dir in ~/gavel/cluster/$RESULTS_DIR/*/; do
    # Only compress if experiment is complete (has summary.txt) and log is uncompressed
    if [ -f "$dir/summary.txt" ] && [ -f "$dir/simulation.log" ]; then
        gzip "$dir/simulation.log"
        count=$((count + 1))
    fi
done

echo "Compressed $count log files."
echo "Space usage: $(du -sh ~/gavel/cluster/$RESULTS_DIR)"
