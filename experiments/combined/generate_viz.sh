#!/bin/bash
# Generate .viz.bin files from FGD experiment logs for the GPU scheduling visualizer.
#
# Usage:
#   ./generate_viz.sh           # process existing logs only
#   ./generate_viz.sh --run     # run Phase D experiments first, then process
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAVEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VIZ_DIR="$(cd "$GAVEL_DIR/../gpu-scheduling-viz" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Optionally run experiments first
if [[ "${1:-}" == "--run" ]]; then
    echo "==> Running Phase D experiments with --save-logs..."
    python3 "$SCRIPT_DIR/run_fgd_experiments.py" --phase d --save-logs
fi

if [ ! -d "$LOG_DIR" ] || [ -z "$(ls "$LOG_DIR"/*.log 2>/dev/null)" ]; then
    echo "No log files found in $LOG_DIR"
    echo "Run with --run to generate logs, or run experiments manually with --save-logs"
    exit 1
fi

mkdir -p "$VIZ_DIR/data"

# The preprocessor uses `from viz.log_parser import ...` which requires the
# gpu-scheduling-viz directory to be importable as `viz`. A symlink in the
# parent directory handles this; PYTHONPATH points there.
VIZ_PARENT="$(dirname "$VIZ_DIR")"
ln -sf "$(basename "$VIZ_DIR")" "$VIZ_PARENT/viz" 2>/dev/null || true
export PYTHONPATH="$VIZ_PARENT"

echo "==> Generating .viz.bin files..."
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    output="$VIZ_DIR/data/fgd_${name}.viz.bin"
    echo "  $name -> $output"
    python3 "$VIZ_DIR/preprocess_viz.py" "$log" "$output" \
        --cluster 36:36:36 --policy "$name" --window-start 0 --window-end 100
done

echo "==> Done. Serve the visualizer with:"
echo "  cd $VIZ_DIR && python3 -m http.server 8080"
