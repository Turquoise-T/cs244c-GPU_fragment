#!/bin/bash
# Launch full comparison experiment with load-ordered dependencies.
# Low load experiments run first; each load level starts after the previous finishes.
#
# Usage: cd ~/gavel/experiments/fgd/slurm && bash launch_full_comparison.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$SCRIPT_DIR/submit_full_comparison.sbatch"

mkdir -p "$SCRIPT_DIR/slurm_logs"

# 13 load levels, 9 experiments per level (3 configs x 3 seeds)
LOADS=(60 85 110 135 160 185 210 235 260 285 310 335 360)
EXPS_PER_LOAD=9

PREV_JOB_ID=""

for i in "${!LOADS[@]}"; do
    load=${LOADS[$i]}
    start=$((i * EXPS_PER_LOAD))
    end=$((start + EXPS_PER_LOAD - 1))

    if [ -z "$PREV_JOB_ID" ]; then
        # First load level: no dependency
        JOB_ID=$(sbatch --parsable --array="${start}-${end}" "$SBATCH_FILE")
    else
        # Subsequent: start after previous load finishes
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB_ID} --array="${start}-${end}" "$SBATCH_FILE")
    fi

    echo "Load ${load} jph: indices ${start}-${end}, job ${JOB_ID}"
    PREV_JOB_ID="$JOB_ID"
done

echo ""
echo "All 13 load levels submitted (117 experiments total)."
echo "Monitor: squeue -u \$USER"
