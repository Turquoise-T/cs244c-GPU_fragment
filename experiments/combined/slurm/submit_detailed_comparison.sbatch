#!/bin/bash
#SBATCH --job-name=gavel-det
#SBATCH --output=slurm_logs/det-%A_%a.out
#SBATCH --error=slurm_logs/det-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# Detailed comparison with full logging (no -q).
# 21 experiments: 3 configs x 7 loads (60-360 jph, 50 jph steps), seed=654
# SLURM stderr will contain TELEMETRY lines for viz conversion.

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase detailed_comparison \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output "$RESULTS_DIR/detailed_cmp_${SLURM_ARRAY_TASK_ID}.json" \
    --save-logs \
    --max-wall-time 28200

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
