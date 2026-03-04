#!/bin/bash
#SBATCH --job-name=gavel-cmp
#SBATCH --output=slurm_logs/cmp-%A_%a.out
#SBATCH --error=slurm_logs/cmp-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# Full comparison: Gavel vs FGD vs Gavel+FGD
# 117 experiments: 3 configs x 13 loads (60-360 jph) x 3 seeds
# Submit via launch_full_comparison.sh for load-ordered dependencies.

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase full_comparison \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output "$RESULTS_DIR/full_cmp_${SLURM_ARRAY_TASK_ID}.json" \
    --save-logs \
    --max-wall-time 28200 \
    -q

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
