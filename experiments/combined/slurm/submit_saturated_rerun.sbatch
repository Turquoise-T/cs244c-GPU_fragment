#!/bin/bash
#SBATCH --job-name=sat-rerun
#SBATCH --output=slurm_logs/sat_rerun-%A_%a.out
#SBATCH --error=slurm_logs/sat_rerun-%A_%a.err
#SBATCH --time=1-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=54-59,109,111-119,219,221-224,267-269,350-365,414-416

# Re-run all 43 experiments that timed out with previous wall-time limits.
# Increased to 24h wall time (from 8-16h).
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_saturated_rerun.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/gavel_replication"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# 24-hour wall-clock budget
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gavel_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 86400 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
