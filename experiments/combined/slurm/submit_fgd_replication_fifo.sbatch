#!/bin/bash
#SBATCH --job-name=fgd-fifo
#SBATCH --output=slurm_logs/fgd_fifo-%A_%a.out
#SBATCH --error=slurm_logs/fgd_fifo-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-179

# FGD Replication with FIFO Policy (180 experiments)
# Same as fgd_replication but using FIFO scheduling to match the FGD paper.
# FIFO is much simpler than MaxMinFairness (no LP), so these should run faster.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fgd_replication_fifo.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/fgd_replication_fifo"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 23 hours (leaves 1 hr margin for SLURM's 24hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase fgd_replication_fifo \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 82800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
