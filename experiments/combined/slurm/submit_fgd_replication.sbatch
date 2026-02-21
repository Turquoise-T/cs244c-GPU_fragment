#!/bin/bash
#SBATCH --job-name=fgd-repl
#SBATCH --output=slurm_logs/fgd_repl-%A_%a.out
#SBATCH --error=slurm_logs/fgd_repl-%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-179

# FGD Replication Experiments (180 experiments)
# Reproduces FGD (ATC 2023) fragmentation results via Gavel scheduler.
# INFO-level logging is kept enabled so that EVENT, TELEMETRY, and
# ALLOCATION lines are available for the viz tool (matching Gavel replication).
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fgd_replication.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/fgd_replication"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 8 hours (leaves 1 hr margin for SLURM's 9hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase fgd_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 28800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
