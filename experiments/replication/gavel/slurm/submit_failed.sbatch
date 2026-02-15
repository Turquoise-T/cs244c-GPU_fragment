#!/bin/bash
#SBATCH --job-name=gavel-retry
#SBATCH --output=slurm_logs/gavel-retry-%A_%a.out
#SBATCH --error=slurm_logs/gavel-retry-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0,10,20,30,40,50,60,70,80,90,105,120,135,150,165

# Retry failed experiments with longer time limit (8 hours)
# These are the high-load experiments that timed out at 4 hours
# Usage: sbatch submit_failed.sbatch

export PATH="$HOME/.local/bin:$PATH"

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

echo "Retrying experiment $SLURM_ARRAY_TASK_ID at $(date)"
cd "$CLUSTER_DIR"

python3 run_single_experiment.py \
    --index $SLURM_ARRAY_TASK_ID \
    --experiments-file experiments.json \
    --output-dir results \
    --scheduler-dir "$GAVEL_DIR/src/scheduler"

echo "Experiment $SLURM_ARRAY_TASK_ID completed at $(date)"
