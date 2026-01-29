#!/bin/bash
#SBATCH --job-name=gavel-fig
#SBATCH --output=slurm_logs/gavel-%A_%a.out
#SBATCH --error=slurm_logs/gavel-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-269

# FarmShare Gavel Figure Reproduction
# Usage: sbatch submit_farmshare.sbatch

# Add user-installed packages to PATH
export PATH="$HOME/.local/bin:$PATH"

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

# Create log directories
mkdir -p "$CLUSTER_DIR/slurm_logs"
mkdir -p "$CLUSTER_DIR/results"

# Run the experiment for this array task
echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
cd "$CLUSTER_DIR"

python3 run_single_experiment.py \
    --index $SLURM_ARRAY_TASK_ID \
    --experiments-file experiments.json \
    --output-dir results \
    --scheduler-dir "$GAVEL_DIR/src/scheduler"

echo "Experiment $SLURM_ARRAY_TASK_ID completed at $(date)"
