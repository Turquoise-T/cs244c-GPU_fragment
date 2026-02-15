#!/bin/bash
#SBATCH --job-name=gavel-fgd-alibaba
#SBATCH --output=slurm_logs/alibaba-%A_%a.out
#SBATCH --error=slurm_logs/alibaba-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-44

# Alibaba FGD Experiment Sweep (45 experiments)
# 5 configs x 3 load rates x 3 seeds on the full 6,200-GPU Alibaba cluster
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_alibaba.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Run single experiment, saving result to per-experiment JSON
# Wall-clock budget: 3.75 hours (leaves 15 min margin for SLURM's 4hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/result_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 13500 \
    -q

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
