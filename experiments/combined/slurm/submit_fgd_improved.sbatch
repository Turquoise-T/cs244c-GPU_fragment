#!/bin/bash
#SBATCH --job-name=fgd-imp
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-79
#SBATCH --output=slurm_logs/fgd_improved-%A-%a.out
#SBATCH --error=slurm_logs/fgd_improved-%A-%a.err

# FGD improvements comparison experiments
# 80 experiments: 8 load levels x 5 methods x 2 seeds

set -e

# Add local pip packages to PATH
export PATH="$HOME/.local/bin:$PATH"

# Change to working directory first
cd $HOME/gavel/experiments/combined

# Use absolute paths for results
RESULTS_DIR="$HOME/gavel/experiments/combined/results/fgd_improved"
mkdir -p "$RESULTS_DIR"
mkdir -p slurm_logs

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

python3 run_fgd_experiments.py \
    --phase fgd_improved \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 12000 \
    -q

echo "========================================"
echo "End time: $(date)"
echo "========================================"
