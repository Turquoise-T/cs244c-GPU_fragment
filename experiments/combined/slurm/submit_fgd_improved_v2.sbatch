#!/bin/bash
#SBATCH --job-name=fgd-v2
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --array=0-179
#SBATCH --output=slurm_logs/fgd_improved_v2-%A-%a.out
#SBATCH --error=slurm_logs/fgd_improved_v2-%A-%a.err

# FGD improvements comparison with Cluster H topology
# 180 experiments: 2 policies x 3 placements x 15 rates x 2 seeds
# Placements: strided, fgd_vanilla, fgd_improved (paper+buddy+dynpen)

set -e

export PATH="$HOME/.local/bin:$PATH"
cd $HOME/gavel/experiments/combined

RESULTS_DIR="$HOME/gavel/experiments/combined/results/fgd_improved_v2"
mkdir -p "$RESULTS_DIR"
mkdir -p slurm_logs

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

python3 run_fgd_experiments.py \
    --phase fgd_improved_v2 \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    -q

echo "========================================"
echo "End time: $(date)"
echo "========================================"
