#!/bin/bash
#SBATCH --job-name=gfgd-all
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-55
#SBATCH --output=slurm_logs/gavelfgd_all-%A-%a.out
#SBATCH --error=slurm_logs/gavelfgd_all-%A-%a.err

# Full Gavel+FGD improvements comparison
# 56 experiments: 7 load levels x 4 configs x 2 seeds
# Configs: gavel, gavelfgd_vanilla, gavelfgd_paper, gavelfgd_full (with dynamic penalty)

set -e

export PATH="$HOME/.local/bin:$PATH"
cd $HOME/gavel/experiments/combined

RESULTS_DIR="$HOME/gavel/experiments/combined/results/gavelfgd_all_improvements"
mkdir -p "$RESULTS_DIR"
mkdir -p slurm_logs

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

python3 run_fgd_experiments.py \
    --phase gavelfgd_all_improvements \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 12000 \
    -q

echo "========================================"
echo "End time: $(date)"
echo "========================================"
