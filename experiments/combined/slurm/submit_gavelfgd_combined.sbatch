#!/bin/bash
#SBATCH --job-name=gfgd-comb
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-41
#SBATCH --output=slurm_logs/gavelfgd_combined-%A-%a.out
#SBATCH --error=slurm_logs/gavelfgd_combined-%A-%a.err

# GavelFGD-Combined: Gavel vs GavelFGD vs GavelFGD-Combined (paper scoring + POA LP)
# 42 experiments: 7 load levels x 3 configs (gavel, gavelfgd, gavelfgd_combined) x 2 seeds

set -e

export PATH="$HOME/.local/bin:$PATH"

cd $HOME/gavel/experiments/combined

RESULTS_DIR="$HOME/gavel/experiments/combined/results/gavelfgd_combined"
mkdir -p "$RESULTS_DIR"
mkdir -p slurm_logs

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

python3 run_fgd_experiments.py \
    --phase gavelfgd_combined \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 12000 \
    -q

echo "========================================"
echo "End time: $(date)"
echo "========================================"
