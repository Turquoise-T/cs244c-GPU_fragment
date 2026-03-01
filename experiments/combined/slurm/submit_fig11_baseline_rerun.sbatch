#!/bin/bash
#SBATCH --job-name=fig11-fix
#SBATCH --output=slurm_logs/fig11_fix-%A_%a.out
#SBATCH --error=slurm_logs/fig11_fix-%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=315-365

# Fig 11 Baseline FTF Re-run (51 experiments, indices 315-365)
# Re-running after fix: FinishTimeFairnessPolicy now sets all throughputs
# to 1.0 (heterogeneity-agnostic), matching MaxMinFairnessPolicy pattern.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fig11_baseline_rerun.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/gavel_replication"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 8 hours (leaves 1 hr margin for SLURM's 9hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gavel_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 28800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
