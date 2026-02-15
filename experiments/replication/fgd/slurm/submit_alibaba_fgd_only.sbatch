#!/bin/bash
#SBATCH --job-name=gavel-fgd-ali-test
#SBATCH --output=slurm_logs/ali_fgd-%A_%a.out
#SBATCH --error=slurm_logs/ali_fgd-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-2

# Alibaba FGD-only bottleneck analysis (3 experiments)
# FGD placement at 60/180/360 jph, seed 0, migration penalty enabled
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_alibaba_fgd_only.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 3.75 hours (leaves 15 min margin for SLURM's 4hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_fgd_only \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/ali_fgd_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 13500 \
    -q

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
