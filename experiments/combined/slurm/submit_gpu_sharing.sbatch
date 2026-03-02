#!/bin/bash
#SBATCH --job-name=gavel-gpu-sharing
#SBATCH --output=slurm_logs/gpu_sharing-%A_%a.out
#SBATCH --error=slurm_logs/gpu_sharing-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-5

# A/B test: GPU sharing with fractional-aware LP vs no sharing.
# 6 experiments: 3 seeds x 2 arms (noshare, share), 60 jph Alibaba.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_gpu_sharing.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting GPU sharing experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gpu_sharing \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output "$RESULTS_DIR/gpu_sharing_${SLURM_ARRAY_TASK_ID}.json" \
    --save-logs \
    --max-wall-time 13800 \
    -q

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
