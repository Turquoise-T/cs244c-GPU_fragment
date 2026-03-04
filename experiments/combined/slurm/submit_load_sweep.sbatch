#!/bin/bash
#SBATCH --job-name=gavel-load-sweep
#SBATCH --output=slurm_logs/load_sweep-%A_%a.out
#SBATCH --error=slurm_logs/load_sweep-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-116%9

# Gavel vs FGD vs Gavel+FGD Load Sweep (117 experiments)
# 3 arms x 13 load levels (60-360 jph, step 25) x 3 seeds
# on the full 6,200-GPU Alibaba cluster.
#
# No -q flag: full INFO logging for TELEMETRY, EVENT, and [Micro-task
# scheduled] lines needed by the viz tool (heatmap + timeseries).
#
# SLURM processes lowest array index first, so load levels run in order:
#   0-8: 60 jph, 9-17: 85 jph, ..., 108-116: 360 jph
#
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_load_sweep.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Run single experiment with full telemetry (no -q).
# Wall-clock budget: 7h45m (leaves 15 min margin for SLURM's 8hr limit).
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase load_sweep \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/load_sweep_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 27900

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
