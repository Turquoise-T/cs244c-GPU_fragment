#!/bin/bash
#SBATCH --job-name=gavel-fgd-philly-mig
#SBATCH --output=slurm_logs/philly_mig-%A_%a.out
#SBATCH --error=slurm_logs/philly_mig-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-26

# Philly FGD + Migration Penalty Sweep (27 experiments)
# {strided, bestfit, fgd} x {60, 180, 360 jph} x 3 seeds
# Philly cluster (36:36:36), 10-min rounds, migration penalty enabled
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_philly_migration.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 3.75 hours (leaves 15 min margin for SLURM's 4hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_philly_migration \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/philly_mig_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 13500 \
    -q

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
