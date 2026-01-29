#!/bin/bash
#SBATCH --job-name=gavel-pilot
#SBATCH --output=slurm_logs/pilot-%A_%a.out
#SBATCH --error=slurm_logs/pilot-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-17

# Pilot: 18 experiments (3 figures x 2 policies x 3 rates x 1 seed)
# Uses run_benchmark.py with saturation detection and comprehensive telemetry

export PATH="$HOME/.local/bin:$PATH"

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

mkdir -p "$CLUSTER_DIR/slurm_logs"
mkdir -p "$CLUSTER_DIR/results_pilot"

cd "$CLUSTER_DIR"

echo "========================================"
echo "Starting pilot experiment $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "========================================"

python3 run_benchmark.py \
    --index $SLURM_ARRAY_TASK_ID \
    --experiments-file experiments_pilot.json \
    --output-dir results_pilot \
    --scheduler-dir "$GAVEL_DIR/src/scheduler"

EXIT_CODE=$?

echo "========================================"
echo "Experiment $SLURM_ARRAY_TASK_ID completed"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
