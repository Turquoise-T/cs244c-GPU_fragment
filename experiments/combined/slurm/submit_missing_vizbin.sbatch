#!/bin/bash
#SBATCH --job-name=missing-viz
#SBATCH --output=slurm_logs/missing_viz-%A_%a.out
#SBATCH --error=slurm_logs/missing_viz-%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=42,43,44,45,46,47,48,49,50,51,52,53,60,61,62,180,181,182,183,184,185,186,187,188,216,217,218,220,225,226,227,228,229,230,232,233,263,264,265,315,316,317,318,319,320,321,322,323,366,367,368,369,370,371,373,374

# Re-run 56 gavel replication experiments that are missing .viz.bin files.
# Original SLURM logs from Job 1434031 were cleaned up before conversion.
# These are mostly low-load (0.2-0.6 jph) + some mid/high baseline points.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_missing_vizbin.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/gavel_replication"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gavel_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 28800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
