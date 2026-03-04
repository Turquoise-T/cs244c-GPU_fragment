#!/bin/bash
#SBATCH --job-name=fig11-perf
#SBATCH --output=slurm_logs/fig11_perf-%A_%a.out
#SBATCH --error=slurm_logs/fig11_perf-%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=405,408,409,410,411,412,413

# Fig 11 Gavel (FTF-perf) re-run for .viz.bin generation.
# These experiments completed before but their SLURM logs were cleaned up.
# Re-running to generate logs for preprocess_viz conversion.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fig11_perf_rerun.sbatch

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
