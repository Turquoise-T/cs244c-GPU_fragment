#!/bin/bash
#SBATCH --job-name=fig11-24h
#SBATCH --output=slurm_logs/fig11_24h-%A_%a.out
#SBATCH --error=slurm_logs/fig11_24h-%A_%a.err
#SBATCH --time=25:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=356,357,359,360,361,362,363,364,365

# Fig 11 Baseline FTF -- 9 saturated experiments re-run with 24h wall time.
# These hit the 8h limit: 2.8jph s2, 3.0jph s0/s2, 3.2jph all, 3.4jph all.
# Uses V100-throughput fix (heterogeneity-agnostic but temporally consistent).
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fig11_saturated_24h.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/gavel_replication"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 24 hours (leaves 1 hr margin for SLURM's 25hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase gavel_replication \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 86400 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
