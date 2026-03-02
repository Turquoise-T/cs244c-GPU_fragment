#!/bin/bash
#SBATCH --job-name=fgd-ch
#SBATCH --output=slurm_logs/fgd_ch-%A_%a.out
#SBATCH --error=slurm_logs/fgd_ch-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-359

# FGD Replication with Cluster H topology (360 experiments)
# Uses the FGD paper's actual cluster: 1200 nodes, single generic GPU type,
# mixed node sizes (462x8 + 310x4 + 228x2 + 200x1 = 5592 GPUs).
# 2 policies x 4 placements x 15 rates x 3 seeds.
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_fgd_replication_cluster_h.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results/fgd_replication_cluster_h"

# Create output directories
mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting experiment $SLURM_ARRAY_TASK_ID at $(date)"
echo "Host: $(hostname)"

# Wall-clock budget: 23 hours (leaves 1 hr margin for SLURM's 24hr limit)
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase fgd_replication_cluster_h \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/exp_${SLURM_ARRAY_TASK_ID}.json" \
    --max-wall-time 82800 \
    --save-logs

EXIT_CODE=$?
echo "Experiment $SLURM_ARRAY_TASK_ID finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
