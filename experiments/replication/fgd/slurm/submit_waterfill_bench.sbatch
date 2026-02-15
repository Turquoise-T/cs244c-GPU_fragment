#!/bin/bash
#SBATCH --job-name=gavel-waterfill
#SBATCH --output=slurm_logs/waterfill_bench-%A_%a.out
#SBATCH --error=slurm_logs/waterfill_bench-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-1

# Water-filling benchmark: ECOS baseline (0) vs waterfill (1) at 60 jph.
# Migration penalty disabled for fair comparison.
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_waterfill_bench.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

NAMES=("ecos_no_penalty" "waterfill")
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting $NAME benchmark (index $SLURM_ARRAY_TASK_ID) at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_waterfill_bench \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/waterfill_bench_${NAME}.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "$NAME finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr for PROFILE: grep PROFILE slurm_logs/waterfill_bench-\${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
exit $EXIT_CODE
