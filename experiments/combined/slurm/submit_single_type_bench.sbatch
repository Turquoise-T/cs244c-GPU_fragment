#!/bin/bash
#SBATCH --job-name=gavel-single-type
#SBATCH --output=slurm_logs/single_type_bench-%A_%a.out
#SBATCH --error=slurm_logs/single_type_bench-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-2

# Single-type benchmark: ECOS+penalty (0) vs single-type (1) vs ECOS no penalty (2)
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_single_type_bench.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

NAMES=("ecos_with_penalty" "single_type_with_stickiness" "ecos_no_penalty")
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting $NAME benchmark (index $SLURM_ARRAY_TASK_ID) at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_single_type_bench \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/single_type_bench_${NAME}.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "$NAME finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr for PROFILE: grep PROFILE slurm_logs/single_type_bench-\${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
exit $EXIT_CODE
