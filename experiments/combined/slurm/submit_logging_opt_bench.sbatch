#!/bin/bash
#SBATCH --job-name=gavel-logopt
#SBATCH --output=slurm_logs/logging_opt_bench-%A.out
#SBATCH --error=slurm_logs/logging_opt_bench-%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# Logging optimization benchmark: demoted [Micro-task scheduled/succeeded] to DEBUG
# Usage: cd ~/gavel/experiments/combined/slurm && sbatch submit_logging_opt_bench.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/combined"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting logging optimization benchmark at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_fgd_skip \
    --index 0 \
    --output "$RESULTS_DIR/logging_opt_bench.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "Finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr for PROFILE: grep PROFILE slurm_logs/logging_opt_bench-\${SLURM_JOB_ID}.err"
exit $EXIT_CODE
