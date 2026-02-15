#!/bin/bash
#SBATCH --job-name=gavel-fgd-skip
#SBATCH --output=slurm_logs/fgd_skip_bench-%A.out
#SBATCH --error=slurm_logs/fgd_skip_bench-%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# FGD skip optimization benchmark: ECOS no-penalty with allocation-unchanged detection
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_fgd_skip_bench.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting FGD skip benchmark at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_fgd_skip \
    --index 0 \
    --output "$RESULTS_DIR/fgd_skip_bench.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "Finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr for PROFILE: grep PROFILE slurm_logs/fgd_skip_bench-\${SLURM_JOB_ID}.err"
exit $EXIT_CODE
