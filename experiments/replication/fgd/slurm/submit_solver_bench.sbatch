#!/bin/bash
#SBATCH --job-name=gavel-solver-bench
#SBATCH --output=slurm_logs/solver_bench-%A_%a.out
#SBATCH --error=slurm_logs/solver_bench-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-2

# Solver benchmark: ECOS (0) vs SCS (1) vs HiGHS (2) at 60 jph Alibaba scale.
# Each experiment runs with --max-wall-time 1200 (20 min).
# No -q flag so PROFILE lines are captured in stderr.
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_solver_bench.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

SOLVERS=("ecos" "scs" "highs")
SOLVER_NAME=${SOLVERS[$SLURM_ARRAY_TASK_ID]}

echo "Starting solver benchmark: $SOLVER_NAME (index $SLURM_ARRAY_TASK_ID) at $(date)"
echo "Host: $(hostname)"

$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_solver_bench \
    --index $SLURM_ARRAY_TASK_ID \
    --output "$RESULTS_DIR/solver_bench_${SOLVER_NAME}.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "Solver benchmark $SOLVER_NAME finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr for PROFILE: grep PROFILE slurm_logs/solver_bench-\${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
exit $EXIT_CODE
