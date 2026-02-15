#!/bin/bash
#SBATCH --job-name=gavel-retry
#SBATCH --output=slurm_logs/retry-%A_%a.out
#SBATCH --error=slurm_logs/retry-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=48,49,50,51,52,53,54,55,56,57,58,118,155,203,231,232,236,238,241,242,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,278,281,284,287,289,290,292,293,294,295,297,298,299,300,301,303,304,305,306,307,308,309,310,311

# Retry failed experiments (61 total)
# - 36 stale file handle (NFS issues)
# - 7 TypeError (fixed in scheduler.py)
# - 24 SolverError (may still fail)

export PATH="$HOME/.local/bin:$PATH"

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

mkdir -p "$CLUSTER_DIR/slurm_logs"
mkdir -p "$CLUSTER_DIR/results_full"

cd "$CLUSTER_DIR"

echo "========================================"
echo "Retrying experiment $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "========================================"

python3 run_benchmark.py \
    --index $SLURM_ARRAY_TASK_ID \
    --experiments-file experiments_full.json \
    --output-dir results_full \
    --scheduler-dir "$GAVEL_DIR/src/scheduler"

EXIT_CODE=$?

# Compress simulation.log after experiment completes
RESULT_DIR=$(python3 -c "import json; e=json.load(open('experiments_full.json'))[$SLURM_ARRAY_TASK_ID]; print(f\"{e['figure']}_{e['policy']}_{e['jobs_per_hr']}jph_{'multi' if e['multi_gpu'] else 'single'}_s{e['seed']}\")")
if [ -f "results_full/$RESULT_DIR/simulation.log" ]; then
    echo "Compressing simulation.log..."
    gzip "results_full/$RESULT_DIR/simulation.log"
fi

echo "========================================"
echo "Experiment $SLURM_ARRAY_TASK_ID completed"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
