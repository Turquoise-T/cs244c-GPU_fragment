#!/bin/bash
#SBATCH --job-name=gavel-full
#SBATCH --output=slurm_logs/full-%A_%a.out
#SBATCH --error=slurm_logs/full-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=0-311

# Full replication: 312 experiments (Figures 9, 10, 11)
# Fig 9: 120 experiments (single-GPU, 2 policies x 20 rates x 3 seeds)
# Fig 10: 90 experiments (multi-GPU, 2 policies x 15 rates x 3 seeds)
# Fig 11: 102 experiments (multi-GPU, 2 policies x 17 rates x 3 seeds)

export PATH="$HOME/.local/bin:$PATH"

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

mkdir -p "$CLUSTER_DIR/slurm_logs"
mkdir -p "$CLUSTER_DIR/results_full"

cd "$CLUSTER_DIR"

echo "========================================"
echo "Starting experiment $SLURM_ARRAY_TASK_ID"
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
