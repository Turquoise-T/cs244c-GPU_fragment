#!/bin/bash
#SBATCH --job-name=gavel_extended
#SBATCH --output=logs/extended_%A_%a.out
#SBATCH --error=logs/extended_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-59

# Extended multi-GPU experiments to see saturation behavior
# Fig 10: 3.2-4.0 jph, Fig 11: 3.6-4.4 jph

cd ~/gavel/cluster
mkdir -p logs results_extended

python3 run_benchmark.py \
    --index $SLURM_ARRAY_TASK_ID \
    --experiments-file experiments_extended.json \
    --output-dir results_extended
