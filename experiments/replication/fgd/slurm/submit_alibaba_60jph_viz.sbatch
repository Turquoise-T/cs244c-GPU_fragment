#!/bin/bash
#SBATCH --job-name=gavel-fgd-60jph-viz
#SBATCH --output=slurm_logs/ali_60jph_viz-%j.out
#SBATCH --error=slurm_logs/ali_60jph_viz-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# Single 60jph FGD experiment with FULL logging (no -q) for viz data.
# Produces INFO-level output: TELEMETRY, EVENT, [Micro-task scheduled].
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_alibaba_60jph_viz.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p slurm_logs
mkdir -p "$RESULTS_DIR"

echo "Starting 60jph FGD experiment (full logging) at $(date)"
echo "Host: $(hostname)"

# No -q flag: scheduler runs at INFO level, emitting all viz data channels.
# Index 0 = gavel_fgd_60jph_s0 in phase_e_alibaba_fgd_only config.
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_fgd_only \
    --index 0 \
    --output "$RESULTS_DIR/ali_60jph_viz.json" \
    --max-wall-time 13500

EXIT_CODE=$?
echo "Experiment finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
