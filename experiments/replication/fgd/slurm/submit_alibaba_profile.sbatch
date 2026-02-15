#!/bin/bash
#SBATCH --job-name=gavel-ali-profile
#SBATCH --output=slurm_logs/ali_profile-%j.out
#SBATCH --error=slurm_logs/ali_profile-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# Bounded profiling run: 60 jph Alibaba experiment with 20-min wall limit.
# Captures ~200-400 rounds of per-section timing data.
# No -q flag so PROFILE summary is emitted to stderr.
# Usage: cd ~/gavel/experiments/fgd/slurm && sbatch submit_alibaba_profile.sbatch

GAVEL_DIR="$HOME/gavel"
FGD_DIR="$GAVEL_DIR/experiments/fgd"
RESULTS_DIR="$FGD_DIR/results"

mkdir -p "$(dirname "$0")/slurm_logs"
mkdir -p "$RESULTS_DIR"

echo "Starting profiling run at $(date)"
echo "Host: $(hostname)"

# Run experiment index 0 (60 jph) from fgd_only config.
# --max-wall-time 1200 = 20 minutes (enough for ~200-400 rounds at Alibaba scale).
# No -q: scheduler logs at INFO level so PROFILE line is captured in stderr.
$HOME/gavel/.venv/bin/python3 "$FGD_DIR/run_fgd_experiments.py" \
    --phase e_alibaba_fgd_only \
    --index 0 \
    --output "$RESULTS_DIR/ali_profile.json" \
    --max-wall-time 1200

EXIT_CODE=$?
echo "Profiling run finished at $(date) with exit code $EXIT_CODE"
echo "Check stderr log for PROFILE line: grep PROFILE slurm_logs/ali_profile-\$SLURM_JOB_ID.err"
exit $EXIT_CODE
