#!/bin/bash
# Setup script for FarmShare
# Run this once after uploading the gavel directory to FarmShare

set -e

GAVEL_DIR="$HOME/gavel"
CLUSTER_DIR="$GAVEL_DIR/cluster"

echo "=== Setting up Gavel on FarmShare ==="

# Check if gavel directory exists
if [ ! -d "$GAVEL_DIR" ]; then
    echo "Error: $GAVEL_DIR not found"
    echo "Please upload the gavel directory to your home folder first:"
    echo "  rsync -avz --exclude='.venv' --exclude='__pycache__' /path/to/gavel <sunetid>@rice.stanford.edu:~/"
    exit 1
fi

# Load Python
module load python/3.9

# Create virtual environment
echo "Creating virtual environment..."
cd "$GAVEL_DIR"
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$CLUSTER_DIR/requirements.txt"

# Generate experiments
echo "Generating experiment configurations..."
cd "$CLUSTER_DIR"
python generate_experiments.py

# Create log directories
mkdir -p slurm_logs results

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To submit all experiments:"
echo "  cd $CLUSTER_DIR"
echo "  sbatch submit_farmshare.sbatch"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To check a specific experiment's output:"
echo "  cat slurm_logs/gavel-<jobid>_<index>.out"
