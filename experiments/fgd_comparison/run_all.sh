#!/bin/bash
# Run the full FGD vs Strided comparison pipeline (all experiments).
#
# Usage:
#   cd experiments/fgd_comparison
#   bash run_all.sh          # runs all configured experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"

echo "========================================="
echo "  FGD vs Strided Comparison Pipeline"
echo "========================================="

# Step 1: Generate configs
echo ""
echo "[Step 1] Generating experiment configs..."
python3 "$SCRIPT_DIR/generate_fgd_experiments.py"

# Step 2: Run experiments
echo ""
echo "[Step 2] Running experiments..."
python3 "$SCRIPT_DIR/run_fgd_experiment.py" --all

# Step 3: Plot results
echo ""
echo "[Step 3] Plotting results..."
python3 "$SCRIPT_DIR/plot_fgd_results.py"

echo ""
echo "========================================="
echo "  Pipeline Complete!"
echo "  Results: results/results_fgd.csv"
echo "  Figures: figures/"
echo "========================================="
