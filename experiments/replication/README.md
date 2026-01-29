# Gavel Replication Experiments

Replication of Figures 9, 10, and 11 from the Gavel paper (OSDI 2020).

## Results Summary

- **504 total experiments** across 3 figures
- **Figures 9, 10, 11**: JCT vs arrival rate for single-GPU and multi-GPU workloads
- Results confirm Gavel's heterogeneity-aware scheduling provides 20-70% JCT improvement

## Directory Structure

```
replication/
├── configs/          # Experiment configurations (JSON)
│   ├── experiments_full.json      # Main 312 experiments
│   ├── experiments_lowrate.json   # Low-rate experiments (scaled windows)
│   ├── experiments_extended.json  # Extended multi-GPU rates
│   └── ...
├── results/          # Experiment outputs
│   ├── results_combined.csv       # Final combined results (504 experiments)
│   ├── results_full.csv           # Original 312 experiments
│   └── results_full/              # Raw experiment directories
├── figures/          # Generated plots
│   ├── gavel_replication_figures.png  # Combined Figs 9, 10, 11
│   └── fig{9,10,11}_replication.png   # Individual figures
├── scripts/          # Experiment scripts
│   ├── run_benchmark.py           # Main experiment runner
│   ├── generate_*.py              # Experiment config generators
│   └── plot_results.py            # Plot generation
├── slurm/            # SLURM submission scripts for FarmShare
├── debug/            # Debug tools and investigation notes
│   ├── extract_telemetry.py       # Log parser for visualization
│   ├── telemetry_viewer.html      # Interactive telemetry dashboard
│   └── 2025-01-27-ecos-solver-failures-research.md
└── legacy/           # Earlier experiment versions (preserved for history)
```

## Key Findings

### Figure 9: Single-GPU Jobs (Max-Min Fairness)
- Rate range: 0.4-8.0 jobs/hr
- Gavel improvement: 20-70% depending on load
- Saturation begins around 5 jobs/hr

### Figures 10 & 11: Multi-GPU Jobs
- Rate range: 0.2-4.4 jobs/hr
- Gavel improvement: 10-30% at moderate loads
- Saturation begins around 3 jobs/hr (earlier than single-GPU due to placement constraints)

## Running Experiments

### Local Testing
```bash
cd experiments/replication/scripts
python run_benchmark.py --index 0 --experiments-file ../configs/experiments_full.json --output-dir ../results/test
```

### FarmShare (SLURM)
```bash
# Sync to FarmShare
rsync -avz experiments/replication/ farmshare:~/gavel/experiments/replication/

# Submit batch job
ssh farmshare "cd ~/gavel/experiments/replication && sbatch slurm/submit_full.sbatch"
```

## Regenerating Plots
```bash
cd experiments/replication/scripts
python plot_results.py
# Output: ../figures/gavel_replication_figures.png
```

## Bugfixes Applied

During replication, we identified and fixed:
1. **V100 hardcoding** in policy throughput lookups (affected heterogeneous clusters)
2. **ECOS solver failures** at certain arrival rates (added SCS fallback)

See `debug/2025-01-27-ecos-solver-failures-research.md` for investigation details.
