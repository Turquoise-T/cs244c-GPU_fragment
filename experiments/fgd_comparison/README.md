# FGD vs Strided Comparison

Run and plot FGD vs Strided placement under GPU sharing (fractional `gpu_milli`).

## Run experiments

```bash
cd experiments/fgd_comparison
# Full 48 runs (2 strategies × 8 rates × 3 seeds)
python3 scripts/run_fgd_experiment.py --all

# Or a subset
python3 scripts/run_fgd_experiment.py --range 0 16
python3 scripts/plot_fgd_results.py
```

Results: `results/results_fgd.csv`, figures in `figures/`.

## Visualize in GPU Scheduling Visualizer

To compare Strided vs FGD **side-by-side** in the web visualizer:

1. **Run experiments with logs kept** (same config, add `--keep-logs`):

   ```bash
   # One pair (e.g. rate=5 jobs/hr, seed=0): run index 4 (strided) and 28 (fgd)
   python3 scripts/run_fgd_experiment.py --index 4 --keep-logs
   python3 scripts/run_fgd_experiment.py --index 28 --keep-logs

   # Or run all 48 with logs (larger disk usage)
   python3 scripts/run_fgd_experiment.py --all --keep-logs
   ```

   Logs are written to `results/logs/<name>/simulation.log`.

2. **Build .viz.bin files** for the visualizer:

   ```bash
   # One (rate, seed) pair
   python3 scripts/build_fgd_viz.py --rate 5 --seed 0

   # All pairs that have both strided and fgd logs
   python3 scripts/build_fgd_viz.py --all

   # List available pairs
   python3 scripts/build_fgd_viz.py --list
   ```

   Output: `figures/viz/fgd_<rate>jph_s<seed>_strided.viz.bin` and `_fgd.viz.bin`.

3. **Open the visualizer** and load the two files:

   ```bash
   cd ../../gpu-scheduling-viz
   python3 -m http.server 8000
   # Browser: http://localhost:8000
   # Use file pickers: Simulation 1 = *_strided.viz.bin, Simulation 2 = *_fgd.viz.bin
   ```

You get Utilization, JCT, Queue, and JCT CDF charts over time, and a GPU grid (per-GPU job assignment is not logged, so grid shows utilization by type only).
