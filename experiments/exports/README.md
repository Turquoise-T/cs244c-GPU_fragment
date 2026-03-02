# Exported Results for Cross-Comparison

These CSVs contain our replication results for the Gavel (OSDI 2020) and FGD (ATC 2023) papers, alongside reference values traced from the original paper figures. They are intended for students in CS 244C to compare their own experiment results against ours and the published paper data.

## Files

| File | Paper | Contents |
|------|-------|----------|
| `gavel_results.csv` | Gavel (OSDI 2020) | Average JCT (hours) vs job arrival rate for Figs 9a, 10a, 11a |
| `fgd_results.csv` | FGD (ATC 2023) | Fragmentation metrics vs demand for all 6 policies |

## Gavel CSV Format

| Column | Description |
|--------|-------------|
| `figure` | Which paper figure: `fig9` (single-GPU, MMF), `fig10` (multi-GPU, MMF), `fig11` (multi-GPU, FTF) |
| `jobs_per_hr` | Input job arrival rate |
| `policy` | `gavel` (heterogeneity-aware) or `baseline` (heterogeneity-agnostic) |
| `mean_jct_hours` | Our measured mean JCT across seeds |
| `std_jct_hours` | Standard deviation across seeds |
| `n_seeds` | Number of seeds averaged |
| `paper_jct_hours` | Reference value traced from the paper figure (null if no matching x-point) |

**Cluster:** 36 V100 + 36 P100 + 36 K80 GPUs (108 total), matching the paper.

## FGD CSV Format

| Column | Description |
|--------|-------------|
| `metric` | One of: `frag_rate_pct`, `frag_over_total_pct`, `unalloc_gpu_pct`, `occupied_nodes` |
| `demand_pct` | Arrived workloads as % of cluster GPU capacity (0-120) |
| `policy` | `random`, `dotprod`, `gpuclustering`, `gpupacking`, `bestfit`, or `fgd` |
| `mean_value` | Our measured mean across seeds |
| `std_value` | Standard deviation across seeds |
| `n_seeds` | Number of seeds averaged |
| `paper_value` | Reference value traced from the paper figure (blank if unavailable) |

**Cluster:** 6,212 GPUs across 1,213 nodes (Alibaba trace), matching the paper's Cluster H.

**Metrics:**
- `frag_rate_pct` -- Fig 7a: % of unallocated GPUs that are fragmented
- `frag_over_total_pct` -- Fig 7b: fragmented GPUs as % of total cluster GPUs
- `unalloc_gpu_pct` -- Fig 9a: % of GPUs not allocated
- `occupied_nodes` -- Fig 9b: number of nodes with at least one allocated GPU

## Quick Start (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# -- Gavel: plot Fig 9a --
df = pd.read_csv('gavel_results.csv')
fig9 = df[df.figure == 'fig9']
for policy, g in fig9.groupby('policy'):
    plt.errorbar(g.jobs_per_hr, g.mean_jct_hours, yerr=g.std_jct_hours,
                 label=f'{policy} (ours)', capsize=3)
# Overlay paper reference
for policy, g in fig9.groupby('policy'):
    ref = g.dropna(subset=['paper_jct_hours'])
    plt.plot(ref.jobs_per_hr, ref.paper_jct_hours, '--', alpha=0.5,
             label=f'{policy} (paper)')
plt.xlabel('Job Arrival Rate (jobs/hr)')
plt.ylabel('Average JCT (hours)')
plt.legend()
plt.show()

# -- FGD: plot Fig 7b (frag/total) --
df = pd.read_csv('fgd_results.csv')
fig7b = df[df.metric == 'frag_over_total_pct']
for policy, g in fig7b.groupby('policy'):
    plt.plot(g.demand_pct, g.mean_value, label=policy)
plt.xlabel('Arrived Workloads (% capacity)')
plt.ylabel('Frag / Total (%)')
plt.legend()
plt.show()
```

## Regenerating

To regenerate from the raw experiment data:

```bash
cd experiments/
python export_results.py
```

## Paper Reference Data

The `paper_*_hours` / `paper_value` columns contain values manually traced from the published figures using a graph digitizer tool. They replace earlier OCR-based approximations which had significant errors. Only 3 FGD policies were traced (random, clustering, fgd) as a representative subset; others have blank paper values.
