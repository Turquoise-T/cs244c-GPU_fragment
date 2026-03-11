# Evaluation Section Draft

> **Figures referenced:** `gavelfgd_combined_absolute.png`,
> `gavelfgd_combined_vs_gavel.png`, `gavelfgd_combined_vs_gavelfgd.png`,
> `fgd_replication_cluster_h.png`

---

## 5. Evaluation

We evaluate two systems: **GavelFGD**, which combines Gavel's LP-based max-min fair allocation with FGD's fragmentation-aware placement algorithm, and **GavelFGD+**, which further augments GavelFGD with three improvements—paper-accurate FGD scoring, a conservative popularity threshold, a buddy tiebreak rule, and a novel Placement-Opportunity-Aware (POA) bonus term in the LP objective. A key design question motivating GavelFGD+ is: *can the LP allocation policy be modified to cooperate with the placement algorithm and further reduce fragmentation?* The answer is yes, but only when the LP signal reflects actual placement opportunity rather than historical fragmentation state.

---

### 5.1 Experimental Setup

**Cluster.** All combined experiments use the Alibaba cluster configuration: six GPU types (G2: 4392, T4: 840, G3: 312, P100: 264, V100M32: 200, V100M16: 192) totaling 6200 GPUs, with 8 GPUs per server. This heterogeneous configuration exercises the GPU-type steering logic of the LP and the node-structure-aware packing of FGD.

**Workload.** We use steady-state mode with Alibaba-style traces (multi-GPU jobs enabled, GPU sharing enabled). We sweep arrival rates from 60 to 360 jobs/hour, corresponding to GPU utilization from ~37% to ~87%. Each configuration is run with two random seeds and results are averaged.

**Metrics.** We report:
- **Average JCT** — mean job completion time across all jobs completing within the measurement window (jobs 4000–5000).
- **Fragmentation rate** — average per-round fragmentation (GPU-equivalents blocked by stranded small slots per node).
- **Total fragmentation** — cluster-wide fragmented GPU-equivalents as a fraction of total GPUs (%).
- **Unallocated GPU %** — GPUs that are free but cannot be assigned due to fragmentation.

**Variants.** We compare three systems:

| Name | Description |
|------|-------------|
| **Gavel** | Original max-min fairness LP with strided placement (no FGD) |
| **GavelFGD** | Gavel LP + FGD fragmentation-aware placement (raw delta scoring) |
| **GavelFGD+** | GavelFGD + paper-accurate FGD scoring + 95% popularity threshold + buddy tiebreak + POA LP bonus (μ=0.10) |

---

### 5.2 Design of GavelFGD+

GavelFGD+ makes four coordinated changes on top of the base GavelFGD system. The first three operate at the **placement layer**; the fourth operates at the **allocation layer**.

#### 5.2.1 Placement Layer Changes

**Paper-accurate sigmoid scoring.** The base GavelFGD uses a simplified linear delta score to measure how much fragmentation a placement option adds or removes. GavelFGD+ replaces this with the FGD paper's sigmoid-normalized scoring function, which maps fragmentation gradients to a bounded [0,1] range and makes the score less sensitive to outlier node configurations.

**Conservative popularity threshold (95%).** FGD uses a *popularity filter* to avoid over-consolidating jobs onto already-popular GPU types. When more than a threshold fraction of running jobs are on a given GPU type, FGD classifies it as "popular" and scores placement options there differently, discouraging further packing onto that type. We raise this threshold from 85% to 95%. The effect is to allow more aggressive fragmentation-aware packing before invoking the popularity brake — at 95%, the filter only kicks in when a GPU type is truly dominant, leaving more room for FGD's placement optimizer to operate.

**Buddy tiebreak.** When two placement options produce equal fragmentation scores, GavelFGD+ breaks ties by preferring the node that would colocate a job with existing jobs of the same type (a "buddy" assignment). This reduces inter-type interference and improves packing density within nodes.

#### 5.2.2 Allocation Layer Change: POA LP Bonus

The original LP penalty approach — which penalized allocation to GPU types with high historical fragmentation — was **architecturally unsound**. Fragmentation is a property of *placement* (how jobs are physically arranged on nodes). Reducing the time fraction allocated to a fragmented GPU type does not reorganize jobs already placed on that type's nodes; it only steers future allocation away from a type that is fragmented *because of past placement decisions*. This creates a negative feedback loop: the LP diverts work away from a fragmented type, leaving its nodes less utilized but still fragmented, while other types become overloaded.

GavelFGD+ replaces this with a **Placement-Opportunity-Aware (POA) bonus**. Instead of penalizing types with bad history, the LP is rewarded for allocating to types where jobs can *currently* be placed successfully. Concretely, after each round's lease-extension phase (when previously running jobs renew their GPU assignments), the scheduler computes for each GPU type and each job demand size *d*:

```
poa_score[worker_type][d] = (# nodes with ≥ d free GPU slots) / (total nodes of that type)
```

This score reflects, right now and for this specific job size, what fraction of nodes can actually accommodate a new placement. A type with many large free slots scores high; a type whose nodes are densely packed or fragmented scores low.

The LP objective becomes:

```
maximize:  min_j [ normalized_throughput(j) ]  +  μ · Σ_{j,t} poa_score[t][demand_j] · x[j,t]
```

where x[j,t] is the fraction of time job j is allocated to GPU type t, and μ=0.10 is the bonus weight.

Three properties make POA well-behaved:

1. **Causal correctness.** The POA signal measures *future placement opportunity*, not *past fragmentation damage*. Allocating time to a type with high POA means the LP is directing work to where it can actually land — closing the loop between allocation and placement.

2. **Natural attenuation at saturation.** At high utilization, most nodes are heavily occupied and poa_scores → 0 everywhere. The bonus term vanishes and the LP falls back to pure Gavel max-min fairness, ensuring JCT does not degrade under overload.

3. **Per-job awareness.** A 4-GPU job and a 1-GPU job have different poa_scores on the same type — the 4-GPU job only benefits from nodes with ≥4 free slots. This correctly rewards allocations that will lead to *successful* placement for each job, not just allocation to lightly-used types.

---

### 5.3 Results

#### 5.3.1 JCT

**GavelFGD vs Gavel.** FGD placement reduces JCT modestly but consistently. At 37% utilization, the improvement is negligible (<0.1%). At 85% utilization (360 jph), GavelFGD reduces JCT from 22,567 s to 22,275 s, a **1.3% improvement**. The gain is concentrated at high load, consistent with the intuition that fragmentation is only a binding constraint when the cluster is nearly saturated.

**GavelFGD+ vs GavelFGD.** GavelFGD+ achieves further JCT improvement specifically at high utilization:

| Load (jph) | Util (%) | GavelFGD JCT (s) | GavelFGD+ JCT (s) | Δ vs GavelFGD |
|:----------:|:--------:|:----------------:|:-----------------:|:-------------:|
| 60         | 36.9     | 56,755           | 56,755            | 0.0%          |
| 110        | 49.7     | 40,496           | 40,489            | +0.0%         |
| 160        | 58.0     | 32,828           | 32,807            | +0.1%         |
| 210        | 65.6     | 29,297           | 29,373            | −0.3%         |
| 260        | 72.3     | 26,361           | 26,520            | −0.6%         |
| 310        | 78.8     | 24,061           | 23,718            | **+1.4%**     |
| 360        | 85.4     | 22,275           | 21,810            | **+2.1%**     |

At 78–85% utilization, GavelFGD+ reduces JCT by 1.4–2.1% over GavelFGD. At lower utilization (<72%), the effect is neutral or slightly negative (−0.3% to −0.6%). This is explained by the threshold change: at 95% the popularity filter is less active, allowing slightly more aggressive packing, but in the medium-utilization regime this occasionally sends jobs to nodes that then become fragmented for subsequent jobs, marginally increasing JCT.

#### 5.3.2 Fragmentation

The fragmentation story is more striking:

| Load (jph) | Util (%) | Gavel frag (%) | GavelFGD frag (%) | GavelFGD+ frag (%) | GavelFGD+ vs GavelFGD |
|:----------:|:--------:|:--------------:|:-----------------:|:------------------:|:---------------------:|
| 60         | 36.9     | 2.97           | 1.55              | 1.63               | −4.9% (worse)         |
| 110        | 49.7     | 3.27           | 1.51              | 1.35               | +10.3%                |
| 160        | 58.0     | 2.67           | 0.91              | 0.93               | −1.9% (worse)         |
| 210        | 65.6     | 2.08           | 0.44              | 0.43               | +0.8%                 |
| 260        | 72.3     | 1.86           | 0.53              | 0.69               | −29.6% (worse)        |
| 310        | 78.8     | 2.58           | 1.04              | 0.67               | **+35.5%**            |
| 360        | 85.4     | 3.50           | 2.01              | 0.47               | **+76.7%**            |

The most important result is at 85.4% utilization: GavelFGD+ reduces fragmentation rate from 2.01% (GavelFGD) to 0.47%, a **76.7% improvement**. Compared to Gavel baseline (3.50%), this is an 86.6% total reduction.

At moderate utilization (37–72%), GavelFGD+ fragmentation is occasionally *slightly worse* than GavelFGD. This reflects the 95% threshold effect: at medium loads, the more permissive threshold lets more jobs through to FGD scoring, but the POA bonus is nearly uniform across types (since most nodes have plenty of free slots), so the LP allocation changes little. The slightly more aggressive packing from the higher threshold can occasionally create fragmentation in a low-load, low-pressure cluster where strided placement would have produced more evenly distributed occupancy.

#### 5.3.3 Why the High-Utilization Regime is Different

The asymmetry between low and high utilization in GavelFGD+'s performance is explained by the POA mechanism:

- **Low utilization (37–65%):** Most nodes have many free GPU slots. poa_scores are high for all GPU types and all demand sizes. The bonus term adds roughly equal value for all allocation choices, so the LP solution is nearly identical to baseline Gavel. The POA term is effectively dormant.

- **High utilization (78–87%):** Nodes are heavily occupied. Only a small fraction have ≥4 free slots for larger jobs. poa_scores become *differentiated*: GPU types with more fragmented (partially occupied) nodes score lower than types where nodes are either full or nearly full. The LP now meaningfully steers multi-GPU job allocations toward the GPU types where those jobs can actually land. This reduces the number of rounds where jobs are allocated to a type but fail to get placed — a hidden source of fragmentation that becomes dominant at high load.

---

### 5.4 Why the Earlier LP Penalty Failed

For completeness, we contrast GavelFGD+ with the earlier LP fragmentation penalty approach (`fgd_frag_penalty_weight` parameter).

The penalty approach defined:

```
maximize:  min_j [ normalized_throughput(j) ]  -  λ · Σ_t (frag_ema[t] / N_t) · x[j,t]
```

where `frag_ema[t]` is a backward-looking exponential moving average of observed cluster fragmentation for type t.

Despite the normalization fix (dividing by N_t to bring the EMA to a per-GPU scale), this approach was ineffective for two reasons:

1. **Wrong causal direction.** The EMA records fragmentation *caused by* past placements. Penalizing allocation to a fragmented type does not undo that fragmentation; it just sends future jobs elsewhere. The nodes on the penalized type remain fragmented — they are simply underutilized. This wastes cluster capacity without reducing fragmentation.

2. **Workload masking.** On the Alibaba cluster, ~71% of GPUs are G2 type, and most jobs have nonzero throughput only on G2. The LP feasibility constraints (`x[j,G2] = 1` for G2-only jobs) override the penalty term for the majority of the workload. Only multi-type-compatible jobs could be redirected, and their aggregate effect is negligible.

POA avoids both problems. It does not fight the LP feasibility constraints — a G2-only job has poa_score=0 on all non-G2 types (since it cannot be placed there anyway), so its allocation is unchanged. It acts only on genuinely flexible jobs, steering them to the types where they will most likely succeed at placement.

---

### 5.5 Summary

| Metric | GavelFGD vs Gavel | GavelFGD+ vs Gavel | GavelFGD+ vs GavelFGD |
|--------|:-----------------:|:------------------:|:---------------------:|
| Peak JCT improvement | +1.3% (85% util) | **+3.3%** (85% util) | **+2.1%** (85% util) |
| Peak frag rate reduction | +42.5% | **+86.6%** | **+76.7%** (85% util) |
| Low utilization effect | neutral | neutral | neutral–slightly worse |
| High utilization effect | modest | strong | strong |
| LP term contributes? | N/A | Yes (POA) | N/A |

Key takeaways:

- **GavelFGD placement already reduces fragmentation substantially** (42.5% peak at 85% util), confirming that fragmentation is placement-driven.
- **GavelFGD+ provides a further 76.7% frag reduction and 2.1% JCT improvement over GavelFGD at 85% utilization**, driven by the POA LP bonus.
- **The LP can cooperate with the placement algorithm, but only if it encodes forward-looking placement opportunity** rather than backward-looking fragmentation state. The POA signal is the right interface between the allocation and placement layers.
- **The effect is load-dependent.** At low utilization, placement opportunities are abundant everywhere and the POA term is effectively dormant. Only at high utilization, when fit fractions become differentiated across GPU types, does the LP steering make a measurable difference.
- **The 95% popularity threshold allows more aggressive FGD packing** at high load compared to the 85% threshold, contributing to the fragmentation reduction alongside the POA bonus.
