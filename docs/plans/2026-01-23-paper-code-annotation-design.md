# Paper-to-Code Inline Annotation Design

CS244C Project - Gavel Validation
Date: 2026-01-23
Updated: 2026-01-23 (post-Codex review)

## Overview

This document describes our approach to annotating Gavel's source code with inline comments that embed the relevant paper text (equations + intuition) directly alongside the implementation. This ensures paper claims and code are perfectly aligned and makes the codebase self-documenting for validation purposes.

## Goals

1. **Traceability**: Every key algorithm in code links back to its paper description
2. **Validation**: Easy to verify implementation matches paper claims
3. **Education**: Future readers can understand the "why" without switching between paper and code
4. **Discrepancy Detection**: Surface any differences between paper and implementation

## Design Decisions

Based on Codex review and discussion:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quote style | **Verbatim** | Exact text for validation, no interpretation drift |
| Tag format | **Compact** | `# PAPER[§X.Y]` - grep-friendly, minimal footprint |
| Sidecar file | **None** | Everything inline, self-contained |
| Execution order | **Code flow** | Follow runtime path: scheduler → policy → policies |
| JSON files | **Separate doc** | `docs/data-schema.md` to avoid polluting data |
| Equation scope | **Implemented + key definitions** | Mark definitions with `[def]` |
| Copyright | **Fair use** | Public repo, educational commentary |

## Annotation Format

### Python Files

**Function-level annotation:**
```python
# PAPER[§4.1] "MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)"
# PAPER[§4.1] Constraints: 0 <= X_mj <= 1, Σ_j X_mj <= 1, Σ_m X_mj * scale_factor_m <= num_workers_j
def get_allocation(self, unflattened_throughputs, scale_factors, ...):
    # PAPER[§4.1] "weighted max-min fairness policy with per-user weights w_m"
    ...
```

**Definition annotation:**
```python
# PAPER[§3.1|def] "effective throughput: time-weighted average throughput across accelerators"
# PAPER[§3.1|def] throughput(m,X) = Σ_j T_mj * X_mj
def get_effective_throughputs(self, X, T):
    ...
```

**Discrepancy annotation:**
```python
# NOTE: Paper says "6-minute rounds" but code uses configurable `round_duration` parameter
```

**Investigation needed:**
```python
# TODO[§5]: Unclear how lease renewal interacts with priority computation
```

### Annotation Tag Reference

| Tag | Purpose | Example |
|-----|---------|---------|
| `# PAPER[§X.Y]` | Direct quote/equation from section | `# PAPER[§4.1] "max-min fairness over..."` |
| `# PAPER[§X.Y\|def]` | Definition (conceptual, not always implemented) | `# PAPER[§3.1\|def] "effective throughput..."` |
| `# PAPER[§X.Y\|alg]` | Algorithm reference | `# PAPER[§5\|alg] Algorithm 1: SCHEDULE_JOBS` |
| `# PAPER[§X.Y\|eq]` | Equation reference | `# PAPER[§4.1\|eq] Eq. 1: constraint formulation` |
| `# NOTE:` | Implementation differs from paper | `# NOTE: Paper says X, code does Y` |
| `# TODO[§X.Y]:` | Needs investigation | `# TODO[§5]: verify round duration logic` |

## File Mapping

### Execution Order (Code Flow)

Files are annotated following the runtime execution path:

| Order | File | Paper Section | Key Concepts |
|-------|------|---------------|--------------|
| 1 | `scheduler.py` | §5 | Round-based mechanism, Algorithm 1, priority computation |
| 2 | `policy.py` | §3.1 | Effective throughput, allocation matrix X, constraints |
| 3 | `max_min_fairness.py` | §4.1 | LAS policy, weighted fairness, optimization problem |
| 4 | `max_min_fairness_water_filling.py` | §4.3 | Hierarchical policies, water filling algorithm |
| 5 | `finish_time_fairness.py` | §4.2 | Themis ρ metric, finish-time optimization |
| 6 | `fifo.py` | §4.2 | FIFO as throughput maximization |
| 7 | `min_total_duration.py` | §4.2 | Makespan minimization |
| 8 | `max_sum_throughput.py` | §4.2 | Cost minimization, SLO constraints |
| 9 | `throughput_estimator.py` | §6 | Matrix completion, fingerprinting |
| 10 | `job_id_pair.py` | §3.1 | Space sharing job combinations |

### Supporting Files

| File | Paper Section | Key Concepts |
|------|---------------|--------------|
| `job.py` | §2.1, §6 | Job state, checkpointing |
| `job_table.py` | §3 | Job management |
| `lease.py` | §5 | Lease renewal mechanism |
| `job_template.py` | §7 | Job types from evaluation |
| `gavel_iterator.py` | §6 | Application API, GavelIterator |
| `isolated.py` | §4.1 | X^equal baseline allocation |
| `proportional.py` | - | Proportional sharing baseline |
| `utils.py` | §3.1 | Placement sensitivity helpers |
| `worker.py` | §5 | Worker state management |

### Baseline Comparison Files

| File | Paper Section | Purpose |
|------|---------------|---------|
| `allox.py` | §7.3, §8 | AlloX baseline for comparison |
| `gandiva.py` | §7.3, §8 | Gandiva baseline for comparison |
| `max_min_fairness_strategy_proof.py` | §4.4 | Strategy proofness extension |

### Data Files

Documented in `docs/data-schema.md` (separate file):

| File | Paper Section | Contents |
|------|---------------|----------|
| `simulation_throughputs.json` | §3.1, Table 2 | Throughput matrix T for simulation |
| `physical_cluster_throughputs.json` | §7.1 | Throughput matrix T for physical cluster |
| `traces/msr/cluster_specs.json` | §7.1 | Cluster configuration |

## Paper-to-Code Concept Mapping

### Section 3.1: Heterogeneity-Aware Policies

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| Allocation matrix X | `policy.py` | `X[m,j]` = fraction of time job m spends on accelerator j |
| Throughput matrix T | `simulation_throughputs.json` | `T[m,j]` = throughput of job m on accelerator j |
| Effective throughput | `policy.py:get_effective_throughputs()` | `throughput(m,X) = Σ_j T_mj * X_mj` |
| Constraint (1) | `policy.py` | `0 <= X_mj <= 1` |
| Constraint (2) | `policy.py` | `Σ_j X_mj <= 1` |
| Constraint (3) | `policy.py` | `Σ_m X_mj * scale_factor_m <= num_workers_j` |
| Space sharing | `job_id_pair.py` | Job combinations `C_m` |
| Placement sensitivity | `utils.py` | Consolidated vs unconsolidated throughputs |

### Section 4.1: Max-Min Fairness

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| LAS objective | `max_min_fairness.py` | `MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)` |
| X^equal | `isolated.py` | Equal time share on each worker |
| Scale factor adjustment | `max_min_fairness.py` | Multiply by `scale_factor_m` for multi-GPU jobs |

### Section 4.2: Other Policies

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| Makespan | `min_total_duration.py` | `MinimizeX max_m num_steps_m / throughput(m,X)` |
| Finish-time fairness ρ | `finish_time_fairness.py` | `ρ(m,X) = (t_m + remaining/throughput) / (t_isolated + remaining/throughput_isolated)` |
| FIFO | `fifo.py` | `MaximizeX Σ_m throughput(m,X)/throughput(m,X^fastest) * (M-m)` |
| Cost minimization | `max_sum_throughput.py` | `MaximizeX Σ_m throughput(m,X) / Σ_m(Σ_j cost_j * X_mj)` |

### Section 4.3: Hierarchical Policies

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| Water filling | `max_min_fairness_water_filling.py` | Iterative LP solving until all jobs bottlenecked |
| Entity weights | `max_min_fairness_water_filling.py` | `w_s` for entity, `w_m^job` for jobs |
| Bottleneck detection | `max_min_fairness_water_filling.py` | Jobs that can't improve without hurting others |

### Section 5: Scheduling Mechanism

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| Round-based scheduling | `scheduler.py` | 6-minute rounds |
| Priority computation | `scheduler.py` | `priorities = X^opt / rounds_received` (element-wise) |
| Algorithm 1 | `scheduler.py:schedule_jobs()` | Greedy job selection by priority |
| Lease renewal | `lease.py`, `gavel_iterator.py` | Jobs can extend if same worker next round |

### Section 6: Implementation

| Paper Concept | Code Location | Equation/Description |
|---------------|---------------|----------------------|
| Throughput estimator | `throughput_estimator.py` | Matrix completion + fingerprinting |
| GavelIterator | `gavel_iterator.py` | `train_loader`, `load_checkpoint`, `save_checkpoint` |
| cvxpy solver | `policies/*.py` | Optimization problem solving |

## Execution Plan

### Phase 1: Core Mechanism (scheduler.py)
- [ ] Add Algorithm 1 pseudocode from paper
- [ ] Document priority computation formula
- [ ] Add round-based allocation logic
- [ ] Document job placement strategy

### Phase 2: Core Abstraction (policy.py)
- [ ] Add effective throughput definition and equations
- [ ] Document allocation matrix X structure
- [ ] Document all three constraints
- [ ] Add space sharing extension formulas

### Phase 3: Primary Policy (max_min_fairness.py)
- [ ] Add LAS objective function
- [ ] Document X^equal normalization
- [ ] Add scale factor adjustment for distributed jobs
- [ ] Document constraint formulation in cvxpy

### Phase 4: Other Policies
- [ ] `max_min_fairness_water_filling.py` - Water filling algorithm
- [ ] `finish_time_fairness.py` - Themis ρ metric
- [ ] `fifo.py` - Priority-weighted throughput
- [ ] `min_total_duration.py` - Makespan objective
- [ ] `max_sum_throughput.py` - Cost objective

### Phase 5: Throughput Estimation
- [ ] `throughput_estimator.py` - Matrix completion
- [ ] `job_id_pair.py` - Space sharing combinations

### Phase 6: Supporting Components
- [ ] `gavel_iterator.py` - Application API
- [ ] `job.py`, `job_table.py`, `lease.py` - State management
- [ ] `utils.py`, `worker.py` - Helpers

### Phase 7: Documentation
- [ ] Create `docs/data-schema.md` for JSON files
- [ ] `allox.py`, `gandiva.py` - Baseline descriptions

## Success Criteria

1. **Complete Coverage**: All 22 Python files have appropriate annotations
2. **Equation Accuracy**: All implemented equations + key definitions appear in code
3. **Discrepancy Documentation**: Any differences between paper and code are noted with `# NOTE:`
4. **Validation Ready**: Annotations enable claim-by-claim verification
5. **Self-Documenting**: New reader can understand algorithm from code + comments alone
6. **Data Documentation**: `docs/data-schema.md` documents all JSON file schemas

## Verification Checklist

After annotation, verify these paper claims are traceable to code:

- [ ] Effective throughput normalizes job throughput across GPU types (§3.1)
- [ ] Policies expressed as optimization problems over effective throughput (§4)
- [ ] Round-based allocation achieves target allocations (§5)
- [ ] Max-min fairness maximizes minimum effective throughput (§4.1)
- [ ] Finish-time fairness equalizes job completion times (§4.2)
- [ ] LAS prioritizes jobs with less accumulated service (§4.1)
- [ ] Round duration balances accuracy vs overhead (§5)
- [ ] Heterogeneity-aware placement improves throughput (§3.1)

## Paper Reference

- **Paper**: "Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads"
- **Authors**: Narayanan, Santhanam, Kazhamiaka, Phanishayee, Zaharia
- **Venue**: OSDI 2020
- **Version**: USENIX open access PDF

## Notes

- Verbosity: Verbatim quotes for validation accuracy
- Format: Compact `# PAPER[§X.Y]` tags
- Preserve existing code comments where relevant
- Use `# NOTE:` prefix for implementation differences
- Use `# TODO[§X.Y]:` prefix for unclear mappings needing investigation
- Public repo: Fair use for educational commentary applies
