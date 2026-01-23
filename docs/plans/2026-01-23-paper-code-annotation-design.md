# Paper-to-Code Inline Annotation Design

CS244C Project - Gavel Validation
Date: 2026-01-23

## Overview

This document describes our approach to annotating Gavel's source code with inline comments that embed the relevant paper text (equations + intuition) directly alongside the implementation. This ensures paper claims and code are perfectly aligned and makes the codebase self-documenting for validation purposes.

## Goals

1. **Traceability**: Every key algorithm in code links back to its paper description
2. **Validation**: Easy to verify implementation matches paper claims
3. **Education**: Future readers can understand the "why" without switching between paper and code
4. **Discrepancy Detection**: Surface any differences between paper and implementation

## Annotation Format

```python
# ══════════════════════════════════════════════════════════════════════════════
# PAPER REFERENCE: Section X.Y - [Section Title]
# ══════════════════════════════════════════════════════════════════════════════
# "[Exact quote from paper providing intuition]"
#
# Equation/Algorithm:
#   [Mathematical formulation or pseudocode from paper]
#
# Constraints:
#   [Any constraints mentioned in paper]
# ══════════════════════════════════════════════════════════════════════════════
def function_name(self, ...):
    # PAPER: "[Inline quote for specific implementation detail]"
    implementation_code_here
```

### Annotation Elements

| Element | Purpose | Example |
|---------|---------|---------|
| Section header | Link to paper location | `Section 4.1 - Max-Min Fairness` |
| Block quote | Intuition/motivation | `"The classical LAS policy implements..."` |
| Equation | Mathematical formulation | `MaximizeX min_m (1/w_m) * throughput(m,X)` |
| Constraints | Validity conditions | `0 <= X_mj <= 1` |
| Inline comment | Line-level mapping | `# PAPER: "priorities are updated as rounds complete"` |
| Discrepancy note | Implementation differences | `# NOTE: Paper says X, code does Y because...` |

## File Mapping

### Core Algorithm Files (Full Annotation)

| Priority | File | Paper Section | Key Concepts |
|----------|------|---------------|--------------|
| 1 | `policy.py` | §3.1 | Effective throughput, allocation matrix X, constraints |
| 2 | `max_min_fairness.py` | §4.1 | LAS policy, weighted fairness, optimization problem |
| 3 | `max_min_fairness_water_filling.py` | §4.3 | Hierarchical policies, water filling algorithm |
| 4 | `finish_time_fairness.py` | §4.2 | Themis ρ metric, finish-time optimization |
| 5 | `fifo.py` | §4.2 | FIFO as throughput maximization |
| 6 | `min_total_duration.py` | §4.2 | Makespan minimization |
| 7 | `max_sum_throughput.py` | §4.2 | Cost minimization, SLO constraints |
| 8 | `scheduler.py` | §5 | Round-based mechanism, Algorithm 1, priority computation |
| 9 | `throughput_estimator.py` | §6 | Matrix completion, fingerprinting |
| 10 | `job_id_pair.py` | §3.1 | Space sharing job combinations |

### Supporting Files (Key Section Annotation)

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

### Baseline Comparison Files (Light Annotation)

| File | Paper Section | Purpose |
|------|---------------|---------|
| `allox.py` | §7.3, §8 | AlloX baseline for comparison |
| `gandiva.py` | §7.3, §8 | Gandiva baseline for comparison |
| `max_min_fairness_strategy_proof.py` | §4.4 | Strategy proofness extension |

### Data Files (Document Schema)

| File | Paper Section | Contents |
|------|---------------|----------|
| `simulation_throughputs.json` | §3.1, Table 2 | Throughput matrix T for simulation |
| `physical_cluster_throughputs.json` | §7.1 | Throughput matrix T for physical cluster |

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

### Phase 1: Core Abstraction (policy.py)
- [ ] Add effective throughput definition and equations
- [ ] Document allocation matrix X structure
- [ ] Document all three constraints
- [ ] Add space sharing extension formulas

### Phase 2: Primary Policy (max_min_fairness.py)
- [ ] Add LAS objective function
- [ ] Document X^equal normalization
- [ ] Add scale factor adjustment for distributed jobs
- [ ] Document constraint formulation in cvxpy

### Phase 3: Other Policies
- [ ] `finish_time_fairness.py` - Themis ρ metric
- [ ] `fifo.py` - Priority-weighted throughput
- [ ] `min_total_duration.py` - Makespan objective
- [ ] `max_sum_throughput.py` - Cost objective

### Phase 4: Hierarchical (max_min_fairness_water_filling.py)
- [ ] Document water filling algorithm
- [ ] Add entity/job weight handling
- [ ] Document bottleneck detection MILP

### Phase 5: Scheduling Mechanism (scheduler.py)
- [ ] Add Algorithm 1 pseudocode
- [ ] Document priority computation
- [ ] Add round-based allocation logic
- [ ] Document job placement strategy

### Phase 6: Supporting Components
- [ ] `throughput_estimator.py` - Matrix completion
- [ ] `gavel_iterator.py` - Application API
- [ ] `job_id_pair.py` - Space sharing combinations
- [ ] `job.py`, `lease.py` - State management

### Phase 7: Baselines and Data
- [ ] `allox.py`, `gandiva.py` - Baseline descriptions
- [ ] `simulation_throughputs.json` - Schema documentation

## Success Criteria

1. **Complete Coverage**: All 22 files have appropriate annotations
2. **Equation Accuracy**: Every paper equation appears in the relevant code file
3. **Discrepancy Documentation**: Any differences between paper and code are noted
4. **Validation Ready**: Annotations enable claim-by-claim verification
5. **Self-Documenting**: New reader can understand algorithm from code + comments alone

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

## Notes

- Verbosity level: Moderate (equations + surrounding context paragraph)
- Preserve existing code comments where relevant
- Use `# NOTE:` prefix for implementation differences
- Use `# TODO:` prefix for unclear mappings needing investigation
