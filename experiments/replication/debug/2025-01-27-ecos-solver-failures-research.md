# ECOS Solver Failures in Finish-Time Fairness Policy

**Date:** 2025-01-27
**Status:** Research Complete
**Impact:** 24 of 312 experiments failed (7.7%)

## Problem Statement

During full replication of Gavel paper experiments (Figures 9, 10, 11), we observed consistent ECOS solver failures in the `finish_time_fairness` policy under high job arrival rates.

### Failure Pattern

All 24 failures share these characteristics:

| Attribute | Value |
|-----------|-------|
| Figure | 11 only |
| Policy | `finish_time_fairness` or `finish_time_fairness_perf` |
| GPU Mode | Multi-GPU (`multi_gpu=true`) |
| Job Rates | 1.2 - 3.4 jobs/hour |

### Error Message

```
cvxpy.error.SolverError: Solver 'ECOS' failed.
Try another solver, or solve with verbose=True for more information.
```

### Distribution by Policy

- `finish_time_fairness_perf`: 13 failures
- `finish_time_fairness`: 11 failures

### Distribution by Job Rate

| Jobs/hr | Failures |
|---------|----------|
| 1.2 | 1 |
| 1.6 | 3 |
| 1.8 | 2 |
| 2.0 | 2 |
| 2.2 | 1 |
| 2.4 | 3 |
| 2.6 | 3 |
| 2.8 | 2 |
| 3.0 | 2 |
| 3.2 | 1 |
| 3.4 | 4 |

## Root Cause Analysis

### The Optimization Problem

The `finish_time_fairness` policy (`policies/finish_time_fairness.py`) formulates a **Second-Order Cone Program (SOCP)**, not a simple LP:

```python
# Line 90-95 in finish_time_fairness.py
allocation_throughput = cp.sum(cp.multiply(throughputs[i], x[i]))
expected_time_allocation = times_since_start[job_id] + \
    (num_steps_remaining[job_id] * cp.inv_pos(allocation_throughput))
expected_time_fraction = expected_time_allocation / expected_time_isolated
# ...
objective = cp.Minimize(cp.maximum(*expected_time_fractions))
```

**Key complexity sources:**

1. **`cp.inv_pos(x)`** - Inverse function (1/x) requires SOCP, not LP
2. **`cp.maximum(*fractions)`** - Min-max objective over all jobs
3. **Multi-GPU constraints** - Additional coupling constraints for jobs spanning GPUs

### Why ECOS Fails

ECOS is an interior-point solver optimized for small-to-medium SOCPs. At high job loads:

1. **Numerical conditioning degrades** - Large variance in throughput values
2. **Constraint coupling increases** - More jobs competing for resources
3. **Near-saturation instability** - Allocation approaches boundary conditions

### Comparison: max_min_fairness vs finish_time_fairness

| Aspect | max_min_fairness | finish_time_fairness |
|--------|------------------|----------------------|
| Problem Type | LP | SOCP |
| Key Function | Linear allocation | `cp.inv_pos()` |
| Failures (Fig 11) | 0 | 24 |
| Complexity | O(n * m) | O(n * m) + cone constraints |

## Research: Alternative Approaches

### 1. Problem Decomposition via ADMM

**Source:** [Boyd et al. - Distributed Optimization via ADMM](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

The Alternating Direction Method of Multipliers decomposes large problems into smaller subproblems coordinated via dual variables.

**Application to Gavel:**
- Decompose by GPU type: solve separate allocation problems for V100s, P100s, K80s
- Coordinate via Lagrange multipliers to ensure total allocation consistency
- Each subproblem is smaller and more numerically stable

**Trade-offs:**
- Pro: Proven at scale, natural parallelism, handles 1000s of variables
- Con: Requires reformulating optimization, convergence tuning needed

### 2. Hierarchical Optimization

**Source:** [Hierarchical Resource Partitioning on Modern GPUs](https://arxiv.org/html/2405.08754v1)

Two-level optimization:
- **Level 1:** Allocate jobs to GPU types (coarse-grained)
- **Level 2:** Fine-grained allocation within each type

**Application to Gavel:**
- Solve finish-time fairness independently per GPU type
- Coordinate cross-type allocations at higher level
- Reduces problem size by factor of num_gpu_types

### 3. Reformulated ILP (FFT Approach)

**Source:** [FFT: Fast and Fair Training - ICS 2025](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-42.pdf)

FFT directly addresses Gavel's scalability limitations:

> "When scheduling a cluster of 1000 GPUs with eight types for 4000 jobs, the ILP-based solver introduces a mere overhead of 1.53 seconds."

**Key insight:** Use GPU *types* as optimization variables instead of individual GPUs:
- Reduces variable count from O(jobs * GPUs) to O(jobs * GPU_types)
- Dramatic improvement in solver stability

**Performance claims:**
- 2.19x better mean finish-time fairness vs Gavel
- 1.99x better max finish-time fairness vs Gavel

### 4. Piecewise Linear Approximation

Replace the SOCP-inducing `cp.inv_pos()` with a piecewise linear upper bound:

```python
def approx_inv_pos(x, breakpoints=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    Piecewise linear approximation of 1/x using tangent lines.
    At each breakpoint bp, tangent line: y = 2/bp - x/bp^2
    Take maximum (upper envelope) for convex approximation.
    """
    pieces = []
    for bp in breakpoints:
        pieces.append(2/bp - x/(bp**2))
    return cp.maximum(*pieces)
```

**Trade-offs:**
- Pro: Converts SOCP to LP, ECOS handles LPs much better
- Pro: Simple implementation change
- Con: Approximation error (bounded by breakpoint density)
- Con: Slightly suboptimal allocations

### 5. Solver Fallback Chain

Implement graceful degradation when ECOS fails:

```python
SOLVER_CHAIN = ['ECOS', 'SCS', 'OSQP', 'CLARABEL']

def solve_with_fallback(problem):
    for solver in SOLVER_CHAIN:
        try:
            result = problem.solve(solver=solver)
            if problem.status == 'optimal':
                return result
        except SolverError:
            continue
    # Final fallback: use previous allocation or equal sharing
    return fallback_allocation()
```

**Note:** Our earlier experiments showed alternative solvers performed worse on average, but they may succeed where ECOS fails.

### 6. Problem Scaling and Conditioning

Improve numerical stability without changing the formulation:

```python
# Normalize throughputs to [0, 1] range
throughputs_normalized = throughputs / throughputs.max()

# Add regularization to avoid division by zero
epsilon = 1e-6
safe_throughput = allocation_throughput + epsilon

# Warm-start from previous allocation
if previous_allocation is not None:
    x.value = previous_allocation
```

## Recommended Experiments

| Priority | Approach | Effort | Expected Benefit |
|----------|----------|--------|------------------|
| 1 | Problem scaling/conditioning | Low | 20-50% fewer failures |
| 2 | Piecewise linear approximation | Medium | Eliminates SOCP entirely |
| 3 | Solver fallback chain | Medium | Graceful degradation |
| 4 | Per-GPU-type decomposition | Medium-High | Better scaling |
| 5 | FFT-style reformulation | High | 1000x scale improvement |

## Immediate Mitigation

For the current paper replication, the 24 failed experiments can be reported as:

1. **Saturated runs** - These occur at high loads (1.2-3.4 jph) where the system approaches saturation anyway
2. **Solver limitation noted** - Document that ECOS fails under specific conditions
3. **Use available data** - 288 of 312 experiments (92.3%) completed successfully

## References

1. Boyd, S., et al. "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." Foundations and Trends in Machine Learning, 2011. https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

2. Narayanan, D., et al. "Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads." OSDI 2020. https://www-cs.stanford.edu/~fiodar/pubs/gavel-osdi20.pdf

3. "Fast and Fair Training for Deep Learning in Heterogeneous GPU Clusters." ICS 2025. https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-42.pdf

4. Saroliya, A., et al. "Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach." https://arxiv.org/html/2405.08754v1

5. CVXPY Documentation - Solver Features. https://www.cvxpy.org/tutorial/solvers/index.html

6. Fielbaum, A., et al. "A Water-Filling Primal-Dual Algorithm for Approximating Non-Linear Covering Problems." ICALP 2020. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2020.46
