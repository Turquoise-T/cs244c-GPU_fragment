# Gavel Paper-to-Code Mapping

This document traces how concepts from the Gavel OSDI 2020 paper map to the implementation code.

**Paper:** Gavel: Heterogeneity-Aware Cluster Scheduling for Machine Learning
**Total Annotations:** 73 across 14 files

---

## Table of Contents

1. [Section 2.1: Workload Model](#section-21-workload-model)
2. [Section 3.1: Problem Formulation](#section-31-problem-formulation)
3. [Section 4.1: Max-Min Fairness](#section-41-max-min-fairness)
4. [Section 4.2: Other Scheduling Objectives](#section-42-other-scheduling-objectives)
5. [Section 4.3: Hierarchical Policies](#section-43-hierarchical-policies)
6. [Section 5: Round-Based Scheduling](#section-5-round-based-scheduling)
7. [Section 6: Throughput Estimation](#section-6-throughput-estimation)

---

## Section 2.1: Workload Model

**Paper Concept:** Jobs have types (model architecture), total training steps, and resource requirements.

### job.py

```python
# PAPER[§2.1] "Job state includes model type, total training steps, and resource requirements"
class Job:
    def __init__(self, job_id, job_type, command, working_directory,
                 num_steps_arg, total_steps, duration, scale_factor=1,
                 priority_weight=1, SLO=None, needs_data_dir=False):
```

**Key attributes:**
- `job_type`: Model architecture (e.g., ResNet-18, Transformer)
- `total_steps`: Total training iterations needed
- `scale_factor`: Workers needed for distributed training
- `priority_weight`: Weight for fairness policies
- `SLO`: Deadline constraint

---

## Section 3.1: Problem Formulation

**Paper Concept:** Gavel formulates scheduling as an optimization problem over an allocation matrix X.

### Core Definitions (policy.py)

```python
# PAPER[§3.1|def] "allocation matrix X where X_mj = fraction of time job m spends on accelerator j"
# PAPER[§3.1|def] "effective throughput: throughput(m,X) = Σ_j T_mj * X_mj"
class Policy:
```

**Allocation Matrix X:**
- `X[m,j]` = fraction of time job m spends on accelerator type j
- Range: 0 <= X_mj <= 1
- Each row sums to at most 1 (job uses at most 100% of time)

**Effective Throughput:**
```
throughput(m, X) = Σ_j T_mj * X_mj
```
Where T_mj is the throughput of job m on accelerator type j.

### Scale Factor (policy.py:21, job.py:2)

```python
# PAPER[§3.1|def] "scale_factor s_m: number of workers needed for distributed training"
def scale_factors_array(self, scale_factors, job_ids, m, n):
```

Jobs using data parallelism need multiple GPUs. The scale factor s_m indicates how many workers a job requires.

### Constraints (policy.py:59-61)

```python
# PAPER[§3.1|eq] Constraint (1): 0 <= X_mj <= 1
# PAPER[§3.1|eq] Constraint (2): Σ_j X_mj <= 1 (job cannot use more than 100% of time)
# PAPER[§3.1|eq] Constraint (3): Σ_m X_mj * scale_factor_m <= num_workers_j (capacity)
def get_base_constraints(self, x, scale_factors_array):
    return [
        x >= 0,
        cp.sum(cp.multiply(scale_factors_array, x), axis=0) <= self._num_workers,
        cp.sum(x, axis=1) <= 1,
    ]
```

### Space Sharing (job_id_pair.py, policy.py:169-170)

```python
# PAPER[§3.1] "Space sharing: multiple jobs can share a single accelerator"
# PAPER[§3.1] "JobIdPair represents either a single job or a pair of co-located jobs"
class JobIdPair():
```

Multiple jobs can share a GPU in a time-multiplexed manner. The `JobIdPair` class represents either:
- A single job (job_id, None)
- A pair of co-located jobs (job_id_1, job_id_2)

---

## Section 4.1: Max-Min Fairness

**Paper Concept:** Maximize the minimum normalized throughput across all jobs.

### Objective (max_min_fairness.py:10-12)

```python
# PAPER[§4.1] "MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)"
# PAPER[§4.1] "Max-min fairness: maximize minimum normalized throughput across jobs"
# PAPER[§4.1|def] "throughput(m,X^equal) = proportional_throughputs (baseline for normalization)"
```

**Optimization Problem:**
```
Maximize_X  min_m  (1/w_m) * throughput(m,X) / throughput(m,X^equal)
```

Where:
- w_m = priority weight of job m
- X^equal = equal time share allocation (baseline)
- throughput(m,X^equal) = what job m would get with fair share

### X^equal Baseline (isolated.py:8-10)

```python
# PAPER[§4.1|def] "X^equal: equal time share baseline allocation"
# PAPER[§4.1] "Each job receives equal fraction of each worker type"
# PAPER[§4.1] "Used to normalize effective throughput in fairness policies"
class IsolatedPolicy(Policy):
```

The isolated/proportional policy gives each job an equal fraction of each worker type. This serves as the fairness baseline.

### Heterogeneity-Aware vs Agnostic (max_min_fairness.py:27, 39)

```python
# PAPER[§4.1] "Heterogeneity-agnostic variant: sets all throughputs to 1.0"
new_unflattened_throughputs[job_id][worker_type] = 1.0

# PAPER[§4.1] MaxMinFairness_Perf: heterogeneity-aware variant using actual throughputs
class MaxMinFairnessPolicyWithPerf(Policy):
```

- **Heterogeneity-agnostic:** Treats all GPU types as equivalent (throughput = 1.0)
- **Heterogeneity-aware (_Perf):** Uses actual measured throughputs

### Scale Factor Adjustment (max_min_fairness.py:70)

```python
# PAPER[§4.1] "scale_factor adjustment: distributed jobs counted as scale_factor jobs"
# Multiply throughputs by scale_factors to ensure that scale_factor
# is taken into account while allocating times to different jobs.
```

---

## Section 4.2: Other Scheduling Objectives

### Finish-Time Fairness / Themis (finish_time_fairness.py)

```python
# PAPER[§4.2] "Finish-time fairness (Themis): equalize completion-time ratio ρ across jobs"
# PAPER[§4.2|eq] "rho(m,X) = (t_m + remaining/throughput) / (t_isolated + remaining/throughput_isolated)"
# PAPER[§4.2] "Objective: MinimizeX max_m rho(m,X)"
```

**The ρ (rho) Metric:**
```
ρ(m,X) = expected_completion_time / expected_completion_time_if_isolated
       = (t_m + remaining_steps / throughput(m,X)) / (t_isolated + remaining_steps / throughput_isolated)
```

**Implementation (finish_time_fairness.py:85-101):**
```python
# PAPER[§4.2] "Cumulative isolated time: tracks time job would have spent in isolation"
if job_ids[i] not in self._cumulative_isolated_time:
    self._cumulative_isolated_time[job_ids[i]] = 0

# PAPER[§4.2] expected_time_isolated = t_isolated + remaining / throughput_isolated
expected_time_isolated = self._cumulative_isolated_time[job_ids[i]] + \
    (num_steps_remaining[job_ids[i]] / isolated_throughputs[i])

# PAPER[§4.2] expected_time_allocation = t_m + remaining / throughput(m,X)
expected_time_allocation = times_since_start[job_ids[i]] + \
    (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))

# PAPER[§4.2] rho = expected_time_allocation / expected_time_isolated
expected_time_fraction = expected_time_allocation / expected_time_isolated
```

### FIFO Policy (fifo.py:10-12)

```python
# PAPER[§4.2] "FIFO: process jobs in arrival order"
# PAPER[§4.2|eq] "MaximizeX Σ_m throughput(m,X)/throughput(m,X^fastest) * (M-m)"
# PAPER[§4.2] "Equivalent to throughput maximization with arrival-order priority"
```

Jobs are processed in order of arrival. Earlier jobs get higher priority weight (M-m where m is arrival order).

### Cost Minimization (max_sum_throughput.py:9-11)

```python
# PAPER[§4.2] "Cost minimization: maximize throughput per unit cost"
# PAPER[§4.2|eq] "MaximizeX Σ_m throughput(m,X) / Σ_m(Σ_j cost_j * X_mj)"
# PAPER[§4.2] "Supports SLO constraints for job completion deadlines"
```

Maximizes aggregate throughput normalized by cost. Different GPU types have different costs.

### Makespan Minimization (min_total_duration.py:9-11)

```python
# PAPER[§4.2] "Makespan minimization: minimize time for all jobs to complete"
# PAPER[§4.2|eq] "MinimizeX max_m num_steps_m / throughput(m,X)"
# PAPER[§4.2] "Binary search over T to find minimum feasible makespan"
```

Finds the allocation that minimizes the time until all jobs complete. Uses binary search over target completion time T.

---

## Section 4.3: Hierarchical Policies

**Paper Concept:** Support multi-level fairness (e.g., fair across users, then fair across jobs within each user).

### Water Filling Algorithm (max_min_fairness_water_filling.py)

```python
# PAPER[§4.3] "Hierarchical max-min fairness using water filling algorithm"
# PAPER[§4.3] "Iteratively solve LP until all jobs are bottlenecked"
class WaterFillingAlgorithm:
```

**Entity Weights (line 21-22):**
```python
# PAPER[§4.3|def] "entity_weights w_s: weight assigned to each entity (user/group)"
# PAPER[§4.3] "job weights w_m^job distributed within entity based on reweighting policy"
```

Entities (users/groups) have weights. Jobs within an entity share that entity's weight based on a reweighting policy (fairness or FIFO within entity).

**Bottleneck Detection (line 157-158):**
```python
# PAPER[§4.3] "bottleneck detection: jobs that cannot improve without hurting others"
# PAPER[§4.3] "uses MILP to find jobs at their maximum achievable throughput"
```

A job is "bottlenecked" when it cannot get more resources without reducing another job's allocation.

**Iterative Algorithm (line 240-241):**
```python
# PAPER[§4.3|alg] "Water filling: iteratively raise allocation until jobs bottleneck"
# PAPER[§4.3] "Each iteration: solve LP, find bottlenecked jobs, freeze them, repeat"
```

1. Solve LP to find optimal allocation
2. Identify bottlenecked jobs (via MILP)
3. Freeze bottlenecked jobs at their current allocation
4. Repeat until all jobs are bottlenecked

---

## Section 5: Round-Based Scheduling

**Paper Concept:** Gavel uses discrete scheduling rounds where allocations are converted to actual job placements.

### Round Duration (scheduler.py:57-58)

```python
# PAPER[§5] "Gavel uses a round-based scheduling mechanism where each round lasts 6 minutes"
# PAPER[§5|def] time_per_iteration=360 (default) corresponds to 6-minute scheduling rounds
```

Each scheduling round is 6 minutes (360 seconds) by default. Jobs receive leases for one round at a time.

### Lease Mechanism (lease.py, gavel_iterator.py)

```python
# PAPER[§5] "Lease: specifies max_steps and max_duration for a scheduling round"
# PAPER[§5] "Jobs receive leases that bound their execution within each round"
class Lease:
    def __init__(self, max_steps, max_duration):
```

**Lease Renewal (gavel_iterator.py:20):**
```python
# PAPER[§5] "Jobs request lease renewal at 75% of lease completion"
LEASE_UPDATE_FRACTION = 0.75
```

**Lease Expiration (gavel_iterator.py:98):**
```python
# PAPER[§5] "Lease expiration triggers job preemption for round-based scheduling"
lease_expired = (self._duration >= self._lease.max_duration or
                 self._steps >= self._lease.max_steps)
```

### Algorithm 1: SCHEDULE_JOBS (scheduler.py:770-772)

```python
# PAPER[§5|alg] Algorithm 1: SCHEDULE_JOBS
# PAPER[§5] "greedily selects jobs in decreasing order of priority until all workers are assigned"
# PAPER[§5] Jobs sorted by (priority, deficit, allocation) for tie-breaking
```

The scheduling algorithm:
1. Sort jobs by priority (descending)
2. Greedily assign workers to highest-priority jobs
3. Use (priority, deficit, allocation) for tie-breaking

### Priority Computation (scheduler.py:2367-2369)

```python
# PAPER[§5] "priority of job m on worker type j is X_mj / fraction_of_time_received_mj"
# PAPER[§5] "jobs with higher priority are scheduled first"
# PAPER[§5] Priority = allocation / fraction_received; new jobs get allocation * 1e9
```

**Priority Formula:**
```
priority[m,j] = allocation[m,j] / fraction_of_time_received[m,j]
```

Jobs that have received less time than allocated get higher priority. New jobs get very high priority (allocation * 1e9).

### Worker Assignment (scheduler.py:716-717)

```python
# PAPER[§5] "strided worker assignment to minimize number of servers used"
# PAPER[§5] Jobs sorted by scale_factor (largest first) to reduce fragmentation
```

Jobs are sorted by scale factor (largest first) to reduce cluster fragmentation.

### Lease Extensions (scheduler.py:865-866)

```python
# PAPER[§5] "jobs that were running in the previous round are given priority on the same workers"
# PAPER[§5] "this minimizes preemption overhead through lease extensions"
```

Jobs that were running get priority to continue on the same workers, avoiding checkpoint/restart overhead.

---

## Section 6: Throughput Estimation

**Paper Concept:** Estimate throughputs for unseen job types using matrix completion and fingerprinting.

### Overview (throughput_estimator.py:15-16)

```python
# PAPER[§6] "Throughput estimation using matrix completion and fingerprinting"
# PAPER[§6] "Match unknown jobs to reference job types using partial measurements"
class ThroughputEstimator:
```

### Partial Profiling (throughput_estimator.py:71)

```python
# PAPER[§6] "Partial profiling: measure subset of throughputs based on profiling_percentage"
def _profile_jobs(self, true_job_type):
```

Only a subset of (job, GPU type) combinations are measured to reduce profiling overhead.

### Fingerprinting (throughput_estimator.py:85-87)

```python
# PAPER[§6] "Fingerprinting: match job to reference using partial throughput profile"
# PAPER[§6] "Matrix completion fills in unmeasured throughput values"
# PAPER[§6] "Cosine distance (1 - similarity) finds closest reference job type"
def match_job_to_reference_job(self, true_job_type):
```

**Algorithm:**
1. Measure partial throughput profile for new job
2. Use matrix completion to fill in unmeasured values
3. Compute cosine distance to each reference job type
4. Match to the closest reference job type

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         Job Submission                          │
│  job_type, total_steps, scale_factor, priority_weight, SLO     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Throughput Estimation (§6)                   │
│  - Partial profiling of new job                                 │
│  - Matrix completion for unmeasured values                      │
│  - Match to reference job type via fingerprinting               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Policy Optimization (§3.1, §4)               │
│  Input: T_mj (throughputs), s_m (scale factors), w_m (weights)  │
│  Output: X_mj (allocation matrix)                               │
│  Constraints: 0 <= X <= 1, row sums <= 1, capacity limits       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Scheduling (§5 Algorithm 1)                  │
│  - Compute priorities: X_mj / fraction_received_mj              │
│  - Sort by (priority, deficit, allocation)                      │
│  - Greedily assign workers to jobs                              │
│  - Issue leases (max_steps, max_duration)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Job Execution (§5)                           │
│  - GavelIterator enforces lease bounds                          │
│  - Request lease renewal at 75% completion                      │
│  - Preempt on lease expiration                                  │
│  - Track fraction_of_time_received for next round               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        [Next Round]
```

---

## Quick Reference

| Section | Key Equation/Algorithm | Implementation |
|---------|----------------------|----------------|
| §3.1 | `throughput(m,X) = Σ_j T_mj * X_mj` | `policy.py` |
| §3.1 | Constraints (1)-(3) | `policy.py:59-67` |
| §4.1 | `max min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)` | `max_min_fairness.py` |
| §4.2 | `ρ(m,X) = expected_time / expected_time_isolated` | `finish_time_fairness.py` |
| §4.3 | Water filling algorithm | `max_min_fairness_water_filling.py` |
| §5 | `priority = allocation / fraction_received` | `scheduler.py:2367` |
| §5 | Algorithm 1: SCHEDULE_JOBS | `scheduler.py:770` |
| §6 | Matrix completion + fingerprinting | `throughput_estimator.py` |
