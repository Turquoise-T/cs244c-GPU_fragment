# Gavel Data Schema Documentation

This document describes the JSON data structures used by the Gavel scheduler, with references to the corresponding paper sections.

## Cluster Specification (`cluster_specs.json`)

**Paper Reference:** Section 3.1 - Problem Formulation

Defines the heterogeneous cluster configuration with counts of each accelerator type.

```json
{
    "trace_name": ["v100_count:p100_count:k80_count"]
}
```

**Example:**
```json
{
    "philly": ["300:300:300"]
}
```

This specifies 300 V100s, 300 P100s, and 300 K80s for the "philly" trace.

**Code Reference:** `src/scheduler/scheduler.py` parses this into `cluster_spec[worker_type] = num_workers`

---

## Throughput Matrix (`simulation_throughputs.json`)

**Paper Reference:** Section 3.1 - Throughput Model, Section 6 - Throughput Estimation

Contains measured throughputs T_mj for each job type m on each accelerator type j.

### Schema

```json
{
    "worker_type": {
        "job_type": {
            "null": isolated_throughput,
            "other_job_type": [throughput_job1, throughput_job2]
        }
    }
}
```

### Fields

| Field | Type | Description | Paper Reference |
|-------|------|-------------|-----------------|
| `worker_type` | string | GPU type (k80, p100, v100) | Section 2 |
| `job_type` | string | Model and batch size tuple | Section 2.1 |
| `null` | float | Isolated throughput (steps/sec) when running alone | Section 3.1, T_mj |
| `other_job_type` | [float, float] | Space-shared throughputs [job1, job2] | Section 3.1 |

### Space Sharing Throughputs

When two jobs share an accelerator (Section 3.1):
- First element: throughput of the primary job
- Second element: throughput of the co-located job
- `[0.0, 0.0]` indicates jobs cannot be co-located (OOM)

**Example:**
```json
{
    "k80": {
        "('ResNet-18 (batch size 16)', 1)": {
            "null": 4.795,
            "('ResNet-50 (batch size 16)', 1)": [2.454, 0.983]
        }
    }
}
```

This means:
- ResNet-18 batch 16 runs at 4.795 steps/sec alone on K80
- When co-located with ResNet-50 batch 16:
  - ResNet-18 gets 2.454 steps/sec
  - ResNet-50 gets 0.983 steps/sec

---

## Job State

**Paper Reference:** Section 2.1, Section 3.1

Jobs are represented internally with the following attributes:

| Attribute | Type | Description | Paper Reference |
|-----------|------|-------------|-----------------|
| `job_id` | int | Unique job identifier | - |
| `job_type` | string | Model type (e.g., "ResNet-18") | Section 2.1 |
| `scale_factor` | int | Number of workers needed (data parallelism) | Section 3.1, s_m |
| `priority_weight` | float | Weight for weighted fairness | Section 4.1, w_m |
| `SLO` | float | Deadline constraint (seconds) | Section 4.2 |
| `total_steps` | int | Total training steps needed | Section 2.1 |

---

## Allocation Matrix X

**Paper Reference:** Section 3.1 - Equation (1-3)

The allocation X is represented as a nested dictionary:

```python
allocation[job_id][worker_type] = fraction  # 0 <= fraction <= 1
```

**Constraints (from paper):**
1. `0 <= X_mj <= 1` for all m, j
2. `sum_j(X_mj) <= 1` for all m (job cannot exceed 100% time)
3. `sum_m(X_mj * scale_factor_m) <= num_workers_j` (capacity)

---

## Lease Structure

**Paper Reference:** Section 5 - Round-based Scheduling

Leases bound job execution within scheduling rounds:

```python
class Lease:
    max_steps: int      # Maximum training steps in this lease
    max_duration: float # Maximum time (seconds) in this lease
```

**Round Duration:** Default 6 minutes (360 seconds) per Section 5.

---

## Priority Computation

**Paper Reference:** Section 5 - Algorithm 1

Priority for scheduling is computed as:

```
priority[job_id][worker_type] = allocation / fraction_received
```

Where `fraction_received` tracks actual time received vs. allocated.

Tie-breakers: `(priority, deficit, allocation)` in descending order.
