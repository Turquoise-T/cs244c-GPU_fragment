# Paper-to-Code Annotation Manifest

Generated: 2026-01-24

Total annotations: 69

## Summary by Section

| Section | Count | Description |
|---------|-------|-------------|
| §2.1    | 1     | Job state and workload model |
| §3.1    | 12    | Problem formulation, allocation matrix, constraints |
| §4.1    | 10    | Max-min fairness, LAS, X^equal |
| §4.2    | 13    | Finish-time fairness, cost minimization, FIFO |
| §4.3    | 8     | Water filling algorithm, hierarchical fairness |
| §5      | 20    | Round-based scheduling, Algorithm 1, priority |
| §6      | 5     | Throughput estimation, matrix completion |

## Annotations by File

### Core Scheduler
- `scheduler.py` (11 annotations) - Round-based scheduling, Algorithm 1, priority computation
- `policy.py` (7 annotations) - Base policy with constraints
- `throughput_estimator.py` (5 annotations) - Matrix completion

### Fairness Policies
- `max_min_fairness.py` (6 annotations) - LAS objective
- `max_min_fairness_water_filling.py` (8 annotations) - Water filling algorithm
- `finish_time_fairness.py` (6 annotations) - Themis rho metric
- `isolated.py` (3 annotations) - X^equal baseline

### Other Policies
- `fifo.py` (3 annotations) - FIFO scheduling
- `min_total_duration.py` (3 annotations) - Makespan minimization
- `max_sum_throughput.py` (3 annotations) - Cost minimization

### Supporting Files
- `job.py` (4 annotations) - Job state attributes
- `job_id_pair.py` (3 annotations) - Space sharing
- `lease.py` (2 annotations) - Lease structure
- `gavel_iterator.py` (4 annotations) - Runtime lease enforcement

## Annotation Format

All annotations follow the format:
```
# PAPER[§X.Y] "description"
# PAPER[§X.Y|type] "description"
```

Where type is one of:
- `def` - Definition
- `eq` - Equation
- `alg` - Algorithm

## Verification Checklist

- [x] All policy files annotated
- [x] Core scheduler mechanism documented
- [x] Algorithm 1 referenced
- [x] Constraint equations marked
- [x] Throughput estimation covered
- [x] Supporting data structures documented
