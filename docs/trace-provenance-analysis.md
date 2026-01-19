# Philly Trace Provenance Analysis

Date: 2026-01-19

## Summary

Gavel's "Philly" traces are **not** the raw Microsoft Philly traces. They are **synthetic traces** that preserve the organizational structure (virtual clusters) from Philly but use generated DL workloads for simulation.

## Official Microsoft Philly Traces

**Source:** [msr-fiddle/philly-traces](https://github.com/msr-fiddle/philly-traces)

**Format:** JSON job logs + CSV utilization data

**Schema (cluster_job_log):**
```json
{
  "status": "Pass|Killed|Failed",
  "vc": "virtual_cluster_id",
  "jobid": "application_xxx",
  "attempts": [{"start_time": "...", "end_time": "...", "detail": [...]}],
  "submitted_time": "2017-10-09 07:01:55",
  "user": "hashed_user_id"
}
```

**Characteristics:**
- 117,325 total jobs
- August 7 - December 22, 2017
- 15 virtual clusters (VCs)
- No workload type information (just job status and timing)
- Real timestamps and GPU allocations

## Gavel's Traces

**Location:** `src/scheduler/traces/msr/`

**Format:** Tab-separated values (TSV)

**Schema:**
```
job_name<TAB>command<TAB>flag<TAB>num_gpus<TAB>total_iterations<TAB>arrival_time<TAB>num_gpus
```

**Example:**
```
Transformer (batch size 128)	cd %s/.../translation && python3 train.py ...	-step	1	20851471	0.000000	1
```

**Characteristics:**
- 54,738 jobs in philly.trace (seed=0)
- Synthetic DL workload types (Transformer, ResNet, LM, CycleGAN, etc.)
- Arrival times in seconds (not real timestamps)
- PyTorch commands for throughput profiling
- 15 virtual clusters matching official Philly VCs

## Key Differences

| Aspect | Official Philly | Gavel Traces |
|--------|-----------------|--------------|
| Format | JSON | TSV |
| Job info | Status, timing, user | Workload type, iterations |
| Timestamps | Real (2017 dates) | Relative (seconds) |
| Workload types | None | Synthetic DL models |
| Purpose | Cluster analysis | Throughput simulation |

## Transformation Applied

Gavel's trace generation process:

1. **Extracted VC structure** from official Philly traces (15 VCs with IDs like `0e4a51`, `11cb48`, etc.)
2. **Generated synthetic jobs** using `scripts/utils/generate_trace.py`:
   - Workload types sampled from `simulation_throughputs.json`
   - Poisson arrival times
   - Random durations and GPU requirements
3. **Created per-VC trace files** preserving the multi-tenant structure
4. **Combined into philly.trace** for full-cluster simulation

## Implications for Validation

1. **Gavel's results cannot be directly compared to real Philly behavior** - the workloads are synthetic
2. **The scheduling dynamics are valid** - arrival patterns and multi-tenancy are preserved
3. **Throughput estimates come from Gavel's profiling** - based on real GPU measurements of the synthetic workloads
4. **Figure reproduction will match the paper** - using the same synthetic traces and throughput profiles

## Virtual Cluster Mapping

All 15 VCs from official Philly are present in Gavel:

| VC ID | Present in Official | Present in Gavel |
|-------|---------------------|------------------|
| 0e4a51 | Yes | Yes |
| 103959 | Yes | Yes |
| 11cb48 | Yes | Yes |
| 23dbec | Yes | Yes |
| 2869ce | Yes | Yes |
| 51b7ef | Yes | Yes |
| 6214e9 | Yes | Yes |
| 6c71a0 | Yes | Yes |
| 795a4c | Yes | Yes |
| 7f04ca | Yes | Yes |
| 925e2b | Yes | Yes |
| b436b2 | Yes | Yes |
| e13805 | Yes | Yes |
| ed69ec | Yes | Yes |
| ee9e8c | Yes | Yes |

## Recommendations

1. For **paper figure reproduction**: Use Gavel's included traces as-is
2. For **real-world validation**: Would need to create new traces from official Philly data with workload type inference
3. For **Alibaba validation** (project goal): This same synthetic approach will work - extract structure, generate DL workloads

## Files

- Official traces: `data/philly-official/trace-data/`
- Gavel traces: `src/scheduler/traces/msr/`
- Trace generator: `src/scheduler/scripts/utils/generate_trace.py`
- Throughput profiles: `src/scheduler/simulation_throughputs.json`
