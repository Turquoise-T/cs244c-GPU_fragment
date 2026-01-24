# Paper-to-Code Mapping Design

CS244C Project - Gavel Validation

Date: 2026-01-19

## Overview

Before running experiments, we need to understand how Gavel's paper claims translate to code. This ensures we understand what we're validating and can identify any discrepancies between the paper and implementation.

## Deliverable

A single comprehensive markdown document (`docs/paper-code-mapping.md`) that:
1. Maps each paper section to corresponding code modules
2. Verifies specific claims from the paper against the implementation
3. Documents the code architecture for team reference

## Document Structure

```markdown
# Gavel: Paper-to-Code Mapping

## Paper Overview
- Citation, key contributions, high-level architecture

## Section-by-Section Mapping

### Section 3: Motivation & Background
- Paper claims -> Code location -> Verification notes

### Section 4: Heterogeneity-Aware Policies
- Effective throughput abstraction
- Policy implementations (max-min fairness, finish-time fairness, etc.)

### Section 5: Scheduling Mechanism
- Round-based allocation
- Lease management
- Job placement

### Section 6: Implementation
- Simulator architecture
- Physical cluster deployment

## Claim Verification Checklist
- [ ] Claim 1: "..." -> Verified in `file.py:line`
- [ ] Claim 2: "..." -> Verified in `file.py:line`
...

## Code Architecture Summary
- Key classes and their relationships
- Data flow diagram
```

## Code Module Mapping

| Paper Section | Primary Code Files |
|--------------|-------------------|
| Effective throughput | `throughput_estimator.py`, `simulation_throughputs.json` |
| Policies | `policies/*.py` |
| Scheduling mechanism | `scheduler.py` |
| Round-based allocation | `scheduler.py` (simulate loop) |
| Job/lease management | `job.py`, `lease.py`, `job_table.py` |

## Claims to Verify

### Core Contributions
1. "Effective throughput abstraction normalizes job throughput across GPU types"
2. "Policies can be expressed as optimization problems over effective throughput"
3. "Round-based allocation achieves target allocations without migration"

### Policy Claims
4. "Max-min fairness maximizes minimum effective throughput across jobs"
5. "Finish-time fairness equalizes job completion times"
6. "LAS (Least Attained Service) prioritizes jobs with less accumulated service"

### Mechanism Claims
7. "Round duration balances allocation accuracy vs. preemption overhead"
8. "Heterogeneity-aware placement improves throughput over random placement"

### Performance Claims (verify experimentally)
9. "1.4x-3.5x improvement in average JCT over heterogeneity-agnostic schedulers"
10. "Policy runtime scales with number of jobs"

## Execution Plan

### Phase 1: Paper Reading (1-2 hours)
- Read paper sections 3-6 carefully
- Extract exact quotes for each claim
- Note any equations or pseudocode

### Phase 2: Code Exploration (2-3 hours)
- Map each claim to code files
- Trace key functions (e.g., `get_allocation()`, `simulate()`)
- Document class relationships

### Phase 3: Verification (1-2 hours)
- For each claim, verify implementation matches description
- Note any simplifications or differences
- Add code references with line numbers

### Phase 4: Documentation (1 hour)
- Write up findings in the markdown doc
- Create code architecture diagram (text-based)
- Add checklist for claim verification

## Success Criteria

- All 8 code claims mapped to specific files/functions
- Any discrepancies between paper and code documented
- Team members can use doc to navigate codebase
