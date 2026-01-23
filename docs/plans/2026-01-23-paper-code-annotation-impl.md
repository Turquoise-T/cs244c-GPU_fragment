# Paper-to-Code Annotation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Annotate Gavel codebase with inline paper references using compact `# PAPER[...]` tags, with per-file Codex review.

**Architecture:** For each file, add paper annotations at function/class level, generate a manifest of annotations, and have Codex validate completeness/accuracy before moving to the next file.

**Tech Stack:** Python comments, structured annotation tags, Codex CLI for review

---

## Workflow Per File

Each file follows this cycle:
1. Read file and identify key functions/classes
2. Add `# PAPER[§X.Y]` annotations with verbatim quotes
3. Generate manifest (list of all annotations added)
4. Phone Codex for review
5. Address any issues Codex identifies
6. Commit and move to next file

---

## Task 1: Annotate `scheduler.py` - Core Scheduling Mechanism

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Reference: Design doc at `docs/plans/2026-01-23-paper-code-annotation-design.md`

**Paper Section:** §5 (Scheduling Mechanism)

**Key Functions to Annotate:**

| Function | Lines | Paper Concept |
|----------|-------|---------------|
| `__init__` | 59-314 | `time_per_iteration=360` is 6-minute rounds (§5) |
| `_schedule_jobs_on_workers_helper` | 766-854 | Algorithm 1: Greedy job selection by priority |
| `_schedule_jobs_on_workers` | 858-957 | Worker assignment, lease extension |
| `_update_priorities` | 2358-2463 | Priority = allocation / fraction_received (§5) |
| `_assign_workers_to_job` | 714-763 | Job placement strategy |
| `simulate` | 1126-1499 | Round-based simulation loop |

**Step 1: Add annotations to `__init__`**

Add at line ~60:
```python
# PAPER[§5] "Gavel uses a round-based scheduling mechanism where each round lasts 6 minutes"
# PAPER[§5|def] time_per_iteration=360 corresponds to 6-minute scheduling rounds
```

**Step 2: Add annotations to `_schedule_jobs_on_workers_helper`**

Add at line ~766:
```python
# PAPER[§5|alg] Algorithm 1: SCHEDULE_JOBS
# PAPER[§5] "greedily selects jobs in decreasing order of priority until all workers are assigned"
# PAPER[§5] Priority computed as: allocation[job][worker_type] / fraction_received[job][worker_type]
```

**Step 3: Add annotations to `_update_priorities`**

Add at line ~2358:
```python
# PAPER[§5] "priority of job m on worker type j is X_mj / rounds_received_mj"
# PAPER[§5] "jobs with higher priority are scheduled first"
```

**Step 4: Add annotations to `_schedule_jobs_on_workers`**

Add at line ~858:
```python
# PAPER[§5] "jobs that were running in the previous round are given priority on the same workers"
# PAPER[§5] "this minimizes preemption overhead through lease extensions"
```

**Step 5: Generate manifest**

Create manifest of all annotations added:
```
scheduler.py Annotation Manifest
================================
Line ~60: PAPER[§5] - 6-minute rounds
Line ~766: PAPER[§5|alg] - Algorithm 1
Line ~2358: PAPER[§5] - Priority computation
Line ~858: PAPER[§5] - Lease extensions
```

**Step 6: Codex review**

Share manifest + file with Codex using phone-a-friend skill. Ask:
- Are all §5 concepts covered?
- Are quotes accurate?
- Any missing mappings?

**Step 7: Address feedback and commit**

```bash
git add src/scheduler/scheduler.py
git commit -m "feat: add paper annotations to scheduler.py (§5)"
```

---

## Task 2: Annotate `policy.py` - Core Abstraction

**Files:**
- Modify: `src/scheduler/policies/policy.py`

**Paper Section:** §3.1 (Heterogeneity-Aware Policies)

**Key Concepts to Annotate:**

| Concept | Paper Reference |
|---------|-----------------|
| Allocation matrix X | `X[m,j]` = fraction of time job m on accelerator j |
| Effective throughput | `throughput(m,X) = Σ_j T_mj * X_mj` |
| Constraint (1) | `0 <= X_mj <= 1` |
| Constraint (2) | `Σ_j X_mj <= 1` |
| Constraint (3) | `Σ_m X_mj * scale_factor_m <= num_workers_j` |

**Step 1: Read policy.py and identify functions**

**Step 2: Add annotations for effective throughput definition**

```python
# PAPER[§3.1|def] "effective throughput: throughput(m,X) = Σ_j T_mj * X_mj"
# PAPER[§3.1] "time-weighted average throughput across all accelerator types"
```

**Step 3: Add constraint annotations**

```python
# PAPER[§3.1|eq] Constraint (1): 0 <= X_mj <= 1
# PAPER[§3.1|eq] Constraint (2): Σ_j X_mj <= 1 (job time constraint)
# PAPER[§3.1|eq] Constraint (3): Σ_m X_mj * scale_factor_m <= num_workers_j (capacity constraint)
```

**Step 4: Generate manifest and Codex review**

**Step 5: Commit**

```bash
git commit -m "feat: add paper annotations to policy.py (§3.1)"
```

---

## Task 3: Annotate `max_min_fairness.py` - LAS Policy

**Files:**
- Modify: `src/scheduler/policies/max_min_fairness.py`

**Paper Section:** §4.1 (Max-Min Fairness / LAS)

**Key Concepts:**

| Concept | Paper Reference |
|---------|-----------------|
| LAS objective | `MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)` |
| X^equal | Equal time share baseline |
| Priority weights | `w_m` per-user weights |

**Step 1: Read file and identify optimization problem**

**Step 2: Add objective function annotation**

```python
# PAPER[§4.1] "MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)"
# PAPER[§4.1] "LAS (Least Attained Service) prioritizes jobs with less accumulated service"
```

**Step 3: Add X^equal reference**

```python
# PAPER[§4.1|def] "X^equal gives each job an equal share of time on each worker type"
```

**Step 4: Generate manifest and Codex review**

**Step 5: Commit**

```bash
git commit -m "feat: add paper annotations to max_min_fairness.py (§4.1)"
```

---

## Task 4: Annotate `max_min_fairness_water_filling.py` - Hierarchical Policies

**Files:**
- Modify: `src/scheduler/policies/max_min_fairness_water_filling.py`

**Paper Section:** §4.3 (Hierarchical Policies)

**Key Concepts:**

| Concept | Paper Reference |
|---------|-----------------|
| Water filling | Iterative LP solving |
| Entity weights | `w_s` for entity |
| Bottleneck detection | Jobs that can't improve |

**Step 1-5: Same pattern as above**

**Commit message:** `feat: add paper annotations to max_min_fairness_water_filling.py (§4.3)`

---

## Task 5: Annotate `finish_time_fairness.py` - Themis Policy

**Files:**
- Modify: `src/scheduler/policies/finish_time_fairness.py`

**Paper Section:** §4.2 (Finish-Time Fairness)

**Key Concepts:**

| Concept | Paper Reference |
|---------|-----------------|
| Themis ρ metric | `ρ(m,X) = (t_m + remaining/throughput) / (t_isolated + remaining/throughput_isolated)` |
| Finish-time optimization | Equalize completion times |

**Commit message:** `feat: add paper annotations to finish_time_fairness.py (§4.2)`

---

## Task 6: Annotate Other Policies

**Files to annotate (same pattern):**

| File | Section | Key Concept |
|------|---------|-------------|
| `fifo.py` | §4.2 | FIFO as throughput maximization |
| `min_total_duration.py` | §4.2 | Makespan minimization |
| `max_sum_throughput.py` | §4.2 | Cost minimization |
| `isolated.py` | §4.1 | X^equal baseline |

**Commit message pattern:** `feat: add paper annotations to <file> (§X.Y)`

---

## Task 7: Annotate `throughput_estimator.py` - Matrix Completion

**Files:**
- Modify: `src/scheduler/throughput_estimator.py`

**Paper Section:** §6 (Implementation)

**Key Concepts:**

| Concept | Paper Reference |
|---------|-----------------|
| Matrix completion | Estimate missing throughput values |
| Fingerprinting | Match jobs to reference models |

**Commit message:** `feat: add paper annotations to throughput_estimator.py (§6)`

---

## Task 8: Annotate Supporting Files

**Files:**
- `job_id_pair.py` - §3.1 (space sharing)
- `job.py` - §2.1, §6 (job state)
- `lease.py` - §5 (lease renewal)
- `gavel_iterator.py` - §6 (application API)

**Commit message pattern:** `feat: add paper annotations to <file> (§X.Y)`

---

## Task 9: Create Data Schema Documentation

**Files:**
- Create: `docs/data-schema.md`

**Content:**

Document JSON file schemas:
- `simulation_throughputs.json` - Throughput matrix T (§3.1, Table 2)
- `physical_cluster_throughputs.json` - Physical cluster T (§7.1)
- `traces/msr/cluster_specs.json` - Cluster configuration (§7.1)

**Commit message:** `docs: add data-schema.md for JSON files (§3.1, §7.1)`

---

## Task 10: Final Verification

**Step 1: Run grep to list all annotations**

```bash
grep -rn "# PAPER\[" src/scheduler/ --include="*.py"
```

**Step 2: Create summary manifest**

Generate `docs/paper-code-annotation-manifest.md` with all annotations.

**Step 3: Final Codex review**

Share complete manifest with Codex for final validation against design doc checklist.

**Step 4: Verify checklist items**

From design doc:
- [ ] Effective throughput normalizes job throughput across GPU types (§3.1)
- [ ] Policies expressed as optimization problems over effective throughput (§4)
- [ ] Round-based allocation achieves target allocations (§5)
- [ ] Max-min fairness maximizes minimum effective throughput (§4.1)
- [ ] Finish-time fairness equalizes job completion times (§4.2)
- [ ] LAS prioritizes jobs with less accumulated service (§4.1)
- [ ] Round duration balances accuracy vs overhead (§5)
- [ ] Heterogeneity-aware placement improves throughput (§3.1)

**Commit message:** `docs: add complete annotation manifest and verification`

---

## Execution Notes

- Each task includes Codex review before commit
- Use `phone-a-friend` skill with `code-review` consultation type
- Share manifest (not full file) to keep context manageable
- Address all Codex feedback before moving to next file
- Commit after each file to maintain clean history

## Success Criteria

1. All 22 Python files have appropriate annotations
2. Every annotation validated by Codex
3. `docs/data-schema.md` documents JSON files
4. Final manifest passes verification checklist
5. Clean commit history with one commit per file
