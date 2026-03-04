# Gavel Repo Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the gavel repo into a paper-centric layout: src/ for core algorithms, experiments/ for per-paper drivers and results, upstream code stripped.

**Architecture:** All changes are file moves (`git mv`), deletions (`git rm`), and a handful of import path fixes. The repo is a separate git repo at `stanford/cs244c/gavel/`. All paths below are relative to that root.

**Tech Stack:** git, Python (sys.path fixes), bash (SLURM script path updates)

**Design doc:** `docs/plans/2026-02-15-gavel-repo-reorganization-design.md`

---

### Task 1: Strip upstream -- delete workloads, RPC, upstream scripts, legacy

These files are never imported by the simulation code. Removing them before moving files keeps the diff clean.

**Files:**
- Delete: `src/workloads/` (entire directory, ~100+ files)
- Delete: `src/scheduler/runtime/` (entire directory, ~20 files)
- Delete: `src/scheduler/scripts/` (entire directory, ~17 files)
- Delete: `src/scheduler/gavel_iterator.py`
- Delete: `src/EXPERIMENTS.md`, `src/LICENSE`, `src/README.md`
- Delete: `experiments/replication/gavel/legacy/` (6 abandoned scripts)

**Step 1: Delete upstream directories and files**

```bash
cd stanford/cs244c/gavel
git rm -r src/workloads/
git rm -r src/scheduler/runtime/
git rm -r src/scheduler/scripts/
git rm src/scheduler/gavel_iterator.py
git rm src/EXPERIMENTS.md src/LICENSE src/README.md
git rm -r experiments/replication/gavel/legacy/
```

**Step 2: Make runtime import lazy in scheduler.py**

The top-level import `from runtime.rpc import scheduler_server, scheduler_client` (line 27) will fail now that `runtime/` is deleted. Move it to lazy imports at the two usage sites.

File: `src/scheduler/scheduler.py`

Remove line 27:
```python
from runtime.rpc import scheduler_server, scheduler_client
```

At line ~338 (the `scheduler_server.serve` call), add a lazy import inside the `if not self._simulate` block that contains it. Find the enclosing block and add:
```python
from runtime.rpc import scheduler_server
```

At line ~3148 (the `scheduler_client.SchedulerRpcClient` call), which is already inside `if not self._simulate:`, add:
```python
from runtime.rpc import scheduler_client
```

Note: These lazy imports will fail at runtime in non-simulation mode since we deleted `runtime/`. That is expected -- we only use simulation mode. The lazy import means Python never executes these lines during simulation.

**Step 3: Run integration tests to verify nothing broke**

```bash
cd src/scheduler/tests
../../../.venv/bin/python -m pytest integration_test.py -v
```

Expected: All tests pass (tests only use simulation mode).

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: strip upstream deployment code (workloads, RPC, legacy scripts)"
```

---

### Task 2: Move fgd_src/ core algorithm to src/fgd/

Split fgd_src into core library (src/fgd/) and experiment tooling (experiments/fgd-standalone/). This task handles the core library move.

**Files:**
- Move: `fgd_src/fgd.py` -> `src/fgd/fgd.py`
- Move: `fgd_src/baselines.py` -> `src/fgd/baselines.py`
- Move: `fgd_src/alibaba_trace_parser.py` -> `src/fgd/alibaba_trace_parser.py`
- Move: `fgd_src/configs/` -> `src/fgd/configs/`
- Move: `fgd_src/data/` -> `src/fgd/data/`
- Move: `fgd_src/tests/` -> `src/fgd/tests/`
- Move: `fgd_src/requirements.txt` -> `src/fgd/requirements.txt`

**Step 1: Create target directory and move core files**

```bash
cd stanford/cs244c/gavel
mkdir -p src/fgd
git mv fgd_src/fgd.py src/fgd/fgd.py
git mv fgd_src/baselines.py src/fgd/baselines.py
git mv fgd_src/alibaba_trace_parser.py src/fgd/alibaba_trace_parser.py
git mv fgd_src/configs src/fgd/configs
git mv fgd_src/data src/fgd/data
git mv fgd_src/tests src/fgd/tests
git mv fgd_src/requirements.txt src/fgd/requirements.txt
```

**Step 2: Update fgd_placement.py import path**

File: `src/scheduler/fgd_placement.py` line 21

```python
# Before:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'fgd_src'))
# After:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fgd'))
```

**Step 3: Update src/fgd/tests/test_simulator.py paths**

Line 21 currently does: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` -- this resolves to the parent of `tests/`, which was `fgd_src/` and is now `src/fgd/`. This still works correctly since it just needs to find `fgd.py` in the parent directory.

Line 72 references `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` for config paths -- also still correct (points to `src/fgd/`).

No change needed for test_simulator.py.

**Step 4: Run FGD placement test to verify import works**

```bash
cd src/scheduler/tests
../../../.venv/bin/python -m pytest test_fgd_placement.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move FGD core algorithm from fgd_src/ to src/fgd/"
```

---

### Task 3: Move fgd_src/ experiment tooling to experiments/fgd-standalone/

Move the experiment drivers, plotting, and results from fgd_src/ to the new experiments directory.

**Files:**
- Move: `fgd_src/simulator.py` -> `experiments/fgd-standalone/simulator.py`
- Move: `fgd_src/run_standalone.py` -> `experiments/fgd-standalone/run_standalone.py`
- Move: `fgd_src/run_evaluation.py` -> `experiments/fgd-standalone/run_evaluation.py`
- Move: `fgd_src/plot_results.py` -> `experiments/fgd-standalone/plot_results.py`
- Move: `fgd_src/paper_reference_curves.json` -> `experiments/fgd-standalone/paper_reference_curves.json`
- Move: `fgd_src/results/` -> `experiments/fgd-standalone/results/`
- Move: `fgd_src/figures/` -> `experiments/fgd-standalone/figures/`
- Delete: `fgd_src/` (now empty)

**Step 1: Create target directory and move files**

```bash
cd stanford/cs244c/gavel
mkdir -p experiments/fgd-standalone
git mv fgd_src/simulator.py experiments/fgd-standalone/simulator.py
git mv fgd_src/run_standalone.py experiments/fgd-standalone/run_standalone.py
git mv fgd_src/run_evaluation.py experiments/fgd-standalone/run_evaluation.py
git mv fgd_src/plot_results.py experiments/fgd-standalone/plot_results.py
git mv fgd_src/paper_reference_curves.json experiments/fgd-standalone/paper_reference_curves.json
git mv fgd_src/results experiments/fgd-standalone/results
git mv fgd_src/figures experiments/fgd-standalone/figures
```

**Step 2: Remove now-empty fgd_src/ directory**

```bash
# Check if anything remains
ls fgd_src/
# If empty (or only __pycache__/.DS_Store), remove
rm -rf fgd_src/__pycache__ fgd_src/.DS_Store
git rm -r fgd_src/ 2>/dev/null; rmdir fgd_src 2>/dev/null
```

**Step 3: Update sys.path in moved experiment scripts**

All four Python files that were in `fgd_src/` used `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` to find `fgd.py` in the same directory. Now `fgd.py` is at `../../src/fgd/` relative to `experiments/fgd-standalone/`.

File: `experiments/fgd-standalone/run_evaluation.py` line 27
```python
# Before:
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# After:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'fgd'))
```

File: `experiments/fgd-standalone/run_standalone.py` line 22
```python
# Before:
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# After:
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'fgd'))
```

File: `experiments/fgd-standalone/simulator.py` -- check if it has a sys.path line. If not, it relies on being in the same directory as fgd.py. Add at the top (after existing imports but before `from fgd import ...`):
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'fgd'))
```

File: `experiments/fgd-standalone/run_standalone.py` line 169 -- references `os.path.dirname(__file__), 'configs', 'cluster_h.json'`. Configs are now at `../../src/fgd/configs/`. Update:
```python
# Before:
os.path.dirname(__file__), 'configs', 'cluster_h.json'
# After:
os.path.dirname(__file__), '..', '..', 'src', 'fgd', 'configs', 'cluster_h.json'
```

File: `experiments/fgd-standalone/plot_results.py` line 334 -- uses `script_dir = os.path.dirname(os.path.abspath(__file__))` for locating results. This still works correctly since results/ moved with the script.

**Step 4: Verify FGD standalone scripts parse correctly**

```bash
cd stanford/cs244c/gavel
.venv/bin/python -c "import sys; sys.path.insert(0, 'experiments/fgd-standalone'); import simulator; print('OK')"
```

Expected: `OK` (module loads without error)

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move FGD experiment tooling to experiments/fgd-standalone/"
```

---

### Task 4: Flatten experiments/replication/gavel/ to experiments/gavel-replication/

**Files:**
- Move: `experiments/replication/gavel/*` -> `experiments/gavel-replication/*`

**Step 1: Move the directory**

```bash
cd stanford/cs244c/gavel
git mv experiments/replication/gavel experiments/gavel-replication
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: flatten experiments/replication/gavel/ to experiments/gavel-replication/"
```

---

### Task 5: Flatten experiments/replication/fgd/ to experiments/combined/

This is the largest experiment directory (19 configs, 117+ results, telemetry, logs, SLURM scripts).

**Files:**
- Move: `experiments/replication/fgd/*` -> `experiments/combined/*`
- Merge: `experiments/combined/slurm/slurm_logs/*` -> `experiments/combined/detailed_logs/`

**Step 1: Move the directory**

```bash
cd stanford/cs244c/gavel
git mv experiments/replication/fgd experiments/combined
```

**Step 2: Merge slurm/slurm_logs/ into detailed_logs/**

The SLURM stderr files from `combined/slurm/slurm_logs/` should merge with `combined/detailed_logs/`:

```bash
# Move SLURM log files into detailed_logs
git mv experiments/combined/slurm/slurm_logs/* experiments/combined/detailed_logs/ 2>/dev/null
# Remove the now-empty slurm_logs dir
rmdir experiments/combined/slurm/slurm_logs 2>/dev/null
```

If `detailed_logs/` doesn't exist yet, create it first: `mkdir -p experiments/combined/detailed_logs`

**Step 3: Remove the now-empty experiments/replication/ directory**

```bash
rm -rf experiments/replication/.DS_Store
git rm -r experiments/replication/ 2>/dev/null; rmdir experiments/replication 2>/dev/null
```

**Step 4: Update combined experiment script paths**

File: `experiments/combined/run_fgd_experiments.py` line 34
```python
# Before:
SCHEDULER_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scheduler')
# After:
SCHEDULER_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scheduler')
```
This path is: `experiments/combined/` -> `..` = `experiments/` -> `..` = `gavel/` -> `src/scheduler/`. This is the SAME depth as before (`experiments/replication/fgd/` -> `..` = `experiments/replication/` -> `..` = `experiments/` -> `..` = `gavel/`).

Wait -- before the path was 3 levels up (`../../..`), now it's 2 levels up (`../..`). Check line 34 carefully:

Before (from `experiments/replication/fgd/`): `os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scheduler')` resolves to `experiments/replication/fgd/../../src/scheduler` = `experiments/src/scheduler`. That's WRONG -- this must have been relying on being 3 dirs deep: `experiments/replication/fgd/../../../src/scheduler` = `src/scheduler`. But the code only has TWO `..` components.

**Must verify the actual code** -- read line 34 exactly and count the `..` components. The correct new path from `experiments/combined/` is `../../src/scheduler` (2 levels up).

If the existing code has `'..', '..', 'src', 'scheduler'` (2 `..`), it was ALREADY WRONG at the old location (would resolve to `experiments/src/scheduler`). If it has `'..', '..', '..', 'src', 'scheduler'` (3 `..`), it needs to change to 2 `..`.

**Action:** Read the exact line. If 3 `..`: change to 2. If 2 `..`: no change needed (it was either broken before or I miscounted nesting).

**Step 5: Update SLURM submit scripts**

All `.sbatch` files in `experiments/combined/slurm/` reference paths like `$GAVEL_DIR/experiments/fgd`. These are FarmShare paths. Update them:

```
# In each .sbatch file:
# Before:
FGD_DIR="$GAVEL_DIR/experiments/fgd"
# After:
FGD_DIR="$GAVEL_DIR/experiments/combined"
```

Also update any comment lines that reference the old path.

**Step 6: Run integration tests**

```bash
cd src/scheduler/tests
../../../.venv/bin/python -m pytest integration_test.py -v
```

Expected: All tests pass.

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: flatten experiments/replication/fgd/ to experiments/combined/"
```

---

### Task 6: Update cross-cutting scripts and documentation

Fix references to old paths in sync scripts, README, and CLAUDE.md.

**Files:**
- Modify: `scripts/sync_fgd_to_farmshare.sh` (references `fgd_src/`)
- Modify: `README.md` (references `experiments/replication/`)
- Modify: `.claude/CLAUDE.md` (references `experiments/replication/`, `fgd_src/`)

**Step 1: Update sync script**

File: `scripts/sync_fgd_to_farmshare.sh`
- Change `"$GAVEL_DIR/fgd_src/"` to `"$GAVEL_DIR/src/fgd/"` (and update the remote path if it mirrors local)
- Update echo messages referencing `fgd_src`

Also check all other scripts in `scripts/` for old path references:
```bash
grep -rn 'fgd_src\|experiments/replication\|experiments/fgd' scripts/
```
Update any matches.

**Step 2: Update README.md**

Replace all references to `experiments/replication/` with the new paths:
- `experiments/replication/README.md` -> `experiments/gavel-replication/README.md`
- `experiments/replication/` -> `experiments/` (general references)
- `experiments/replication/results/` -> `experiments/gavel-replication/results/`
- `experiments/replication/debug/` -> `experiments/gavel-replication/debug/`
- Update the directory table at line ~203

**Step 3: Update .claude/CLAUDE.md**

Replace references to old paths:
- `fgd_src/` -> `src/fgd/` (for core) or `experiments/fgd-standalone/` (for tooling)
- `experiments/replication/` -> `experiments/`
- Update the "Key Files" table
- Keep the warning: `set_queue.py` is a runtime dependency

**Step 4: Update .gitignore if needed**

Check if `.gitignore` has rules specific to old paths:
```bash
grep -n 'fgd_src\|experiments/replication' .gitignore
```
Update any matches.

**Step 5: Commit**

```bash
git add -A
git commit -m "docs: update README, CLAUDE.md, and scripts for new directory structure"
```

---

### Task 7: Final verification and cleanup

**Step 1: Run full test suite**

```bash
cd stanford/cs244c/gavel
# Integration tests
cd src/scheduler/tests && ../../../.venv/bin/python -m pytest integration_test.py -v && cd ../../..
# FGD placement tests
cd src/scheduler/tests && ../../../.venv/bin/python -m pytest test_fgd_placement.py -v && cd ../../..
# FGD unit tests
cd src/fgd/tests && ../../../.venv/bin/python -m pytest test_simulator.py -v && cd ../../..
```

Expected: All tests pass.

**Step 2: Verify directory structure matches design**

```bash
# Should show: scheduler/, fgd/
ls src/
# Should show: gavel-replication/, fgd-standalone/, combined/
ls experiments/
# Should NOT exist
ls fgd_src/ 2>&1    # "No such file or directory"
ls experiments/replication/ 2>&1  # "No such file or directory"
ls src/workloads/ 2>&1  # "No such file or directory"
```

**Step 3: Check for stale references**

```bash
grep -rn 'fgd_src\|experiments/replication\|src/workloads\|gavel_iterator' --include='*.py' --include='*.sh' --include='*.md' --include='*.sbatch' .
```

Any remaining references (except in `docs/plans/` design docs which are historical) need updating.

**Step 4: Verify no untracked files were left behind**

```bash
git status
```

Clean up any `.DS_Store`, `__pycache__/`, or orphaned empty directories.

**Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: final cleanup after repo reorganization"
```
