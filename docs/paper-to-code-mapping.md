# Gavel: A Guided Tour Through the Code

*A conversational walkthrough of the Gavel scheduler, as if the authors were sitting next to you.*

---

Hey! We built Gavel because we were frustrated.

GPU clusters are expensive, and we kept seeing jobs waiting forever while other GPUs sat idle - just because those GPUs were "the wrong type." A ResNet training job would wait in line for a V100 while K80s collected dust, even though it could run perfectly fine on a K80 (just a bit slower).

We thought: **what if the scheduler was smart about which jobs run best on which hardware?**

That's Gavel - a heterogeneity-aware cluster scheduler. It knows that your Transformer model flies on V100s but crawls on K80s, while your recommendation model doesn't really care. It uses this knowledge to make better scheduling decisions.

Let's walk through how we built it. We'll follow a single training job from the moment it arrives until it finishes. Along the way, you'll see every piece of the system and why we designed it that way.

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Your Job   │───▶│  How fast on    │───▶│  What's fair?   │───▶│   Actual     │
│  Arrives    │    │  each GPU?      │    │  (Policy)       │    │  Scheduling  │
└─────────────┘    └─────────────────┘    └─────────────────┘    └──────────────┘
                                                                        │
                   ┌─────────────────┐    ┌─────────────────┐           │
                   │     Done!       │◀───│  Running with   │◀──────────┘
                   │                 │    │  Leases         │
                   └─────────────────┘    └─────────────────┘
```

Ready? Let's go.

---

## Chapter 1: Your Job Arrives

When your job arrives, we need to understand a few things about it. Think of it like checking into a hospital - they need your basics before they can help you.

Here's what we ask:

**What kind of model are you training?** (`job_type`)

This tells us your "personality." Some models love V100s and their tensor cores. Others are memory-bound and don't care much about GPU type. A Transformer with large batch sizes behaves very differently from a small CNN.

**How long is your journey?** (`total_steps`)

Are you a quick 1,000-step fine-tuning job, or a million-step training marathon? This affects how we think about fairness - a job that's been running for days has different needs than one that just started.

**Do you need a team?** (`scale_factor`)

Some jobs need 8 GPUs working together (data parallelism). Others work solo. If you need 4 GPUs, we can't give you 3 - it's all or nothing. This constraint shapes what allocations are even possible.

**How important are you?** (`priority_weight`)

In a fair system, everyone starts equal. But sometimes a deadline job needs a boost, or a researcher is paying more for priority access. This weight lets us tune the fairness calculation.

**Do you have a hard deadline?** (`SLO`)

"I must finish by 5pm" changes everything. If you have a service-level objective, we might need to give you more resources now to ensure you finish on time.

Here's the actual code that captures all this:

```python
# src/scheduler/job.py

# PAPER[§2.1] "Job state includes model type, total training steps, and resource requirements"
# PAPER[§3.1|def] "scale_factor: number of workers job needs for data parallelism"
# PAPER[§4.1] "priority_weight w_m: weight for weighted max-min fairness"
# PAPER[§4.2] "SLO: deadline constraint for finish-time fairness policies"
class Job:
    def __init__(self, job_id, job_type, command, working_directory,
                 num_steps_arg, total_steps, duration, scale_factor=1,
                 priority_weight=1, SLO=None, needs_data_dir=False):
        self._job_id = job_id
        self._job_type = job_type              # What model? (ResNet-18, Transformer, etc.)
        self._command = command                 # How to run it
        self._working_directory = working_directory
        self._total_steps = total_steps         # How many training steps total?
        self._duration = duration               # Expected duration
        self._scale_factor = scale_factor       # How many GPUs needed together?
        self._priority_weight = priority_weight # Fairness weight
        self._SLO = SLO                         # Deadline (if any)
```

Notice how simple this is. We're not trying to understand *everything* about your job - just the essentials that affect scheduling decisions.

---

## Chapter 2: The Big Picture - What Are We Even Optimizing?

Here's our key insight, and it took us a while to see it clearly:

**Scheduling isn't about assigning jobs to GPUs. It's about assigning *time*.**

Let us explain. Imagine you have 10 V100 GPUs and 20 jobs that want them. You can't give everyone a dedicated GPU. But you *can* give everyone a fraction of the GPU time.

Think of it like a pizza. The pizza represents all the GPU-hours available. Our job is to slice up that pizza fairly among all the hungry jobs.

### The Allocation Matrix

We represent this with a table we call the **allocation matrix X**. Each cell X[m,j] says: "What fraction of GPU type j's time does job m get?"

Here's a concrete example. Say we have 3 jobs and 2 GPU types:

```
              V100    K80
Job A:        0.5     0.2    ← Job A gets 50% of V100 time, 20% of K80 time
Job B:        0.3     0.4    ← Job B gets 30% of V100 time, 40% of K80 time
Job C:        0.2     0.4    ← Job C gets 20% of V100 time, 40% of K80 time
              ───     ───
Column sums:  1.0     1.0    ← All the time is accounted for
```

This is powerful because:
1. **It's continuous** - We can give you 30% of V100 time, not just "all or nothing"
2. **It's flexible** - Different jobs can get different mixes of GPU types
3. **It's optimizable** - We can use math to find the "best" allocation

### The Constraints

Of course, not every allocation makes sense. We have rules:

**Rule 1: Fractions must be between 0 and 1**

You can't get negative time, and you can't get more than 100% of a resource.

**Rule 2: A job can't use more than 100% of its time**

If you add up all the GPU time we give you, it can't exceed 1.0. You can't be in two places at once.

**Rule 3: We can't allocate more GPUs than we have**

If we only have 10 V100s, and Job A needs 4 GPUs (scale_factor=4), then Job A's allocation times 4 plus everyone else's allocations times their scale factors must not exceed 10.

Here's how we encode these constraints:

```python
# src/scheduler/policies/policy.py

# PAPER[§3.1|def] "allocation matrix X where X_mj = fraction of time job m spends on accelerator j"
# PAPER[§3.1|def] "effective throughput: throughput(m,X) = Σ_j T_mj * X_mj"
class Policy:

    # PAPER[§3.1|def] "scale_factor s_m: number of workers needed for distributed training"
    def scale_factors_array(self, scale_factors, job_ids, m, n):
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]
        return scale_factors_array

    # PAPER[§3.1|eq] Constraint (1): 0 <= X_mj <= 1
    # PAPER[§3.1|eq] Constraint (2): Σ_j X_mj <= 1 (job cannot use more than 100% of time)
    # PAPER[§3.1|eq] Constraint (3): Σ_m X_mj * scale_factor_m <= num_workers_j (capacity)
    def get_base_constraints(self, x, scale_factors_array):
        return [
            x >= 0,                                                          # Rule 1 (lower bound)
            cp.sum(x, axis=1) <= 1,                                         # Rule 2 (row sums)
            cp.sum(cp.multiply(scale_factors_array, x), axis=0) <= self._num_workers,  # Rule 3 (capacity)
        ]
```

We use a library called CVXPY to express these constraints. It lets us write the math almost exactly as we'd write it on a whiteboard, and then it solves the optimization problem for us.

### Effective Throughput

Now here's where it gets interesting. We don't just care about *time* - we care about *progress*. A job getting 50% of V100 time makes more progress than the same job getting 50% of K80 time (assuming V100s are faster for that job).

We capture this with **effective throughput**:

```
throughput(job, allocation) = Σ (throughput on GPU type j) × (fraction of time on GPU type j)
```

In math notation: `throughput(m, X) = Σ_j T_mj × X_mj`

This number tells us: "Given this allocation, how many training steps per second will this job complete?"

This is the number we actually care about optimizing. Not time, but *progress*.

---

## Chapter 3: How Fast Would You Run?

Before we can schedule fairly, we need to know something crucial: **how fast does each job run on each GPU type?**

This is the throughput matrix T. If we have 5 job types and 3 GPU types, T is a 5×3 table where T[m,j] = "steps per second for job m on GPU type j."

### The Problem

Here's a chicken-and-egg problem: to schedule you fairly, we need to know how fast you'd run on a V100 vs a K80. But we can't actually run you on every GPU type just to find out - that would waste the very resources we're trying to save!

If you have 10 GPU types and 100 job types, that's 1,000 combinations to measure. And every new job type means measuring 10 more combinations. This doesn't scale.

### Our Solution: Fingerprinting

Our solution is like Shazam for jobs.

When you hum a song into Shazam, it doesn't compare your humming to every song ever recorded. It extracts a "fingerprint" - a compressed signature - and matches that against a database of known fingerprints.

We do the same thing:

1. **Partial profiling**: We only measure your job on *some* GPU types. Maybe 20% of combinations. This gives us a partial picture.

2. **Matrix completion**: Remember how Netflix predicts what movies you'd like based on your partial ratings? Same idea. We use matrix completion to fill in the gaps.

3. **Fingerprinting**: Now we compare your partial profile to known job types. "Your performance pattern looks a lot like ResNet-50's pattern." We match you to the closest reference job and use its throughputs.

Here's the core algorithm:

```python
# src/scheduler/throughput_estimator.py

# PAPER[§6] "Throughput estimation using matrix completion and fingerprinting"
# PAPER[§6] "Match unknown jobs to reference job types using partial measurements"
class ThroughputEstimator:

    # PAPER[§6] "Partial profiling: measure subset of throughputs based on profiling_percentage"
    def _profile_jobs(self, true_job_type):
        """Only measure some (job, GPU type) combinations."""
        true_job_type_idx = self._job_types.index(true_job_type)
        profiled_jobs = {}
        for i, worker_type in enumerate(self._worker_types):
            profiled_jobs[worker_type] = {}
            for j, reference_job_type in enumerate(self._reference_job_types):
                r = self._rng.uniform(0, 1)
                if r <= self._profiling_percentage:  # Only measure some combinations
                    offset = i * len(self._reference_job_types) + j
                    profiled_jobs[worker_type][reference_job_type] = \
                        self._normalized_throughputs[true_job_type_idx][offset]
        return profiled_jobs

    # PAPER[§6] "Fingerprinting: match job to reference using partial throughput profile"
    # PAPER[§6] "Matrix completion fills in unmeasured throughput values"
    # PAPER[§6] "Cosine distance (1 - similarity) finds closest reference job type"
    def match_job_to_reference_job(self, true_job_type):
        """Match an unknown job to a known reference job type."""
        profiled_jobs = self._profile_jobs(true_job_type)

        # Build matrix with reference jobs + the new job (last row)
        throughputs_matrix = np.zeros((self._reference_throughputs.shape[0] + 1,
                                       self._reference_throughputs.shape[1]))
        throughputs_matrix[:-1,:] = self._reference_throughputs  # Known jobs

        # Fill in what we measured for the new job
        mask = np.zeros(throughputs_matrix.shape)
        mask[:-1,:] = 1  # We know all reference throughputs

        for i, worker_type in enumerate(sorted(profiled_jobs.keys())):
            for j, reference_job_type in enumerate(self._reference_job_types):
                if reference_job_type in profiled_jobs[worker_type]:
                    offset = i * len(self._reference_job_types) + j
                    throughputs_matrix[-1][offset] = profiled_jobs[worker_type][reference_job_type]
                    mask[-1][offset] = 1  # We measured this one

        # Use matrix completion to fill in unmeasured values
        if np.min(mask) == 0:
            estimated_throughputs = matrix_completion.pmf_solve(
                throughputs_matrix, mask, k=10, mu=1e-2)
            throughputs_matrix = np.where(mask, throughputs_matrix,
                                         np.clip(estimated_throughputs, 0, 1))[0]

        # Find the reference job with the most similar pattern (smallest cosine distance)
        distances = []
        for i, reference_job_type in enumerate(self._reference_job_types):
            distance = cosine_distance(throughputs_matrix[i], throughputs_matrix[-1])
            distances.append((reference_job_type, distance))

        distances.sort(key=lambda x: x[1])  # Sort by distance
        return distances[0][0]  # Return the closest match
```

The `cosine_distance` function measures how different two vectors are. If your job's performance profile points in the same "direction" as ResNet-50's profile (fast on V100s, slower on K80s, etc.), the cosine distance is small, and we match you to ResNet-50.

This is how Gavel handles new job types efficiently - it doesn't need to profile everything, just enough to fingerprint you.

---

## Chapter 4: What Does "Fair" Even Mean?

Here's a question that sounds simple but isn't:

**If you have 10 GPUs and 5 jobs, what's the "fair" allocation?**

Give everyone 2 GPUs? But what if Job A runs 10x faster on V100s while Job B doesn't care? What if Job C has been waiting for hours while Job D just arrived? What if Job E has a deadline in 30 minutes?

There's no single right answer. "Fair" depends on what you value.

So we implemented several definitions of fairness. Your cluster admin picks the one that matches their values. Let's walk through each.

### Max-Min Fairness: "Help the Worst-Off First"

This is the socialist approach to scheduling.

The idea: **maximize the minimum**. Make the worst-off job as well-off as possible. If someone's getting screwed, fix that first - even if it means the rich jobs get a bit less.

But we need to define "well-off." We use **normalized throughput**: how fast are you going compared to what you'd get with a perfectly equal share?

```
normalized_throughput(job) = actual_throughput / throughput_with_equal_share
```

If this number is 1.0, you're getting exactly your fair share. Below 1.0, you're getting less than fair. Above 1.0, you're getting more.

The max-min fair allocation maximizes the minimum normalized throughput across all jobs:

```
Maximize min_over_all_jobs (1/weight) × (throughput / throughput_with_equal_share)
```

Here's the code:

```python
# src/scheduler/policies/max_min_fairness.py

# PAPER[§4.1] "MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)"
# PAPER[§4.1] "Max-min fairness: maximize minimum normalized throughput across jobs"
# PAPER[§4.1|def] "throughput(m,X^equal) = proportional_throughputs (baseline for normalization)"
class MaxMinFairnessPolicy(Policy):

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       priority_weights, cluster_spec):
        # ... setup code ...

        # Get what each job would get with equal share (the baseline)
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)

        # Normalize: actual throughput / fair share throughput
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        x = cp.Variable(throughputs.shape)

        # PAPER[§4.1|eq] Objective: Maximize min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)
        objective = cp.Maximize(
            cp.min(cp.sum(cp.multiply(
                np.multiply(throughputs * priority_weights.reshape((m, 1)),
                            scale_factors_array), x), axis=1)))

        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
```

The key insight: we're not maximizing total throughput. We're maximizing the *minimum* throughput. This naturally helps struggling jobs.

### Finish-Time Fairness (Themis): "Everyone Finishes Together"

Imagine a group project where everyone should finish at the same time. Someone who started early and is behind should get more help to catch up.

Themis uses a metric called **rho (ρ)**: how long will you take to finish compared to how long you *would* have taken with dedicated resources?

```
ρ = expected_completion_time / expected_completion_time_if_you_had_dedicated_resources
```

If ρ = 1.0, you're finishing exactly when you would have with dedicated access. If ρ = 2.0, you're taking twice as long as you "should."

The goal: minimize the maximum ρ. Nobody should be delayed more than necessary.

```python
# src/scheduler/policies/finish_time_fairness.py

# PAPER[§4.2] "Finish-time fairness (Themis): equalize completion-time ratio ρ across jobs"
# PAPER[§4.2|eq] "rho(m,X) = (t_m + remaining/throughput) / (t_isolated + remaining/throughput_isolated)"
# PAPER[§4.2] "Objective: MinimizeX max_m rho(m,X)"
class FinishTimeFairnessPolicyWithPerf(Policy):

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, times_since_start,
                       num_steps_remaining, cluster_spec):
        # ... setup code ...

        expected_time_fractions = []
        for i in range(len(job_ids)):
            # PAPER[§4.2] "Cumulative isolated time: tracks time job would have spent in isolation"
            if job_ids[i] not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[job_ids[i]] = 0

            # Track how long job would have run in isolation
            if job_ids[i] in self._num_steps_remaining_prev_iteration:
                self._cumulative_isolated_time[job_ids[i]] += (
                    self._num_steps_remaining_prev_iteration[job_ids[i]] -
                    num_steps_remaining[job_ids[i]]) / \
                    self._isolated_throughputs_prev_iteration[job_ids[i]]

            allocation_throughput = cp.sum(cp.multiply(throughputs[i], x[i]))

            # PAPER[§4.2] expected_time_isolated = t_isolated + remaining / throughput_isolated
            expected_time_isolated = self._cumulative_isolated_time[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] / isolated_throughputs[i])

            # PAPER[§4.2] expected_time_allocation = t_m + remaining / throughput(m,X)
            expected_time_allocation = times_since_start[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))

            # PAPER[§4.2] rho = expected_time_allocation / expected_time_isolated
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)

        # Minimize the maximum rho
        objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)
```

This policy is great when jobs have different lengths. A quick job that's almost done shouldn't be starved just because a marathon job also needs resources.

### Other Policies

We also implemented:

**FIFO**: First come, first served. Simple, predictable, but doesn't account for heterogeneity.

```python
# PAPER[§4.2] "FIFO: process jobs in arrival order"
# PAPER[§4.2] "Equivalent to throughput maximization with arrival-order priority"
```

**Cost Minimization**: Maximize throughput per dollar spent. Different GPU types cost different amounts.

```python
# PAPER[§4.2] "Cost minimization: maximize throughput per unit cost"
# PAPER[§4.2|eq] "MaximizeX Σ_m throughput(m,X) / Σ_m(Σ_j cost_j * X_mj)"
```

**Makespan Minimization**: Minimize the time until ALL jobs finish. Good when you have a batch of jobs and want the whole batch done ASAP.

```python
# PAPER[§4.2] "Makespan minimization: minimize time for all jobs to complete"
# PAPER[§4.2|eq] "MinimizeX max_m num_steps_m / throughput(m,X)"
```

The point isn't that one policy is "right." The point is that Gavel gives you choices, and all of them are heterogeneity-aware.

---

## Chapter 5: From Theory to Reality - Actually Scheduling Jobs

So we've computed this beautiful allocation matrix that says "Job A should get 30% of V100 time." Great.

But GPUs don't do percentages. A job either runs on a GPU or it doesn't. How do we turn fractions into actual schedules?

### Rounds: Discretizing Time

Our answer: **rounds**. Think of it like a restaurant that seats people in 1-hour slots.

We break time into 6-minute rounds. Each round, we convert the ideal allocation into actual GPU assignments. Over many rounds, the actual time each job receives converges to the ideal allocation.

```python
# src/scheduler/scheduler.py

# PAPER[§5] "Gavel uses a round-based scheduling mechanism where each round lasts 6 minutes"
# PAPER[§5|def] time_per_iteration=360 (default) corresponds to 6-minute scheduling rounds
```

Why 6 minutes? It's a tradeoff. Shorter rounds mean we can react faster to changing conditions, but they also mean more overhead from checkpointing and switching jobs. 6 minutes felt like a good balance.

### Priority: Who's Behind?

Here's the clever part. Say Job A should get 30% of time but has only received 10% so far. Job B should get 20% and has received 25%. Who should go next?

We compute a **priority**:

```
priority = allocation / time_received
```

Job A's priority is 30/10 = 3.0. It should have gotten 3x more time than it has.
Job B's priority is 20/25 = 0.8. It's actually ahead of schedule.

Job A clearly goes first. It's the most behind.

New jobs get very high priority (we multiply by 1 billion) so they get a chance to run quickly.

```python
# src/scheduler/scheduler.py

# PAPER[§5] "priority of job m on worker type j is X_mj / fraction_of_time_received_mj"
# PAPER[§5] "jobs with higher priority are scheduled first"
# PAPER[§5] Priority = allocation / fraction_received; new jobs get allocation * 1e9
def _update_priorities(self):
    for job_id in self._allocation:
        for worker_type in self._allocation[job_id]:
            allocation = self._allocation[job_id][worker_type]
            fraction_received = self._fraction_of_time_received[job_id][worker_type]

            if fraction_received == 0:
                # New job - give it very high priority
                self._priority[job_id][worker_type] = allocation * 1e9
            else:
                # Priority = how much you should get / how much you've gotten
                self._priority[job_id][worker_type] = allocation / fraction_received
```

### Algorithm 1: The Scheduling Loop

Now we have priorities. Here's the actual scheduling algorithm:

```python
# src/scheduler/scheduler.py

# PAPER[§5|alg] Algorithm 1: SCHEDULE_JOBS
# PAPER[§5] "greedily selects jobs in decreasing order of priority until all workers are assigned"
# PAPER[§5] Jobs sorted by (priority, deficit, allocation) for tie-breaking
def _schedule_jobs(self):
    available_workers = copy.deepcopy(self._cluster_spec)
    scheduled_jobs = {}

    # Sort jobs by priority (highest first)
    # Tie-breaker: deficit (how far behind), then allocation
    job_priority_list = []
    for job_id in self._priority:
        for worker_type in self._priority[job_id]:
            priority = self._priority[job_id][worker_type]
            deficit = self._deficit[job_id][worker_type]
            allocation = self._allocation[job_id][worker_type]
            job_priority_list.append((job_id, worker_type, priority, deficit, allocation))

    # Sort by (priority DESC, deficit DESC, allocation DESC)
    job_priority_list.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)

    # Greedily assign GPUs to highest-priority jobs
    for job_id, worker_type, priority, deficit, allocation in job_priority_list:
        scale_factor = self._scale_factors[job_id]

        if available_workers[worker_type] >= scale_factor:
            # We have enough GPUs - assign this job
            scheduled_jobs[job_id] = worker_type
            available_workers[worker_type] -= scale_factor

    return scheduled_jobs
```

It's a greedy algorithm: look at the highest-priority (job, GPU type) pair, assign it if we have capacity, repeat.

### Minimizing Fragmentation

One more detail: we try to minimize fragmentation.

```python
# PAPER[§5] "strided worker assignment to minimize number of servers used"
# PAPER[§5] Jobs sorted by scale_factor (largest first) to reduce fragmentation
```

If you're assigning 8-GPU jobs and 1-GPU jobs, assign the 8-GPU jobs first. Otherwise, you might scatter 1-GPU jobs across your cluster and have no contiguous 8-GPU slots left.

---

## Chapter 6: Running Your Job - Leases and Preemption

Congratulations! You've been assigned to a V100 for this round. But here's the thing - we're not giving you the GPU forever. You get a **lease**: permission to run for a limited time.

### Why Leases?

Because the world changes. New jobs arrive. Priorities shift. That "fair" allocation we computed 6 minutes ago might not be fair anymore.

Leases let us course-correct. At the end of each lease, we re-evaluate and possibly reassign GPUs.

```python
# src/scheduler/lease.py

# PAPER[§5] "Lease: specifies max_steps and max_duration for a scheduling round"
# PAPER[§5] "Jobs receive leases that bound their execution within each round"
class Lease:
    def __init__(self, max_steps, max_duration):
        self._max_steps = max_steps      # You can run at most this many training steps
        self._max_duration = max_duration # You can run at most this many seconds
```

Your lease says: "You may run for up to 360 seconds OR 10,000 training steps, whichever comes first."

### The GavelIterator: Enforcing Leases

Here's where the magic happens. We provide a wrapper called `GavelIterator` that wraps your PyTorch DataLoader. On every training step, it checks: "Am I still allowed to run?"

```python
# src/scheduler/gavel_iterator.py

# PAPER[§5] "GavelIterator wraps PyTorch DataLoader to enforce lease-based execution"
# PAPER[§5] "Runtime component that communicates with scheduler for lease renewal"
class GavelIterator:
    def __init__(self, data_loader, checkpoint_dir, load_checkpoint_func,
                 save_checkpoint_func, ...):
        self._data_loader = data_loader
        self._lease = Lease(0, 0)
        # ... setup ...

    def __next__(self):
        # Update elapsed time
        cur_time = time.time()
        elapsed_time = cur_time - self._prev_time
        self._duration += elapsed_time
        self._prev_time = cur_time

        # Should we renew our lease?
        if (self._steps_until_next_lease_update <= 0 or
            self._time_until_next_lease_update <= 0):
            self._update_lease()

        # PAPER[§5] "Lease expiration triggers job preemption for round-based scheduling"
        lease_expired = (self._duration >= self._lease.max_duration or
                         self._steps >= self._lease.max_steps)

        if lease_expired:
            self._done = True
            raise StopIteration  # Stop training - lease is up

        # Still have lease - return next batch
        val = next(self._iterator)
        self._steps += 1
        return val
```

### Lease Renewal

At 75% of your lease, we phone home: "Hey scheduler, I'm still going - can I get an extension?"

```python
# PAPER[§5] "Jobs request lease renewal at 75% of lease completion"
LEASE_UPDATE_FRACTION = 0.75
```

If you're still the highest priority for this GPU, you get more time seamlessly - no checkpoint needed. If someone else needs it more, your current lease expires and you checkpoint.

### Preemption: Saving Your Game

When your lease expires (and isn't renewed), you're not killed - you're **preempted**.

Think of it like saving your game before someone else needs the console. You checkpoint your model weights and optimizer state, release the GPU, and go back to the waiting pool.

Next round, you might get scheduled again. When you do, you load your checkpoint and continue from where you left off. The training continues seamlessly (minus the checkpoint/restart overhead).

```python
# PAPER[§5] "jobs that were running in the previous round are given priority on the same workers"
# PAPER[§5] "this minimizes preemption overhead through lease extensions"
```

We try to minimize preemptions by:
1. Renewing leases when possible (lease extensions)
2. Giving jobs priority on the same GPUs they were using (avoids checkpoint/restart)

---

## Chapter 7: Advanced Topics

You now understand the core of Gavel. But we built a few advanced features that are worth knowing about.

### Hierarchical Fairness: Fair Across Groups, Then Within Groups

So far, we've talked about fairness between jobs. But what if you're running a shared cluster with multiple research groups?

You want:
1. Fairness between **groups** first (Group A and Group B each get 50% of resources)
2. Then fairness between **jobs within each group** (Jobs in Group A share Group A's 50% fairly)

This is hierarchical fairness, and we solve it with the **water filling algorithm**.

Imagine filling up water glasses at a dinner table. You don't fill one glass completely before moving to the next - you pour a little in each, raising all water levels together.

When one glass is full (a job is "bottlenecked" - it literally cannot use more resources productively), you skip it and keep filling the others.

```python
# src/scheduler/policies/max_min_fairness_water_filling.py

# PAPER[§4.3] "Hierarchical max-min fairness using water filling algorithm"
# PAPER[§4.3] "Iteratively solve LP until all jobs are bottlenecked"
class WaterFillingAlgorithm:

    # PAPER[§4.3|def] "entity_weights w_s: weight assigned to each entity (user/group)"
    # PAPER[§4.3] "job weights w_m^job distributed within entity based on reweighting policy"
    def _compute_priority_weights(self, entity_weights, priority_weights,
                                   entity_to_job_mapping, ...):
        """Distribute entity weight among jobs within that entity."""
        for entity_id in entity_to_job_mapping:
            entity_weight = entity_weights[entity_id]

            # Sum up weights of all non-bottlenecked jobs in this entity
            total_job_priority_in_entity = 0.0
            for job_id in entity_to_job_mapping[entity_id]:
                if job_id not in final_normalized_effective_throughputs:  # Not bottlenecked
                    total_job_priority_in_entity += float(priority_weights[job_id])

            # Distribute entity weight proportionally
            for job_id in entity_to_job_mapping[entity_id]:
                if job_id in final_normalized_effective_throughputs:  # Bottlenecked
                    returned_priority_weights[job_id] = 0.0
                else:
                    returned_priority_weights[job_id] = \
                        entity_weight * (float(priority_weights[job_id]) /
                                        total_job_priority_in_entity)
        return returned_priority_weights
```

The algorithm:

```python
    # PAPER[§4.3|alg] "Water filling: iteratively raise allocation until jobs bottleneck"
    # PAPER[§4.3] "Each iteration: solve LP, find bottlenecked jobs, freeze them, repeat"
    def _run_get_allocation_iterations(self, ...):
        done = False
        final_normalized_effective_throughputs = {}  # Bottlenecked jobs

        while not done:
            # Solve LP to raise everyone's allocation
            x, c, mask = self._get_allocation(...)

            # PAPER[§4.3] "bottleneck detection: jobs that cannot improve without hurting others"
            # PAPER[§4.3] "uses MILP to find jobs at their maximum achievable throughput"
            z = self._get_bottleneck_jobs(...)

            # Freeze bottlenecked jobs
            for i, job_id in enumerate(job_ids):
                if job_id not in final_normalized_effective_throughputs and not z[i]:
                    final_normalized_effective_throughputs[job_id] = current_throughput[i]

            if len(final_normalized_effective_throughputs) == num_jobs:
                done = True  # Everyone is bottlenecked

        return x
```

### Space Sharing: Two Jobs, One GPU

Here's a GPU trick: some jobs don't use 100% of a GPU's memory. A small ResNet-18 might only use 4GB on a 16GB V100.

What if we ran two small jobs on the same GPU simultaneously? This is **space sharing**.

```python
# src/scheduler/job_id_pair.py

# PAPER[§3.1] "Space sharing: multiple jobs can share a single accelerator"
# PAPER[§3.1] "JobIdPair represents either a single job or a pair of co-located jobs"
class JobIdPair():
    def __init__(self, job0, job1):
        # Can represent a single job (job0=id, job1=None)
        # Or a pair of co-located jobs (job0=id1, job1=id2)
        self._job0 = job0
        self._job1 = job1
        self._is_pair = job0 is not None and job1 is not None

    # PAPER[§3.1] "is_pair() returns True for space-shared job combinations"
    def is_pair(self):
        return self._is_pair
```

The throughput matrix now has entries for *pairs* of jobs:
- How fast does ResNet-18 run alone on V100? → T[ResNet-18, V100] = 100 steps/sec
- How fast do ResNet-18 and LanguageModel run together on V100? → T[(ResNet-18, LM), V100] = [80, 50] steps/sec

Their combined throughput (80 + 50 = 130) is higher than either alone! It's like carpooling - two jobs make progress using resources that would otherwise be wasted.

```python
# src/scheduler/policies/policy.py

# PAPER[§3.1|eq] Same constraints as Policy but extended for job packing (space sharing)
# PAPER[§3.1] "space sharing: multiple jobs share the same GPU in a time-multiplexed manner"
class PolicyWithPacking(Policy):
    """Extended policy that considers space sharing combinations."""
```

---

## The Complete Picture

And that's Gavel! Let's recap the journey your job took:

1. **You arrived** with your job type, steps, scale factor, and priority
2. **We estimated your throughputs** on each GPU type (fingerprinting + matrix completion)
3. **We computed a fair allocation** based on your cluster's chosen policy (max-min, Themis, etc.)
4. **Every 6 minutes**, we converted that allocation into GPU assignments using priorities
5. **You ran with a lease**, checkpointing when preempted
6. **Eventually, you finished** training

The beautiful thing is: this all happens automatically. You submit a job, and Gavel handles the heterogeneity-aware, fairness-driven scheduling behind the scenes.

---

## Quick Reference: Where to Look

| "I want to understand..." | Start here |
|---------------------------|------------|
| How jobs are represented | `job.py` |
| The allocation matrix and constraints | `policy.py:59-67` |
| Max-min fairness | `max_min_fairness.py` |
| Finish-time fairness (Themis) | `finish_time_fairness.py` |
| Hierarchical water filling | `max_min_fairness_water_filling.py` |
| How priorities are computed | `scheduler.py:2367` |
| The scheduling loop (Algorithm 1) | `scheduler.py:770` |
| Lease enforcement | `gavel_iterator.py` |
| Throughput estimation | `throughput_estimator.py` |
| Space sharing | `job_id_pair.py`, `policy.py:169` |

---

## Key Equations at a Glance

**Effective Throughput** (how fast a job progresses with allocation X):
```
throughput(m, X) = Σ_j T_mj × X_mj
```

**Max-Min Fairness Objective**:
```
Maximize min_m (1/w_m) × throughput(m,X) / throughput(m, X^equal)
```

**Themis ρ Metric** (finish-time fairness):
```
ρ(m,X) = (time_so_far + remaining_steps/throughput) / (isolated_time + remaining_steps/isolated_throughput)
```

**Priority** (for round-based scheduling):
```
priority[m,j] = allocation[m,j] / fraction_of_time_received[m,j]
```

---

*Thanks for walking through Gavel with us. We hope this makes the code easier to navigate. If something's still confusing, the comments in the code (marked with `# PAPER[§X.Y]`) will point you back to the relevant section of the paper.*

*Happy scheduling!*
