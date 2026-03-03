import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking, solve_with_fallback
from proportional import ProportionalPolicy

# PAPER[§4.1] "MaximizeX min_m (1/w_m) * throughput(m,X) / throughput(m,X^equal)"
# PAPER[§4.1] "Max-min fairness: maximize minimum normalized throughput across jobs"
# PAPER[§4.1|def] "throughput(m,X^equal) = proportional_throughputs (baseline for normalization)"
class MaxMinFairnessPolicy(Policy):

    def __init__(self, solver, solver_kwargs=None):
        self._name = 'MaxMinFairness'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessPolicyWithPerf(solver, solver_kwargs=solver_kwargs)

    def set_migration_context(self, migration_times, time_per_iteration):
        self._max_min_fairness_perf_policy.set_migration_context(
            migration_times, time_per_iteration)

    def set_fragmentation_context(self, frag_ema, penalty_weight):
        self._max_min_fairness_perf_policy.set_fragmentation_context(
            frag_ema, penalty_weight)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       priority_weights, cluster_spec, gpu_demands=None):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        # PAPER[§4.1] "Heterogeneity-agnostic variant: sets all throughputs to 1.0"
        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        return self._max_min_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, priority_weights,
            cluster_spec, gpu_demands=gpu_demands)


# PAPER[§4.1] MaxMinFairness_Perf: heterogeneity-aware variant using actual throughputs
class MaxMinFairnessPolicyWithPerf(Policy):

    def __init__(self, solver, solver_kwargs=None):
        Policy.__init__(self, solver, solver_kwargs=solver_kwargs)
        self._name = 'MaxMinFairness_Perf'
        self._proportional_policy = ProportionalPolicy()
        # Track previous allocation for switching penalty and warm-start
        self._prev_allocation = None  # {job_id: {worker_type: fraction}}
        # DPP problem cache: reuse compiled problem when shape matches.
        # Only used when migration penalty is active (penalty stabilizes
        # allocations, giving high cache hit rate).
        self._dpp_cache = None  # dict with shape, x, t, params, problem
        # Fragmentation-aware allocation: penalize allocation to GPU types
        # with high fragmentation EMA.
        self._frag_ema = None  # {worker_type: float}
        self._frag_penalty_weight = 0.0

    def set_fragmentation_context(self, frag_ema, penalty_weight):
        """Set fragmentation context for fragmentation-aware allocation.

        Args:
            frag_ema: Dict mapping worker_type to fragmentation EMA value.
            penalty_weight: Weight (lambda) for fragmentation penalty in LP.
        """
        self._frag_ema = frag_ema
        self._frag_penalty_weight = penalty_weight

    def _build_dpp_problem(self, m, n):
        """Build a DPP-parametrized LP for shape (m, n).

        Uses explicit auxiliary variables for the abs term so that all
        parameters appear affinely in the canonicalized problem:
        - coeff_param: objective coefficients (affine in objective)
        - sf_param: scale factors in capacity constraint (affine in RHS)
        - alpha_param: penalty weight (affine in objective)
        - x_prev_param: previous allocation (affine in constraint RHS)
        - mask_param: zero-throughput mask (affine in constraint RHS)
        """
        x = cp.Variable((m, n))
        # Explicit auxiliary for |x - x_prev| to ensure DPP compliance.
        # cp.abs(x - param) is not recognized as DPP by cvxpy, but manual
        # reformulation with t >= x - param, t >= param - x is.
        t = cp.Variable((m, n), nonneg=True)

        coeff_param = cp.Parameter((m, n))
        sf_param = cp.Parameter((m, n), nonneg=True)
        alpha_param = cp.Parameter(m, nonneg=True)
        x_prev_param = cp.Parameter((m, n))
        mask_param = cp.Parameter((m, n), nonneg=True)

        per_job_throughput = cp.sum(cp.multiply(coeff_param, x), axis=1)
        switch_per_job = cp.sum(t, axis=1)
        penalty = cp.multiply(alpha_param, switch_per_job)

        objective = cp.Maximize(cp.min(per_job_throughput - penalty))

        constraints = [
            x >= 0,
            cp.sum(cp.multiply(sf_param, x), axis=0) <= self._num_workers,
            cp.sum(x, axis=1) <= 1,
            x <= mask_param,
            # Manual abs reformulation: t >= |x - x_prev|
            t >= x - x_prev_param,
            t >= x_prev_param - x,
        ]

        problem = cp.Problem(objective, constraints)

        return {
            'shape': (m, n),
            'x': x,
            't': t,
            'coeff': coeff_param,
            'sf': sf_param,
            'alpha': alpha_param,
            'x_prev': x_prev_param,
            'mask': mask_param,
            'problem': problem,
        }

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       gpu_demands=None):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        # Capacity array: use gpu_demands for fractional GPU capacity
        # in the LP constraint, else fall back to scale_factors.
        # This lets four 0.25-GPU jobs share one GPU slot in the LP.
        if gpu_demands is not None:
            capacity_array = self.scale_factors_array(
                gpu_demands, job_ids, m, n)
        else:
            capacity_array = scale_factors_array

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        # Coefficients still use scale_factors_array (NOT capacity_array).
        # Throughput is already scaled by gpu_request at scheduler.py:2800,
        # so using gpu_demands here would double-penalize fractional jobs.
        coefficients = np.multiply(
            throughputs * priority_weights.reshape((m, 1)),
            scale_factors_array)

        # Build previous allocation matrix (used for switching penalty
        # and warm-start seeding). New jobs get x_prev=0.
        x_prev = None
        if self._prev_allocation is not None:
            x_prev = np.zeros((m, n))
            for i, job_id in enumerate(job_ids):
                if job_id in self._prev_allocation:
                    for j, wt in enumerate(worker_types):
                        x_prev[i, j] = self._prev_allocation[job_id].get(
                            wt, 0.0)

        # Choose code path:
        #  1. Standard (no penalty): preserves exact determinism for tests
        #  2. DPP-cached (penalty, small problems): compiles cone form once
        #  3. Fresh penalty (penalty, large problems): builds LP each round
        #     to avoid DPP compilation overhead at scale
        use_penalty = (self._migration_times is not None
                       and self._time_per_iteration is not None
                       and self._time_per_iteration > 0)

        # DPP compilation is expensive for large parameter matrices, but
        # the cost is one-time and amortized over hundreds of rounds.
        # 50000 comfortably covers Alibaba-scale (3500*6 = 21000).
        _DPP_MAX_ELEMENTS = 50000

        if use_penalty and m * n <= _DPP_MAX_ELEMENTS:
            solved_x = self._solve_dpp(
                m, n, job_ids, worker_types, throughputs,
                coefficients, capacity_array, x_prev)
        elif use_penalty:
            solved_x = self._solve_penalty_fresh(
                m, n, job_ids, throughputs,
                coefficients, capacity_array, x_prev)
        else:
            solved_x = self._solve_standard(
                m, n, throughputs, coefficients, capacity_array, x_prev,
                worker_types)

        # Always save allocation for warm-start seeding and switching penalty
        self._prev_allocation = {}
        for i, job_id in enumerate(job_ids):
            self._prev_allocation[job_id] = {}
            for j, wt in enumerate(worker_types):
                self._prev_allocation[job_id][wt] = float(solved_x[i, j])

        return super().unflatten(solved_x, index)

    def _solve_standard(self, m, n, throughputs, coefficients,
                        scale_factors_array, x_prev, worker_types=None):
        """Original non-cached solve path (no migration penalty).

        When fragmentation context is set (via set_fragmentation_context),
        adds a penalty term to the objective to discourage allocation to
        GPU types with high fragmentation.
        """
        x = cp.Variable((m, n))
        per_job_throughput = cp.sum(cp.multiply(coefficients, x), axis=1)

        # Build objective with optional fragmentation penalty.
        # Penalty = lambda * sum_m sum_t (frag_ema[t] * X[m,t])
        # This discourages allocating jobs to GPU types with high fragmentation.
        if (self._frag_ema is not None
                and self._frag_penalty_weight > 0
                and worker_types is not None):
            # Build fragmentation weight vector [frag_v100, frag_p100, ...]
            frag_weights = np.array([
                self._frag_ema.get(wt, 0.0) for wt in worker_types
            ])
            # Total fragmentation penalty: sum over all allocations weighted by
            # per-GPU-type fragmentation. This is a scalar.
            frag_penalty = cp.sum(x @ frag_weights)
            objective = cp.Maximize(
                cp.min(per_job_throughput) - self._frag_penalty_weight * frag_penalty
            )
        else:
            objective = cp.Maximize(cp.min(per_job_throughput))

        constraints = self.get_base_constraints(x, scale_factors_array)
        for i in range(m):
            for j in range(n):
                if throughputs[i, j] == 0:
                    constraints.append(x[i, j] == 0)

        cvxprob = cp.Problem(objective, constraints)

        use_warm_start = False
        if x_prev is not None:
            x.value = x_prev
            use_warm_start = True

        solve_with_fallback(cvxprob, self._solver, warm_start=use_warm_start,
                             solver_kwargs=self._solver_kwargs)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return x.value.clip(min=0.0).clip(max=1.0)

    def _solve_penalty_fresh(self, m, n, job_ids, throughputs,
                             coefficients, scale_factors_array, x_prev):
        """Non-cached solve with migration penalty for large problems.

        Builds the LP fresh each round using concrete numpy values.
        At this scale, cp.abs(x - numpy_array) works natively without
        the DPP auxiliary-variable trick, and we avoid the expensive
        DPP compilation pass over large parameter matrices.
        """
        x = cp.Variable((m, n))

        # Compute penalty weights from migration times.
        alpha = np.zeros(m)
        if x_prev is not None:
            for i, job_id in enumerate(job_ids):
                mt = self._migration_times.get(job_id, 0)
                migration_frac = mt / self._time_per_iteration
                alpha[i] = migration_frac * np.max(coefficients[i]) / 2.0

        per_job_throughput = cp.sum(cp.multiply(coefficients, x), axis=1)

        if x_prev is not None and np.any(alpha > 0):
            switching_cost = cp.sum(cp.abs(x - x_prev), axis=1)
            penalty = cp.multiply(alpha, switching_cost)
            objective = cp.Maximize(cp.min(per_job_throughput - penalty))
        else:
            objective = cp.Maximize(cp.min(per_job_throughput))

        constraints = self.get_base_constraints(x, scale_factors_array)
        # Vectorized zero-throughput mask instead of O(m*n) individual
        # constraints (significant at Alibaba scale with 21000+ entries).
        mask = (throughputs > 0).astype(np.float64)
        constraints.append(x <= mask)

        cvxprob = cp.Problem(objective, constraints)

        use_warm_start = False
        if x_prev is not None:
            x.value = x_prev
            use_warm_start = True

        solve_with_fallback(cvxprob, self._solver, warm_start=use_warm_start,
                             solver_kwargs=self._solver_kwargs)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return x.value.clip(min=0.0).clip(max=1.0)

    def _solve_dpp(self, m, n, job_ids, worker_types, throughputs,
                   coefficients, scale_factors_array, x_prev):
        """DPP-cached solve path (migration penalty active)."""
        # Build or reuse DPP-cached problem.
        if self._dpp_cache is None or self._dpp_cache['shape'] != (m, n):
            self._dpp_cache = self._build_dpp_problem(m, n)

        cache = self._dpp_cache
        x = cache['x']

        # Compute penalty alpha vector.
        alpha = np.zeros(m)
        if x_prev is not None:
            migration_frac = np.zeros(m)
            for i, job_id in enumerate(job_ids):
                mt = self._migration_times.get(job_id, 0)
                migration_frac[i] = mt / self._time_per_iteration
            peak_throughput = np.max(coefficients, axis=1)
            alpha = migration_frac * peak_throughput / 2.0

        # Update parameter values for this round.
        cache['coeff'].value = coefficients
        cache['sf'].value = scale_factors_array
        cache['alpha'].value = alpha
        cache['x_prev'].value = x_prev if x_prev is not None else np.zeros((m, n))
        cache['mask'].value = (throughputs > 0).astype(np.float64)

        # Warm-start: seed x with previous allocation.
        use_warm_start = False
        if x_prev is not None:
            x.value = x_prev
            use_warm_start = True

        solve_with_fallback(cache['problem'], self._solver,
                             warm_start=use_warm_start,
                             solver_kwargs=self._solver_kwargs)

        if cache['problem'].status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return x.value.clip(min=0.0).clip(max=1.0)


class MaxMinFairnessPolicyWithPacking(PolicyWithPacking):

    def __init__(self, solver, solver_kwargs=None):
        PolicyWithPacking.__init__(self, solver, solver_kwargs=solver_kwargs)
        self._name = 'MaxMinFairness_Packing'
        self._proportional_policy = ProportionalPolicy()

    def get_allocation_using_job_type_throughputs(
            self, unflattened_throughputs, job_id_to_job_type_key,
            scale_factors, unflattened_priority_weights, cluster_spec):
        job_ids = sorted(job_id_to_job_type_key.keys())
        if len(job_ids) == 0:
            return None
        job_type_keys = sorted(unflattened_throughputs.keys())
        worker_types = sorted(cluster_spec.keys())
        num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Create a map from job type to list of job indexes.
        job_type_key_to_job_idx = {}
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            if job_type_key not in job_type_key_to_job_idx:
                job_type_key_to_job_idx[job_type_key] = []
            job_type_key_to_job_idx[job_type_key].append(i)

        # Num jobs.
        n = len(job_ids)
        # Num job_types.
        a = len(unflattened_throughputs.keys())
        # Num worker_types.
        m = len(worker_types)
        # Num varibles per job.
        num_vars_per_job = 1 + a

        # Set up scale factors.
        flattened_scale_factors = \
            np.reshape([scale_factors[job_id] for job_id in job_ids], (n, 1))
        scale_factors_array = np.tile(flattened_scale_factors,
                                        (1, num_vars_per_job * m))

        # Set up flattened job type throughputs.
        flattened_throughputs = np.zeros(shape=(a, (1 + a) * m),
                                         dtype=np.float32)
        for i, job_type_key in enumerate(job_type_keys):
            for k, worker_type in enumerate(worker_types):
                for j, other_job_type_key in enumerate([None] + job_type_keys):
                    if j > 0 and other_job_type_key[1] != job_type_key[1]:
                        flattened_throughputs[i,k*(1+a)+j] = 0.0
                    else:
                        flattened_throughputs[i,k*(1+a)+j] = \
                            unflattened_throughputs[job_type_key][worker_type][other_job_type_key]

        # Set up masks to avoid double-counting allocation values when
        # computing constraint that the sum of allocation values of each
        # worker type must be <= the number of workers of that worker type.
        # TODO: Change this if we ever consider combinations larger than pairs.
        masks = np.full(shape=(n, num_vars_per_job), fill_value=0.5)
        masks[:,0] = 1.0

        # Allocation matrix.
        x = cp.Variable((n, num_vars_per_job * m))

        constraints = [
            # All allocation values must be >= 0.
            x >= 0,
            # The sum of allocation values for each job must be <= 1.
            cp.sum(x, axis=1) <= 1
        ]

        # The sum of allocation values for each worker type must be <=
        # the number of workers of that type.
        per_worker_type_allocations = []
        for i in range(m):
            relevant_vars = \
                x[:,i*num_vars_per_job:(i+1)*num_vars_per_job]
            relevant_scale_factors = \
                scale_factors_array[:,i*num_vars_per_job:(i+1)*num_vars_per_job]
            per_worker_type_allocations.append(
                cp.sum(cp.multiply(relevant_vars,
                                   cp.multiply(relevant_scale_factors,
                                               masks))))
        constraints.append(
                cp.hstack(per_worker_type_allocations) <= num_workers)

        # Set the following constraints:
        # for all job type pairs a, b:
        #   sum of allocation of all jobs of type a paired with type b ==
        #   sum of allocation of all jobs of type b paired with type a
        lhs = []
        rhs = []
        for i, job_type_key_0 in enumerate(job_type_keys):
            for j, job_type_key_1 in enumerate(job_type_keys):
                if j <= i:
                    continue
                elif job_type_key_0[1] != job_type_key_1[1]:
                    continue

                # Retrieve the list of jobs of each type.
                job_type_0_jobs = job_type_key_to_job_idx[job_type_key_0]
                job_type_1_jobs = job_type_key_to_job_idx[job_type_key_1]

                for k in range(m):
                    job_type_0_mask = np.zeros(x.shape)
                    job_type_1_mask = np.zeros(x.shape)

                    # Allocation of job_type_0 jobs when paired with job_type_1
                    for job_idx in job_type_0_jobs:
                        offset = k * num_vars_per_job + 1 + j
                        job_type_0_mask[job_idx,offset] = 1

                    # Allocation of job_type_1 jobs when paired with job_type_0
                    for job_idx in job_type_1_jobs:
                        offset = k * num_vars_per_job + 1 + i
                        job_type_1_mask[job_idx,offset] = 1

                    lhs.append(cp.sum(x[job_type_0_mask == 1]))
                    rhs.append(cp.sum(x[job_type_1_mask == 1]))

        assert (len(lhs) == len(rhs))
        if len(lhs) > 0:
            constraints.append(cp.hstack(lhs) == cp.hstack(rhs))

        # Add constraints to make all variables of the form i-A where job i
        # is of job type A equal.
        for i, job_type_key in enumerate(job_type_keys):
            for k in range(m):
                same_job_type_vars = []
                job_type_jobs = job_type_key_to_job_idx[job_type_key]

                # Find all variables for job-job_type pairs where the job
                # types match.
                offset = k * num_vars_per_job + 1 + i
                for job_idx in job_type_jobs:
                    same_job_type_vars.append(x[job_idx, offset])

                # Constrain the variables to all be equal.
                c = cp.Variable()
                constraints.append(cp.hstack(same_job_type_vars) == c)

        throughputs_no_packed_jobs = np.zeros((len(job_ids), len(worker_types)))
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[job_type_key][worker_type][None]
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (job_ids, worker_types),
            cluster_spec)

        # Allocation coefficients.
        all_coefficients = np.zeros((n, num_vars_per_job * m))
        for i, job_id in enumerate(job_ids):
            job_type_key = job_id_to_job_type_key[job_id]
            job_type_idx = job_type_keys.index(job_type_key)
            if len(job_type_key_to_job_idx[job_type_key]) == 1:
                for k, worker_type in enumerate(worker_types):
                    offset = k * num_vars_per_job + 1 + job_type_idx
                    constraints.append(x[i,offset] == 0.0)
            proportional_throughput = proportional_throughputs[i]
            all_coefficients[i] = \
                np.multiply(flattened_throughputs[job_type_idx],
                            scale_factors_array[i]) /\
                    (unflattened_priority_weights[job_id] * proportional_throughput)
        objective = \
            cp.Maximize(cp.min(cp.sum(cp.multiply(all_coefficients, x),
                                      axis=1)))

        cvxprob = cp.Problem(objective, constraints)
        result = solve_with_fallback(cvxprob, self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        allocation = x.value.clip(min=0.0).clip(max=1.0)

        # Unflatten allocation.
        unflattened_allocation = {}
        for i, job_id in enumerate(job_ids):
            unflattened_allocation[job_id] = {}
            for j, worker_type in enumerate(worker_types):
                unflattened_allocation[job_id][worker_type] = {}
                for k, job_type_key in enumerate([None] + job_type_keys):
                    unflattened_allocation[job_id][worker_type][job_type_key] = \
                        allocation[i, j * num_vars_per_job + k]

        return self.convert_job_type_allocation(unflattened_allocation,
                                                job_id_to_job_type_key)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       gpu_demands=None):
        all_throughputs, index = \
            self.flatten(d=unflattened_throughputs,
                         cluster_spec=cluster_spec,
                         priority_weights=unflattened_priority_weights)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        x = cp.Variable((m, n))

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        throughputs_no_packed_jobs = np.zeros((len(single_job_ids), n))
        for i, single_job_id in enumerate(single_job_ids):
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[single_job_id][worker_type]
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (single_job_ids, worker_types),
            cluster_spec)

        objective_terms = []
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs.
        import scipy.sparse as sp
        idx = []
        tputs = []
        # compute the obejctive in a vectorized fashion
        for i in range(len(all_throughputs)):
            indexes = relevant_combinations[single_job_ids[i]]
            idx += indexes
            proportional_throughput = float(proportional_throughputs[i])
            curr_throughputs = np.multiply(
                    all_throughputs[i][indexes],
                    scale_factors_array[indexes]) / proportional_throughput
            tputs.append(curr_throughputs)

        tputs = sp.csc_matrix(np.vstack(tputs))
        indexed_vars = x[idx]
        realized_tputs = cp.multiply(tputs, indexed_vars)
        # reshape so that the sum of each row gives the throughput
        realized_tputs_mat = cp.reshape(realized_tputs,
                (len(all_throughputs),
                int(np.prod(realized_tputs.shape) / len(all_throughputs))),
                order='C')

        objective_fn = cp.min(cp.sum(realized_tputs_mat, axis=1))

        objective = cp.Maximize(objective_fn)

        # Make sure the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, single_job_ids,
                                               scale_factors_array,
                                               relevant_combinations)

        # Explicitly constrain all allocation values with an effective scale
        # factor of 0 to be 0.
        # NOTE: This is not strictly necessary because these allocation values
        # do not affect the optimal allocation for nonzero scale factor
        # combinations.
        for i in range(m):
            for j in range(n):
                if scale_factors_array[i,j] == 0:
                    constraints.append(x[i,j] == 0)
        cvxprob = cp.Problem(objective, constraints)
        if self._solver == 'SCS':
            # anderson acceleration is sometimes unstable, and adds
            # significant overhead
            kwargs = {'acceleration_lookback': 0}
        else:
            kwargs = {}

        result = solve_with_fallback(cvxprob, self._solver, **kwargs)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return self.unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
