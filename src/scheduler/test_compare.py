#!/usr/bin/env python
"""Quick comparison: strided vs fgd placement under GPU sharing."""
import sys
sys.path.append('.')
import scheduler, utils

cluster_spec = {'v100': 4, 'p100': 0, 'k80': 0}
num_gpus_per_server = {'v100': 4, 'p100': 4, 'k80': 4}

for strategy in ['strided', 'fgd']:
    policy = utils.get_policy('fifo', seed=0, solver='ECOS')
    sched = scheduler.Scheduler(
        policy, throughputs_file='simulation_throughputs.json',
        seed=0, time_per_iteration=360,
        simulate=True, placement_strategy=strategy, gpu_sharing_mode=True)
    sched.simulate(cluster_spec, lam=0.0, num_total_jobs=50,
                   num_gpus_per_server=num_gpus_per_server,
                   gpu_milli_generator_func=utils._generate_gpu_milli_sharing)
    jct = sched.get_average_jct(verbose=False)
    makespan = sched.get_current_timestamp()
    sched.shutdown()
    print('RESULT %s: avg_jct=%.2f, makespan=%.2f' % (strategy, jct, makespan))
