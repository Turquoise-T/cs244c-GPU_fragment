import csv
from datetime import datetime
import json
import os
import pickle
import psutil
import random
import re
import socket
import subprocess

from job import Job
from job_table import JobTable
from policies import allox, fifo, finish_time_fairness, gandiva, isolated, \
    max_min_fairness, max_min_fairness_water_filling, max_sum_throughput, \
    min_total_duration

def _generate_scale_factor(rng):
    # Sample the scale factor from the Philly distribution.
    scale_factor = 1
    r = rng.uniform(0, 1)
    if 0.7 <= r <= 0.8:
        scale_factor = 2
    elif 0.8 <= r <= 0.95:
        scale_factor = 4
    elif 0.95 <= r:
        scale_factor = 8
    return scale_factor


def _generate_scale_factor_fragmentation_friendly(rng):
    """Alibaba-like mix: many small jobs (1–2 GPU), some 4-GPU.
    Creates fragmentation on 4-GPU-per-server clusters so FGD vs strided shows difference.
    ~55% 1-GPU, ~30% 2-GPU, ~15% 4-GPU (no 8 to avoid always filling 2 servers)."""
    r = rng.uniform(0, 1)
    if r < 0.55:
        return 1
    if r < 0.85:
        return 2
    return 4


def _generate_gpu_milli_sharing(rng):
    """Generate fractional GPU requests (gpu_milli) for GPU sharing workloads.

    Alibaba-like distribution: mix of small and large GPU fractions.
    Returns gpu_milli value (0-1000, where 1000 = 1.0 GPU).

    Distribution:
      ~20% request 200 milli (0.2 GPU)
      ~20% request 300 milli (0.3 GPU)
      ~25% request 500 milli (0.5 GPU)
      ~15% request 700 milli (0.7 GPU)
      ~20% request 1000 milli (1.0 full GPU)
    """
    r = rng.uniform(0, 1)
    if r < 0.20:
        return 200
    elif r < 0.40:
        return 300
    elif r < 0.65:
        return 500
    elif r < 0.80:
        return 700
    else:
        return 1000


def format_job_for_print(job, index=None):
    """Return a short readable string for one job (for debugging/samples)."""
    prefix = "" if index is None else f"[{index}] "
    return (
        f"{prefix}job_type={job.job_type!r} scale_factor={job.scale_factor} "
        f"total_steps={job.total_steps} priority={job.priority_weight} "
        f"gpu_milli={getattr(job, 'gpu_milli', job.scale_factor * 1000)}"
    )

def _generate_duration(rng):
    # Sample the job duration from the Philly distribution.
    if rng.random() >= 0.8:
        run_time = 60 * (10 ** rng.uniform(3, 4))
    else:
        run_time = 60 * (10 ** rng.uniform(1.5, 3))
    return run_time

def generate_job(throughputs, reference_worker_type='v100', rng=None,
                 job_id=None, fixed_job_duration=None,
                 generate_multi_gpu_jobs=False,
                 generate_multi_priority_jobs=False, run_dir=None,
                 scale_factor_generator_func=_generate_scale_factor,
                 duration_generator_func=_generate_duration,
                 scale_factor_rng=None, duration_rng=None, SLO_rng=None,
                 always_generate_scale_factor=True,
                 gpu_milli_generator_func=None):
    """Generates a new job.

       Args:
         throughputs: A dict containing pre-measured throughputs.
         reference_worker_type: The worker type to use when calculating steps.
         rng: A random number generator for selecting job parameters.
         job_id: The job's ID.
         fixed_job_duration: If set, fixes the duration to the specified value.
         generate_multi_gpu_jobs: If set, generate a scale factor >= 1.
         generate_multi_priority_jobs: If set, generate a priority >= 1.
         run_dir: The directory to run the job from.
         scale_factor_generator_func: A function that accepts an RNG parameter
                                      and returns a job size.
         duration_generator_func: A function that accepts an RNG parameter and
                                  returns a job duration in seconds.
         scale_factor_rng: A random number generator specifically for
                           generating scale factors.
         duration_rng: A random number generator specifically for generating
                       durations.
         SLO_rng: If set, generate an SLO >= 1 using this RNG.
         always_generate_scale_factor: If set, generate a scale factor
                                       regardless of whether user has
                                       requested multi-GPU jobs.
         gpu_milli_generator_func: If set, a function that accepts an RNG and
                                   returns a gpu_milli value (0-1000) for
                                   fractional GPU sharing. Fractional jobs
                                   (gpu_milli < 1000) get scale_factor=1.
      Returns:
        The generated Job.
    """

    if rng is None:
        rng = random.Random()
    if scale_factor_rng is None:
        scale_factor_rng = rng
    if duration_rng is None:
        duration_rng = rng

    # Generate gpu_milli for GPU sharing mode
    gpu_milli_value = None
    if gpu_milli_generator_func is not None:
        gpu_milli_value = gpu_milli_generator_func(rng)

    job_template = None

    if always_generate_scale_factor:
        scale_factor = scale_factor_generator_func(scale_factor_rng)
    else:
        # NOTE: We select the job template here to maintain backwards
        # compatability with scripts/utils/generate_trace.py
        job_template = rng.choice(JobTable)
        if generate_multi_gpu_jobs and job_template.distributed:
            scale_factor = scale_factor_generator_func(scale_factor_rng)
        else:
            scale_factor = 1

    if fixed_job_duration:
        run_time = fixed_job_duration
    else:
        run_time = duration_generator_func(duration_rng)
    if not generate_multi_gpu_jobs:
        scale_factor = 1

    # For GPU sharing: fractional jobs (gpu_milli < 1000) always use scale_factor=1
    if gpu_milli_value is not None and gpu_milli_value < 1000:
        scale_factor = 1

    assert(run_time > 0)
    assert(scale_factor >= 1 and scale_factor <= 8)

    # Sample the job type.
    if job_template is None:
        while True:
            job_template = rng.choice(JobTable)
            if (scale_factor == 1 or
                (scale_factor > 1 and job_template.distributed)):
                break
    job_type = job_template.model

    # Complete the job command with the run directory.
    command = job_template.command
    if run_dir is not None:
        if job_template.needs_data_dir:
            command = command % (run_dir, run_dir)
        else:
            command = command % (run_dir)

    # Compute the number of steps the job will run for given its duration.
    key = (job_type, scale_factor)
    assert(key in throughputs[reference_worker_type])
    num_steps = run_time * throughputs[reference_worker_type][key]['null']
    assert(num_steps > 0)

    # Optionally assign a priority to the job.
    priority_weight = 1.0
    if generate_multi_priority_jobs:
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.2:
            priority_weight = 5.0

    # Optionally assign an SLO to the job.
    SLO = None
    if SLO_rng is not None:
        r = SLO_rng.uniform(0, 1)
        if 0.0 <= r < 0.33:
            SLO = 1.2
        elif 0.33 <= r < 0.67:
            SLO = 2.0
        else:
            SLO = 10.0

    job = Job(job_id=job_id,
              job_type=job_type,
              command=command,
              working_directory=job_template.working_directory,
              num_steps_arg=job_template.num_steps_arg,
              total_steps=num_steps,
              duration=run_time,
              scale_factor=scale_factor,
              priority_weight=priority_weight,
              SLO=SLO,
              needs_data_dir=job_template.needs_data_dir,
              gpu_milli=gpu_milli_value)

    return job

def load_philly_job_distribution():
    with open('philly_job_distribution.pickle', 'rb') as f:
        return pickle.load(f)

def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def get_num_gpus():
    command = 'nvidia-smi -L'
    output = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                            shell=True).stdout.decode('utf-8').strip()
    return len(output.split('\n'))

def get_pid_for_job(command):
    pids = []
    for proc in psutil.process_iter():
        cmdline = ' '.join(proc.cmdline())
        if cmdline == command:
            pids.append(proc.pid)
    return min(pids)

def get_gpu_processes():
    output = subprocess.check_output('nvidia-smi').decode('utf-8')
    gpu_processes = {}
    processes_flag = False
    for line in output.split('\n'):
        if 'Processes' in line:
            processes_flag = True
            continue
        if processes_flag:
            res = re.search('(\d+) +(\d+) +(\w+) +(.+) +(\d+)MiB', line)
            if res is not None:
                gpu_id = int(res.group(1))
                if gpu_id not in gpu_processes:
                    gpu_processes[gpu_id] = []
                pid = int(res.group(2))
                process_name = res.group(4)
                if process_name != 'nvidia-cuda-mps-server':
                    gpu_processes[gpu_id].append(pid)
    return gpu_processes

def get_available_policies():
    return ['allox',
            'fifo', 'fifo_perf', 'fifo_packed',
            'finish_time_fairness',
            'finish_time_fairness_perf',
            'finish_time_fairness_packed',
            'gandiva',
            'isolated',
            'max_min_fairness',
            'max_min_fairness_perf',
            'max_min_fairness_packed',
            'max_min_fairness_water_filling',
            'max_min_fairness_water_filling_perf',
            'max_min_fairness_water_filling_packed',
            'max_sum_throughput_perf',
            'max_sum_throughput_normalized_by_cost_perf',
            'max_sum_throughput_normalized_by_cost_perf_SLOs',
            'max_sum_throughput_normalized_by_cost_packed_SLOs',
            'min_total_duration',
            'min_total_duration_perf',
            'min_total_duration_packed',
            ]

def read_per_instance_type_spot_prices_aws(directory):
    # TODO: Make this flexible.
    directory = os.path.join(directory, 'us-east-1')
    per_instance_type_spot_prices = {}
    for filename in os.listdir(directory):
        full_filepath = os.path.join(directory, filename)
        with open(full_filepath, 'r') as f:
            json_obj = json.load(f)
            for x in json_obj['SpotPriceHistory']:
                instance_type = x['InstanceType']
                if instance_type not in per_instance_type_spot_prices:
                    per_instance_type_spot_prices[instance_type] = []
                per_instance_type_spot_prices[instance_type].append(x)
    return per_instance_type_spot_prices

def read_per_instance_type_spot_prices_azure(directory):
    per_instance_type_spot_prices = {}
    for filename in os.listdir(directory):
        full_filepath = os.path.join(directory, filename)
        with open(full_filepath, 'r') as f:
            zone = filename.replace(".csv", "")
            reader = csv.reader(f)
            i = 0
            for row in reader:
                if i == 0:
                    header = row
                    for header_elem in header[1:]:
                        if header_elem not in per_instance_type_spot_prices:
                            per_instance_type_spot_prices[header_elem] = {}
                else:
                    for (header_elem, row_elem) in zip(header[1:], row[1:]):
                        if (zone not in per_instance_type_spot_prices[header_elem]):
                            per_instance_type_spot_prices[header_elem][zone] = []
                        date = datetime.strptime(row[0], '%m/%d/%Y')
                        per_instance_type_spot_prices[header_elem][zone].append((date, row_elem))
                i += 1
    return per_instance_type_spot_prices

def read_per_instance_type_spot_prices_json(directory):
    per_instance_type_spot_prices = {}
    per_instance_type_spot_prices['aws'] = \
        read_per_instance_type_spot_prices_aws(os.path.join(directory,
                                                            'aws/logs'))
    per_instance_type_spot_prices['azure'] = \
        read_per_instance_type_spot_prices_azure(os.path.join(directory,
                                                              'azure/logs'))
    per_instance_type_spot_prices['gcp'] = {
        'v100': 0.74,
        'p100': 0.43,
        'k80': 0.135
    }
    return per_instance_type_spot_prices

def get_latest_price_for_worker_type_aws(worker_type, current_time,
                                         per_instance_type_spot_prices):
    # TODO: Make this function more efficient.
    if worker_type == 'v100':
        instance_type = 'p3.2xlarge'
    elif worker_type == 'p100':
        # NOTE: AWS does not have single P100 instances, use 1.5x K80 price
        # as a proxy.
        instance_type = 'p2.xlarge'
    elif worker_type == 'k80':
        instance_type = 'p2.xlarge'

    timestamps = [datetime.strptime(x['Timestamp'], '%Y-%m-%dT%H:%M:%S.000Z')
                  for x in per_instance_type_spot_prices[instance_type]]
    timestamps.sort()

    availability_zones = \
        [x['AvailabilityZone']
         for x in per_instance_type_spot_prices[instance_type]]
    latest_prices = []
    for availability_zone in set(availability_zones):
        per_instance_type_spot_prices[instance_type].sort(
            key=lambda x: datetime.strptime(x['Timestamp'],
                                            '%Y-%m-%dT%H:%M:%S.000Z'))
        latest_price = None
        for x in per_instance_type_spot_prices[instance_type]:
            if x['AvailabilityZone'] != availability_zone:
                continue
            timestamp = (datetime.strptime(x['Timestamp'],
                                          '%Y-%m-%dT%H:%M:%S.000Z') -
                         timestamps[0]).total_seconds()
            if timestamp > current_time and latest_price is not None:
                break
            latest_price = float(x['SpotPrice'])
        assert(latest_price is not None)
        latest_prices.append(latest_price)

    # NOTE: AWS does not have single P100 instances, use 1.5x K80 price
    # as a proxy.
    if worker_type == 'p100':
        return min(latest_prices) * 1.5
    else:
        return min(latest_prices)

def get_latest_price_for_worker_type_gcp(worker_type, current_time,
                                         per_instance_type_spot_prices):
    return per_instance_type_spot_prices[worker_type]

def get_latest_price_for_worker_type_azure(worker_type, current_time,
                                           per_instance_type_spot_prices):
    if worker_type == 'k80':
        instance_type = 'NC6'
    elif worker_type == 'p100':
        instance_type = 'NC6s v2'
    elif worker_type == 'v100':
        instance_type = 'NC6s v3'

    earliest_timestamps = []
    for zone in per_instance_type_spot_prices[instance_type]:
        per_instance_type_spot_prices[instance_type][zone].sort(
            key=lambda x: x[0])
        earliest_timestamps.append(
            per_instance_type_spot_prices[instance_type][zone][0][0])
    earliest_timestamp = min(earliest_timestamps)
    latest_prices = []
    for zone in per_instance_type_spot_prices[instance_type]:
        latest_price = None
        for x in per_instance_type_spot_prices[instance_type][zone]:
            timestamp = (x[0] - earliest_timestamp).total_seconds()
            if timestamp > current_time and latest_price is not None:
                break
            elif x[1] == '':
                continue
            else:
                # Remove '$' character.
                latest_price = float(x[1][1:])
    return latest_price

def get_latest_price_for_worker_type(worker_type, current_time,
                                     per_instance_type_spot_prices,
                                     available_clouds):
    assert(len(available_clouds) > 0)
    prices = []
    if 'aws' in available_clouds:
        aws_price = \
            get_latest_price_for_worker_type_aws(
                    worker_type, current_time,
                    per_instance_type_spot_prices['aws'])
        prices.append(aws_price)
    if 'gcp' in available_clouds:
        gcp_price = \
            get_latest_price_for_worker_type_gcp(
                    worker_type, current_time,
                    per_instance_type_spot_prices['gcp'])
        prices.append(gcp_price)
    if 'azure' in available_clouds:
        azure_price = \
            get_latest_price_for_worker_type_azure(
                    worker_type, current_time,
                    per_instance_type_spot_prices['azure'])
        prices.append(azure_price)

    return min(prices)

def parse_job_type_str(job_type):
    if job_type is None:
        return None
    match = re.match('(.*) \(scale factor (\d+)\)', job_type)
    if match is None:
        return (job_type, 1)
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)

def stringify_throughputs(throughputs):
    stringified_throughputs = {}
    for worker_type in throughputs:
        stringified_throughputs[worker_type] = {}
        for key in throughputs[worker_type]:
            stringified_throughputs[worker_type][str(key)] = {}
            for other_key in throughputs[worker_type][key]:
                stringified_throughputs[worker_type][str(key)][str(other_key)] = \
                    throughputs[worker_type][key][other_key]
    return stringified_throughputs

def read_all_throughputs_json_v2(file_name):
    with open(file_name, 'r') as f:
        raw_throughputs = json.load(f)
    parsed_throughputs = {}
    for worker_type in raw_throughputs:
        parsed_throughputs[worker_type] = {}
        for job_type in raw_throughputs[worker_type]:
            key = parse_job_type_tuple(job_type)
            assert(key is not None)
            parsed_throughputs[worker_type][key] = {}
            for other_job_type in raw_throughputs[worker_type][job_type]:
                if other_job_type == 'null':
                    other_key = other_job_type
                else:
                    other_key = parse_job_type_tuple(other_job_type)
                    assert(other_key is not None)
                parsed_throughputs[worker_type][key][other_key] =\
                    raw_throughputs[worker_type][job_type][other_job_type]
    return parsed_throughputs

def read_all_throughputs_json(throughputs_file):
    with open(throughputs_file, 'r') as f:
        throughputs = json.load(f)
    return throughputs

def get_policy(policy_name, solver=None, seed=None,
               priority_reweighting_policies=None):
    if policy_name.startswith('allox'):
        if policy_name == 'allox':
            alpha = 1.0
        else:
            alpha = float(policy_name.split("allox_alpha=")[1])
        policy = allox.AlloXPolicy(alpha=alpha)
    elif policy_name == 'fifo':
        policy = fifo.FIFOPolicy(seed=seed)
    elif policy_name == 'fifo_perf':
        policy = fifo.FIFOPolicyWithPerf()
    elif policy_name == 'fifo_packed':
        policy = fifo.FIFOPolicyWithPacking()
    elif policy_name == 'finish_time_fairness':
        policy = finish_time_fairness.FinishTimeFairnessPolicy(solver=solver)
    elif policy_name == 'finish_time_fairness_perf':
        policy = \
            finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver=solver)
    elif policy_name == 'finish_time_fairness_packed':
        policy = \
            finish_time_fairness.FinishTimeFairnessPolicyWithPacking(
                solver=solver)
    elif policy_name == 'gandiva':
        policy = gandiva.GandivaPolicy(seed=seed)
    elif policy_name == 'isolated':
        policy = isolated.IsolatedPolicy()
    elif policy_name == 'max_min_fairness':
        policy = max_min_fairness.MaxMinFairnessPolicy(solver=solver)
    elif policy_name == 'max_min_fairness_perf':
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver=solver)
    elif policy_name == 'max_min_fairness_packed':
        policy = \
            max_min_fairness.MaxMinFairnessPolicyWithPacking(solver=solver)
    elif policy_name == 'max_min_fairness_water_filling':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicy(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_min_fairness_water_filling_perf':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_min_fairness_water_filling_packed':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPacking(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_sum_throughput_perf':
        policy = max_sum_throughput.ThroughputSumWithPerf(solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_perf':
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
                    solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_perf_SLOs':
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerfSLOs(
                    solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_packed_SLOs':
        policy = \
            max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
                                                        solver=solver)
    elif policy_name == 'min_total_duration':
        policy = min_total_duration.MinTotalDurationPolicy(solver=solver)
    elif policy_name == 'min_total_duration_perf':
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(solver=solver)
    elif policy_name == 'min_total_duration_packed':
        policy = \
            min_total_duration.MinTotalDurationPolicyWithPacking(solver=solver)
    else:
        raise ValueError('Unknown policy!')
    return policy

def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            (job_type, command, working_directory, num_steps_arg,
             needs_data_dir, total_steps, scale_factor, priority_weight, SLO,
             arrival_time) = line.split('\t')
            assert(int(scale_factor) >= 1)
            jobs.append(Job(job_id=None,
                            job_type=job_type,
                            command=command,
                            working_directory=working_directory,
                            needs_data_dir=bool(int(needs_data_dir)),
                            num_steps_arg=num_steps_arg,
                            total_steps=int(total_steps),
                            duration=None,
                            scale_factor=int(scale_factor),
                            priority_weight=float(priority_weight),
                            SLO=float(SLO)))
            arrival_times.append(float(arrival_time))
    return jobs, arrival_times


def parse_trace_msr(trace_file):
    """
    Parse MSR/Philly-style trace file (used by Gavel experiments).
    
    Format: job_type<TAB>command<TAB>num_steps_arg<TAB>scale_factor<TAB>total_steps<TAB>arrival_time<TAB>priority_weight
    
    Example line:
        Transformer (batch size 128)<TAB>cd %s/... && python3 train.py...<TAB>-step<TAB>1<TAB>95121<TAB>0.000000<TAB>1
    
    Args:
        trace_file: Path to the MSR-format trace file.
        
    Returns:
        Tuple of (jobs, arrival_times) where jobs is a list of Job objects.
    """
    jobs = []
    arrival_times = []
    
    with open(trace_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            
            job_type = parts[0]
            command = parts[1]
            num_steps_arg = parts[2]
            scale_factor = int(parts[3])
            total_steps = int(parts[4])
            arrival_time = float(parts[5])
            priority_weight = float(parts[6]) if len(parts) > 6 else 1.0
            
            jobs.append(Job(
                job_id=None,
                job_type=job_type,
                command=command,
                working_directory='.',
                needs_data_dir=False,
                num_steps_arg=num_steps_arg,
                total_steps=total_steps,
                duration=None,
                scale_factor=scale_factor,
                priority_weight=priority_weight,
                SLO=None
            ))
            arrival_times.append(arrival_time)
    
    return jobs, arrival_times


def parse_trace_alibaba_fgd(trace_file, throughput_per_gpu=100, 
                            gpu_only=True, normalize_time=True,
                            map_to_gavel_types=True, seed=42):
    """
    Parse Alibaba FGD trace file (CSV format from cluster-trace-gpu-v2023).
    
    This converts the Alibaba production cluster trace format to Gavel's Job format.
    FGD traces support GPU sharing (gpu_milli) and multi-GPU jobs.
    
    CSV Format:
        name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,
        creation_time,deletion_time,scheduled_time
    
    Args:
        trace_file: Path to the Alibaba FGD CSV trace file.
        throughput_per_gpu: Assumed throughput (steps/second) per GPU for
                           converting job duration to total_steps.
        gpu_only: If True, only include jobs that request GPUs (num_gpu > 0 or gpu_milli > 0).
        normalize_time: If True, normalize arrival times to start from 0.
        map_to_gavel_types: If True, map job types to known Gavel job types
                           (required for simulation with throughput oracle).
        seed: Random seed for job type mapping.
        
    Returns:
        Tuple of (jobs, arrival_times) where jobs is a list of Job objects.
    """
    import csv
    import random
    
    # Set random seed for reproducibility
    rng = random.Random(seed)
    
    jobs = []
    arrival_times = []
    min_creation_time = float('inf')
    
    # Known Gavel job types (from job_table.py) - these have throughput data
    # Format: (job_type, scale_factors_supported)
    GAVEL_JOB_TYPES = [
        # ResNet-18 variants
        ('ResNet-18 (batch size 16)', [1, 2, 4]),
        ('ResNet-18 (batch size 32)', [1, 2, 4]),
        ('ResNet-18 (batch size 64)', [1, 2, 4]),
        ('ResNet-18 (batch size 128)', [1, 2, 4]),
        # ResNet-50 variants
        ('ResNet-50 (batch size 16)', [1, 2, 4]),
        ('ResNet-50 (batch size 32)', [1, 2, 4]),
        ('ResNet-50 (batch size 64)', [1, 2, 4]),
        ('ResNet-50 (batch size 128)', [1, 2, 4]),
        # Transformer variants
        ('Transformer (batch size 16)', [1, 2, 4]),
        ('Transformer (batch size 32)', [1, 2, 4]),
        ('Transformer (batch size 64)', [1, 2, 4]),
        ('Transformer (batch size 128)', [1, 2, 4]),
        ('Transformer (batch size 256)', [1, 2, 4]),
        # LM variants
        ('LM (batch size 5)', [1]),
        ('LM (batch size 10)', [1]),
        ('LM (batch size 20)', [1]),
        ('LM (batch size 40)', [1]),
        ('LM (batch size 80)', [1]),
        # Recommendation variants
        ('Recommendation (batch size 512)', [1]),
        ('Recommendation (batch size 1024)', [1]),
        ('Recommendation (batch size 2048)', [1]),
        ('Recommendation (batch size 4096)', [1]),
        ('Recommendation (batch size 8192)', [1]),
        # Other
        ('CycleGAN', [1]),
    ]
    
    # QoS to priority mapping (higher = more important)
    qos_priority = {
        'LS': 2.0,         # Latency Sensitive - highest priority
        'Burstable': 1.5,  # Burstable
        'BE': 1.0,         # Best Effort - lowest priority
        'Guaranteed': 2.5, # Guaranteed - highest
    }
    
    # First pass: find minimum creation time for normalization
    if normalize_time:
        with open(trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                creation_time = float(row['creation_time'])
                if creation_time < min_creation_time:
                    min_creation_time = creation_time
    else:
        min_creation_time = 0
    
    # Second pass: parse jobs
    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse fields
            name = row['name']
            num_gpu = int(row['num_gpu'])
            gpu_milli = int(row['gpu_milli']) if row['gpu_milli'] else 0
            gpu_spec = row.get('gpu_spec', '')
            qos = row.get('qos', 'BE')
            pod_phase = row.get('pod_phase', '')
            creation_time = float(row['creation_time'])
            deletion_time = float(row['deletion_time'])
            
            # Skip non-GPU jobs if gpu_only is True
            if gpu_only and num_gpu == 0 and gpu_milli == 0:
                continue
            
            # Skip failed or pending jobs
            if pod_phase in ['Failed', 'Pending']:
                continue
            
            # Calculate scale_factor (number of GPUs)
            # For GPU sharing (num_gpu=1 with gpu_milli<1000), still use scale_factor=1
            # Gavel doesn't support fractional GPUs natively
            if num_gpu > 0:
                scale_factor = num_gpu
            elif gpu_milli > 0:
                # GPU sharing job - treat as 1 GPU
                scale_factor = 1
            else:
                scale_factor = 1
            
            # Calculate job duration and convert to steps
            duration = max(deletion_time - creation_time, 1)  # minimum 1 second
            # Estimate total_steps based on duration and assumed throughput
            total_steps = int(duration * throughput_per_gpu * scale_factor)
            
            # Determine job type
            if map_to_gavel_types:
                # Map to a known Gavel job type based on scale_factor
                # Filter job types that support this scale_factor
                compatible_types = [
                    jt for jt, sfs in GAVEL_JOB_TYPES 
                    if scale_factor in sfs or scale_factor == 1
                ]
                if not compatible_types:
                    # Fallback: use single-GPU jobs and adjust scale_factor
                    compatible_types = [jt for jt, _ in GAVEL_JOB_TYPES]
                    scale_factor = 1
                
                job_type = rng.choice(compatible_types)
            else:
                # Use descriptive job type (won't work with Gavel simulation)
                if gpu_spec and gpu_spec != 'nan' and gpu_spec.strip():
                    job_type = f"GPU-{scale_factor}x-{gpu_spec}"
                elif num_gpu > 1:
                    job_type = f"MultiGPU-{num_gpu}"
                elif gpu_milli > 0 and gpu_milli < 1000:
                    job_type = f"GPUShare-{gpu_milli}m"
                elif num_gpu == 1:
                    job_type = "SingleGPU"
                else:
                    job_type = "CPUOnly"
            
            # Get priority from QoS
            priority_weight = qos_priority.get(qos, 1.0)
            
            # Normalize arrival time
            arrival_time = creation_time - min_creation_time
            
            # Store gpu_milli for GPU sharing support
            # If gpu_milli is not specified, assume full GPU (1000)
            job_gpu_milli = gpu_milli if gpu_milli > 0 else 1000
            
            jobs.append(Job(
                job_id=None,
                job_type=job_type,
                command=f"# Alibaba FGD trace: {name}",
                working_directory='.',
                needs_data_dir=False,
                num_steps_arg='--steps',
                total_steps=total_steps,
                duration=duration,
                scale_factor=scale_factor,
                priority_weight=priority_weight,
                SLO=None,
                gpu_milli=job_gpu_milli  # FGD: preserve fractional GPU request
            ))
            arrival_times.append(arrival_time)
    
    # Sort by arrival time
    if jobs:
        sorted_pairs = sorted(zip(arrival_times, jobs), key=lambda x: x[0])
        arrival_times, jobs = zip(*sorted_pairs)
        arrival_times = list(arrival_times)
        jobs = list(jobs)
    
    return jobs, arrival_times


def get_alibaba_fgd_trace_stats(trace_file):
    """
    Get statistics about an Alibaba FGD trace file.
    
    Args:
        trace_file: Path to the Alibaba FGD CSV trace file.
        
    Returns:
        Dictionary with trace statistics.
    """
    import csv
    from collections import Counter
    
    stats = {
        'total_pods': 0,
        'gpu_pods': 0,
        'cpu_only_pods': 0,
        'multi_gpu_pods': 0,
        'gpu_share_pods': 0,
        'gpu_spec_distribution': Counter(),
        'num_gpu_distribution': Counter(),
        'qos_distribution': Counter(),
        'pod_phase_distribution': Counter(),
        'min_creation_time': float('inf'),
        'max_deletion_time': 0,
        'total_duration_seconds': 0,
    }
    
    with open(trace_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['total_pods'] += 1
            
            num_gpu = int(row['num_gpu'])
            gpu_milli = int(row['gpu_milli']) if row['gpu_milli'] else 0
            gpu_spec = row.get('gpu_spec', '')
            qos = row.get('qos', 'Unknown')
            pod_phase = row.get('pod_phase', 'Unknown')
            creation_time = float(row['creation_time'])
            deletion_time = float(row['deletion_time'])
            
            # Update time stats
            if creation_time < stats['min_creation_time']:
                stats['min_creation_time'] = creation_time
            if deletion_time > stats['max_deletion_time']:
                stats['max_deletion_time'] = deletion_time
            
            # Categorize pods
            if num_gpu > 0 or gpu_milli > 0:
                stats['gpu_pods'] += 1
                if num_gpu > 1:
                    stats['multi_gpu_pods'] += 1
                if gpu_milli > 0 and gpu_milli < 1000:
                    stats['gpu_share_pods'] += 1
            else:
                stats['cpu_only_pods'] += 1
            
            # Update distributions
            stats['num_gpu_distribution'][num_gpu] += 1
            stats['qos_distribution'][qos] += 1
            stats['pod_phase_distribution'][pod_phase] += 1
            if gpu_spec and gpu_spec != 'nan':
                stats['gpu_spec_distribution'][gpu_spec] += 1
    
    # Calculate total duration
    if stats['total_pods'] > 0:
        stats['total_duration_seconds'] = stats['max_deletion_time'] - stats['min_creation_time']
        stats['total_duration_hours'] = stats['total_duration_seconds'] / 3600
    
    return stats

def print_allocation(allocation, current_time=None):
    """Prints the allocation.

       Debug method used for printing the allocation of each job on each
       worker type.
    """
    print('=' * 80)
    if current_time is not None:
        print('Allocation\t(Current_time: %f)' % (current_time))
        print('-' * 80)
    for job_id in sorted(list(allocation.keys())):
        allocation_str = 'Job ID %s:' % (job_id)
        for worker_type in sorted(list(allocation[job_id].keys())):
            value = allocation[job_id][worker_type]
            allocation_str += ' [%s: %f]' % (worker_type, value)
        print(allocation_str)
    print('=' * 80)
