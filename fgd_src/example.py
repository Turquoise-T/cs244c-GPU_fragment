# example.py

from scheduler import FGDScheduler
from job import Job

def main():
    # Create scheduler
    scheduler = FGDScheduler(time_per_iteration=360, simulate=True)
    
    # Define cluster spec
    cluster_spec = {
        'node-0': {'cpu': 16, 'memory': 64, 'num_gpus': 4, 'gpu_type': 'V100'},
        'node-1': {'cpu': 32, 'memory': 128, 'num_gpus': 8, 'gpu_type': 'A100'},
        'node-2': {'cpu': 16, 'memory': 64, 'num_gpus': 4, 'gpu_type': 'V100'},
    }
    
    # Create jobs
    jobs = [
        Job(job_id='', job_type='resnet50', command='train.py',
            working_directory='/workspace', num_steps_arg='--steps',
            total_steps=1000, cpu_request=4, gpu_request=0.5,
            memory_request=16),
        Job(job_id='', job_type='bert', command='train.py',
            working_directory='/workspace', num_steps_arg='--steps',
            total_steps=2000, cpu_request=2, gpu_request=0.25,
            memory_request=8),
        Job(job_id='', job_type='vgg16', command='train.py',
            working_directory='/workspace', num_steps_arg='--steps',
            total_steps=1500, cpu_request=8, gpu_request=1.0,
            memory_request=32),
        Job(job_id='', job_type='inception', command='train.py',
            working_directory='/workspace', num_steps_arg='--steps',
            total_steps=1200, cpu_request=4, gpu_request=0.5,
            memory_request=16),
        Job(job_id='', job_type='mobilenet', command='train.py',
            working_directory='/workspace', num_steps_arg='--steps',
            total_steps=800, cpu_request=16, gpu_request=2.0,
            memory_request=48),
        ]
    
    # Run simulation
    scheduler.simulate(jobs, cluster_spec)

if __name__ == '__main__':
    main()