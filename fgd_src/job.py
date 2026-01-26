from typing import Optional

class Job:
    """
    Represents a training/inference job requesting GPU resources
    Similar to Gavel's Job but focused on GPU sharing
    """
    
    def __init__(self, job_id: str, job_type: str, command: str,
                 working_directory: str, num_steps_arg: str,
                 total_steps: int, duration: Optional[float] = None,
                 cpu_request: float = 1.0, gpu_request: float = 1.0,
                 memory_request: float = 1.0, gpu_type: Optional[str] = None,
                 scale_factor: int = 1, priority_weight: float = 1.0):
        self._job_id = job_id
        self._job_type = job_type
        self._command = command
        self._working_directory = working_directory
        self._num_steps_arg = num_steps_arg
        self._total_steps = total_steps
        self._duration = duration
        
        # Resource requests
        self._cpu_request = cpu_request
        self._gpu_request = gpu_request
        self._memory_request = memory_request
        self._gpu_type = gpu_type
        
        # For distributed training
        self._scale_factor = scale_factor
        self._priority_weight = priority_weight
        
        # Runtime tracking
        self._steps_completed = 0
        self._time_elapsed = 0.0
        self._assigned_node = None
        self._assigned_gpus = None
    
    @property
    def job_id(self):
        return self._job_id
    
    @property
    def job_type(self):
        return self._job_type
    
    @property
    def cpu_request(self):
        return self._cpu_request
    
    @property
    def gpu_request(self):
        return self._gpu_request
    
    @property
    def memory_request(self):
        return self._memory_request
    
    @property
    def gpu_type(self):
        return self._gpu_type
    
    @property
    def total_steps(self):
        return self._total_steps
    
    @property
    def scale_factor(self):
        return self._scale_factor
    
    @property
    def priority_weight(self):
        return self._priority_weight
    
    @property
    def steps_completed(self):
        return self._steps_completed
    
    @steps_completed.setter
    def steps_completed(self, value):
        self._steps_completed = value
    
    @property
    def remaining_steps(self):
        return self._total_steps - self._steps_completed
    
    def __str__(self):
        return (f'Job({self._job_id}, type={self._job_type}, '
                f'cpu={self._cpu_request}, gpu={self._gpu_request}, '
                f'steps={self._steps_completed}/{self._total_steps})')