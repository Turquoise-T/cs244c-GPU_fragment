import logging
from typing import List, Optional, Tuple

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

class WorkerRpcClient:
    """
    RPC client running on worker nodes to communicate with scheduler
    """
    
    def __init__(self, worker_type: str, worker_addr: str, worker_port: int,
                 scheduler_addr: str, scheduler_port: int):
        self._worker_type = worker_type
        self._worker_addr = worker_addr
        self._worker_port = worker_port
        self._scheduler_addr = scheduler_addr
        self._scheduler_port = scheduler_port
        
        logger = logging.getLogger('worker_rpc_client')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
        
        self._logger.info(
            f'Initialized worker RPC client: {worker_addr}:{worker_port} -> '
            f'{scheduler_addr}:{scheduler_port}'
        )
    
    def register_worker(self, num_gpus: int) -> Tuple[List[int], float, Optional[str]]:
        """
        Register this worker with the scheduler
        
        Args:
            num_gpus: Number of GPUs on this worker
            
        Returns:
            Tuple of (worker_ids, round_duration, error_message)
        """
        self._logger.info(
            f'Registering worker with scheduler: type={self._worker_type}, '
            f'num_gpus={num_gpus}'
        )
        
        # In a real implementation, make RPC call to scheduler
        # For now, return mock data
        worker_ids = list(range(num_gpus))
        round_duration = 360.0
        error = None
        
        return worker_ids, round_duration, error
    
    def init_job(self, job_id: str) -> Tuple[int, float, float]:
        """
        Initialize a job and get its initial lease
        
        Args:
            job_id: ID of the job to initialize
            
        Returns:
            Tuple of (max_steps, max_duration, extra_time)
        """
        self._logger.info(f'Initializing job {job_id}')
        
        # In a real implementation, make RPC call
        max_steps = 1000
        max_duration = 360.0
        extra_time = 0.0
        
        return max_steps, max_duration, extra_time
    
    def update_lease(self, job_id: str, worker_id: int, steps: int,
                    duration: float, max_steps: int, max_duration: float) -> Tuple[int, float]:
        """
        Request lease update for a running job
        
        Args:
            job_id: ID of the job
            worker_id: ID of the worker running the job
            steps: Steps completed so far
            duration: Time elapsed so far
            max_steps: Current max steps in lease
            max_duration: Current max duration in lease
            
        Returns:
            Tuple of (updated_max_steps, updated_max_duration)
        """
        self._logger.info(
            f'Requesting lease update for job {job_id}: '
            f'steps={steps}/{max_steps}, duration={duration:.2f}/{max_duration:.2f}'
        )
        
        # In a real implementation, make RPC call
        # For now, just return the same values
        return max_steps, max_duration
    
    def done(self, job_id: str, worker_id: int, all_num_steps: List[int],
            all_execution_times: List[float], all_iterator_logs: Optional[List[str]] = None):
        """
        Notify scheduler that job has completed
        
        Args:
            job_id: ID of the completed job
            worker_id: ID of the worker that ran the job
            all_num_steps: List of steps completed (for each sub-job in distributed case)
            all_execution_times: List of execution times
            all_iterator_logs: Optional iterator logs
        """
        self._logger.info(
            f'Notifying scheduler of job completion: job_id={job_id}, '
            f'worker_id={worker_id}, steps={all_num_steps}, time={all_execution_times}'
        )
        
        # In a real implementation, make RPC call to scheduler


class IteratorRpcClient:
    """
    RPC client used by GavelIterator to communicate with scheduler
    Similar to WorkerRpcClient but specifically for iterator use
    """
    
    def __init__(self, job_id: str, worker_id: int, 
                 scheduler_addr: str, scheduler_port: int, logger):
        self._job_id = job_id
        self._worker_id = worker_id
        self._scheduler_addr = scheduler_addr
        self._scheduler_port = scheduler_port
        self._logger = logger
        
        self._logger.info(
            f'Initialized iterator RPC client for job {job_id} on worker {worker_id}'
        )
    
    def init(self) -> Tuple[int, float, float]:
        """
        Initialize job and get initial lease
        
        Returns:
            Tuple of (max_steps, max_duration, extra_time)
        """
        self._logger.info(f'Iterator initializing job {self._job_id}')
        
        # Make RPC call to scheduler
        max_steps = 1000
        max_duration = 360.0
        extra_time = 0.0
        
        return max_steps, max_duration, extra_time
    
    def update_lease(self, steps: int, duration: float,
                    max_steps: int, max_duration: float) -> Tuple[int, float]:
        """
        Request lease update
        
        Args:
            steps: Steps completed so far
            duration: Time elapsed so far
            max_steps: Current max steps
            max_duration: Current max duration
            
        Returns:
            Tuple of (updated_max_steps, updated_max_duration)
        """
        self._logger.debug(
            f'Iterator requesting lease update: steps={steps}/{max_steps}, '
            f'duration={duration:.2f}/{max_duration:.2f}'
        )
        
        # Make RPC call to scheduler
        return max_steps, max_duration