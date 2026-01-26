import logging
import socket
from typing import Optional, Tuple, List

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

class SchedulerRpcClient:
    """
    RPC client for communicating with the scheduler from workers
    """
    
    def __init__(self, scheduler_addr: str, scheduler_port: int):
        self._scheduler_addr = scheduler_addr
        self._scheduler_port = scheduler_port
        
        logger = logging.getLogger('scheduler_rpc_client')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
        
        self._logger.info(
            f'Initialized scheduler RPC client for {scheduler_addr}:{scheduler_port}'
        )
    
    def register_worker(self, worker_type: str, num_gpus: int,
                       worker_addr: str, worker_port: int) -> Tuple[List[int], float, Optional[str]]:
        """
        Register this worker with the scheduler
        
        Returns:
            Tuple of (worker_ids, round_duration, error_message)
        """
        self._logger.info(
            f'Registering worker: type={worker_type}, gpus={num_gpus}'
        )
        
        # In a real implementation, this would make an RPC call
        # For now, return mock data
        worker_ids = list(range(num_gpus))
        round_duration = 360.0
        error = None
        
        return worker_ids, round_duration, error
    
    def run_job(self, job_descriptions: List[Tuple], worker_id: int, round_id: int):
        """
        Request to run a job on this worker
        
        Args:
            job_descriptions: List of tuples containing job information
            worker_id: ID of the worker
            round_id: Current round ID
        """
        self._logger.info(
            f'Run job request: worker_id={worker_id}, round_id={round_id}, '
            f'num_jobs={len(job_descriptions)}'
        )
    
    def kill_job(self, job_id: str):
        """Request to kill a running job"""
        self._logger.info(f'Kill job request: job_id={job_id}')
    
    def reset(self):
        """Reset the worker"""
        self._logger.info('Reset worker request')
    
    def shutdown(self):
        """Shutdown the worker"""
        self._logger.info('Shutdown worker request')


class WorkerRpcClient:
    """
    RPC client for communicating with workers from the scheduler
    Similar to SchedulerRpcClient but reversed direction
    """
    
    def __init__(self, worker_addr: str, worker_port: int):
        self.addr = worker_addr
        self.port = worker_port
        
        logger = logging.getLogger('worker_rpc_client')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
        
        self._logger.info(
            f'Initialized worker RPC client for {worker_addr}:{worker_port}'
        )
    
    def run_job(self, job_descriptions: List[Tuple], worker_id: int, round_id: int):
        """
        Send job to worker for execution
        
        Args:
            job_descriptions: List of (job_id, command, working_dir, needs_data_dir, 
                                      num_steps_arg, num_steps)
            worker_id: Worker ID
            round_id: Current scheduling round ID
        """
        self._logger.info(
            f'Sending job to worker {worker_id} in round {round_id}'
        )
        
        # In a real implementation, make RPC call to worker
        for job_desc in job_descriptions:
            job_id = job_desc[0]
            self._logger.debug(f'  Job {job_id}: {job_desc[1]}')
    
    def kill_job(self, job_id: str):
        """Kill a running job on the worker"""
        self._logger.info(f'Sending kill signal for job {job_id}')
    
    def reset(self):
        """Reset the worker"""
        self._logger.info('Sending reset signal to worker')
    
    def shutdown(self):
        """Shutdown the worker"""
        self._logger.info('Sending shutdown signal to worker')