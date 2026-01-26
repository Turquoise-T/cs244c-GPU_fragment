import logging
import threading
import time
from typing import Callable, Dict, List, Tuple

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

class WorkerRpcServer:
    """
    RPC server running on each worker node
    Handles requests from the scheduler
    """
    
    def __init__(self, port: int, callbacks: Dict[str, Callable]):
        self._port = port
        self._callbacks = callbacks
        self._running = False
        
        logger = logging.getLogger('worker_rpc_server')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
    
    def run_job(self, jobs: List[Tuple], worker_id: int, round_id: int):
        """Handle run job request from scheduler"""
        self._logger.info(
            f'Received run job request: worker_id={worker_id}, '
            f'round_id={round_id}, num_jobs={len(jobs)}'
        )
        
        if 'RunJob' in self._callbacks:
            self._callbacks['RunJob'](jobs, worker_id, round_id)
    
    def kill_job(self, job_id: str):
        """Handle kill job request"""
        self._logger.info(f'Received kill job request: job_id={job_id}')
        
        if 'KillJob' in self._callbacks:
            self._callbacks['KillJob'](job_id)
    
    def reset(self):
        """Handle reset request"""
        self._logger.info('Received reset request')
        
        if 'Reset' in self._callbacks:
            self._callbacks['Reset']()
    
    def shutdown(self):
        """Handle shutdown request"""
        self._logger.info('Received shutdown request')
        
        if 'Shutdown' in self._callbacks:
            self._callbacks['Shutdown']()
        
        self.stop()
    
    def start(self):
        """Start the worker RPC server"""
        self._running = True
        self._logger.info(f'Starting worker RPC server on port {self._port}')
    
    def stop(self):
        """Stop the worker RPC server"""
        self._running = False
        self._logger.info('Stopping worker RPC server')


def serve(port: int, callbacks: Dict[str, Callable]):
    """
    Start the worker RPC server
    
    Args:
        port: Port number to listen on
        callbacks: Dictionary mapping RPC method names to callback functions
    """
    server = WorkerRpcServer(port, callbacks)
    server.start()
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()