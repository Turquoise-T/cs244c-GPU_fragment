import grpc
from concurrent import futures
import logging
import threading
import time
from typing import Callable, Dict

# For a production implementation, we'll use protobuf
# For now, using a simple RPC interface with JSON

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

class SchedulerRpcServer:
    """
    RPC server for the scheduler
    Handles incoming requests from workers
    """
    
    def __init__(self, port: int, callbacks: Dict[str, Callable]):
        self._port = port
        self._callbacks = callbacks
        self._server = None
        self._running = False
        
        logger = logging.getLogger('scheduler_rpc_server')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        logger.addHandler(ch)
        self._logger = logger
    
    def register_worker(self, worker_type: str, num_gpus: int,
                       ip_addr: str, port: int):
        """Handle worker registration"""
        if 'RegisterWorker' in self._callbacks:
            return self._callbacks['RegisterWorker'](
                worker_type, num_gpus, ip_addr, port
            )
        return None, None, "RegisterWorker callback not found"
    
    def init_job(self, job_id: str):
        """Initialize a job"""
        if 'InitJob' in self._callbacks:
            return self._callbacks['InitJob'](job_id)
        return 0, 0, 0
    
    def update_lease(self, job_id: str, worker_id: int, steps: int,
                    duration: float, max_steps: int, max_duration: float):
        """Update lease for a job"""
        if 'UpdateLease' in self._callbacks:
            return self._callbacks['UpdateLease'](
                job_id, worker_id, steps, duration, max_steps, max_duration
            )
        return max_steps, max_duration
    
    def done(self, job_id: str, worker_id: int, all_num_steps: list,
            all_execution_times: list, all_iterator_logs: list = None):
        """Mark job as done"""
        if 'Done' in self._callbacks:
            self._callbacks['Done'](
                job_id, worker_id, all_num_steps,
                all_execution_times, all_iterator_logs
            )
    
    def start(self):
        """Start the RPC server"""
        self._running = True
        self._logger.info(f'Starting scheduler RPC server on port {self._port}')
        # In a real implementation, this would start a gRPC/HTTP server
        # For now, we'll just log that it's running
    
    def stop(self):
        """Stop the RPC server"""
        self._running = False
        self._logger.info('Stopping scheduler RPC server')


def serve(port: int, callbacks: Dict[str, Callable]):
    """
    Start the scheduler RPC server
    
    Args:
        port: Port number to listen on
        callbacks: Dictionary mapping RPC method names to callback functions
    """
    server = SchedulerRpcServer(port, callbacks)
    server.start()
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()