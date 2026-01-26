import threading
from typing import List, Dict, Optional, Tuple

class Node:
    """Represents a compute node with GPUs"""
    
    def __init__(self, node_id: str, total_cpu: float, total_memory: float,
                 num_gpus: int, gpu_type: str = "V100"):
        self.node_id = node_id
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type
        
        # Track allocated resources
        self.allocated_cpu = 0.0
        self.allocated_memory = 0.0
        # Each GPU's available capacity (0.0 - 1.0)
        self.gpu_capacities = [1.0] * num_gpus
        
        # Track job assignments per GPU
        self.gpu_job_assignments = [[] for _ in range(num_gpus)]
        
        self._lock = threading.Lock()
    
    @property
    def available_cpu(self) -> float:
        return self.total_cpu - self.allocated_cpu
    
    @property
    def available_memory(self) -> float:
        return self.total_memory - self.allocated_memory
    
    def get_gpu_scalar(self) -> float:
        """
        Map GPU vector to scalar: u = f + p
        where f = number of fully unallocated GPUs
              p = maximum unallocated partial GPU
        """
        full_gpus = sum(1 for cap in self.gpu_capacities if cap == 1.0)
        partial_gpus = [cap for cap in self.gpu_capacities if 0 < cap < 1.0]
        max_partial = max(partial_gpus) if partial_gpus else 0.0
        return full_gpus + max_partial
    
    def can_fit_task(self, job_id: str, cpu_req: float, gpu_req: float,
                     memory_req: float, gpu_type: Optional[str] = None) -> bool:
        """Check if node has sufficient resources for the task"""
        with self._lock:
            if self.available_cpu < cpu_req:
                return False
            if self.available_memory < memory_req:
                return False
            if gpu_type and gpu_type != self.gpu_type:
                return False
            
            gpu_scalar = self.get_gpu_scalar()
            if gpu_scalar < gpu_req:
                return False
            
            return True
    
    def find_suitable_gpus(self, gpu_req: float) -> Optional[List[int]]:
        """Find which GPU(s) can accommodate the task"""
        with self._lock:
            if gpu_req == 0:
                return []
            
            # For partial GPU request (0 < gpu < 1)
            if 0 < gpu_req < 1:
                suitable = []
                for idx, capacity in enumerate(self.gpu_capacities):
                    if capacity >= gpu_req:
                        suitable.append(idx)
                return suitable if suitable else None
            
            # For full GPU request (1, 2, 4, 8, etc.)
            num_gpus_needed = int(gpu_req)
            full_gpus = [idx for idx, cap in enumerate(self.gpu_capacities) 
                        if cap == 1.0]
            
            if len(full_gpus) >= num_gpus_needed:
                return full_gpus[:num_gpus_needed]
            
            return None
    
    def allocate_resources(self, job_id: str, cpu_req: float, gpu_req: float,
                          memory_req: float, gpu_indices: List[int]):
        """Allocate resources to a job"""
        with self._lock:
            self.allocated_cpu += cpu_req
            self.allocated_memory += memory_req
            
            if gpu_req > 0:
                if 0 < gpu_req < 1:
                    # Partial GPU
                    gpu_idx = gpu_indices[0]
                    self.gpu_capacities[gpu_idx] -= gpu_req
                    self.gpu_job_assignments[gpu_idx].append(job_id)
                else:
                    # Full GPU(s)
                    for idx in gpu_indices:
                        self.gpu_capacities[idx] = 0.0
                        self.gpu_job_assignments[idx].append(job_id)
    
    def release_resources(self, job_id: str, cpu_req: float, gpu_req: float,
                         memory_req: float, gpu_indices: List[int]):
        """Release resources from a completed job"""
        with self._lock:
            self.allocated_cpu -= cpu_req
            self.allocated_memory -= memory_req
            
            if gpu_req > 0:
                if 0 < gpu_req < 1:
                    # Partial GPU
                    gpu_idx = gpu_indices[0]
                    self.gpu_capacities[gpu_idx] += gpu_req
                    if job_id in self.gpu_job_assignments[gpu_idx]:
                        self.gpu_job_assignments[gpu_idx].remove(job_id)
                else:
                    # Full GPU(s)
                    for idx in gpu_indices:
                        self.gpu_capacities[idx] = 1.0
                        if job_id in self.gpu_job_assignments[idx]:
                            self.gpu_job_assignments[idx].remove(job_id)
    
    def get_state_dict(self) -> Dict:
        """Get current state for debugging/logging"""
        with self._lock:
            return {
                'node_id': self.node_id,
                'available_cpu': self.available_cpu,
                'total_cpu': self.total_cpu,
                'available_memory': self.available_memory,
                'total_memory': self.total_memory,
                'gpu_capacities': self.gpu_capacities.copy(),
                'gpu_job_assignments': [jobs.copy() for jobs in self.gpu_job_assignments]
            }