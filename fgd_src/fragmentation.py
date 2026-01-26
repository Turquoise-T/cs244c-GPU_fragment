import copy
from typing import Dict, List
from node import Node
from job import Job
from workload import Workload

class FragmentationCalculator:
    """
    Implements the FGD fragmentation measure from the paper
    """
    
    @staticmethod
    def compute_node_fragmentation(node: Node, job: Job) -> float:
        """
        Compute F_n(m): fragmentation of node n measured by job m.
        Returns the amount of GPU resources that cannot be allocated to job m.
        
        Cases:
        1. Task cannot run (Q-I, Q-II, Q-IV, or x-axis): all GPUs fragmented
        2. Task can run (Q-III): check each GPU individually
        """
        state = node.get_state_dict()
        available_cpu = state['available_cpu']
        available_gpu_scalar = node.get_gpu_scalar()
        gpu_capacities = state['gpu_capacities']
        
        # Case 1: Task cannot run - all unallocated GPUs are fragmented
        if (job.cpu_request > available_cpu or 
            job.gpu_request > available_gpu_scalar or
            job.gpu_request == 0):
            return sum(gpu_capacities)
        
        # Case 2: Task can run - check each GPU individually
        fragmented = 0.0
        for gpu_capacity in gpu_capacities:
            # GPU is fragmented if it has insufficient capacity
            min_needed = min(job.gpu_request, 1.0)
            if gpu_capacity < min_needed:
                fragmented += gpu_capacity
        
        return fragmented
    
    @staticmethod
    def compute_node_fragmentation_for_workload(node: Node, 
                                               workload: Workload) -> float:
        """
        Compute F_n(M): expected fragmentation for workload M.
        F_n(M) = Σ p_m * F_n(m) for all jobs m in workload M
        """
        total_frag = 0.0
        for job_type_id, job in workload.job_types.items():
            popularity = workload.popularity.get(job_type_id, 0.0)
            job_frag = FragmentationCalculator.compute_node_fragmentation(
                node, job
            )
            total_frag += popularity * job_frag
        
        return total_frag
    
    @staticmethod
    def compute_cluster_fragmentation(nodes: List[Node], 
                                     workload: Workload) -> float:
        """
        Compute F_N(M): cluster-level fragmentation.
        F_N(M) = Σ F_n(M) for all nodes n in cluster N
        """
        return sum(
            FragmentationCalculator.compute_node_fragmentation_for_workload(
                node, workload
            )
            for node in nodes
        )
    
    @staticmethod
    def compute_fragmentation_rate(node: Node, workload: Workload) -> float:
        """
        Compute fragmentation rate: ratio of fragmented to unallocated GPUs
        """
        state = node.get_state_dict()
        total_unallocated = sum(state['gpu_capacities'])
        
        if total_unallocated == 0:
            return 0.0
        
        fragmentation = FragmentationCalculator.compute_node_fragmentation_for_workload(
            node, workload
        )
        
        return fragmentation / total_unallocated