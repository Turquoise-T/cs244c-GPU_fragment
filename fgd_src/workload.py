from typing import Dict
from job import Job

class Workload:

    # Represents the target workload with task popularity distribution (similar to the paper's definition)
    
    def __init__(self):
        self.job_types: Dict[str, Job] = {}
        self.popularity: Dict[str, float] = {}  # Normalized popularity (sums to 1)
    
    def add_job_type(self, job: Job, popularity: float):
        """Add a job type with its popularity"""
        self.job_types[job.job_id] = job
        self.popularity[job.job_id] = popularity
    
    def normalize_popularity(self):
        """Ensure popularity sums to 1"""
        total = sum(self.popularity.values())
        if total > 0:
            for job_id in self.popularity:
                self.popularity[job_id] /= total
    
    def get_job_type(self, job_id: str) -> Job:
        """Get job type by ID"""
        return self.job_types.get(job_id)