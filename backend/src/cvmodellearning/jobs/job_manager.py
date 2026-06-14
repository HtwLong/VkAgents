import threading
from typing import Dict, Any, Optional

class JobManager:
    """
    Singleton manager for centralized, thread-safe tracking of background jobs.
    """
    # Dictionary to hold all job data: {job_id: {status: str, result: Any, error: Optional[str]}}
    _jobs: Dict[str, Dict[str, Any]] = {}
    
    # Lock for thread-safe access to the job store
    _job_lock = threading.Lock()
    
    def create_job(self, job_id: str):
        """Initializes a new job entry with 'running' status."""
        with self._job_lock:
            self._jobs[job_id] = {"status": "running"}

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Safely retrieve a job's details."""
        with self._job_lock:
            return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: str, **kwargs):
        """
        Updates the status of an existing job, optionally adding result or error data.
        """
        if status not in ["running", "completed", "error"]:
            raise ValueError(f"Invalid job status: {status}")
            
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = status
                job.update(kwargs)
                if status != "running":
                    job.pop("error", None) # Clear error if status is not 'error'
            else:
                # Handle case where job ID might be invalid or race condition occurs
                print(f"Warning: Attempted to update non-existent job ID: {job_id}")

    def delete_job(self, job_id: str) -> bool:
        """Removes a job from the store."""
        with self._job_lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

# Create a single global instance of the manager
JOB_MANAGER = JobManager()