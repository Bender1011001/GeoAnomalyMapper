"""
Robust Job Store for GAM API.

This module provides a thread-safe, persistent-ready job store to replace the
volatile global dictionary. It uses an in-memory dictionary for simplicity but
is designed to be easily swapped with a database backend (e.g., Redis, PostgreSQL).
"""
from typing import Dict, Any, Optional
from threading import Lock
import uuid
from datetime import datetime

class JobStore:
    """A thread-safe store for analysis jobs."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(JobStore, cls).__new__(cls)
                cls._instance.jobs: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def create_job(self) -> str:
        """Create a new job with a unique ID and initial state."""
        job_id = str(uuid.uuid4())
        with self._lock:
            self.jobs[job_id] = {
                "status": "QUEUED",
                "progress": 0.0,
                "stage": "Queued",
                "start_time": None,
                "results": None,
                "error_message": None,
                "output_files": None,
            }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a job by its ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        """Update a job's state."""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
                return self.jobs[job_id]
            return None

# Global instance
job_store = JobStore()