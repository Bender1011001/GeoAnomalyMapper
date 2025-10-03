from __future__ import annotations

from typing import Dict, Any, Iterator, MutableMapping, Optional
from threading import RLock
from datetime import datetime
import uuid


class JobStore(MutableMapping[str, Dict[str, Any]]):
    """
    Thread-safe in-memory job store with dict-like behavior.

    Supports:
      - 'in' containment: job_id in job_store
      - subscription: job_store[job_id]
      - assignment: job_store[job_id] = {...}
      - iteration over keys, len(), keys()/items()/values()
    Provides helpers for the API:
      - create_job() -> str
      - get_job(job_id) -> Optional[dict]
      - update_job(job_id, updates: dict) -> dict
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    # -------- Dict-like API required by tests --------

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            return key in self._jobs

    def __getitem__(self, key: str) -> Dict[str, Any]:
        with self._lock:
            return self._jobs[key]

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("JobStore values must be dicts")
        with self._lock:
            self._jobs[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._jobs[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            # Return an iterator over a snapshot of keys to avoid race with mutation
            return iter(list(self._jobs.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def keys(self):
        with self._lock:
            return list(self._jobs.keys())

    def items(self):
        with self._lock:
            return list(self._jobs.items())

    def values(self):
        with self._lock:
            return list(self._jobs.values())

    # -------- Convenience helpers for API usage --------

    def create_job(self) -> str:
        """Create a new job id and seed default record."""
        job_id = uuid.uuid4().hex
        now = datetime.utcnow().isoformat()
        default_record = {
            "status": "QUEUED",
            "progress": 0.0,
            "stage": "Queued",
            "message": None,
            "error_message": None,
            "results": None,
            "output_files": None,
            "create_time": now,
            "start_time": None,
            "end_time": None,
        }
        with self._lock:
            self._jobs[job_id] = default_record
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return job dict or None if missing."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow-merge updates into a job record, creating it if needed."""
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dict")
        with self._lock:
            if job_id not in self._jobs:
                self._jobs[job_id] = {}
            self._jobs[job_id].update(updates)
            return self._jobs[job_id]

    def clear(self) -> None:
        """Clear all jobs (useful for tests)."""
        with self._lock:
            self._jobs.clear()


# Public singleton used by API and tests
job_store = JobStore()