from __future__ import annotations

import asyncio
import uuid
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import JobStatus

LOGGER = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    generated_count: int = 0
    failed_count: int = 0
    message: str | None = None
    output_path: str | None = None
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000))
    subscribers: set[asyncio.Queue] = field(default_factory=set)


class JobService:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create_job(self, job_type: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, job_type=job_type)
        self._jobs[job_id] = record
        LOGGER.info("Created job job_id=%s type=%s", job_id, job_type)
        return record

    def get_job(self, job_id: str) -> JobRecord:
        return self._jobs[job_id]

    def publish_event(self, job_id: str, event: dict[str, Any]) -> None:
        record = self._jobs[job_id]
        record.events.append(event)
        LOGGER.info("Job event job_id=%s type=%s", job_id, event.get("event_type"))
        for queue in list(record.subscribers):
            queue.put_nowait(event)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._jobs[job_id].subscribers.add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        self._jobs[job_id].subscribers.discard(queue)


job_service = JobService()
