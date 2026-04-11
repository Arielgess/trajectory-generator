from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from app.models.schemas import JobStatus
from app.services.job_service import job_service
from app.writers.base_writer import BaseWriter

LOGGER = logging.getLogger(__name__)


class BaseTrajectoryGenerator(ABC):
    def __init__(self, job_id: str, params: Any, writer: BaseWriter) -> None:
        self.job_id = job_id
        self.params = params
        self.writer = writer

    def publish_progress(self, generated_count: int, total_count: int, message: str) -> None:
        record = job_service.get_job(self.job_id)
        progress = min(1.0, generated_count / max(1, total_count))
        record.progress = progress
        record.generated_count = generated_count
        record.message = message
        record.status = JobStatus.RUNNING
        job_service.publish_event(
            self.job_id,
            {
                "event_type": "progress",
                "payload": {"progress": progress, "generated_count": generated_count, "message": message},
            },
        )

    def publish_preview(self, preview: dict[str, Any]) -> None:
        job_service.publish_event(self.job_id, {"event_type": "preview", "payload": preview})

    def run(self) -> None:
        record = job_service.get_job(self.job_id)
        record.status = JobStatus.RUNNING
        LOGGER.info("Generator started job_id=%s generator=%s", self.job_id, self.__class__.__name__)
        try:
            self.generate_trajectories()
            record.status = JobStatus.COMPLETED
            record.progress = 1.0
            LOGGER.info("Generator completed job_id=%s", self.job_id)
            job_service.publish_event(self.job_id, {"event_type": "done", "payload": {"job_id": self.job_id}})
        except Exception as exc:
            record.status = JobStatus.FAILED
            record.message = str(exc)
            LOGGER.exception("Generator failed job_id=%s error=%s", self.job_id, exc)
            job_service.publish_event(self.job_id, {"event_type": "error", "payload": {"message": str(exc)}})

    @abstractmethod
    def generate_trajectories(self) -> None:
        raise NotImplementedError
