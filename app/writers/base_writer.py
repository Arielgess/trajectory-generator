from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWriter(ABC):
    @abstractmethod
    def append_trajectory(self, trajectory: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_trajectories(self, trajectories: list[dict[str, Any]]) -> None:
        raise NotImplementedError
