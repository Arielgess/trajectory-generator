from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config.defaults import now_stamp
from app.writers.base_writer import BaseWriter


class FileWriter(BaseWriter):
    def __init__(self, output_dir: str, trajectory_type: str, important_hparams: dict[str, Any]) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self._output_dir / self._build_filename(trajectory_type, important_hparams)

    @property
    def output_path(self) -> str:
        return str(self._output_path)

    def append_trajectory(self, trajectory: dict[str, Any]) -> None:
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory) + "\n")

    def write_trajectories(self, trajectories: list[dict[str, Any]]) -> None:
        with self._output_path.open("w", encoding="utf-8") as handle:
            for trajectory in trajectories:
                handle.write(json.dumps(trajectory) + "\n")

    def _build_filename(self, trajectory_type: str, important_hparams: dict[str, Any]) -> str:
        date_part = now_stamp()
        kv_tokens = []
        for key, value in sorted(important_hparams.items()):
            token = f"{key}-{value}"
            token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
            kv_tokens.append(token)
        params_part = "_".join(kv_tokens) if kv_tokens else "default"
        return f"{trajectory_type}_{date_part}_{params_part}.jsonl"
