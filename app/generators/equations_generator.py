from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from app.generators.base_generator import BaseTrajectoryGenerator
from app.local_lib.route_generation import TrajectoryState, generate_composite_trajectory
from app.models.schemas import EquationsRequest, NumericParam, SegmentType

LOGGER = logging.getLogger(__name__)


def _sample_param(param: NumericParam) -> float:
    if param.mode.value == "range":
        return random.uniform(param.min_value, param.max_value)
    return param.value


def _to_plot_payload(noisy: np.ndarray, clean: np.ndarray) -> dict[str, Any]:
    return {
        "noisy": noisy.tolist(),
        "clean": clean.tolist(),
    }


class EquationTrajectoryGenerator(BaseTrajectoryGenerator):
    def __init__(self, job_id: str, params: EquationsRequest, writer: Any) -> None:
        super().__init__(job_id=job_id, params=params, writer=writer)

    def _segment_to_params(self, block: Any, observation_noise: float) -> tuple[str, int, dict[str, Any]]:
        params: dict[str, Any] = {"measurement_noise_std": observation_noise}
        if block.model_type == SegmentType.CV and block.vel_change_std is not None:
            params["vel_change_std"] = _sample_param(block.vel_change_std)
        elif block.model_type == SegmentType.CA and block.accel_noise_std is not None:
            params["accel_noise_std"] = _sample_param(block.accel_noise_std)
        elif block.model_type == SegmentType.CT:
            if block.omega is None:
                raise ValueError("CT block requires omega")
            params["omega"] = _sample_param(block.omega)
            params["omega_noise_std"] = _sample_param(block.omega_noise_std or NumericParam(value=0.0))
        elif block.model_type == SegmentType.SINGER:
            if block.tau is None:
                raise ValueError("SINGER block requires tau")
            params["tau"] = _sample_param(block.tau)
            params["sigma_a"] = _sample_param(block.sigma_a or NumericParam(value=0.5))
            params["noise_std"] = observation_noise
        return block.model_type.value, block.steps, params

    def generate_trajectories(self) -> None:
        payloads: list[dict[str, Any]] = []
        total = self.params.num_trajectories
        LOGGER.info("Equations generation started job_id=%s total=%s", self.job_id, total)

        for idx in range(total):
            obs_noise = _sample_param(self.params.observation_noise)
            segments = [self._segment_to_params(block, obs_noise) for block in self.params.blocks]
            initial_state = None
            velocity_values = None
            if self.params.initial_velocity_params is not None:
                velocity_values = [_sample_param(p) for p in self.params.initial_velocity_params]
            elif self.params.initial_velocity is not None:
                velocity_values = self.params.initial_velocity
            if velocity_values is not None:
                dim = self.params.dim
                initial_state = TrajectoryState(
                    position=np.zeros(dim, dtype=float),
                    velocity=np.asarray(velocity_values, dtype=float),
                    acceleration=(
                        np.asarray(self.params.initial_acceleration, dtype=float)
                        if self.params.initial_acceleration is not None
                        else None
                    ),
                )
            noisy, clean, _ = generate_composite_trajectory(
                trajectory_segments=segments,
                dt=self.params.dt,
                dim=self.params.dim,
                initial_state=initial_state,
                seed=None if self.params.seed is None else self.params.seed + idx,
                randomize_blueprint=self.params.randomize_from_current_blocks,
                min_segment_length=self.params.min_segment_length,
                max_segment_length=self.params.max_segment_length,
                target_T=self.params.target_total_steps,
            )
            trajectory_config = {
                "dt": self.params.dt,
                "dim": self.params.dim,
                "seed": None if self.params.seed is None else self.params.seed + idx,
                "observation_noise_std": obs_noise,
                "randomize_from_current_blocks": self.params.randomize_from_current_blocks,
                "min_segment_length": self.params.min_segment_length,
                "max_segment_length": self.params.max_segment_length,
                "target_total_steps": self.params.target_total_steps,
                "initial_velocity": velocity_values,
                "initial_acceleration": self.params.initial_acceleration,
                "segments": [
                    {"model_type": model_type, "steps": steps, "params": params}
                    for model_type, steps, params in segments
                ],
            }
            item = {
                "id": idx,
                "type": "equations",
                "trajectory_config": trajectory_config,
                "noisy": noisy.tolist(),
                "clean": clean.tolist(),
            }
            payloads.append(item)
            if idx < 10:
                self.publish_preview(_to_plot_payload(noisy, clean))
            self.publish_progress(idx + 1, total, f"Generated {idx + 1}/{total}")
            LOGGER.info("Equations trajectory generated job_id=%s index=%s", self.job_id, idx)

        self.writer.write_trajectories(payloads)
        LOGGER.info("Equations trajectories written job_id=%s count=%s", self.job_id, len(payloads))
