from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AppDefaults:
    equations_dt: float = 0.04
    equations_dim: int = 3
    equations_num_trajectories: int = 20
    equations_steps_per_segment: int = 60
    equations_observation_noise: float = 0.1
    equations_observation_noise_min: float = 0.0
    equations_observation_noise_max: float = 1.0
    equations_initial_velocity_2d: tuple[float, float] = (5.0, 5.0)
    equations_initial_velocity_3d: tuple[float, float, float] = (5.0, 5.0, 1.0)
    equations_initial_acceleration_2d: tuple[float, float] = (0.6, 0.2)
    equations_initial_acceleration_3d: tuple[float, float, float] = (0.6, 0.2, 0.1)
    px4_num_trajectories: int = 1
    px4_duration_s: float = 20.0
    px4_dt_s: float = 0.04
    px4_observation_noise: float = 0.1
    px4_observation_noise_min: float = 0.1
    px4_observation_noise_max: float = 0.3
    px4_connection_uri: str = "udpin://0.0.0.0:14540"


DEFAULTS = AppDefaults()


def default_output_dir() -> str:
    return str(Path.cwd() / "output")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
