from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParamMode(str, Enum):
    FIXED = "fixed"
    RANGE = "range"


class NumericParam(BaseModel):
    mode: ParamMode = ParamMode.FIXED
    value: float = 0.0
    min_value: float = 0.0
    max_value: float = 1.0

    @model_validator(mode="after")
    def validate_range(self) -> "NumericParam":
        if self.mode == ParamMode.RANGE and self.min_value > self.max_value:
            raise ValueError("min_value must be <= max_value")
        return self


class SegmentType(str, Enum):
    CV = "CV"
    CA = "CA"
    CT = "CT"
    SINGER = "SINGER"


class SegmentBlock(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: SegmentType
    steps: int = Field(..., ge=2, le=5000)
    vel_change_std: NumericParam | None = None
    accel_noise_std: NumericParam | None = None
    omega: NumericParam | None = None
    omega_noise_std: NumericParam | None = None
    tau: NumericParam | None = None
    sigma_a: NumericParam | None = None


class EquationsRequest(BaseModel):
    output_dir: str
    num_trajectories: int = Field(..., ge=1, le=10000)
    dt: float = Field(..., gt=0)
    dim: int = Field(default=3)
    seed: int | None = None
    initial_velocity: list[float] | None = None
    initial_velocity_params: list[NumericParam] | None = None
    initial_acceleration: list[float] | None = None
    observation_noise: NumericParam
    randomize_from_current_blocks: bool = False
    min_segment_length: int = 20
    max_segment_length: int = 60
    target_total_steps: int | None = None
    blocks: list[SegmentBlock]

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, value: int) -> int:
        if value not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        return value

    @model_validator(mode="after")
    def validate_blocks(self) -> "EquationsRequest":
        if not self.blocks:
            raise ValueError("At least one segment block is required")
        if self.randomize_from_current_blocks and self.target_total_steps is None:
            raise ValueError("target_total_steps is required when randomize_from_current_blocks is true")
        if self.initial_velocity is not None and len(self.initial_velocity) != self.dim:
            raise ValueError("initial_velocity length must match dim")
        if self.initial_velocity_params is not None and len(self.initial_velocity_params) != self.dim:
            raise ValueError("initial_velocity_params length must match dim")
        if self.initial_acceleration is not None and len(self.initial_acceleration) != self.dim:
            raise ValueError("initial_acceleration length must match dim")
        return self


class Px4MotionConfig(BaseModel):
    num_waypoints_min: NumericParam
    num_waypoints_max: NumericParam
    waypoint_xy_min: NumericParam
    waypoint_xy_max: NumericParam
    waypoint_z_min: NumericParam
    waypoint_z_max: NumericParam
    max_speed: NumericParam
    accel: NumericParam
    waypoint_tolerance: NumericParam


class Px4Request(BaseModel):
    output_dir: str
    num_trajectories: int = Field(..., ge=1, le=10000)
    duration_s: float = Field(..., gt=0)
    dt_s: float = Field(..., gt=0)
    observation_noise: NumericParam
    connection_uri: str
    seed: int | None = None
    profile_name: str = "default"
    motion: Px4MotionConfig


class JobType(str, Enum):
    EQUATIONS = "equations"
    PX4 = "px4"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobAcceptedResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    generated_count: int
    failed_count: int
    message: str | None = None
    output_path: str | None = None


class ProgressEvent(BaseModel):
    job_id: str
    event_type: str
    payload: dict[str, Any]
