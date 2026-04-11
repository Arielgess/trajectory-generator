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


class EquationsMissionParams(BaseModel):
    """Equations-generator source for the Eq→PX4 mission-mode profile.

    Generates the full trajectory upfront from the configured blocks, uploads it
    as a MAVLink mission, and lets PX4 execute it autonomously.  Because all
    waypoints are on the vehicle before takeoff, this works correctly at any
    sim speed factor — no Python timing loop is required during flight.
    """

    model_config = ConfigDict(protected_namespaces=())
    dt: float = Field(..., gt=0)
    dim: int = 3
    seed: int | None = None
    randomize_from_current_blocks: bool = False
    min_segment_length: int = 20
    max_segment_length: int = 60
    target_total_steps: int | None = None
    blocks: list[SegmentBlock]
    initial_velocity: list[float] | None = None
    initial_velocity_params: list[NumericParam] | None = None
    initial_acceleration: list[float] | None = None
    # Mission upload / waypoint settings
    mission_max_waypoints: int = Field(default=900, ge=3, le=900)
    mission_min_step_m: float = Field(default=0.5, gt=0)
    waypoint_acceptance_radius_m: float = Field(default=2.0, gt=0)
    # PX4 flight dynamics — when randomize_flight_dynamics=True each of the five MPC params
    # below is applied before the flight using its NumericParam mode (fixed or sampled range).
    randomize_flight_dynamics: bool = False
    mpc_acc_hor_max: NumericParam = Field(
        default_factory=lambda: NumericParam(mode=ParamMode.RANGE, value=12.5, min_value=5.0, max_value=20.0)
    )
    mpc_jerk_max: NumericParam = Field(
        default_factory=lambda: NumericParam(mode=ParamMode.RANGE, value=20.0, min_value=5.0, max_value=35.0)
    )
    mpc_xy_p: NumericParam = Field(
        default_factory=lambda: NumericParam(mode=ParamMode.RANGE, value=1.5, min_value=1.0, max_value=2.0)
    )
    mpc_tiltmax_air: NumericParam = Field(
        default_factory=lambda: NumericParam(mode=ParamMode.RANGE, value=62.5, min_value=45.0, max_value=80.0)
    )
    mpc_xy_vel_p_acc: NumericParam = Field(
        default_factory=lambda: NumericParam(mode=ParamMode.RANGE, value=2.65, min_value=1.8, max_value=3.5)
    )
    # Safety: shift the whole trajectory upward if any point is below this altitude above home.
    # Equations start at z=0 (ground in NED), so a non-zero floor is essential.
    min_altitude_m: float = Field(default=10.0, gt=0)

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, value: int) -> int:
        if value != 3:
            raise ValueError("equations_mission requires dim=3 for PX4")
        return value

    @model_validator(mode="after")
    def validate_blocks(self) -> "EquationsMissionParams":
        if not self.blocks:
            raise ValueError("At least one segment block is required")
        if self.randomize_from_current_blocks and self.target_total_steps is None:
            raise ValueError("target_total_steps is required when randomize_from_current_blocks is true")
        if self.min_segment_length > self.max_segment_length:
            raise ValueError("min_segment_length must be <= max_segment_length")
        if self.initial_velocity is not None and len(self.initial_velocity) != self.dim:
            raise ValueError("initial_velocity length must match dim")
        if self.initial_velocity_params is not None and len(self.initial_velocity_params) != self.dim:
            raise ValueError("initial_velocity_params length must match dim")
        if self.initial_acceleration is not None and len(self.initial_acceleration) != self.dim:
            raise ValueError("initial_acceleration length must match dim")
        return self


class Px4Request(BaseModel):
    output_dir: str
    num_trajectories: int = Field(..., ge=1, le=10000)
    duration_s: float = Field(..., gt=0)
    dt_s: float = Field(..., gt=0)
    observation_noise: NumericParam
    connection_uri: str
    seed: int | None = None
    profile_name: str = "default"
    motion: Px4MotionConfig | None = None
    equations_mission: EquationsMissionParams | None = None

    @model_validator(mode="after")
    def validate_profile_payload(self) -> "Px4Request":
        if self.profile_name == "equations_mission":
            if self.equations_mission is None:
                raise ValueError("equations_mission profile requires equations_mission")
        elif self.motion is None:
            raise ValueError("motion is required for this PX4 profile")
        return self


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
