from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from mavsdk import System

from app.generators.base_generator import BaseTrajectoryGenerator
from app.generators.equations_generator import sample_equations_trajectory_pair
from app.local_lib.px4_generation import (
    apply_flight_dynamics,
    equations_clean_to_traj,
    fly_equations_mission_and_log,
    fly_trajectory_and_log,
    generate_base_uav_traj,
    generate_figure8_uav_traj,
    generate_s_turn_uav_traj,
    get_sim_speed_factor,
)
from app.models.schemas import NumericParam, Px4Request  # NumericParam used for zero-noise placeholder

LOGGER = logging.getLogger(__name__)


def _sample(param: NumericParam) -> float:
    if param.mode.value == "range":
        return random.uniform(param.min_value, param.max_value)
    return param.value


class PX4TrajectoryGenerator(BaseTrajectoryGenerator):
    def __init__(self, job_id: str, params: Px4Request, writer: Any) -> None:
        super().__init__(job_id=job_id, params=params, writer=writer)

    def _motion_cfg(self) -> dict[str, Any]:
        m = self.params.motion
        return {
            "num_waypoints_min": int(_sample(m.num_waypoints_min)),
            "num_waypoints_max": int(_sample(m.num_waypoints_max)),
            "waypoint_xy_range": {"min": _sample(m.waypoint_xy_min), "max": _sample(m.waypoint_xy_max)},
            "waypoint_z_range": {"min": _sample(m.waypoint_z_min), "max": _sample(m.waypoint_z_max)},
            "max_speed": _sample(m.max_speed),
            "accel": _sample(m.accel),
            "waypoint_tolerance": _sample(m.waypoint_tolerance),
        }

    def _build_trajectory(self) -> tuple[list[dict[str, float]], dict[str, Any]]:
        if self.params.profile_name == "figure8":
            return generate_figure8_uav_traj(self.params.duration_s, self.params.dt_s), {
                "profile_name": "figure8",
                "ignores_user_motion_hyperparameters": True,
            }
        if self.params.profile_name == "s_turn":
            return generate_s_turn_uav_traj(self.params.duration_s, self.params.dt_s), {
                "profile_name": "s_turn",
                "ignores_user_motion_hyperparameters": True,
            }
        motion_cfg = self._motion_cfg()
        return generate_base_uav_traj(self.params.duration_s, self.params.dt_s, motion_cfg), motion_cfg

    async def _run_one(self, idx: int, drone: System) -> dict[str, Any]:
        observation_noise_std = _sample(self.params.observation_noise)
        speed_factor = get_sim_speed_factor()

        if self.params.profile_name == "equations_mission":
            em = self.params.equations_mission
            assert em is not None
            seed = self.params.seed if self.params.seed is not None else em.seed
            # Observation noise is irrelevant for equations generation here — we only
            # use the clean trajectory.  PX4 telemetry noise is applied separately via
            # Px4Request.observation_noise when logging the flight.
            _, clean, eq_cfg = sample_equations_trajectory_pair(
                blocks=em.blocks,
                dt=em.dt,
                dim=em.dim,
                seed=seed,
                trajectory_index=idx,
                observation_noise=NumericParam(value=0.0),
                randomize_from_current_blocks=em.randomize_from_current_blocks,
                min_segment_length=em.min_segment_length,
                max_segment_length=em.max_segment_length,
                target_total_steps=em.target_total_steps,
                initial_velocity=em.initial_velocity,
                initial_velocity_params=em.initial_velocity_params,
                initial_acceleration=em.initial_acceleration,
            )
            duration_s = float(clean.shape[0]) * em.dt
            trajectory_metadata = {
                "duration_s": duration_s,
                "connection_uri": self.params.connection_uri,
                "profile_name": self.params.profile_name,
                "wait_px4_health_s": 60.0,
                "source_type": "equations_clean",
                "equations_trajectory_config": eq_cfg,
            }
            flight_dynamics = (
                {
                    "mpc_acc_hor_max": _sample(em.mpc_acc_hor_max),
                    "mpc_jerk_max": _sample(em.mpc_jerk_max),
                    "mpc_xy_p": _sample(em.mpc_xy_p),
                    "mpc_tiltmax_air": _sample(em.mpc_tiltmax_air),
                    "mpc_xy_vel_p_acc": _sample(em.mpc_xy_vel_p_acc),
                }
                if em.randomize_flight_dynamics
                else None
            )
            logged = await fly_equations_mission_and_log(
                drone,
                clean,
                em.dt,
                trajectory_metadata,
                speed_factor,
                observation_noise_std=observation_noise_std,
                mission_max_waypoints=em.mission_max_waypoints,
                mission_min_step_m=em.mission_min_step_m,
                waypoint_acceptance_radius_m=em.waypoint_acceptance_radius_m,
                min_altitude_m=em.min_altitude_m,
                flight_dynamics=flight_dynamics,
            )
            return {
                "id": idx,
                "type": "px4",
                "trajectory_config": {
                    "dt": em.dt,
                    "observation_noise_std": observation_noise_std,
                    "metadata": trajectory_metadata,
                },
                "setpoints": logged.get("setpoints", []),
                "clean": logged.get("clean", []),
                "noisy": logged.get("noisy", []),
            }

        traj, motion_cfg = self._build_trajectory()
        trajectory_metadata = {
            "duration_s": self.params.duration_s,
            "connection_uri": self.params.connection_uri,
            "profile_name": self.params.profile_name,
            "wait_px4_health_s": 60.0,
        }
        if self.params.profile_name == "default":
            trajectory_metadata["motion"] = motion_cfg
        else:
            trajectory_metadata["profile_settings"] = motion_cfg

        logged = await fly_trajectory_and_log(
            drone,
            traj,
            self.params.dt_s,
            trajectory_metadata,
            speed_factor,
            observation_noise_std=observation_noise_std,
        )
        return {
            "id": idx,
            "type": "px4",
            "trajectory_config": {
                "dt": self.params.dt_s,
                "observation_noise_std": observation_noise_std,
                "metadata": trajectory_metadata,
            },
            "setpoints": logged.get("setpoints", []),
            "clean": logged.get("clean", []),
            "noisy": logged.get("noisy", []),
        }

    async def _run_all(self) -> None:
        drone = System()
        await drone.connect(system_address=self.params.connection_uri)
        LOGGER.info("PX4 connected job_id=%s uri=%s", self.job_id, self.params.connection_uri)
        total = self.params.num_trajectories
        for idx in range(total):
            data = await self._run_one(idx, drone)
            if data:
                self.writer.append_trajectory(data)
                self.publish_preview(
                    {
                        "clean": data.get("clean", []),
                        "noisy": data.get("noisy", []),
                        "setpoints": data.get("setpoints", []),
                    }
                )
                self.publish_progress(idx + 1, total, f"Generated {idx + 1}/{total}")
                LOGGER.info("PX4 trajectory generated job_id=%s index=%s", self.job_id, idx)
            else:
                self.publish_progress(idx + 1, total, f"Trajectory {idx + 1}/{total} failed")
                LOGGER.warning("PX4 trajectory failed job_id=%s index=%s", self.job_id, idx)

    def generate_trajectories(self) -> None:
        asyncio.run(self._run_all())
