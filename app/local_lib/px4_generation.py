from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from typing import Any

import numpy as np
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.mission_raw import MissionRawError
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

from app.local_lib.mission_conversion import (
    build_nav_waypoint_mission_items,
    clean_positions_to_path_points,
    thin_path_points,
)

LOGGER = logging.getLogger(__name__)


def get_sim_speed_factor() -> float:
    return float(os.environ.get("PX4_SIM_SPEED_FACTOR", 1.0))


def sample_param(params: dict[str, Any], name: str, default: float) -> float:
    val = params.get(name, default)
    if isinstance(val, dict):
        return random.uniform(float(val.get("min", default)), float(val.get("max", default)))
    return float(val)


def sample_range_pair(params: dict[str, Any], name: str, default_min: float, default_max: float) -> tuple[float, float]:
    val = params.get(name, {"min": default_min, "max": default_max})
    return float(val.get("min", default_min)), float(val.get("max", default_max))


def generate_base_uav_traj(duration: float, dt: float, motion_cfg: dict[str, Any]) -> list[dict[str, float]]:
    num_wp_min = int(motion_cfg.get("num_waypoints_min", 3))
    num_wp_max = int(motion_cfg.get("num_waypoints_max", 8))
    num_wp = random.randint(num_wp_min, num_wp_max)
    xy_min, xy_max = sample_range_pair(motion_cfg, "waypoint_xy_range", -80.0, 80.0)
    z_min, z_max = sample_range_pair(motion_cfg, "waypoint_z_range", -30.0, -5.0)
    max_speed = sample_param(motion_cfg, "max_speed", 7.0)
    accel = sample_param(motion_cfg, "accel", 4.0)
    wp_tol = float(motion_cfg.get("waypoint_tolerance", 3.0))

    waypoints = [{"x": 0.0, "y": 0.0, "z": 0.0}]
    for _ in range(num_wp - 1):
        waypoints.append({"x": random.uniform(xy_min, xy_max), "y": random.uniform(xy_min, xy_max), "z": random.uniform(z_min, z_max)})

    px, py, pz = waypoints[0]["x"], waypoints[0]["y"], waypoints[0]["z"]
    vx, vy, vz = 0.0, 0.0, 0.0
    current_wp_index = 1
    points = []
    t = 0.0

    while t <= duration:
        if current_wp_index < len(waypoints):
            target = waypoints[current_wp_index]
            dx, dy, dz = target["x"] - px, target["y"] - py, target["z"] - pz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < wp_tol:
                current_wp_index += 1
                dx, dy, dz, dist = 0.0, 0.0, 0.0, 0.0
            if dist > 1e-3:
                vx_target = (dx / dist) * max_speed
                vy_target = (dy / dist) * max_speed
                vz_target = (dz / dist) * max_speed
            else:
                vx_target, vy_target, vz_target = 0.0, 0.0, 0.0
        else:
            vx_target, vy_target, vz_target = 0.0, 0.0, 0.0
        alpha = min(1.0, accel * dt / max(max_speed, 1e-3))
        vx += (vx_target - vx) * alpha
        vy += (vy_target - vy) * alpha
        vz += (vz_target - vz) * alpha
        px += vx * dt
        py += vy * dt
        pz += vz * dt
        yaw = math.degrees(math.atan2(vy, vx)) if (abs(vx) + abs(vy) > 1e-3) else 0.0
        points.append({"t": t, "x": px, "y": py, "z": pz, "yaw": yaw})
        t += dt
    return points


def _climb_profile(progress: float, cruise_altitude_m: float) -> float:
    climb_progress = min(1.0, max(0.0, progress / 0.15))
    smooth = 0.5 - 0.5 * math.cos(math.pi * climb_progress)
    return cruise_altitude_m * smooth


def generate_figure8_uav_traj(duration: float, dt: float) -> list[dict[str, float]]:
    amplitude_x = 18.0
    amplitude_y = 12.0
    cruise_altitude_m = -10.0
    altitude_wave_m = 1.5
    omega = (2.0 * math.pi) / max(duration, dt)
    points: list[dict[str, float]] = []
    t = 0.0

    while t <= duration:
        progress = t / max(duration, dt)
        x = amplitude_x * math.sin(omega * t)
        y = amplitude_y * math.sin(omega * t) * math.cos(omega * t)
        z = _climb_profile(progress, cruise_altitude_m) + altitude_wave_m * math.sin(0.5 * omega * t)
        vx = amplitude_x * omega * math.cos(omega * t)
        vy = amplitude_y * omega * math.cos(2.0 * omega * t)
        yaw = math.degrees(math.atan2(vy, vx)) if (abs(vx) + abs(vy) > 1e-6) else 0.0
        points.append({"t": t, "x": x, "y": y, "z": z, "yaw": yaw})
        t += dt

    return points


def generate_s_turn_uav_traj(duration: float, dt: float) -> list[dict[str, float]]:
    forward_distance_m = 50.0
    lateral_amplitude_m = 12.0
    cruise_altitude_m = -10.0
    altitude_wave_m = 1.0
    omega = (2.0 * math.pi) / max(duration, dt)
    vx_nominal = forward_distance_m / max(duration, dt)
    points: list[dict[str, float]] = []
    t = 0.0

    while t <= duration:
        progress = t / max(duration, dt)
        x = forward_distance_m * progress
        y = lateral_amplitude_m * math.sin(omega * t)
        z = _climb_profile(progress, cruise_altitude_m) + altitude_wave_m * math.sin(omega * t)
        vx = vx_nominal
        vy = lateral_amplitude_m * omega * math.cos(omega * t)
        yaw = math.degrees(math.atan2(vy, vx)) if (abs(vx) + abs(vy) > 1e-6) else 0.0
        points.append({"t": t, "x": x, "y": y, "z": z, "yaw": yaw})
        t += dt

    return points


async def randomize_flight_dynamics(drone: System) -> None:
    """Apply random MPC dynamics (used by the offboard PX4 profile)."""
    await apply_flight_dynamics(
        drone,
        mpc_acc_hor_max=random.uniform(5.0, 20.0),
        mpc_jerk_max=random.uniform(5.0, 35.0),
        mpc_xy_p=random.uniform(1.0, 2.0),
        mpc_tiltmax_air=random.uniform(45.0, 80.0),
        mpc_xy_vel_p_acc=random.uniform(1.8, 3.5),
    )


async def apply_flight_dynamics(
    drone: System,
    *,
    mpc_acc_hor_max: float,
    mpc_jerk_max: float,
    mpc_xy_p: float,
    mpc_tiltmax_air: float,
    mpc_xy_vel_p_acc: float,
) -> None:
    """Set specific MPC flight-controller parameters on the connected drone."""
    try:
        await drone.param.set_param_float("MPC_ACC_HOR_MAX", mpc_acc_hor_max)
        await drone.param.set_param_float("MPC_JERK_MAX", mpc_jerk_max)
        await drone.param.set_param_float("MPC_XY_P", mpc_xy_p)
        await drone.param.set_param_float("MPC_TILTMAX_AIR", mpc_tiltmax_air)
        await drone.param.set_param_float("MPC_XY_VEL_P_ACC", mpc_xy_vel_p_acc)
    except Exception:
        return


async def wait_for_px4_health(drone: System, timeout_s: float, speed_factor: float) -> None:
    start = time.time()
    async for health in drone.telemetry.health():
        current_time = time.time()
        if health.is_global_position_ok and health.is_local_position_ok and health.is_home_position_ok:
            await asyncio.sleep(10.0 / speed_factor)
            return
        if current_time - start > (timeout_s / speed_factor):
            await asyncio.sleep(5.0 / speed_factor)
            return


async def fly_trajectory_and_log(
    drone: System,
    base_traj: list[dict[str, float]],
    dt: float,
    metadata: dict[str, Any],
    speed_factor: float,
    observation_noise_std: float = 0.0,
) -> dict[str, Any]:
    yaws_rad = [math.radians(p["yaw"]) for p in base_traj]
    unwrapped = [yaws_rad[0]]
    for i in range(1, len(yaws_rad)):
        diff = (yaws_rad[i] - yaws_rad[i - 1] + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(unwrapped[-1] + diff)
    for i, p in enumerate(base_traj):
        p["yaw"] = math.degrees(unwrapped[i])

    for i in range(len(base_traj)):
        if i < len(base_traj) - 1:
            dt_step = base_traj[i + 1]["t"] - base_traj[i]["t"]
            if dt_step > 0:
                base_traj[i]["vx"] = (base_traj[i + 1]["x"] - base_traj[i]["x"]) / dt_step
                base_traj[i]["vy"] = (base_traj[i + 1]["y"] - base_traj[i]["y"]) / dt_step
                base_traj[i]["vz"] = (base_traj[i + 1]["z"] - base_traj[i]["z"]) / dt_step
            else:
                base_traj[i]["vx"] = base_traj[i]["vy"] = base_traj[i]["vz"] = 0.0
        else:
            base_traj[i]["vx"] = base_traj[i - 1]["vx"]
            base_traj[i]["vy"] = base_traj[i - 1]["vy"]
            base_traj[i]["vz"] = base_traj[i - 1]["vz"]

    await wait_for_px4_health(drone, metadata["wait_px4_health_s"], speed_factor)
    await randomize_flight_dynamics(drone)

    arm_attempts = 0
    while arm_attempts < 10:
        try:
            await drone.action.arm()
            break
        except ActionError:
            arm_attempts += 1
            if arm_attempts >= 10:
                return {}
            await asyncio.sleep(5.0 / speed_factor)

    await drone.action.takeoff()
    async for pos in drone.telemetry.position():
        if pos.relative_altitude_m > 2.0:
            break

    origin_n = origin_e = origin_d = 0.0
    async for pos in drone.telemetry.position_velocity_ned():
        origin_n, origin_e, origin_d = pos.position.north_m, pos.position.east_m, pos.position.down_m
        break

    initial_pos = PositionNedYaw(origin_n, origin_e, origin_d, base_traj[0]["yaw"])
    initial_vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    offboard_started = False
    for _ in range(5):
        for _ in range(10):
            await drone.offboard.set_position_velocity_ned(initial_pos, initial_vel)
            await asyncio.sleep(0.01)
        try:
            await drone.offboard.start()
            offboard_started = True
            break
        except OffboardError:
            pass
    if not offboard_started:
        await drone.action.land()
        return {}

    clean_log: list[dict[str, float]] = []
    noisy_log: list[dict[str, float]] = []
    setpoints_log: list[dict[str, float]] = []
    step_dt_wall = dt / speed_factor
    wall_start = time.perf_counter()
    next_wall_t = wall_start + step_dt_wall
    for p in base_traj:
        await drone.offboard.set_position_velocity_ned(
            PositionNedYaw(p["x"] + origin_n, p["y"] + origin_e, p["z"] + origin_d, p["yaw"]),
            VelocityNedYaw(p["vx"], p["vy"], p["vz"], 0.0),
        )
        # Store z as positive altitude above origin (up = positive) for output.
        setpoints_log.append({**p, "z": -p["z"]})
        actual_sim_t = (time.perf_counter() - wall_start) * speed_factor
        async for pos in drone.telemetry.position_velocity_ned():
            sample = {
                "t": actual_sim_t,
                "x": pos.position.north_m - origin_n,
                "y": pos.position.east_m - origin_e,
                "z": origin_d - pos.position.down_m,   # positive = above home
                "vx": pos.velocity.north_m_s,
                "vy": pos.velocity.east_m_s,
                "vz": -pos.velocity.down_m_s,           # positive = climbing
            }
            clean_log.append(sample)
            noisy_log.append(
                {
                    **sample,
                    "x": sample["x"] + random.gauss(0.0, observation_noise_std),
                    "y": sample["y"] + random.gauss(0.0, observation_noise_std),
                    "z": sample["z"] + random.gauss(0.0, observation_noise_std),
                }
            )
            break
        # Accumulating timer: compensates for loop overhead and doesn't add up sleep delays.
        # If we're already behind schedule (high speed_factor), skip sleep and continue.
        wait_s = next_wall_t - time.perf_counter()
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        next_wall_t += step_dt_wall

    traj_wall_s = time.perf_counter() - wall_start
    LOGGER.info(
        "Offboard trajectory done — wall: %.1fs  sim: %.1fs  speed_factor: %.1f",
        traj_wall_s,
        traj_wall_s * speed_factor,
        speed_factor,
    )

    try:
        await drone.offboard.stop()
        await drone.action.land()
    except Exception:
        pass
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            break
    try:
        await drone.action.disarm()
    except Exception:
        pass

    return {"metadata": metadata, "setpoints": setpoints_log, "clean": clean_log, "noisy": noisy_log}


def equations_clean_to_traj(clean: np.ndarray, dt: float) -> list[dict[str, float]]:
    """Convert equations `clean` position array to the path-point format used by fly_trajectory_and_log."""
    return clean_positions_to_path_points(clean, dt)


def _normalize_path_altitude(
    path_points: list[dict[str, float]],
    min_altitude_m: float,
) -> list[dict[str, float]]:
    """Shift the whole trajectory upward if any point would be below min_altitude_m above home.

    Equations generate paths in arbitrary space and often start at z=0 (ground in NED).
    This ensures every waypoint is at least `min_altitude_m` above the home/arm position,
    preserving relative height differences within the trajectory.
    """
    if not path_points:
        return path_points
    # NED: z more negative = higher altitude.  altitude_m = -z_NED.
    # We need -z <= min_altitude_m for all points  →  z >= -min_altitude_m for the max z.
    max_z = max(p["z"] for p in path_points)
    if max_z > -min_altitude_m:
        shift = -min_altitude_m - max_z  # negative shift = move everything upward
        return [{**p, "z": p["z"] + shift} for p in path_points]
    return path_points


async def fly_equations_mission_and_log(
    drone: System,
    clean: np.ndarray,
    dt: float,
    metadata: dict[str, Any],
    speed_factor: float,
    observation_noise_std: float,
    mission_max_waypoints: int,
    mission_min_step_m: float,
    waypoint_acceptance_radius_m: float,
    min_altitude_m: float,
    flight_dynamics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convert `clean` equation positions to a PX4 mission, upload upfront, fly, and log NED telemetry.

    Designed for fast-simulation runs: all waypoints are uploaded before the flight starts
    so PX4 drives its own timing — no Python sleep loop is needed during execution.

    Execution sequence (multicopter):
    1. Wait for health → fetch home lat/lon at ground level.
    2. Optionally randomise PX4 MPC flight-dynamics parameters for diverse dataset dynamics.
    3. Normalize trajectory altitude so all waypoints are above min_altitude_m.
    4. Thin waypoints and upload as a global MAVLink mission.
    4. Arm.  PX4 Mission mode handles takeoff automatically.
    5. start_mission() — must come AFTER arm, never before.
    6. Poll for completion; log NED telemetry asynchronously.
    7. Land and disarm.
    """
    path_points = clean_positions_to_path_points(clean, dt)
    path_points = _normalize_path_altitude(path_points, min_altitude_m)
    thinned = thin_path_points(path_points, mission_min_step_m, mission_max_waypoints)

    await wait_for_px4_health(drone, metadata["wait_px4_health_s"], speed_factor)

    if flight_dynamics is not None:
        await apply_flight_dynamics(drone, **flight_dynamics)
        LOGGER.info("PX4 flight dynamics applied: %s", flight_dynamics)

    # Capture home lat/lon and actual NED position before arming.
    # PX4's local NED frame origin is the EKF reference point set at simulator startup —
    # it is NOT necessarily at the drone's home/arm position.  We must record the drone's
    # actual NED position here and subtract it from all telemetry, otherwise the logged
    # clean positions will be offset by the spawn-to-EKF-origin distance (often 100+ m).
    home_lat = home_lon = 0.0
    async for pos in drone.telemetry.position():
        home_lat, home_lon = pos.latitude_deg, pos.longitude_deg
        break

    origin_n = origin_e = origin_d = 0.0
    async for pv in drone.telemetry.position_velocity_ned():
        origin_n = pv.position.north_m
        origin_e = pv.position.east_m
        origin_d = pv.position.down_m
        break
    LOGGER.info("Home NED origin captured: (%.2f, %.2f, %.2f)", origin_n, origin_e, origin_d)

    mission_items = build_nav_waypoint_mission_items(
        thinned,
        0.0,
        0.0,
        0.0,
        home_lat,
        home_lon,
        acceptance_radius_m=waypoint_acceptance_radius_m,
    )

    # Upload before arming — commander needs a valid mission before mode switch.
    try:
        try:
            await drone.mission_raw.clear_mission()
        except MissionRawError:
            pass
        await drone.mission_raw.upload_mission(mission_items)
        LOGGER.info("Mission uploaded (%s waypoints); waiting for navigator to validate.", len(mission_items))
        await asyncio.sleep(1.5 / speed_factor)
    except MissionRawError as exc:
        LOGGER.warning("Mission upload failed: %s", exc)
        return {}

    # Arm.
    arm_attempts = 0
    while arm_attempts < 10:
        try:
            await drone.action.arm()
            break
        except ActionError:
            arm_attempts += 1
            if arm_attempts >= 10:
                return {}
            await asyncio.sleep(5.0 / speed_factor)

    # Store setpoints with positive z (altitude above home, up = positive).
    setpoints_log = [dict(p, z=-p["z"]) for p in thinned]

    # Mission mode is speed-factor-agnostic: PX4 drives all timing autonomously.
    # The telemetry logger streams whatever PX4 pushes (no artificial sleep) and
    # assigns t = sample_index * dt so the output is consistent regardless of
    # simulation speed.
    #
    # The initial climb from ground to min_altitude_m IS part of the logged data —
    # it is controlled by the min_altitude_m parameter and can be trimmed in
    # post-processing by discarding samples where z < min_altitude_m.
    clean_log: list[dict[str, float]] = []
    noisy_log: list[dict[str, float]] = []
    stop_logging = asyncio.Event()

    async def _log_telemetry() -> None:
        sample_idx = 0
        async for pos in drone.telemetry.position_velocity_ned():
            if stop_logging.is_set():
                break
            sample = {
                "t": sample_idx * dt,
                "x": pos.position.north_m - origin_n,
                "y": pos.position.east_m - origin_e,
                "z": origin_d - pos.position.down_m,   # positive = above home
                "vx": pos.velocity.north_m_s,
                "vy": pos.velocity.east_m_s,
                "vz": -pos.velocity.down_m_s,           # positive = climbing
            }
            clean_log.append(sample)
            noisy_log.append(
                {
                    **sample,
                    "x": sample["x"] + random.gauss(0.0, observation_noise_std),
                    "y": sample["y"] + random.gauss(0.0, observation_noise_std),
                    "z": sample["z"] + random.gauss(0.0, observation_noise_std),
                }
            )
            sample_idx += 1

    # Start mission and begin logging immediately — the initial climb from ground to
    # min_altitude_m is captured in the data so the user can see exactly when the
    # waypoint-following phase begins.
    mission_wall_start = time.perf_counter()
    try:
        await drone.mission_raw.start_mission()
        LOGGER.info(
            "Mission started — initial climb to %.1f m AGL included in log.  "
            "Trim clean[z < %.1f] in post-processing to isolate the waypoint phase.",
            min_altitude_m, min_altitude_m,
        )
    except MissionRawError as exc:
        LOGGER.warning("Mission start failed: %s", exc)
        try:
            await drone.action.land()
        except Exception:
            pass
        return {}

    log_task = asyncio.create_task(_log_telemetry())

    # Detect mission completion by watching the flight mode.
    # PX4 exits MISSION mode (→ HOLD) when the last waypoint is reached.
    # This is more reliable than mission_progress().current >= total which PX4
    # does not always emit for the final waypoint.
    # Timeout is in real wall time — no speed factor needed.
    est_duration_wall_s = (max(float(clean.shape[0]) * dt, 5.0) * 2.0 + 60.0) / speed_factor

    async def _wait_mission_done() -> None:
        from mavsdk.telemetry import FlightMode
        in_mission = False
        async for fm in drone.telemetry.flight_mode():
            if fm == FlightMode.MISSION:
                in_mission = True
            elif in_mission:
                LOGGER.info("Flight mode left MISSION → %s; mission complete.", fm)
                break

    try:
        await asyncio.wait_for(_wait_mission_done(), timeout=est_duration_wall_s)
        mission_wall_s = time.perf_counter() - mission_wall_start
        LOGGER.info(
            "Mission finished — wall: %.1fs  clean_samples: %d  setpoints: %d",
            mission_wall_s,
            len(clean_log),
            len(setpoints_log),
        )
    except asyncio.TimeoutError:
        LOGGER.warning("Mission timeout (%.0fs wall) exceeded.", est_duration_wall_s)
    finally:
        stop_logging.set()
        await log_task

    try:
        await drone.action.land()
    except Exception:
        pass
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            break
    try:
        await drone.action.disarm()
    except Exception:
        pass


    # min_altitude_m is the post-processing trim threshold: samples where
    # z < min_altitude_m are the initial climb and can be discarded to isolate
    # the waypoint-following phase.
    metadata["min_altitude_m"] = min_altitude_m
    return {"metadata": metadata, "setpoints": setpoints_log, "clean": clean_log, "noisy": noisy_log}
