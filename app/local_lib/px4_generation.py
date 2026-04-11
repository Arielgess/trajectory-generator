from __future__ import annotations

import asyncio
import math
import os
import random
import time
from typing import Any

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw


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
    try:
        await drone.param.set_param_float("MPC_ACC_HOR_MAX", random.uniform(5.0, 20.0))
        await drone.param.set_param_float("MPC_JERK_MAX", random.uniform(5.0, 35.0))
        await drone.param.set_param_float("MPC_XY_P", random.uniform(1.0, 2.0))
        await drone.param.set_param_float("MPC_TILTMAX_AIR", random.uniform(45.0, 80.0))
        await drone.param.set_param_float("MPC_XY_VEL_P_ACC", random.uniform(1.8, 3.5))
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
    start_real_time = time.time()
    for p in base_traj:
        await drone.offboard.set_position_velocity_ned(
            PositionNedYaw(p["x"] + origin_n, p["y"] + origin_e, p["z"] + origin_d, p["yaw"]),
            VelocityNedYaw(p["vx"], p["vy"], p["vz"], 0.0),
        )
        setpoints_log.append(p)
        actual_sim_t = (time.time() - start_real_time) * speed_factor
        async for pos in drone.telemetry.position_velocity_ned():
            sample = {
                "t": actual_sim_t,
                "x": pos.position.north_m - origin_n,
                "y": pos.position.east_m - origin_e,
                "z": pos.position.down_m - origin_d,
                "vx": pos.velocity.north_m_s,
                "vy": pos.velocity.east_m_s,
                "vz": pos.velocity.down_m_s,
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
        await asyncio.sleep(dt / speed_factor)

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
