from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrajectoryState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: Optional[np.ndarray] = None
    omega: Optional[float] = None
    tau: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float)
            if len(self.acceleration) != len(self.position):
                raise ValueError("Acceleration dimension must match position dimension")
        if len(self.velocity) != len(self.position):
            raise ValueError("Velocity dimension must match position dimension")
        if self.tau is not None:
            self.tau = np.asarray(self.tau, dtype=float)
            if len(self.tau) != len(self.position):
                raise ValueError("Tau dimension must match position dimension")


def generate_cv_trajectory(
    T: int,
    dt: float,
    initial_state: TrajectoryState,
    vel_change_std: float | np.ndarray = 0.0,
    measurement_noise_std: float | np.ndarray = 0.0,
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    dim = len(initial_state.position)
    rng = np.random.default_rng(seed)
    vel_change_std = np.full(dim, float(vel_change_std)) if np.isscalar(vel_change_std) else np.asarray(vel_change_std, dtype=float)
    measurement_noise_std = (
        np.full(dim, float(measurement_noise_std))
        if np.isscalar(measurement_noise_std)
        else np.asarray(measurement_noise_std, dtype=float)
    )

    trajectories = []
    for _ in range(number_of_trajectories):
        state = TrajectoryState(position=initial_state.position.copy(), velocity=initial_state.velocity.copy())
        clean = np.empty((T, dim), dtype=float)
        clean[0] = state.position
        for t in range(1, T):
            prev_velocity = state.velocity.copy()
            state.velocity += rng.normal(0.0, vel_change_std, size=dim)
            state.position += prev_velocity * dt
            clean[t] = state.position
        noisy = clean + rng.normal(0.0, measurement_noise_std, size=(T, dim))
        trajectories.append((noisy, clean, state))
    return trajectories


def generate_ca_trajectory(
    T: int,
    dt: float,
    initial_state: TrajectoryState,
    measurement_noise_std: float | np.ndarray = 0.0,
    accel_noise_std: float | np.ndarray = 0.0,
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    dim = len(initial_state.position)
    rng = np.random.default_rng(seed)
    measurement_noise_std = (
        np.full(dim, float(measurement_noise_std))
        if np.isscalar(measurement_noise_std)
        else np.asarray(measurement_noise_std, dtype=float)
    )
    accel_noise_std = np.full(dim, float(accel_noise_std)) if np.isscalar(accel_noise_std) else np.asarray(accel_noise_std, dtype=float)

    trajectories = []
    for _ in range(number_of_trajectories):
        state = TrajectoryState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
            acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else np.zeros(dim, dtype=float),
        )
        clean = np.zeros((T, dim), dtype=float)
        for t in range(T):
            clean[t] = state.position
            state.position += state.velocity * dt + 0.5 * state.acceleration * dt**2
            state.velocity += state.acceleration * dt
            state.acceleration += rng.normal(0.0, accel_noise_std, size=dim)
        noisy = clean + rng.normal(0.0, measurement_noise_std, size=(T, dim))
        trajectories.append((noisy, clean, state))
    return trajectories


def generate_ct_trajectory_simple(
    T: int,
    dt: float,
    omega: float,
    dim: int = 2,
    initial_state: Optional[TrajectoryState] = None,
    omega_noise_std: float = 0.0,
    measurement_noise_std: float | np.ndarray = 0.0,
    z_acceleration: Optional[float] | np.ndarray = None,
    number_of_trajectories: int = 1,
    seed: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    rng = np.random.default_rng(seed)
    measurement_noise_std = (
        np.full(dim, float(measurement_noise_std))
        if np.isscalar(measurement_noise_std)
        else np.asarray(measurement_noise_std, dtype=float)
    )
    trajectories = []
    for _ in range(number_of_trajectories):
        if initial_state is None:
            state = TrajectoryState(position=rng.uniform(-10, 10, size=dim), velocity=rng.uniform(-5, 5, size=dim), omega=omega)
        else:
            state = TrajectoryState(position=initial_state.position.copy(), velocity=initial_state.velocity.copy(), omega=omega)
        if dim == 3:
            if state.acceleration is not None:
                az = state.acceleration[2]
            elif z_acceleration is not None:
                az = z_acceleration
            else:
                az = rng.normal(0, 0.1)
                state.acceleration = np.array([0.0, 0.0, az])

        clean = np.zeros((T, dim), dtype=float)
        clean[0] = state.position
        x, y = state.position[0], state.position[1]
        vx, vy = state.velocity[0], state.velocity[1]
        for t in range(1, T):
            current_omega = omega + rng.normal(0, omega_noise_std)
            x += vx * dt
            y += vy * dt
            cos_omega_dt = np.cos(current_omega * dt)
            sin_omega_dt = np.sin(current_omega * dt)
            vx_new = vx * cos_omega_dt - vy * sin_omega_dt
            vy_new = vx * sin_omega_dt + vy * cos_omega_dt
            state.position[0], state.position[1] = x, y
            state.velocity[0], state.velocity[1] = vx_new, vy_new
            state.omega = current_omega
            vx, vy = vx_new, vy_new
            if dim == 3:
                z, vz = state.position[2], state.velocity[2]
                z_new = z + vz * dt + 0.5 * az * dt**2
                vz_new = vz + az * dt
                state.position[2], state.velocity[2] = z_new, vz_new
                state.acceleration[2] = az
            clean[t] = state.position
        noisy = clean + rng.normal(0, measurement_noise_std, size=(T, dim)) if np.any(measurement_noise_std > 0) else clean.copy()
        trajectories.append((noisy, clean, state))
    return trajectories


def generate_singer_trajectory(
    T: int,
    dt: float,
    tau: float | np.ndarray,
    dim: int = 2,
    sigma_a: float | np.ndarray = 0.5,
    initial_state: Optional[TrajectoryState] = None,
    noise_std: float | np.ndarray = 0.0,
    number_of_trajectories: int = 1,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, TrajectoryState]]:
    rng = np.random.default_rng(seed)
    tau = np.full(dim, float(tau)) if np.isscalar(tau) else np.asarray(tau, dtype=float)
    sigma_a = np.full(dim, float(sigma_a)) if np.isscalar(sigma_a) else np.asarray(sigma_a, dtype=float)
    noise_std = np.full(dim, float(noise_std)) if np.isscalar(noise_std) else np.asarray(noise_std, dtype=float)
    alpha = 1.0 / tau
    exp_alpha_dt = np.exp(-alpha * dt)
    exp_2alpha_dt = np.exp(-2.0 * alpha * dt)
    accel_process_std = sigma_a * np.sqrt((1.0 - exp_2alpha_dt) / (2.0 * alpha))

    trajectories = []
    for _ in range(number_of_trajectories):
        if initial_state is None:
            state = TrajectoryState(
                position=rng.uniform(0, 4, size=dim),
                velocity=rng.uniform(-0.4, 0.4, size=dim),
                acceleration=rng.uniform(-0.1, 0.1, size=dim),
                tau=tau,
            )
        else:
            state = TrajectoryState(
                position=initial_state.position.copy(),
                velocity=initial_state.velocity.copy(),
                acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else np.zeros(dim, dtype=float),
                tau=tau,
            )
        clean = np.empty((T, dim), dtype=float)
        clean[0] = state.position
        for t in range(1, T):
            state.acceleration = exp_alpha_dt * state.acceleration + rng.normal(0.0, accel_process_std, size=dim)
            small_x_mask = (alpha * dt) < 1e-6
            delta_v = np.zeros(dim, dtype=float)
            delta_x = np.zeros(dim, dtype=float)
            for i in range(dim):
                if small_x_mask[i]:
                    delta_v[i] = state.acceleration[i] * dt * (1.0 - alpha[i] * dt / 2.0)
                    delta_x[i] = state.velocity[i] * dt + state.acceleration[i] * dt**2 / 2.0 * (1.0 - alpha[i] * dt / 3.0)
                else:
                    int_exp = (1.0 - exp_alpha_dt[i]) / alpha[i]
                    delta_v[i] = state.acceleration[i] * int_exp
                    delta_x[i] = state.velocity[i] * dt + state.acceleration[i] * (dt - int_exp)
            state.velocity += delta_v
            state.position += delta_x
            clean[t] = state.position
        noisy = clean + rng.normal(0.0, noise_std, size=(T, dim))
        trajectories.append((noisy, clean, state))
    return trajectories


def _generate_blueprint_segments(
    blueprint_segments: list[tuple[str, int, dict]],
    target_T: int,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> list[tuple[str, int, dict]]:
    generated_segments: list[tuple[str, int, dict]] = []
    current_length = 0
    while current_length < target_T:
        remaining = target_T - current_length
        if remaining < min_length:
            generated_segments.append(("CV", remaining, {"vel_change_std": 0.1, "measurement_noise_std": 0.3}))
            break
        blueprint_idx = rng.integers(0, len(blueprint_segments))
        model_type, _, params = blueprint_segments[blueprint_idx]
        segment_length = rng.integers(min_length, min(max_length, remaining) + 1)
        generated_segments.append((model_type, segment_length, params))
        current_length += segment_length
    return generated_segments


def generate_composite_trajectory(
    trajectory_segments: list[tuple[str, int, dict]],
    dt: float,
    dim: int = 2,
    initial_state: Optional[TrajectoryState] = None,
    seed: int | None = None,
    randomize_order: bool = False,
    randomize_blueprint: bool = False,
    min_segment_length: int = 20,
    max_segment_length: int = 60,
    target_T: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, TrajectoryState]:
    rng = np.random.default_rng(seed)
    if randomize_blueprint:
        if target_T is None:
            raise ValueError("target_T must be specified when randomize_blueprint=True")
        trajectory_segments = _generate_blueprint_segments(trajectory_segments, target_T, min_segment_length, max_segment_length, rng)
    elif randomize_order:
        trajectory_segments = trajectory_segments.copy()
        rng.shuffle(trajectory_segments)

    if initial_state is None:
        state = TrajectoryState(position=rng.uniform(0, 4, size=dim), velocity=rng.uniform(-0.4, 0.4, size=dim))
    else:
        state = TrajectoryState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
            acceleration=initial_state.acceleration.copy() if initial_state.acceleration is not None else None,
            omega=initial_state.omega,
            tau=initial_state.tau,
        )

    all_noisy: list[np.ndarray] = []
    all_clean: list[np.ndarray] = []
    for model_type, steps, params in trajectory_segments:
        if model_type == "CV":
            results = generate_cv_trajectory(
                T=steps,
                dt=dt,
                initial_state=state,
                vel_change_std=params.get("vel_change_std", 0.0),
                measurement_noise_std=params.get("measurement_noise_std", 0.0),
                number_of_trajectories=1,
                seed=rng.integers(0, 2**32),
            )
        elif model_type == "CA":
            state.acceleration = params.get("acceleration", state.acceleration)
            results = generate_ca_trajectory(
                T=steps,
                dt=dt,
                initial_state=state,
                measurement_noise_std=params.get("measurement_noise_std", 0.0),
                accel_noise_std=params.get("accel_noise_std", 0.0),
                number_of_trajectories=1,
                seed=rng.integers(0, 2**32),
            )
        elif model_type == "CT":
            results = generate_ct_trajectory_simple(
                T=steps,
                dt=dt,
                omega=params["omega"],
                dim=dim,
                initial_state=state,
                omega_noise_std=params.get("omega_noise_std", 0.0),
                measurement_noise_std=params.get("measurement_noise_std", 0.0),
                z_acceleration=params.get("z_acceleration", None),
                number_of_trajectories=1,
                seed=rng.integers(0, 2**32),
            )
        elif model_type == "SINGER":
            results = generate_singer_trajectory(
                T=steps,
                dt=dt,
                tau=params["tau"],
                dim=dim,
                initial_state=state,
                sigma_a=params.get("sigma_a", 0.5),
                noise_std=params.get("noise_std", 0.0),
                number_of_trajectories=1,
                seed=rng.integers(0, 2**32),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        noisy, clean, final_state = results[0]
        if len(all_noisy) == 0:
            all_noisy.append(noisy)
            all_clean.append(clean)
        else:
            all_noisy.append(noisy[1:])
            all_clean.append(clean[1:])
        state = final_state

    return np.vstack(all_noisy), np.vstack(all_clean), state
