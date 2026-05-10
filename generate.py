"""
Standalone trajectory generation API.

Two public functions — ``generate_equations`` and ``generate_px4`` — are the
primary entry points for this project.  Each function accepts all generation
parameters as explicit keyword arguments with documented defaults so that any
caller (script, LLM, CI pipeline) can invoke them without reading any other
module.

Both functions are **synchronous and blocking**: they run the full generation
loop in the calling thread and return the path to the written ``.jsonl`` file
when done.

Quick start::

    from generate import generate_equations

    path = generate_equations(
        num_trajectories=50,
        dt=0.04,
        dim=3,
        blocks=[
            {"model_type": "CV", "steps": 60, "vel_change_std": 0.5},
            {"model_type": "CT", "steps": 60, "omega": 0.2, "omega_noise_std": 0.02},
        ],
    )
    print("Wrote", path)
"""

from __future__ import annotations

from app.config.defaults import DEFAULTS, default_output_dir
from app.generators.equations_generator import EquationTrajectoryGenerator
from app.generators.px4_generator import PX4TrajectoryGenerator
from app.models.schemas import EquationsRequest, Px4Request
from app.services.job_service import job_service
from app.writers.file_writer import FileWriter


_NUMERIC_PARAM_BLOCK_FIELDS = frozenset(
    {"vel_change_std", "accel_noise_std", "omega", "omega_noise_std", "tau", "sigma_a"}
)


def _wrap_noise(value: float | dict) -> dict:
    """Normalise a bare float to a fixed NumericParam dict."""
    if isinstance(value, (int, float)):
        return {"mode": "fixed", "value": float(value)}
    return value


def _normalize_blocks(blocks: list[dict]) -> list[dict]:
    """Wrap any bare float values in NumericParam fields inside each block dict."""
    normalized = []
    for block in blocks:
        block = dict(block)
        for field in _NUMERIC_PARAM_BLOCK_FIELDS:
            if field in block and isinstance(block[field], (int, float)):
                block[field] = _wrap_noise(block[field])
        normalized.append(block)
    return normalized


def generate_equations(
    *,
    output_dir: str | None = None,
    num_trajectories: int = DEFAULTS.equations_num_trajectories,
    dt: float = DEFAULTS.equations_dt,
    dim: int = DEFAULTS.equations_dim,
    blocks: list[dict],
    observation_noise: float | dict = DEFAULTS.equations_observation_noise,
    seed: int | None = None,
    randomize_from_current_blocks: bool = False,
    min_segment_length: int = 20,
    max_segment_length: int = DEFAULTS.equations_steps_per_segment,
    target_total_steps: int | None = None,
    initial_velocity: list[float] | None = None,
    initial_acceleration: list[float] | None = None,
    initial_velocity_params: list[dict] | None = None,
    _job_id: str | None = None,
) -> str:
    """Generate physically plausible trajectories from kinematic motion equations.

    Trajectories are composed of one or more *segments*, each driven by a
    different motion model.  Results are written as newline-delimited JSON
    (``.jsonl``), one trajectory object per line.

    Parameters
    ----------
    output_dir:
        Directory where the ``.jsonl`` file is written.  Created automatically
        if it does not exist.  Defaults to ``./output`` relative to the current
        working directory.
    num_trajectories:
        Number of independent trajectories to generate.  Must be between 1 and
        10 000.
    dt:
        Simulation time-step in seconds.  Smaller values produce finer-grained
        trajectories.  Must be > 0.
    dim:
        Spatial dimensionality of each trajectory.  Must be 2 (x, y) or 3
        (x, y, z).
    blocks:
        **Required.**  Ordered list of motion-segment descriptors.  Each entry
        is a dict with the following keys:

        ``model_type`` *(str, required)*
            One of ``"CV"`` (constant velocity), ``"CA"`` (constant
            acceleration), ``"CT"`` (coordinated turn), or ``"SINGER"``
            (Singer acceleration model).

        ``steps`` *(int, required)*
            Number of time steps to simulate for this segment.  Range: 2–5000.

        Model-specific noise fields (each can be a bare ``float`` for a fixed
        value, or a dict ``{"mode": "range", "min_value": lo, "max_value": hi}``
        to sample uniformly per trajectory):

        * **CV** — ``vel_change_std``: std-dev of random velocity perturbations.
        * **CA** — ``accel_noise_std``: std-dev of acceleration noise.
        * **CT** — ``omega`` *(required)*: turn rate in rad/s; ``omega_noise_std``:
          noise on turn rate.
        * **SINGER** — ``tau`` *(required)*: manoeuvre time constant (s);
          ``sigma_a``: acceleration power spectral density.

        Example::

            blocks=[
                {"model_type": "CV", "steps": 80, "vel_change_std": 0.3},
                {"model_type": "CT", "steps": 60, "omega": 0.15, "omega_noise_std": 0.01},
                {"model_type": "SINGER", "steps": 50, "tau": 20.0, "sigma_a": 0.5},
            ]

    observation_noise:
        Measurement noise added to the clean trajectory to produce the noisy
        observation.  Pass a ``float`` for a fixed standard deviation, or a
        dict for a sampled range::

            observation_noise=0.1                                   # fixed
            observation_noise={"mode": "range", "min_value": 0.05, "max_value": 0.3}

    seed:
        Integer random seed for full reproducibility.  ``None`` means
        non-deterministic.
    randomize_from_current_blocks:
        When ``True`` the generator ignores the explicit ``steps`` values in
        ``blocks`` and instead randomly draws segment lengths between
        ``min_segment_length`` and ``max_segment_length`` until
        ``target_total_steps`` is reached.  Requires ``target_total_steps``.
    min_segment_length:
        Minimum number of steps per segment when ``randomize_from_current_blocks``
        is ``True``.
    max_segment_length:
        Maximum number of steps per segment when ``randomize_from_current_blocks``
        is ``True``.
    target_total_steps:
        Total trajectory length in steps when ``randomize_from_current_blocks``
        is ``True``.
    initial_velocity:
        Fixed initial velocity vector, e.g. ``[5.0, 5.0, 1.0]`` for 3-D.
        Length must equal ``dim``.  Mutually exclusive with
        ``initial_velocity_params``.
    initial_acceleration:
        Fixed initial acceleration vector.  Length must equal ``dim``.
    initial_velocity_params:
        Per-axis ``NumericParam`` dicts for a *sampled* initial velocity.
        Length must equal ``dim``.  Mutually exclusive with
        ``initial_velocity``.

    Returns
    -------
    str
        Absolute path to the written ``.jsonl`` file.

    Raises
    ------
    pydantic.ValidationError
        If any parameter value fails schema validation.
    """
    if output_dir is None:
        output_dir = default_output_dir()

    params_dict: dict = {
        "output_dir": output_dir,
        "num_trajectories": num_trajectories,
        "dt": dt,
        "dim": dim,
        "blocks": _normalize_blocks(blocks),
        "observation_noise": _wrap_noise(observation_noise),
        "seed": seed,
        "randomize_from_current_blocks": randomize_from_current_blocks,
        "min_segment_length": min_segment_length,
        "max_segment_length": max_segment_length,
        "target_total_steps": target_total_steps,
        "initial_velocity": initial_velocity,
        "initial_acceleration": initial_acceleration,
        "initial_velocity_params": initial_velocity_params,
    }

    request = EquationsRequest.model_validate(params_dict)
    record = job_service.get_job(_job_id) if _job_id is not None else job_service.create_job("equations")
    writer = FileWriter(
        output_dir=request.output_dir,
        trajectory_type="equations",
        important_hparams={"count": request.num_trajectories, "dim": request.dim, "dt": request.dt},
    )
    record.output_path = writer.output_path
    generator = EquationTrajectoryGenerator(job_id=record.job_id, params=request, writer=writer)
    if _job_id is not None:
        generator.run()
    else:
        generator.generate_trajectories()
    return writer.output_path


def generate_px4(
    *,
    output_dir: str | None = None,
    num_trajectories: int = DEFAULTS.px4_num_trajectories,
    duration_s: float = DEFAULTS.px4_duration_s,
    dt_s: float = DEFAULTS.px4_dt_s,
    connection_uri: str = DEFAULTS.px4_connection_uri,
    profile_name: str = "default",
    observation_noise: float | dict = DEFAULTS.px4_observation_noise,
    seed: int | None = None,
    motion: dict | None = None,
    equations_mission: dict | None = None,
    _job_id: str | None = None,
) -> str:
    """Generate trajectories by flying a PX4 autopilot simulation via MAVSDK.

    Connects to a running PX4 SITL simulator, executes the requested flight
    profile for each trajectory, logs the vehicle state at ``dt_s`` intervals,
    and writes results as newline-delimited JSON (``.jsonl``).

    **Prerequisites:** a PX4 SITL instance must be reachable at
    ``connection_uri`` before this function is called.

    Parameters
    ----------
    output_dir:
        Directory where the ``.jsonl`` file is written.  Created automatically
        if it does not exist.  Defaults to ``./output``.
    num_trajectories:
        Number of independent flights to execute and log.  Must be between 1
        and 10 000.
    duration_s:
        Maximum allowed flight duration per trajectory in seconds.  Must be
        > 0.
    dt_s:
        Logging time-step in seconds.  Must be > 0.
    connection_uri:
        MAVSDK connection string for the PX4 SITL instance.  Common values:

        * ``"udpin://0.0.0.0:14540"`` — UDP, default PX4 SITL port (default)
        * ``"serial:///dev/ttyUSB0:57600"`` — serial port

    profile_name:
        Flight profile to execute.  One of:

        ``"default"``
            Flies random waypoints using the ``motion`` config.
        ``"figure8"``
            Scripted figure-eight pattern.  Requires ``motion``.
        ``"s_turn"``
            Scripted S-turn pattern.  Requires ``motion``.
        ``"equations_mission"``
            Generates a trajectory from kinematic equations (same as
            ``generate_equations``), uploads it as a MAVLink mission, and lets
            PX4 fly it autonomously.  Requires ``equations_mission``.

    observation_noise:
        Measurement noise added to clean logged positions.  Pass a ``float``
        for a fixed standard deviation, or a range dict::

            observation_noise=0.1
            observation_noise={"mode": "range", "min_value": 0.1, "max_value": 0.3}

    seed:
        Integer random seed.  ``None`` means non-deterministic.
    motion:
        Required for ``profile_name`` in ``("default", "figure8", "s_turn")``.
        Dict matching ``Px4MotionConfig``.  All field values are either a bare
        ``float`` (treated as fixed) or a ``NumericParam`` dict
        ``{"mode": "range", "min_value": lo, "max_value": hi}``.

        Keys:

        * ``num_waypoints_min`` / ``num_waypoints_max`` — waypoint count bounds.
        * ``waypoint_xy_min`` / ``waypoint_xy_max`` — XY extent in metres.
        * ``waypoint_z_min`` / ``waypoint_z_max`` — altitude bounds in metres.
        * ``max_speed`` — maximum horizontal speed in m/s.
        * ``accel`` — acceleration limit in m/s².
        * ``waypoint_tolerance`` — waypoint acceptance radius in metres.

        Example::

            motion={
                "num_waypoints_min": 3, "num_waypoints_max": 8,
                "waypoint_xy_min": -50.0, "waypoint_xy_max": 50.0,
                "waypoint_z_min": 10.0, "waypoint_z_max": 40.0,
                "max_speed": 5.0, "accel": 2.0, "waypoint_tolerance": 1.0,
            }

    equations_mission:
        Required when ``profile_name="equations_mission"``.  Dict matching
        ``EquationsMissionParams``.  Shares the same ``blocks``, ``dt``,
        and segment-randomisation fields as ``generate_equations`` (see that
        function for ``blocks`` format), plus PX4-specific fields:

        * ``mission_max_waypoints`` *(int, default 900)* — cap on uploaded
          waypoints.
        * ``mission_min_step_m`` *(float, default 0.5)* — minimum distance
          between consecutive waypoints in metres (path thinning).
        * ``waypoint_acceptance_radius_m`` *(float, default 2.0)* — PX4
          waypoint acceptance radius in metres.
        * ``randomize_flight_dynamics`` *(bool, default False)* — randomise MPC
          parameters before each flight for physical diversity.
        * MPC tuning params (each a ``NumericParam`` dict or float):
          ``mpc_acc_hor_max``, ``mpc_jerk_max``, ``mpc_xy_p``,
          ``mpc_tiltmax_air``, ``mpc_xy_vel_p_acc``.
        * ``min_altitude_m`` *(float, default 10.0)* — minimum flight altitude
          above home in metres; the trajectory is shifted upward if any point
          falls below this value.

        Example::

            equations_mission={
                "dt": 0.04,
                "blocks": [
                    {"model_type": "CV", "steps": 80, "vel_change_std": 0.3},
                    {"model_type": "CT", "steps": 60, "omega": 0.15},
                ],
                "min_altitude_m": 15.0,
                "randomize_flight_dynamics": True,
            }

    Returns
    -------
    str
        Absolute path to the written ``.jsonl`` file.

    Raises
    ------
    pydantic.ValidationError
        If any parameter value fails schema validation.
    """
    if output_dir is None:
        output_dir = default_output_dir()

    def _wrap_motion_values(motion_dict: dict) -> dict:
        """Wrap bare float values inside a motion config to NumericParam dicts."""
        return {k: _wrap_noise(v) if isinstance(v, (int, float)) else v for k, v in motion_dict.items()}

    params_dict: dict = {
        "output_dir": output_dir,
        "num_trajectories": num_trajectories,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "connection_uri": connection_uri,
        "profile_name": profile_name,
        "observation_noise": _wrap_noise(observation_noise),
        "seed": seed,
        "motion": _wrap_motion_values(motion) if motion is not None else None,
        "equations_mission": (
            {
                **equations_mission,
                "blocks": _normalize_blocks(equations_mission["blocks"]),
            }
            if equations_mission is not None and "blocks" in equations_mission
            else equations_mission
        ),
    }

    request = Px4Request.model_validate(params_dict)
    record = job_service.get_job(_job_id) if _job_id is not None else job_service.create_job("px4")
    writer = FileWriter(
        output_dir=request.output_dir,
        trajectory_type="px4",
        important_hparams={"count": request.num_trajectories, "dt": request.dt_s, "duration": request.duration_s},
    )
    record.output_path = writer.output_path
    generator = PX4TrajectoryGenerator(job_id=record.job_id, params=request, writer=writer)
    if _job_id is not None:
        generator.run()
    else:
        generator.generate_trajectories()
    return writer.output_path
