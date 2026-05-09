# Trajectory Generator

A Python library for generating trajectories from kinematic motion models and PX4 autopilot simulations.

The generator produces pairs of clean and noisy state sequences that can be used as training data for tracking, prediction, and state-estimation models.  Output is written as newline-delimited JSON (`.jsonl`), one trajectory object per line.

Two generation pipelines are available:

| Pipeline | Description |
|----------|-------------|
| **Equations** | Pure-Python simulation using kinematic motion models (CV, CA, CT, SINGER). Fast, no hardware required. |
| **PX4** | Connects to a PX4 SITL simulator via MAVSDK, flies a configurable profile, and logs the vehicle state. Produces realistic UAV flight dynamics. |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Generator API

The entry point is [`generate.py`](generate.py) at the repository root.  Import and call either function directly from any Python script or notebook.

### `generate_equations`

Generates trajectories from sequences of kinematic motion segments.

```python
from generate import generate_equations

output_path = generate_equations(
    num_trajectories=100,
    dt=0.04,
    dim=3,
    blocks=[
        {"model_type": "CV", "steps": 60, "vel_change_std": 0.3},
        {"model_type": "CT", "steps": 60, "omega": 0.15, "omega_noise_std": 0.01},
        {"model_type": "SINGER", "steps": 60, "tau": 20.0, "sigma_a": 0.5},
    ],
    observation_noise=0.1,
    seed=42,
)
print("Saved to", output_path)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str \| None` | `./output` | Directory for the output `.jsonl` file. Created if absent. |
| `num_trajectories` | `int` | `20` | Number of trajectories to generate (1–10 000). |
| `dt` | `float` | `0.04` | Time-step in seconds. |
| `dim` | `int` | `3` | Spatial dimensions: `2` (x, y) or `3` (x, y, z). |
| `blocks` | `list[dict]` | — | **Required.** Ordered list of motion-segment descriptors (see [Blocks](#blocks-format) below). |
| `observation_noise` | `float \| dict` | `0.1` | Measurement noise std-dev, or a [NumericParam dict](#numericparam-format) for a sampled range. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |
| `randomize_from_current_blocks` | `bool` | `False` | Randomly resample segment lengths between `min_segment_length` and `max_segment_length` until `target_total_steps` is reached. |
| `min_segment_length` | `int` | `20` | Minimum steps per segment when randomising. |
| `max_segment_length` | `int` | `60` | Maximum steps per segment when randomising. |
| `target_total_steps` | `int \| None` | `None` | Required when `randomize_from_current_blocks=True`. |
| `initial_velocity` | `list[float] \| None` | `None` | Fixed initial velocity vector, e.g. `[5.0, 5.0, 1.0]`. Length must equal `dim`. |
| `initial_acceleration` | `list[float] \| None` | `None` | Fixed initial acceleration vector. Length must equal `dim`. |
| `initial_velocity_params` | `list[dict] \| None` | `None` | Per-axis [NumericParam dicts](#numericparam-format) for a *sampled* initial velocity. Mutually exclusive with `initial_velocity`. |

**Returns:** `str` — path to the written `.jsonl` file.

---

### `generate_px4`

Executes flight profiles in a PX4 SITL simulator and logs vehicle state.

> **Requires a running PX4 SITL instance.** 
```python
from generate import generate_px4

output_path = generate_px4(
    num_trajectories=5,
    duration_s=30.0,
    dt_s=0.04,
    connection_uri="udpin://0.0.0.0:14540",
    profile_name="default",
    observation_noise=0.1,
    motion={
        "num_waypoints_min": 3, "num_waypoints_max": 8,
        "waypoint_xy_min": -50.0, "waypoint_xy_max": 50.0,
        "waypoint_z_min": 10.0, "waypoint_z_max": 40.0,
        "max_speed": 5.0, "accel": 2.0, "waypoint_tolerance": 1.0,
    },
)
print("Saved to", output_path)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str \| None` | `./output` | Directory for the output `.jsonl` file. |
| `num_trajectories` | `int` | `1` | Number of flights to execute and log (1–10 000). |
| `duration_s` | `float` | `20.0` | Maximum flight duration per trajectory in seconds. |
| `dt_s` | `float` | `0.04` | Logging time-step in seconds. |
| `connection_uri` | `str` | `"udpin://0.0.0.0:14540"` | MAVSDK connection string for the PX4 SITL instance. |
| `profile_name` | `str` | `"default"` | Flight profile: `"default"`, `"figure8"`, `"s_turn"`, or `"equations_mission"`. |
| `observation_noise` | `float \| dict` | `0.1` | Measurement noise std-dev, or a [NumericParam dict](#numericparam-format). |
| `seed` | `int \| None` | `None` | Random seed. |
| `motion` | `dict \| None` | `None` | Required for `default`, `figure8`, and `s_turn` profiles. See [Motion config](#motion-config) below. |
| `equations_mission` | `dict \| None` | `None` | Required for `equations_mission` profile. See [Equations mission config](#equations-mission-config) below. |

**Returns:** `str` — path to the written `.jsonl` file.

#### Motion config

Used with `profile_name` in `("default", "figure8", "s_turn")`.  All values are a bare `float` (fixed) or a [NumericParam dict](#numericparam-format) (sampled per trajectory).

| Key | Description |
|-----|-------------|
| `num_waypoints_min` / `num_waypoints_max` | Waypoint count bounds. |
| `waypoint_xy_min` / `waypoint_xy_max` | XY position bounds in metres. |
| `waypoint_z_min` / `waypoint_z_max` | Altitude bounds in metres. |
| `max_speed` | Maximum horizontal speed in m/s. |
| `accel` | Acceleration limit in m/s². |
| `waypoint_tolerance` | Waypoint acceptance radius in metres. |

#### Equations mission config

Used with `profile_name="equations_mission"`.  Generates a trajectory from kinematic equations, uploads it as a MAVLink mission, and lets PX4 fly it autonomously.

Shares the same `blocks`, `dt`, and segment-randomisation fields as `generate_equations`, plus:

| Key | Default | Description |
|-----|---------|-------------|
| `mission_max_waypoints` | `900` | Cap on uploaded waypoints. |
| `mission_min_step_m` | `0.5` | Minimum distance between consecutive waypoints in metres (path thinning). |
| `waypoint_acceptance_radius_m` | `2.0` | PX4 waypoint acceptance radius in metres. |
| `min_altitude_m` | `10.0` | Minimum flight altitude above home in metres. |
| `randomize_flight_dynamics` | `False` | Randomise MPC parameters before each flight for physical diversity. |
| `mpc_acc_hor_max` | range 5–20 | Horizontal acceleration limit (m/s²). |
| `mpc_jerk_max` | range 5–35 | Jerk limit (m/s³). |
| `mpc_xy_p` | range 1–2 | XY position controller proportional gain. |
| `mpc_tiltmax_air` | range 45–80 | Maximum tilt angle in degrees. |
| `mpc_xy_vel_p_acc` | range 1.8–3.5 | XY velocity controller proportional gain. |

---

## Parameter Reference

### Blocks format

Each entry in the `blocks` list describes one motion segment:

```python
# Constant Velocity
{"model_type": "CV", "steps": 60, "vel_change_std": 0.3}

# Constant Acceleration
{"model_type": "CA", "steps": 60, "accel_noise_std": 0.2}

# Coordinated Turn
{"model_type": "CT", "steps": 60, "omega": 0.15, "omega_noise_std": 0.01}

# Singer (correlated acceleration)
{"model_type": "SINGER", "steps": 60, "tau": 20.0, "sigma_a": 0.5}
```

| Key | Models | Description |
|-----|--------|-------------|
| `model_type` | all | `"CV"`, `"CA"`, `"CT"`, or `"SINGER"` |
| `steps` | all | Number of time steps (2–5000) |
| `vel_change_std` | CV | Std-dev of random velocity perturbations |
| `accel_noise_std` | CA | Std-dev of acceleration noise |
| `omega` | CT | Turn rate in rad/s (required) |
| `omega_noise_std` | CT | Noise on turn rate |
| `tau` | SINGER | Manoeuvre time constant in seconds (required) |
| `sigma_a` | SINGER | Acceleration power spectral density |

### NumericParam format

Noise and motion fields accept either a bare `float` (fixed value) or a dict to sample uniformly per trajectory:

```python
# Fixed value
0.3

# Sampled uniformly from [0.1, 0.5] each trajectory
{"mode": "range", "min_value": 0.1, "max_value": 0.5}

# Explicit fixed form (equivalent to bare float)
{"mode": "fixed", "value": 0.3}
```

---

## Output Format

Each line in the output `.jsonl` file is a JSON object with the following fields:

| Field | Description |
|-------|-------------|
| `id` | Zero-based trajectory index. |
| `type` | `"equations"` or `"px4"`. |
| `trajectory_config` | Dict of all parameters used to generate this trajectory (dt, seed, segments, noise, etc.). |
| `clean` | `[[x, y(, z)], ...]` — noise-free state sequence. |
| `noisy` | `[[x, y(, z)], ...]` — observation sequence with measurement noise applied. |

Reading trajectories:

```python
import json

with open(output_path) as f:
    trajectories = [json.loads(line) for line in f]

first = trajectories[0]
print(first["type"])              # "equations"
print(len(first["clean"]))        # number of time steps
print(first["trajectory_config"]) # generation parameters
```

---

## Running from a Config File

For batch or automated runs you can describe all parameters in a JSON file and invoke the generator from the command line without writing any Python.

```bash
python run_from_config.py configs/example_equations.json
python run_from_config.py configs/example_px4.json
```

The config file must contain a `"type"` key set to `"equations"` or `"px4"`.  All other keys map directly to the parameters of `generate_equations` or `generate_px4` described above.

```json
{
  "type": "equations",
  "num_trajectories": 50,
  "dt": 0.04,
  "dim": 3,
  "observation_noise": 0.1,
  "blocks": [
    { "model_type": "CV", "steps": 60, "vel_change_std": 0.3 },
    { "model_type": "CT", "steps": 60, "omega": 0.15 }
  ]
}
```

Ready-to-use example configs are in [`configs/`](configs/).

The `run_config` function in [`run_from_config.py`](run_from_config.py) can also be imported and called programmatically:

```python
from run_from_config import run_config

output_path = run_config("configs/example_equations.json")
```

---

## Graphical Interface

A Streamlit web application provides an interactive front-end to both generators.  It is useful for exploring parameters visually and previewing trajectories in 3-D before running large generation jobs programmatically.

```bash
python run_ui.py
# or equivalently:
streamlit run app/ui/streamlit_app.py
```

Streamlit opens the app automatically in your browser at `http://localhost:8501`.

The UI mirrors every parameter documented above through form controls, and renders live Plotly previews of recently generated trajectories.  All generation still goes through the same underlying code as the Python API.
