# Trajectory Generator

Interactive Streamlit app for generating trajectory datasets from:

- Mathematical motion models such as `CV`, `CA`, `CT`, and `SINGER`
- PX4-driven flights, including `default`, `figure8`, and `s_turn` profiles

The app writes output as newline-delimited JSON (`.jsonl`) so each generated trajectory is stored as one JSON object per line.

## What The App Includes

- `Equations` tab for synthetic trajectory generation from motion equations
- `PX4` tab for PX4-based trajectory execution and logging
- `Output Format` tab that explains the generated JSON structure
- Plot previews for recent trajectories directly in the UI

## Repository Layout

Key files:

- `run_ui.py`: convenience launcher for the Streamlit app
- `app/ui/streamlit_app.py`: main UI
- `app/generators/equations_generator.py`: equations trajectory generation
- `app/generators/px4_generator.py`: PX4 trajectory generation
- `app/local_lib/route_generation.py`: equations-side motion generation utilities
- `app/local_lib/px4_generation.py`: PX4-side trajectory generation and logging
- `app/config/defaults.py`: UI defaults
- `requirements.txt`: Python dependencies

## Prerequisites

Recommended:

- Python `3.12`
- `pip`
- A virtual environment tool such as `venv`

For PX4 generation:

- A running PX4 SITL or compatible MAVSDK-accessible vehicle
- Default connection URI is `udpin://0.0.0.0:14540`

If you only want to use the equations generator, you do not need PX4 running.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running The App

You can start the UI either through the helper script:

```bash
python run_ui.py
```

Or directly with Streamlit:

```bash
streamlit run app/ui/streamlit_app.py
```

Streamlit usually opens the app automatically in your browser. If it does not, look for a local URL similar to:

```text
http://localhost:8501
```

## Basic Usage

### Equations Tab

Use this tab to:

- Choose output directory
- Set global parameters such as trajectory count, `dt`, and dimension
- Configure observation noise
- Build multi-segment motion blueprints
- Generate and preview synthetic trajectories

The equations generator writes trajectories with normalized fields such as:

- `id`
- `type`
- `trajectory_config`
- `clean`
- `noisy`

### PX4 Tab

Use this tab to:

- Choose output directory
- Configure PX4 connection settings
- Select a profile: `default`, `figure8`, or `s_turn`
- Configure observation noise
- Generate and preview PX4 trajectories

Notes:

- PX4 trajectories are always 3D
- `default` uses the motion hyperparameter controls
- `figure8` and `s_turn` use scripted shapes and disable unused motion controls in the UI
- Output is normalized to match the equations format as closely as possible

## Output Files

By default, output is written under:

```text
./output
```

Each generated file is a `.jsonl` file. Every line is one trajectory record.

Typical normalized top-level fields:

- `id`
- `type`
- `trajectory_config`
- `clean`
- `noisy`

PX4 records also include:

- `setpoints`

The `trajectory_config` object contains common fields such as:

- `dt`
- `observation_noise_std`

PX4-specific values that differ from equations are stored under:

- `trajectory_config.metadata`

## Example Commands

Install and run from scratch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_ui.py
```

Run without the helper script:

```bash
streamlit run app/ui/streamlit_app.py
```

## Troubleshooting

### `streamlit: command not found`

Your environment is probably not activated. Activate the virtual environment first, then retry.

### PX4 generation does not connect

Check:

- PX4 SITL is running
- The MAVSDK endpoint matches the UI value
- The default URI `udpin://0.0.0.0:14540` is correct for your setup

### Browser does not open automatically

Open the local Streamlit URL manually, usually `http://localhost:8501`.

### Output files are not where expected

Check the `Output directory` field in the UI. If unchanged, files are written into `output/` under the current working directory used to launch the app.

## Development Notes

- The UI is implemented with Streamlit and Plotly
- Generated files are written by `app/writers/file_writer.py`
- Validation is handled with Pydantic models in `app/models/schemas.py`
- The app runs in a local Streamlit-only mode and does not require a separate backend server

## Testing

This repository includes `pytest` in `requirements.txt`. If you add tests later, run them with:

```bash
pytest
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_ui.py
```
