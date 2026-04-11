# Trajectory Generator

Interactive Streamlit app for generating trajectory datasets from:

- Mathematical motion models such as `CV`, `CA`, `CT`, and `SINGER`
- PX4-driven flights, including `default`, `figure8`, and `s_turn` profiles

The app writes output as newline-delimited JSON (`.jsonl`) so each generated trajectory is stored as one JSON object per line.

## Overview

- `Equations` tab for synthetic trajectory generation from motion equations
- `PX4` tab for PX4-based trajectory execution and logging
- `Output Format` tab that explains the generated JSON structure
- Plot previews for recent trajectories directly in the UI

## Installation And Launch

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

## Application Sections

### Equations Tab

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
