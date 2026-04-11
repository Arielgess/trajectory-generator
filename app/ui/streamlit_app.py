from __future__ import annotations

from html import escape
import logging
import time
from threading import Thread
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from app.config.defaults import DEFAULTS, default_output_dir
from app.config.logging_config import setup_logging
from app.generators.equations_generator import EquationTrajectoryGenerator
from app.generators.px4_generator import PX4TrajectoryGenerator
from app.models.schemas import EquationsRequest, JobStatus, Px4Request
from app.services.job_service import job_service
from app.writers.file_writer import FileWriter

setup_logging()
LOGGER = logging.getLogger(__name__)


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .tg-section-title {
            margin-bottom: 0.15rem;
            font-weight: 600;
        }
        .tg-section-caption {
            margin-bottom: 0.75rem;
            color: rgba(214, 226, 255, 0.92);
            font-size: 0.92rem;
        }
        .tg-help-label {
            margin-bottom: 0.1rem;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .tg-help-text {
            margin-bottom: 0.5rem;
            color: rgba(214, 226, 255, 0.92);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .tg-inline-note {
            color: rgba(214, 226, 255, 0.92);
            font-size: 0.88rem;
            margin-top: -0.2rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_intro(title: str, caption: str) -> None:
    st.markdown(f"<div class='tg-section-title'>{escape(title)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tg-section-caption'>{escape(caption)}</div>", unsafe_allow_html=True)


def _help_label(label: str, help_text: str) -> None:
    st.markdown(f"<div class='tg-help-label'>{escape(label)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tg-help-text'>{escape(help_text)}</div>", unsafe_allow_html=True)


def _numeric_param_ui(
    label: str,
    defaults: tuple[float, float, float],
    key_prefix: str,
    help_text: str,
    disabled: bool = False,
) -> dict[str, Any]:
    _help_label(label, help_text)
    mode = st.radio(
        f"{label} mode",
        ["fixed", "range"],
        horizontal=True,
        key=f"{key_prefix}_mode",
        label_visibility="collapsed",
        disabled=disabled,
    )
    if mode == "fixed":
        value = st.number_input("Value", value=defaults[0], key=f"{key_prefix}_value", disabled=disabled)
        return {"mode": "fixed", "value": value, "min_value": defaults[1], "max_value": defaults[2]}
    min_col, max_col = st.columns(2)
    with min_col:
        min_value = st.number_input("Min", value=defaults[1], key=f"{key_prefix}_min", disabled=disabled)
    with max_col:
        max_value = st.number_input("Max", value=defaults[2], key=f"{key_prefix}_max", disabled=disabled)
    return {"mode": "range", "value": defaults[0], "min_value": min_value, "max_value": max_value}


def _param_defaults(param: dict[str, Any] | None, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not param:
        return fallback
    return (
        float(param.get("value", fallback[0])),
        float(param.get("min_value", fallback[1])),
        float(param.get("max_value", fallback[2])),
    )


def _start_equations_job(payload: dict[str, Any]) -> str | None:
    LOGGER.info("Equations generate clicked")
    try:
        params = EquationsRequest.model_validate(payload)
    except Exception as exc:
        LOGGER.exception("Equations validation error: %s", exc)
        st.error(f"Invalid equations request: {exc}")
        return None

    record = job_service.create_job("equations")
    writer = FileWriter(
        output_dir=params.output_dir,
        trajectory_type="equations",
        important_hparams={"dt": params.dt, "dim": params.dim, "count": params.num_trajectories},
    )
    record.output_path = writer.output_path
    generator = EquationTrajectoryGenerator(job_id=record.job_id, params=params, writer=writer)
    Thread(target=generator.run, daemon=True).start()
    LOGGER.info("Equations job started job_id=%s output=%s", record.job_id, record.output_path)
    return record.job_id


def _start_px4_job(payload: dict[str, Any]) -> str | None:
    LOGGER.info("PX4 generate clicked")
    try:
        params = Px4Request.model_validate(payload)
    except Exception as exc:
        LOGGER.exception("PX4 validation error: %s", exc)
        st.error(f"Invalid PX4 request: {exc}")
        return None

    record = job_service.create_job("px4")
    writer = FileWriter(
        output_dir=params.output_dir,
        trajectory_type="px4",
        important_hparams={"dt": params.dt_s, "duration": params.duration_s, "count": params.num_trajectories},
    )
    record.output_path = writer.output_path
    generator = PX4TrajectoryGenerator(job_id=record.job_id, params=params, writer=writer)
    Thread(target=generator.run, daemon=True).start()
    LOGGER.info("PX4 job started job_id=%s output=%s", record.job_id, record.output_path)
    return record.job_id


def _format_start_end(start: list[float], end: list[float]) -> str:
    return f"Start: [{', '.join(f'{v:.3f}' for v in start)}] | End: [{', '.join(f'{v:.3f}' for v in end)}]"


def _plot_preview(preview: dict[str, Any], title: str) -> None:
    if "clean" in preview:
        clean_pts = preview["clean"]
        noisy_pts = preview.get("noisy", [])
        if clean_pts and isinstance(clean_pts[0], dict):
            setpoints_pts = preview.get("setpoints", [])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=[p["x"] for p in clean_pts],
                    y=[p["y"] for p in clean_pts],
                    z=[p.get("z", 0.0) for p in clean_pts],
                    mode="lines",
                    name="clean",
                )
            )
            if noisy_pts:
                fig.add_trace(
                    go.Scatter3d(
                        x=[p["x"] for p in noisy_pts],
                        y=[p["y"] for p in noisy_pts],
                        z=[p.get("z", 0.0) for p in noisy_pts],
                        mode="markers",
                        marker={"size": 4},
                        name="noisy",
                    )
                )
            if setpoints_pts:
                fig.add_trace(
                    go.Scatter3d(
                        x=[p["x"] for p in setpoints_pts],
                        y=[p["y"] for p in setpoints_pts],
                        z=[p.get("z", 0.0) for p in setpoints_pts],
                        mode="lines",
                        name="setpoints",
                    )
                )
            fig.update_layout(title=title, width=420, height=420)
            st.plotly_chart(fig, use_container_width=False)
            start = [clean_pts[0]["x"], clean_pts[0]["y"], clean_pts[0].get("z", 0.0)]
            end = [clean_pts[-1]["x"], clean_pts[-1]["y"], clean_pts[-1].get("z", 0.0)]
            st.caption(_format_start_end(start, end))
            return

        dim = len(clean_pts[0])
        if dim == 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[p[0] for p in clean_pts], y=[p[1] for p in clean_pts], z=[p[2] for p in clean_pts], mode="lines", name="clean"))
            if noisy_pts:
                fig.add_trace(
                    go.Scatter3d(
                        x=[p[0] for p in noisy_pts],
                        y=[p[1] for p in noisy_pts],
                        z=[p[2] for p in noisy_pts],
                        mode="markers",
                        marker={"size": 2},
                        name="noisy",
                    )
                )
            fig.update_layout(title=title, width=420, height=420)
            st.plotly_chart(fig, use_container_width=False)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[p[0] for p in clean_pts], y=[p[1] for p in clean_pts], mode="lines", name="clean"))
            if noisy_pts:
                fig.add_trace(
                    go.Scatter(x=[p[0] for p in noisy_pts], y=[p[1] for p in noisy_pts], mode="markers", marker={"size": 3}, name="noisy")
                )
            fig.update_layout(title=title, width=420, height=420, yaxis={"scaleanchor": "x", "scaleratio": 1})
            st.plotly_chart(fig, use_container_width=False)
        st.caption(_format_start_end(clean_pts[0], clean_pts[-1]))
        return

    if "executed" in preview:
        clean_pts = preview["executed"]
        noisy_pts = preview.get("observed", [])
        setpoints_pts = preview.get("setpoints", [])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in clean_pts],
                y=[p["y"] for p in clean_pts],
                z=[p.get("z", 0.0) for p in clean_pts],
                mode="lines",
                name="clean",
            )
        )
        if noisy_pts:
            fig.add_trace(
                go.Scatter3d(
                    x=[p["x"] for p in noisy_pts],
                    y=[p["y"] for p in noisy_pts],
                    z=[p.get("z", 0.0) for p in noisy_pts],
                    mode="markers",
                    marker={"size": 4},
                    name="noisy",
                )
            )
        if setpoints_pts:
            fig.add_trace(
                go.Scatter3d(
                    x=[p["x"] for p in setpoints_pts],
                    y=[p["y"] for p in setpoints_pts],
                    z=[p.get("z", 0.0) for p in setpoints_pts],
                    mode="lines",
                    name="setpoints",
                )
            )
        fig.update_layout(title=title, width=420, height=420)
        st.plotly_chart(fig, use_container_width=False)
        start = [clean_pts[0]["x"], clean_pts[0]["y"], clean_pts[0].get("z", 0.0)]
        end = [clean_pts[-1]["x"], clean_pts[-1]["y"], clean_pts[-1].get("z", 0.0)]
        st.caption(_format_start_end(start, end))


def _render_job_monitor(job_id: str, keep_last: int = 10) -> None:
    try:
        record = job_service.get_job(job_id)
    except KeyError:
        st.error("Job not found in local job registry.")
        return
    status = {
        "job_id": record.job_id,
        "status": record.status.value,
        "progress": record.progress,
        "generated_count": record.generated_count,
        "failed_count": record.failed_count,
        "message": record.message,
        "output_path": record.output_path,
    }
    st.progress(float(record.progress))
    st.write(status)
    events = list(record.events)
    error_events = [e for e in events if e.get("event_type") == "error"]
    if error_events:
        st.error(f"Latest error: {error_events[-1]['payload'].get('message', 'Unknown error')}")
    previews = [e["payload"] for e in events if e["event_type"] == "preview"][-keep_last:]
    preview_cols = st.columns(2)
    for i, preview in enumerate(previews):
        with preview_cols[i % 2]:
            _plot_preview(preview, f"Preview {i + 1}")
    if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        st.caption("Auto-refreshing active job every 2 seconds")
        time.sleep(2)
        st.rerun()


def _default_block(model_type: str = "CV") -> dict[str, Any]:
    block: dict[str, Any] = {"model_type": model_type, "steps": DEFAULTS.equations_steps_per_segment}
    if model_type == "CV":
        block["vel_change_std"] = {"mode": "fixed", "value": 0.1, "min_value": 0.0, "max_value": 0.5}
    elif model_type == "CA":
        block["accel_noise_std"] = {"mode": "fixed", "value": 0.1, "min_value": 0.0, "max_value": 0.5}
    elif model_type == "CT":
        block["omega"] = {"mode": "fixed", "value": 1.2, "min_value": -3.0, "max_value": 3.0}
        block["omega_noise_std"] = {"mode": "fixed", "value": 0.0, "min_value": 0.0, "max_value": 0.5}
    elif model_type == "SINGER":
        block["tau"] = {"mode": "fixed", "value": 1.0, "min_value": 0.1, "max_value": 5.0}
        block["sigma_a"] = {"mode": "fixed", "value": 0.5, "min_value": 0.1, "max_value": 2.0}
    return block


def _equation_templates() -> dict[str, list[dict[str, Any]]]:
    return {
        "Custom": [_default_block("CV")],
        "CV-CA-CT Mix": [_default_block("CV"), _default_block("CA"), _default_block("CT")],
    }


def _nominal_velocity_from_params(initial_velocity_params: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for param in initial_velocity_params:
        if param["mode"] == "range":
            values.append(float((param["min_value"] + param["max_value"]) / 2.0))
        else:
            values.append(float(param["value"]))
    return values


def _build_figure8_template(initial_velocity_params: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    v = _nominal_velocity_from_params(initial_velocity_params)
    speed_xy = (v[0] ** 2 + v[1] ** 2) ** 0.5 if len(v) >= 2 else abs(v[0])
    speed_xy = max(speed_xy, 0.5)
    target_radius = max(3.0, speed_xy)
    omega = speed_xy / target_radius
    # Full turn closes one circle: T = 2*pi/omega, steps = T/dt.
    steps_per_segment = max(20, int(round((2.0 * 3.141592653589793) / (omega * max(dt, 1e-4)))))

    first = _default_block("CT")
    second = _default_block("CT")
    first["omega"] = {"mode": "fixed", "value": omega, "min_value": omega, "max_value": omega}
    second["omega"] = {"mode": "fixed", "value": -omega, "min_value": -omega, "max_value": -omega}
    first["steps"] = steps_per_segment
    second["steps"] = steps_per_segment
    first["omega_noise_std"] = {"mode": "fixed", "value": 0.0, "min_value": 0.0, "max_value": 0.0}
    second["omega_noise_std"] = {"mode": "fixed", "value": 0.0, "min_value": 0.0, "max_value": 0.0}
    return [first, second]


def _build_s_shape_template(initial_velocity_params: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    v = _nominal_velocity_from_params(initial_velocity_params)
    speed_xy = (v[0] ** 2 + v[1] ** 2) ** 0.5 if len(v) >= 2 else abs(v[0])
    speed_xy = max(speed_xy, 0.5)
    target_radius = max(3.0, speed_xy)
    omega = speed_xy / target_radius
    # Half turn then opposite half turn gives S-shape.
    steps_per_segment = max(20, int(round((3.141592653589793) / (omega * max(dt, 1e-4)))))

    ca = _default_block("CA")
    ca["steps"] = 100
    first = _default_block("CT")
    second = _default_block("CT")
    first["omega"] = {"mode": "fixed", "value": omega, "min_value": omega, "max_value": omega}
    second["omega"] = {"mode": "fixed", "value": -omega, "min_value": -omega, "max_value": -omega}
    first["steps"] = steps_per_segment
    second["steps"] = steps_per_segment
    first["omega_noise_std"] = {"mode": "fixed", "value": 0.0, "min_value": 0.0, "max_value": 0.0}
    second["omega_noise_std"] = {"mode": "fixed", "value": 0.0, "min_value": 0.0, "max_value": 0.0}
    return [ca, first, second]


def equations_tab() -> None:
    _inject_ui_styles()
    st.subheader("Mathematical Equations")
    st.caption("Configure global options, build ordered segment blocks, then submit a generation job.")

    with st.container(border=True):
        _section_intro("Run setup", "Choose where data is written and set the global trajectory settings.")
        top_cols = st.columns([2.4, 1, 1, 1])
        with top_cols[0]:
            output_dir = st.text_input("Output directory", value=default_output_dir(), key="eq_output_dir")
        with top_cols[1]:
            num_trajectories = st.number_input("Trajectories", min_value=1, value=DEFAULTS.equations_num_trajectories)
        with top_cols[2]:
            dt = st.number_input("dt", min_value=0.0001, value=DEFAULTS.equations_dt, help="Sampling interval in seconds.")
        with top_cols[3]:
            dim = st.selectbox("Dimension", [2, 3], index=1, help="Number of spatial dimensions to generate.")

    if dim == 2:
        v_defaults = DEFAULTS.equations_initial_velocity_2d
        a_defaults = DEFAULTS.equations_initial_acceleration_2d
    else:
        v_defaults = DEFAULTS.equations_initial_velocity_3d
        a_defaults = DEFAULTS.equations_initial_acceleration_3d
    axis_names = ["vx", "vy", "vz"][:dim]
    initial_velocity_params = []
    state_cols = st.columns([1, 1.4])
    with state_cols[0]:
        with st.container(border=True):
            _section_intro("Measurement and sampling", "Configure observation noise and optional block randomization.")
            observation_noise = _numeric_param_ui(
                "Observation noise",
                (
                    DEFAULTS.equations_observation_noise,
                    DEFAULTS.equations_observation_noise_min,
                    DEFAULTS.equations_observation_noise_max,
                ),
                key_prefix="eq_obs",
                help_text="Common measurement noise applied to all segments in a generated trajectory.",
            )

            randomize = st.toggle(
                "Randomize from current blocks",
                value=False,
                help="Re-sample segment durations from your current blueprint instead of keeping exact block lengths.",
            )
            target_total_steps = (
                st.number_input(
                    "Target total steps",
                    min_value=10,
                    value=400,
                    help="Approximate total steps used when randomization is enabled.",
                )
                if randomize
                else None
            )
            st.markdown(
                "<div class='tg-inline-note'>Templates adapt to the current initial velocity and sampling rate.</div>",
                unsafe_allow_html=True,
            )

    with state_cols[1]:
        with st.container(border=True):
            _section_intro("Initial velocity", "Set the starting motion state used to seed each generated trajectory.")
            velocity_cols = st.columns(dim)
            for i, axis_name in enumerate(axis_names):
                with velocity_cols[i]:
                    initial_velocity_params.append(
                        _numeric_param_ui(
                            axis_name,
                            (float(v_defaults[i]), float(v_defaults[i]) * 0.5, float(v_defaults[i]) * 1.5),
                            key_prefix=f"eq_init_v_{axis_name}",
                            help_text=f"Initial {axis_name} (fixed) or sample range per trajectory.",
                        )
                    )

    if "eq_blocks" not in st.session_state:
        st.session_state["eq_blocks"] = [_default_block("CV")]
    if "eq_blocks_version" not in st.session_state:
        st.session_state["eq_blocks_version"] = 0

    with st.container(border=True):
        _section_intro("Trajectory creation lab", "Build an ordered list of motion segments or start from a template.")
        templates = _equation_templates()
        template_cols = st.columns([2.2, 1, 1, 2.2])
        with template_cols[0]:
            selected_template = st.selectbox(
                "Predefined structure",
                ["Custom", "Figure-8", "S-turn"] + list(templates.keys())[1:],
                index=0,
            )
        if template_cols[1].button("Apply template", use_container_width=True):
            if selected_template == "Figure-8":
                st.session_state["eq_blocks"] = _build_figure8_template(initial_velocity_params, dt)
            elif selected_template == "S-turn":
                st.session_state["eq_blocks"] = _build_s_shape_template(initial_velocity_params, dt)
            else:
                st.session_state["eq_blocks"] = [dict(item) for item in templates[selected_template]]
            st.session_state["eq_blocks_version"] += 1
            st.rerun()
        if template_cols[2].button("Add block", use_container_width=True):
            st.session_state["eq_blocks"].append(_default_block("CV"))
        with template_cols[3]:
            st.markdown(
                "<div class='tg-inline-note'>Use the segment list below to mix CV, CA, CT, and SINGER motion patterns.</div>",
                unsafe_allow_html=True,
            )

        for idx, block in enumerate(st.session_state["eq_blocks"]):
            version = st.session_state["eq_blocks_version"]
            with st.expander(f"Segment {idx + 1}: {block['model_type']}", expanded=True):
                cols = st.columns([4, 2, 1, 1, 1])
                with cols[0]:
                    old_type = block["model_type"]
                    new_type = st.selectbox(
                        f"Type {idx + 1}",
                        ["CV", "CA", "CT", "SINGER"],
                        key=f"type_{version}_{idx}",
                        index=["CV", "CA", "CT", "SINGER"].index(block["model_type"]),
                    )
                    if new_type != old_type:
                        st.session_state["eq_blocks"][idx] = _default_block(new_type)
                        st.session_state["eq_blocks"][idx]["steps"] = block["steps"]
                        st.rerun()
                with cols[1]:
                    block["steps"] = st.number_input(
                        f"Steps {idx + 1}",
                        min_value=2,
                        value=int(block["steps"]),
                        key=f"steps_{version}_{idx}",
                        help="Number of time steps generated for this segment before moving to the next block.",
                    )
                if cols[2].button("Up", key=f"up_{idx}", use_container_width=True) and idx > 0:
                    st.session_state["eq_blocks"][idx - 1], st.session_state["eq_blocks"][idx] = (
                        st.session_state["eq_blocks"][idx],
                        st.session_state["eq_blocks"][idx - 1],
                    )
                    st.rerun()
                if cols[3].button("Down", key=f"down_{idx}", use_container_width=True) and idx < len(st.session_state["eq_blocks"]) - 1:
                    st.session_state["eq_blocks"][idx + 1], st.session_state["eq_blocks"][idx] = (
                        st.session_state["eq_blocks"][idx],
                        st.session_state["eq_blocks"][idx + 1],
                    )
                    st.rerun()
                if cols[4].button("Delete", key=f"del_{idx}", use_container_width=True):
                    st.session_state["eq_blocks"].pop(idx)
                    st.rerun()

                model_type = block["model_type"]
                if model_type == "CV":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["vel_change_std"] = _numeric_param_ui(
                            "CV velocity-change std",
                            _param_defaults(block.get("vel_change_std"), (0.1, 0.0, 0.5)),
                            key_prefix=f"cv_v_{version}_{idx}",
                            help_text="Process noise for the CV.",
                        )
                elif model_type == "CA":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["accel_noise_std"] = _numeric_param_ui(
                            "CA acceleration-noise std",
                            _param_defaults(block.get("accel_noise_std"), (0.1, 0.0, 0.5)),
                            key_prefix=f"ca_a_{version}_{idx}",
                            help_text="Controls process noise applied to acceleration in CA segment.",
                        )
                elif model_type == "CT":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["omega"] = _numeric_param_ui(
                            "CT omega",
                            _param_defaults(block.get("omega"), (1.2, -3.0, 3.0)),
                            key_prefix=f"ct_o_{version}_{idx}",
                            help_text="Turn rate in radians/second for coordinated turn.",
                        )
                    with param_cols[1]:
                        block["omega_noise_std"] = _numeric_param_ui(
                            "CT omega-noise std",
                            _param_defaults(block.get("omega_noise_std"), (0.0, 0.0, 0.5)),
                            key_prefix=f"ct_on_{version}_{idx}",
                            help_text="Noise on turn rate per step.",
                        )
                elif model_type == "SINGER":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["tau"] = _numeric_param_ui(
                            "SINGER tau",
                            _param_defaults(block.get("tau"), (1.0, 0.1, 5.0)),
                            key_prefix=f"sg_t_{version}_{idx}",
                            help_text="Time constant of acceleration correlation.",
                        )
                    with param_cols[1]:
                        block["sigma_a"] = _numeric_param_ui(
                            "SINGER sigma_a",
                            _param_defaults(block.get("sigma_a"), (0.5, 0.1, 2.0)),
                            key_prefix=f"sg_s_{version}_{idx}",
                            help_text="Steady-state acceleration standard deviation.",
                        )

    has_ca_block = any(b["model_type"] == "CA" for b in st.session_state["eq_blocks"])
    initial_acceleration = None
    if has_ca_block:
        with st.container(border=True):
            _section_intro("Initial acceleration", "Used only when at least one CA segment is present in the trajectory.")
            a_cols = st.columns(dim)
            initial_acceleration = [
                a_cols[i].number_input(
                    f"a{i} ({axis_names[i]})",
                    value=float(a_defaults[i]),
                    key=f"eq_a_{i}",
                )
                for i in range(dim)
            ]

    if st.button("Generate equations trajectories", type="primary"):
        st.session_state.pop("equations_job_id", None)
        if len(st.session_state["eq_blocks"]) == 0:
            st.error("Add at least one segment block before generating.")
            return
        body = {
            "output_dir": output_dir,
            "num_trajectories": int(num_trajectories),
            "dt": dt,
            "dim": dim,
            "initial_velocity_params": initial_velocity_params,
            "initial_acceleration": initial_acceleration,
            "observation_noise": observation_noise,
            "randomize_from_current_blocks": randomize,
            "target_total_steps": int(target_total_steps) if target_total_steps else None,
            "blocks": st.session_state["eq_blocks"],
        }
        job_id = _start_equations_job(body)
        if job_id is None:
            return
        st.session_state["equations_job_id"] = job_id

    if "equations_job_id" in st.session_state:
        _render_job_monitor(st.session_state["equations_job_id"], keep_last=10)


def px4_tab() -> None:
    _inject_ui_styles()
    st.subheader("PX4")
    st.caption("Generate and execute PX4 trajectories. Hyperparameters can be fixed or sampled from ranges.")
    with st.container(border=True):
        _section_intro("Run setup", "Choose output, runtime settings, and the PX4 connection profile.")
        top_cols = st.columns([2.4, 1, 1, 1, 1.4])
        with top_cols[0]:
            output_dir = st.text_input("Output directory", value=default_output_dir(), key="px4_output_dir")
        with top_cols[1]:
            num_trajectories = st.number_input("Trajectories", min_value=1, value=DEFAULTS.px4_num_trajectories)
        with top_cols[2]:
            duration_s = st.number_input("Duration [s]", min_value=1.0, value=DEFAULTS.px4_duration_s)
        with top_cols[3]:
            dt_s = st.number_input("dt [s]", min_value=0.01, value=DEFAULTS.px4_dt_s)
        with top_cols[4]:
            profile_name = st.selectbox("Predefined profile", ["default", "figure8", "s_turn"])

        connection_uri = st.text_input("PX4 connection URI", value=DEFAULTS.px4_connection_uri)

    motion_controls_disabled = profile_name in {"figure8", "s_turn"}
    info_cols = st.columns([1, 2.2])
    with info_cols[0]:
        with st.container(border=True):
            _section_intro("Measurement", "Apply sampled or fixed observation noise to the logged PX4 trajectory.")
            observation_noise = _numeric_param_ui(
                "Observation noise",
                (
                    DEFAULTS.px4_observation_noise,
                    DEFAULTS.px4_observation_noise_min,
                    DEFAULTS.px4_observation_noise_max,
                ),
                "px4_obs",
                "Gaussian noise standard deviation applied to logged position observations.",
            )
            st.markdown(
                "<div class='tg-inline-note'>Previews show clean flight data, noisy observations, and commanded setpoints.</div>",
                unsafe_allow_html=True,
            )

    with info_cols[1]:
        with st.container(border=True):
            _section_intro("Motion hyperparameters", "Compact controls for waypoint sampling and vehicle motion behavior.")
            if motion_controls_disabled:
                st.markdown(
                    "<div class='tg-inline-note'>These controls are disabled because the selected profile uses a fixed scripted shape.</div>",
                    unsafe_allow_html=True,
                )
            motion_cols = st.columns(3)
            motion_items = [
                ("num_waypoints_min", "Min waypoints", (4, 3, 10), "px4_nw_min", "Minimum number of waypoints sampled for each trajectory."),
                ("num_waypoints_max", "Max waypoints", (8, 4, 15), "px4_nw_max", "Maximum number of waypoints sampled for each trajectory."),
                ("waypoint_xy_min", "Waypoint XY min [m]", (-80.0, -120.0, -30.0), "px4_xy_min", "Lower XY bound for sampled waypoints."),
                ("waypoint_xy_max", "Waypoint XY max [m]", (80.0, 30.0, 120.0), "px4_xy_max", "Upper XY bound for sampled waypoints."),
                ("waypoint_z_min", "Waypoint Z min [m]", (-30.0, -60.0, -5.0), "px4_z_min", "Lower Z bound in the NED frame."),
                ("waypoint_z_max", "Waypoint Z max [m]", (-5.0, -20.0, -2.0), "px4_z_max", "Upper Z bound in the NED frame."),
                ("max_speed", "Max speed [m/s]", (7.0, 2.0, 15.0), "px4_spd", "Cruise speed target used while moving between waypoints."),
                ("accel", "Acceleration gain", (4.0, 1.0, 10.0), "px4_acc", "How quickly the vehicle converges toward its target velocity."),
                ("waypoint_tolerance", "Waypoint tolerance [m]", (3.0, 1.0, 8.0), "px4_tol", "Distance threshold used to mark a waypoint as reached."),
            ]
            motion = {}
            for idx, (key, label, defaults, key_prefix, help_text) in enumerate(motion_items):
                with motion_cols[idx % 3]:
                    motion[key] = _numeric_param_ui(label, defaults, key_prefix, help_text, disabled=motion_controls_disabled)

    if st.button("Generate PX4 trajectories", type="primary"):
        st.session_state.pop("px4_job_id", None)
        body = {
            "output_dir": output_dir,
            "num_trajectories": int(num_trajectories),
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
            "observation_noise": observation_noise,
            "connection_uri": connection_uri,
            "profile_name": profile_name,
            "motion": motion,
        }
        job_id = _start_px4_job(body)
        if job_id is None:
            return
        st.session_state["px4_job_id"] = job_id

    if "px4_job_id" in st.session_state:
        _render_job_monitor(st.session_state["px4_job_id"], keep_last=10)


def output_format_tab() -> None:
    _inject_ui_styles()
    st.subheader("Output Format")
    st.caption("Reference for the JSONL files written by the equations and PX4 generators.")

    with st.container(border=True):
        _section_intro("File structure", "Each output file is newline-delimited JSON, with one trajectory record per line.")
        st.markdown(
            """
            - Output files are written as `.jsonl`.
            - Each line is a complete JSON object for one generated trajectory.
            - The filename includes a timestamp and a few key hyperparameters.
            """
        )

    cols = st.columns(2)

    with cols[0]:
        with st.container(border=True):
            _section_intro("Equations records", "Mathematical trajectories store clean and noisy sequences together with the sampled configuration.")
            st.code(
                """{
  "id": 0,
  "type": "equations",
  "trajectory_config": {
    "dt": 0.04,
    "dim": 3,
    "observation_noise_std": 0.1,
    "segments": [...]
  },
  "noisy": [[x, y, z], ...],
  "clean": [[x, y, z], ...]
}""",
                language="json",
            )

    with cols[1]:
        with st.container(border=True):
            _section_intro("PX4 records", "PX4 trajectories store commanded setpoints together with normalized clean/noisy flight logs.")
            st.code(
                """{
  "id": 0,
  "type": "px4",
  "trajectory_config": {
    "dt": 0.1,
    "observation_noise_std": 0.1,
    "metadata": {
      "duration_s": 20.0,
      "profile_name": "default",
      "motion": {...}
    }
  },
  "setpoints": [{"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}, ...],
  "clean": [{"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}, ...],
  "noisy": [{"t": 0.0, "x": 0.1, "y": -0.1, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}, ...]
}""",
                language="json",
            )

    with st.container(border=True):
        _section_intro("Coordinate conventions", "How to interpret the stored values.")
        st.markdown(
            """
            - PX4 trajectories are always 3D and use NED-style position fields: `x`, `y`, and `z`.
            - `setpoints` are commanded reference states sent to PX4.
            - `clean` is the logged vehicle state returned by telemetry.
            - `noisy` is the clean position with added Gaussian observation noise.
            - Common fields stay under `trajectory_config`, while PX4-specific details live under `trajectory_config.metadata`.
            - Equation trajectories are 2D or 3D depending on the selected `dim`.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Trajectory Generator Lab", layout="wide")
    st.title("Trajectory Generator Lab")
    tab_eq, tab_px4, tab_output = st.tabs(["Equations", "PX4", "Output Format"])
    with tab_eq:
        equations_tab()
    with tab_px4:
        px4_tab()
    with tab_output:
        output_format_tab()


if __name__ == "__main__":
    main()
