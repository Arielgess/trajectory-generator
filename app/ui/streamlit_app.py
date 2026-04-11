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


def _render_equation_blocks_panel(
    dt: float,
    initial_velocity_params: list[dict[str, Any]],
    *,
    blocks_key: str,
    version_key: str,
    widget_prefix: str,
) -> None:
    """Shared segment-block editor (templates + per-block params). Uses distinct session keys per tab."""
    if blocks_key not in st.session_state:
        st.session_state[blocks_key] = [_default_block("CV")]
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    with st.container(border=True):
        _section_intro("Trajectory creation lab", "Build an ordered list of motion segments or start from a template.")
        templates = _equation_templates()
        template_cols = st.columns([2.2, 1, 1, 2.2])
        with template_cols[0]:
            selected_template = st.selectbox(
                "Predefined structure",
                ["Custom", "Figure-8", "S-turn"] + list(templates.keys())[1:],
                index=0,
                key=f"{widget_prefix}_tpl_select",
            )
        if template_cols[1].button("Apply template", use_container_width=True, key=f"{widget_prefix}_tpl_apply"):
            if selected_template == "Figure-8":
                st.session_state[blocks_key] = _build_figure8_template(initial_velocity_params, dt)
            elif selected_template == "S-turn":
                st.session_state[blocks_key] = _build_s_shape_template(initial_velocity_params, dt)
            else:
                st.session_state[blocks_key] = [dict(item) for item in templates[selected_template]]
            st.session_state[version_key] += 1
            st.rerun()
        if template_cols[2].button("Add block", use_container_width=True, key=f"{widget_prefix}_add_block"):
            st.session_state[blocks_key].append(_default_block("CV"))
        with template_cols[3]:
            st.markdown(
                "<div class='tg-inline-note'>Use the segment list below to mix CV, CA, CT, and SINGER motion patterns.</div>",
                unsafe_allow_html=True,
            )

        blocks: list[dict[str, Any]] = st.session_state[blocks_key]
        for idx, block in enumerate(blocks):
            version = st.session_state[version_key]
            with st.expander(f"Segment {idx + 1}: {block['model_type']}", expanded=True):
                cols = st.columns([4, 2, 1, 1, 1])
                with cols[0]:
                    old_type = block["model_type"]
                    new_type = st.selectbox(
                        f"Type {idx + 1}",
                        ["CV", "CA", "CT", "SINGER"],
                        key=f"{widget_prefix}_type_{version}_{idx}",
                        index=["CV", "CA", "CT", "SINGER"].index(block["model_type"]),
                    )
                    if new_type != old_type:
                        st.session_state[blocks_key][idx] = _default_block(new_type)
                        st.session_state[blocks_key][idx]["steps"] = block["steps"]
                        st.rerun()
                with cols[1]:
                    block["steps"] = st.number_input(
                        f"Steps {idx + 1}",
                        min_value=2,
                        value=int(block["steps"]),
                        key=f"{widget_prefix}_steps_{version}_{idx}",
                        help="Number of time steps generated for this segment before moving to the next block.",
                    )
                if cols[2].button("Up", key=f"{widget_prefix}_up_{idx}", use_container_width=True) and idx > 0:
                    st.session_state[blocks_key][idx - 1], st.session_state[blocks_key][idx] = (
                        st.session_state[blocks_key][idx],
                        st.session_state[blocks_key][idx - 1],
                    )
                    st.rerun()
                if (
                    cols[3].button("Down", key=f"{widget_prefix}_down_{idx}", use_container_width=True)
                    and idx < len(st.session_state[blocks_key]) - 1
                ):
                    st.session_state[blocks_key][idx + 1], st.session_state[blocks_key][idx] = (
                        st.session_state[blocks_key][idx],
                        st.session_state[blocks_key][idx + 1],
                    )
                    st.rerun()
                if cols[4].button("Delete", key=f"{widget_prefix}_del_{idx}", use_container_width=True):
                    st.session_state[blocks_key].pop(idx)
                    st.rerun()

                model_type = block["model_type"]
                if model_type == "CV":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["vel_change_std"] = _numeric_param_ui(
                            "CV velocity-change std",
                            _param_defaults(block.get("vel_change_std"), (0.1, 0.0, 0.5)),
                            key_prefix=f"{widget_prefix}_cv_v_{version}_{idx}",
                            help_text="Process noise for the CV.",
                        )
                elif model_type == "CA":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["accel_noise_std"] = _numeric_param_ui(
                            "CA acceleration-noise std",
                            _param_defaults(block.get("accel_noise_std"), (0.1, 0.0, 0.5)),
                            key_prefix=f"{widget_prefix}_ca_a_{version}_{idx}",
                            help_text="Controls process noise applied to acceleration in CA segment.",
                        )
                elif model_type == "CT":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["omega"] = _numeric_param_ui(
                            "CT omega",
                            _param_defaults(block.get("omega"), (1.2, -3.0, 3.0)),
                            key_prefix=f"{widget_prefix}_ct_o_{version}_{idx}",
                            help_text="Turn rate in radians/second for coordinated turn.",
                        )
                    with param_cols[1]:
                        block["omega_noise_std"] = _numeric_param_ui(
                            "CT omega-noise std",
                            _param_defaults(block.get("omega_noise_std"), (0.0, 0.0, 0.5)),
                            key_prefix=f"{widget_prefix}_ct_on_{version}_{idx}",
                            help_text="Noise on turn rate per step.",
                        )
                elif model_type == "SINGER":
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        block["tau"] = _numeric_param_ui(
                            "SINGER tau",
                            _param_defaults(block.get("tau"), (1.0, 0.1, 5.0)),
                            key_prefix=f"{widget_prefix}_sg_t_{version}_{idx}",
                            help_text="Time constant of acceleration correlation.",
                        )
                    with param_cols[1]:
                        block["sigma_a"] = _numeric_param_ui(
                            "SINGER sigma_a",
                            _param_defaults(block.get("sigma_a"), (0.5, 0.1, 2.0)),
                            key_prefix=f"{widget_prefix}_sg_s_{version}_{idx}",
                            help_text="Steady-state acceleration standard deviation.",
                        )


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

    _render_equation_blocks_panel(
        dt,
        initial_velocity_params,
        blocks_key="eq_blocks",
        version_key="eq_blocks_version",
        widget_prefix="eq",
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


def equations_px4_mission_tab() -> None:
    """Build equation segment blocks, generate a clean trajectory, upload as a PX4 mission, and log flight."""
    _inject_ui_styles()
    st.subheader("Equations → PX4")
    st.caption(
        "Design a trajectory with motion blocks, upload it as a MAVLink mission, and let PX4 fly it "
        "autonomously at any sim speed. The output 'clean' is what the vehicle actually flew; "
        "the initial climb to the configured altitude is included and can be trimmed in post-processing."
    )

    v_defaults = DEFAULTS.equations_initial_velocity_3d
    a_defaults = DEFAULTS.equations_initial_acceleration_3d
    axis_names = ["vx", "vy", "vz"]

    # ── Run setup ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro("Run setup", "Output, batch size, and simulator connection.")
        row = st.columns([2.4, 1, 2])
        with row[0]:
            output_dir = st.text_input("Output directory", value=default_output_dir(), key="eqpx4_output_dir")
        with row[1]:
            num_trajectories = st.number_input(
                "Trajectories", min_value=1, value=DEFAULTS.px4_num_trajectories, key="eqpx4_n"
            )
        with row[2]:
            connection_uri = st.text_input(
                "PX4 connection URI", value=DEFAULTS.px4_connection_uri, key="eqpx4_uri"
            )

    # ── Trajectory settings ───────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "Trajectory",
            "The path length is determined by the block steps × dt. "
            "Enable randomisation to sample a random assortment of blocks up to a target step count.",
        )
        traj_cols = st.columns([1, 1, 1])
        with traj_cols[0]:
            em_dt = st.number_input(
                "dt [s]",
                min_value=0.0001,
                value=DEFAULTS.equations_dt,
                key="eqpx4_em_dt",
                help="Time step for the equation path. Each step becomes a potential mission waypoint.",
            )
        with traj_cols[1]:
            randomize = st.toggle(
                "Randomize blocks",
                value=False,
                key="eqpx4_rand",
                help="Draw a random assortment from the block palette rather than using the fixed order.",
            )
        with traj_cols[2]:
            seed = st.number_input("Seed (blank = random)", min_value=0, value=0, key="eqpx4_seed")
            seed_val: int | None = int(seed) if seed else None

        if randomize:
            rand_cols = st.columns(3)
            with rand_cols[0]:
                em_min_seg = st.number_input("Min segment steps", min_value=2, value=20, key="eqpx4_min_seg")
            with rand_cols[1]:
                em_max_seg = st.number_input("Max segment steps", min_value=2, value=60, key="eqpx4_max_seg")
            with rand_cols[2]:
                target_total_steps = st.number_input(
                    "Target total steps",
                    min_value=10,
                    value=400,
                    key="eqpx4_tgt_steps",
                    help="Blocks are drawn until the total step count reaches this value.",
                )
        else:
            em_min_seg, em_max_seg, target_total_steps = 20, 60, None

    # ── Initial velocity ──────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "Initial velocity (3D)",
            "Starting velocity for the equations path. "
            "Use a range to vary it across trajectories in a batch.",
        )
        vel_cols = st.columns(3)
        initial_velocity_params: list[dict[str, Any]] = []
        for i, axis_name in enumerate(axis_names):
            with vel_cols[i]:
                initial_velocity_params.append(
                    _numeric_param_ui(
                        axis_name,
                        (float(v_defaults[i]), float(v_defaults[i]) * 0.5, float(v_defaults[i]) * 1.5),
                        key_prefix=f"eqpx4_init_v_{axis_name}",
                        help_text=f"Initial {axis_name} applied to the equations generator.",
                    )
                )

    # ── Block palette ─────────────────────────────────────────────────────────
    sync_cols = st.columns([1, 3])
    with sync_cols[0]:
        if st.button("Copy blocks from Equations tab", key="eqpx4_sync_eq"):
            src = st.session_state.get("eq_blocks")
            if src:
                st.session_state["eq_mission_blocks"] = [dict(b) for b in src]
                st.session_state["eq_mission_blocks_version"] = (
                    st.session_state.get("eq_mission_blocks_version", 0) + 1
                )
                st.success("Blocks copied.")
                st.rerun()
            else:
                st.warning("Equations tab has no blocks yet.")
    with sync_cols[1]:
        st.markdown(
            "<div class='tg-inline-note'>Block state is independent from the Equations tab.</div>",
            unsafe_allow_html=True,
        )

    _render_equation_blocks_panel(
        float(em_dt),
        initial_velocity_params,
        blocks_key="eq_mission_blocks",
        version_key="eq_mission_blocks_version",
        widget_prefix="eqpx4",
    )

    blocks = st.session_state.get("eq_mission_blocks") or []

    # ── Initial acceleration (only when a CA block exists) ────────────────────
    has_ca_block = any(b.get("model_type") == "CA" for b in blocks)
    initial_acceleration: list[float] | None = None
    if has_ca_block:
        with st.container(border=True):
            _section_intro("Initial acceleration (3D)", "Required because at least one CA segment is present.")
            a_cols = st.columns(3)
            initial_acceleration = [
                float(
                    a_cols[i].number_input(
                        f"a{i} ({axis_names[i]})", value=float(a_defaults[i]), key=f"eqpx4_a_{i}"
                    )
                )
                for i in range(3)
            ]

    # ── Advanced settings (collapsed by default) ──────────────────────────────
    with st.expander("PX4 & mission settings", expanded=False):

        st.markdown("**Telemetry noise** — applied to PX4 flight logs, not to the equations path.")
        observation_noise = _numeric_param_ui(
            "Observation noise (PX4 telemetry)",
            (
                DEFAULTS.px4_observation_noise,
                DEFAULTS.px4_observation_noise_min,
                DEFAULTS.px4_observation_noise_max,
            ),
            key_prefix="eqpx4_obs",
            help_text="Gaussian noise added to x/y/z in the 'noisy' telemetry log.",
        )

        st.divider()
        st.markdown(
            "**Flight dynamics** — set or randomise PX4 MPC controller parameters before each flight. "
            "Use *range* mode on any parameter to sample a fresh value per trajectory, "
            "or *fixed* to apply an exact value every time."
        )
        randomize_dynamics = st.toggle(
            "Apply PX4 flight dynamics",
            value=False,
            key="eqpx4_rand_dyn",
            help=(
                "When on, the five MPC parameters below are written to PX4 before each flight. "
                "Leave off to fly with PX4 defaults."
            ),
        )
        _MPC_ITEMS = [
            ("mpc_acc_hor_max", "MPC_ACC_HOR_MAX [m/s²]", (12.5, 5.0, 20.0), "eqpx4_mpc_acc",
             "Maximum horizontal acceleration. Range 5–20 m/s²."),
            ("mpc_jerk_max",    "MPC_JERK_MAX [m/s³]",   (20.0, 5.0, 35.0), "eqpx4_mpc_jrk",
             "Maximum jerk limit. Range 5–35 m/s³."),
            ("mpc_xy_p",        "MPC_XY_P",               (1.5, 1.0, 2.0),   "eqpx4_mpc_xyp",
             "Position loop P-gain for XY axes. Range 1–2."),
            ("mpc_tiltmax_air", "MPC_TILTMAX_AIR [°]",   (62.5, 45.0, 80.0), "eqpx4_mpc_tlt",
             "Maximum tilt angle in air. Range 45–80 °."),
            ("mpc_xy_vel_p_acc","MPC_XY_VEL_P_ACC",      (2.65, 1.8, 3.5),  "eqpx4_mpc_vel",
             "Velocity loop P-gain (accel feed-forward). Range 1.8–3.5."),
        ]
        dynamics_params: dict[str, Any] = {}
        if randomize_dynamics:
            dyn_cols = st.columns(3)
            for idx, (field, label, defs, key_pfx, help_txt) in enumerate(_MPC_ITEMS):
                with dyn_cols[idx % 3]:
                    dynamics_params[field] = _numeric_param_ui(label, defs, key_pfx, help_txt)
        else:
            # Send schema defaults (fixed mode) — ignored by backend when toggle is off.
            for field, _label, (val, lo, hi), _key, _help in _MPC_ITEMS:
                dynamics_params[field] = {"mode": "range", "value": val, "min_value": lo, "max_value": hi}

        st.divider()
        st.markdown(
            "**Mission upload** — controls waypoint thinning and how PX4 flies the path. "
            "The initial climb from ground to *min altitude* is included in the logged data; "
        )
        adv_cols = st.columns(3)
        with adv_cols[0]:
            mission_min_step_m = st.number_input(
                "Min waypoint spacing [m]",
                min_value=0.05, value=0.5, key="eqpx4_min_step",
                help=(
                    "Adjacent waypoints closer than this are merged before upload. "
                    "Increase to reduce the number of uploaded waypoints."
                ),
            )
        with adv_cols[1]:
            waypoint_acceptance_radius_m = st.number_input(
                "Acceptance radius [m]",
                min_value=0.1, value=2.0, key="eqpx4_accept_r",
                help="PX4 marks a waypoint as reached when the drone is within this radius.",
            )
        with adv_cols[2]:
            min_altitude_m = st.number_input(
                "Min altitude [m AGL]",
                min_value=1.0, value=10.0, key="eqpx4_min_alt",
                help=(
                    "The whole path is shifted upward so its lowest point is at least "
                    "this many metres above the home/arm position."
                ),
            )
        # mission_max_waypoints is set permanently to the schema maximum (900).
        # Users should control density via min_waypoint_spacing instead.
        mission_max_waypoints = 900

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("Generate & fly on PX4", type="primary", key="eqpx4_run"):
        st.session_state.pop("eq_px4_mission_job_id", None)
        if not blocks:
            st.error("Add at least one segment block.")
            return
        if randomize and not target_total_steps:
            st.error("Target total steps is required when randomisation is enabled.")
            return
        # duration_s is computed from trajectory length (steps × dt) in the generator;
        # we pass a placeholder here — the generator overwrites it with the actual value.
        computed_duration_s = (
            float(target_total_steps) * float(em_dt) if randomize
            else sum(b.get("steps", 0) for b in blocks) * float(em_dt)
        )
        equations_mission_payload: dict[str, Any] = {
            "dt": float(em_dt),
            "dim": 3,
            "seed": seed_val,
            "randomize_from_current_blocks": randomize,
            "min_segment_length": int(em_min_seg),
            "max_segment_length": int(em_max_seg),
            "target_total_steps": int(target_total_steps) if target_total_steps is not None else None,
            "blocks": [dict(b) for b in blocks],
            "initial_velocity_params": initial_velocity_params,
            "initial_acceleration": initial_acceleration,
            "mission_max_waypoints": int(mission_max_waypoints),
            "mission_min_step_m": float(mission_min_step_m),
            "waypoint_acceptance_radius_m": float(waypoint_acceptance_radius_m),
            "min_altitude_m": float(min_altitude_m),
            "randomize_flight_dynamics": bool(randomize_dynamics),
            **dynamics_params,
        }
        body: dict[str, Any] = {
            "output_dir": output_dir,
            "num_trajectories": int(num_trajectories),
            "duration_s": max(computed_duration_s, 1.0),
            "dt_s": float(em_dt),
            "observation_noise": observation_noise,
            "connection_uri": connection_uri,
            "profile_name": "equations_mission",
            "motion": None,
            "equations_mission": equations_mission_payload,
        }
        job_id = _start_px4_job(body)
        if job_id is None:
            return
        st.session_state["eq_px4_mission_job_id"] = job_id

    if "eq_px4_mission_job_id" in st.session_state:
        _render_job_monitor(st.session_state["eq_px4_mission_job_id"], keep_last=10)


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
    st.subheader("Usage Guide")
    st.caption(
        "How each generator works, when to use it, and the format of the output files."
    )

    # ── Overview ──────────────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro("Overview", "Three engines, two philosophies.")
        st.markdown(
            """
            This tool provides three trajectory generation engines.  Two of them
            (**Equations** and **PX4**) are independent; the third (**Eq → PX4**) is a
            pipeline that feeds the output of the equations engine into PX4 as a
            pre-planned mission.

            The choice of engine depends on your goal:

            | Engine | Physics source | Timing control | Speed factor | Recommended for |
            |---|---|---|---|---|
            | Equations | Mathematical models | Instant (no simulation) | N/A | Large dataset generation, filter prototyping |
            | PX4 | Real PX4 flight stack | Python loop (`asyncio.sleep`) | Works reliably up to ~3–5× | Realistic PX4 dynamics, one-off runs |
            | Eq → PX4 | Real PX4 flight stack | PX4 drives itself (mission mode) | Any speed factor | Large-scale PX4 datasets with diverse shapes |
            """
        )

    # ── Equations engine ──────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "Equations engine",
            "Pure mathematics, no simulator required.",
        )
        st.markdown(
            """
            The equations engine generates trajectories by numerically integrating
            stochastic differential equations.  No PX4, no Gazebo, and no network
            connection are involved — a batch of thousands of trajectories can be
            created in seconds.

            #### Motion models

            Each trajectory is built from one or more sequential **segments**, each
            governed by one of four motion models:

            - **CV — Constant Velocity.**  The target moves at a fixed velocity.
              Small random perturbations (process noise) are applied each step, so the
              path is smooth but gently curved rather than perfectly straight.
            - **CA — Constant Acceleration.**  The target accelerates at a roughly
              constant rate.  Process noise causes the acceleration to drift slowly
              over time.  Produces arcing, banana-shaped segments.
            - **CT — Coordinated Turn.**  The target moves at constant speed through a
              banked turn at a fixed turn rate (omega).  Process noise on the turn rate
              makes the curve slightly irregular.  Good for modelling aircraft-style
              manoeuvres.
            - **SINGER — Singer model.**  Models acceleration as a first-order
              Markov process with a user-specified time constant (tau) and variance
              (sigma_a).  The trajectory is smooth yet highly manoeuvrable, making it
              the most realistic model for agile targets.

            #### Segment blocks and randomisation

            Segments are configured as a list of **blocks**.  Each block specifies the
            motion model, the number of time steps, and model-specific parameters such
            as noise standard deviations or turn rates.

            When **Randomize from current blocks** is enabled, the generator treats the
            block list as a *palette* rather than a fixed plan.  For each trajectory it
            draws a random number of blocks, picks each one from the palette uniformly
            at random, and assigns it a random length sampled between *min segment
            length* and *max segment length*.  The draw continues until the total step
            count reaches or exceeds *target total steps*.  This produces a large
            variety of trajectory shapes from a compact configuration.

            #### Observation noise

            After the clean trajectory is integrated, independent zero-mean Gaussian
            noise with the configured standard deviation is added to each position
            axis at every time step.  The result is the **noisy** sequence, which
            simulates realistic sensor measurements.  The **clean** sequence is the
            noiseless ground truth.

            #### Dimensionality

            The generator supports **2D** (x, y) and **3D** (x, y, z).  In 2D mode
            the z axis is omitted entirely.  In 3D mode all four motion models operate
            independently on each axis, except CT which couples x and y through the
            turn rate while treating z as a separate CV/CA process.
            """
        )

    # ── PX4 engine ────────────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "PX4 engine",
            "Realistic drone dynamics powered by the PX4 flight stack.",
        )
        st.markdown(
            """
            The PX4 engine connects to a running PX4 SITL (Software-In-The-Loop)
            simulator via MAVSDK and flies the drone in **offboard mode**.  In offboard
            mode Python acts as the high-rate external controller: it sends a new
            position + velocity setpoint to PX4 at every time step and PX4's internal
            control loops (position controller → velocity controller → attitude
            controller → motor mixer) handle the actual stabilisation.

            #### Execution sequence

            1. Wait for PX4 health checks (GPS fix, estimator convergence).
            2. Arm the drone and command a takeoff to a safe initial altitude.
            3. Capture the current NED position as the trajectory origin.
            4. Enter offboard mode and begin streaming setpoints from the pre-generated
               path one step at a time.
            5. After each setpoint is sent, one telemetry sample is read back and
               stored as the **clean** log entry.
            6. Land and disarm.

            #### Timing and speed factor

            The step interval is `dt / PX4_SIM_SPEED_FACTOR`.  The code uses an
            accumulating wall-clock timer so that loop overhead is compensated across
            steps rather than added to each sleep.  At speed factors up to roughly 3–5×
            this works well.  At higher speed factors the required sleep interval falls
            below Python's scheduler resolution (~10 ms on most systems), and the drone
            effectively receives all setpoints in rapid succession rather than spaced
            over the correct simulated time — trajectory quality degrades significantly.
            For high speed factors, use the **Eq → PX4** engine instead.

            #### Built-in trajectory profiles

            - **Default.**  Random waypoints are generated inside a configurable 3D
              box.  The drone visits them in order using smooth jerk-limited motion
              whose speed and acceleration limits you control.  The number of waypoints
              and their spread can be fixed or drawn from a range, making each
              trajectory unique.
            - **Figure-8.**  A deterministic figure-of-eight path in the horizontal
              plane at a fixed altitude.  Useful for repeatable benchmarking.
            - **S-turn.**  A sinusoidal lateral sweep combined with a gradual climb,
              simulating a search pattern.

            #### What the clean log contains

            Because PX4's controllers never track setpoints perfectly — there is
            always lag, overshoot, and vibration — the **clean** and **setpoints**
            sequences are genuinely different.  The clean log is the *actual* vehicle
            state reported by the on-board estimator, making it realistic flight data
            rather than an idealized reference.
            """
        )

    # ── Eq → PX4 engine ───────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "Eq → PX4 engine",
            "Equations-generated shapes flown by PX4 in mission mode — works at any sim speed.",
        )
        st.markdown(
            """
            This engine is a two-stage pipeline that combines the diversity of the
            equations generator with the physical realism of the PX4 flight stack,
            while avoiding the Python timing limitations of offboard mode.

            #### Stage 1 — trajectory generation (equations)

            A full position-over-time trajectory is generated offline using the same
            block-based equations engine described above.  All blocks, randomisation,
            and motion models are available.  The generator always runs in 3D (z is
            the NED down axis) and the entire path is available in memory before any
            flight begins.

            The path is then post-processed in two ways:
            - **Altitude normalisation.**  Equation paths start at z = 0 (ground
              level in NED).  The entire path is shifted upward so that the lowest
              point sits at least *min altitude* metres above the home/arm position.
              This prevents PX4 from receiving waypoints below ground.
            - **Waypoint thinning.**  Dense equation paths (one point per dt) may
              contain thousands of positions.  Consecutive waypoints closer than *min
              step* metres are merged, and the result is capped at *max waypoints*.
              This produces a compact mission that still faithfully represents the
              original path shape.

            #### Stage 2 — mission upload and flight (PX4)

            The thinned waypoints are converted to global coordinates (lat/lon +
            altitude relative to home) and uploaded to PX4 as a **MAVLink mission**
            before the drone arms.  The execution sequence is:

            1. Upload the complete mission.  Wait 1–2 seconds for PX4's navigator to
               validate it.
            2. Arm.
            3. Call `start_mission()`.  PX4 then handles everything autonomously:
               takeoff to the first waypoint altitude, waypoint following.
               **Python does not send any setpoints during the flight.**
            4. Telemetry logging begins immediately at mission start — **including the
               initial climb from the ground to `min altitude` metres.**
            5. Mission completion is detected by watching PX4's flight mode: when it
               transitions from MISSION → HOLD, logging stops.

            #### Initial climb and post-processing trim

            The initial climb from ground to `min altitude` (which you configure in the
            UI) is intentionally included in the `clean` and `noisy` logs.  The climb
            height is stored in `trajectory_config.metadata.min_altitude_m`.

            To isolate only the waypoint-following phase in post-processing:

            ```python
            min_alt = record["trajectory_config"]["metadata"]["min_altitude_m"]
            clean = [s for s in record["clean"] if s["z"] >= min_alt]
            ```

            This gives you a clean start from the first equation waypoint with no
            need for any on-the-fly detection logic.

            #### Why this works at any speed factor

            Because all waypoints are on the vehicle before takeoff, PX4's navigator
            drives the timing entirely.  There is no Python sleep loop that needs to
            match simulated time.  Whether `PX4_SIM_SPEED_FACTOR` is 1× or 20×,
            Python is only polling for completion and streaming telemetry — both of
            which have large tolerance for timing jitter.

            #### Setpoints vs. clean in this mode

            - **Setpoints** are the thinned mission waypoints (the uploaded path shape).
              They are sparse by design.
            - **Clean** is the continuous telemetry logged at the observation rate
              while PX4 flies between those waypoints.  PX4's mission executor
              generates smooth, jerk-limited paths between waypoints, so the clean
              trajectory is denser and smoother than the setpoints, and reflects real
              flight dynamics rather than the exact equation path.  This is intentional:
              the equations define the *shape* of the flight; PX4 determines the
              *dynamics*.
            """
        )

    # ── Output format ─────────────────────────────────────────────────────────
    with st.container(border=True):
        _section_intro(
            "Output format",
            "All generators write newline-delimited JSON (.jsonl) — one trajectory object per line.",
        )
        st.markdown(
            """
            #### Filename convention

            Files are named `{engine}_{timestamp}_count-{n}_dt-{dt}_duration-{d}.jsonl`.
            The timestamp is local wall time at job start.

            #### Common fields (all engines)

            | Field | Description |
            |---|---|
            | `id` | Zero-based index of this trajectory within the file. |
            | `type` | `"equations"` or `"px4"`. |
            | `trajectory_config` | Hyperparameters used to generate this trajectory. |
            | `clean` | Ground-truth positions (equations) or vehicle telemetry (PX4). |
            | `noisy` | `clean` positions with independent per-axis Gaussian noise added. |

            #### Equations-specific fields

            `clean` and `noisy` are arrays of position vectors `[x, y]` (2D) or
            `[x, y, z]` (3D), one per time step.  There are no velocity fields because
            velocity can be estimated by finite difference if needed.

            #### PX4-specific fields

            `setpoints` is an array of commanded reference states, each containing
            `{t, x, y, z, yaw}`.  Both `clean` and `noisy` are arrays of logged vehicle
            states containing `{t, x, y, z, vx, vy, vz}`.  Positions are in the NED
            frame relative to the home/arm position; velocities are NED body-frame from
            the on-board estimator.

            #### Coordinate conventions

            All PX4 outputs use **NED** (North-East-Down): `x` is north, `y` is east,
            `z` is down (so negative `z` means above ground).  Yaw is measured in
            degrees, increasing clockwise from north.  Equations-only outputs use
            abstract Cartesian coordinates with no physical orientation assumed.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Trajectory Generator Lab", layout="wide")
    st.title("Trajectory Generator Lab")
    tab_eq, tab_eq_px4, tab_px4, tab_output = st.tabs(
        ["Equations", "Eq → PX4", "PX4", "Usage"]
    )
    with tab_eq:
        equations_tab()
    with tab_eq_px4:
        equations_px4_mission_tab()
    with tab_px4:
        px4_tab()
    with tab_output:
        output_format_tab()


if __name__ == "__main__":
    main()
