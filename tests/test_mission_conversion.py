import math

import numpy as np
import pytest

from app.local_lib.mission_conversion import (
    build_nav_waypoint_mission_items,
    clean_positions_to_path_points,
    ned_from_home_to_wgs84_relative_alt,
    thin_path_points,
)


def test_ned_from_home_origin() -> None:
    lat, lon, rel = ned_from_home_to_wgs84_relative_alt(47.0, 8.0, 0.0, 0.0, 0.0)
    assert abs(lat - 47.0) < 1e-9
    assert abs(lon - 8.0) < 1e-9
    assert rel == 0.0


def test_ned_offset_matches_approximate_meters() -> None:
    home_lat, home_lon = 47.398, 8.5456
    north_m, east_m, down_m = 100.0, 50.0, -10.0
    lat, lon, rel = ned_from_home_to_wgs84_relative_alt(home_lat, home_lon, north_m, east_m, down_m)
    assert rel == 10.0
    assert lat > home_lat
    assert lon > home_lon


def test_clean_positions_to_path_points_yaw() -> None:
    dt = 0.1
    clean = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [1.0, 1.0, -1.0]], dtype=float)
    pts = clean_positions_to_path_points(clean, dt)
    assert len(pts) == 3
    assert pts[0]["yaw"] == pytest.approx(0.0)
    assert pts[1]["yaw"] == pytest.approx(90.0)


def test_thin_path_points_respects_max() -> None:
    pts = [{"t": float(i), "x": float(i), "y": 0.0, "z": 0.0, "yaw": 0.0} for i in range(100)]
    out = thin_path_points(pts, min_step_m=0.1, max_points=10)
    assert len(out) <= 10
    assert out[0]["x"] == 0.0
    assert out[-1]["x"] == 99.0


def test_build_mission_items_count_and_seq() -> None:
    path = [
        {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        {"t": 1.0, "x": 10.0, "y": 0.0, "z": -5.0, "yaw": 45.0},
    ]
    items = build_nav_waypoint_mission_items(path, 0.0, 0.0, 0.0, 47.0, 8.0, acceptance_radius_m=3.0)
    assert len(items) == 2
    assert items[0].seq == 0
    assert items[1].seq == 1
    assert items[0].current == 1
    assert items[1].current == 0
    assert not math.isnan(items[0].param4)
