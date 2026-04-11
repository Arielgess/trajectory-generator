from __future__ import annotations

import math
import numpy as np
from mavsdk.mission_raw import MissionItem

from app.local_lib.mavlink_constants import (
    MAV_CMD_NAV_WAYPOINT,
    MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    MAV_MISSION_TYPE_MISSION,
)

EARTH_RADIUS_M = 6378137.0


def ned_from_home_to_wgs84_relative_alt(
    home_lat_deg: float,
    home_lon_deg: float,
    north_m: float,
    east_m: float,
    down_m: float,
) -> tuple[float, float, float]:
    """Convert home-relative NED (m) to WGS84 lat/lon (deg) and relative altitude (m, up positive)."""
    lat0 = math.radians(home_lat_deg)
    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    cos_lat = math.cos(lat0)
    denom = EARTH_RADIUS_M * max(cos_lat, 1e-9)
    dlon = (east_m / denom) * (180.0 / math.pi)
    lat_deg = home_lat_deg + dlat
    lon_deg = home_lon_deg + dlon
    rel_alt_m = -down_m
    return lat_deg, lon_deg, rel_alt_m


def clean_positions_to_path_points(clean: np.ndarray, dt: float) -> list[dict[str, float]]:
    """Build local NED-style path points with time and yaw from consecutive XY motion."""
    if clean.ndim != 2 or clean.shape[1] not in (2, 3):
        raise ValueError("clean must be (T, 2) or (T, 3)")
    points: list[dict[str, float]] = []
    n = clean.shape[0]
    for i in range(n):
        x = float(clean[i, 0])
        y = float(clean[i, 1])
        z = float(clean[i, 2]) if clean.shape[1] >= 3 else 0.0
        if i + 1 < n:
            dx = float(clean[i + 1, 0] - clean[i, 0])
            dy = float(clean[i + 1, 1] - clean[i, 1])
            yaw = math.degrees(math.atan2(dy, dx)) if abs(dx) + abs(dy) > 1e-6 else 0.0
        elif points:
            yaw = points[-1]["yaw"]
        else:
            yaw = 0.0
        points.append({"t": float(i) * dt, "x": x, "y": y, "z": z, "yaw": yaw})
    return points


def thin_path_points(
    points: list[dict[str, float]],
    min_step_m: float,
    max_points: int,
) -> list[dict[str, float]]:
    """Keep first and last; retain points when 3D distance from last kept exceeds min_step_m."""
    if len(points) <= 2:
        return list(points)
    kept: list[dict[str, float]] = [points[0]]
    min_step = max(min_step_m, 1e-6)
    end = points[-1]

    def dist(a: dict[str, float], b: dict[str, float]) -> float:
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dz = b["z"] - a["z"]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    for p in points[1:-1]:
        if dist(p, kept[-1]) >= min_step:
            kept.append(p)
    if dist(end, kept[-1]) > 1e-6:
        kept.append(end)
    else:
        kept[-1] = end

    if len(kept) <= max_points:
        return kept

    # Evenly subsample including endpoints
    idx = [round(j * (len(kept) - 1) / (max_points - 1)) for j in range(max_points)]
    idx[0] = 0
    idx[-1] = len(kept) - 1
    return [kept[int(i)] for i in sorted(set(idx))]


def build_nav_waypoint_mission_items(
    path_points: list[dict[str, float]],
    origin_north_m: float,
    origin_east_m: float,
    origin_down_m: float,
    home_lat_deg: float,
    home_lon_deg: float,
    acceptance_radius_m: float = 2.0,
) -> list[MissionItem]:
    """Build MissionRaw MissionItem list for MAV_CMD_NAV_WAYPOINT in global relative-alt frame."""
    items: list[MissionItem] = []
    for seq, p in enumerate(path_points):
        n_abs = origin_north_m + p["x"]
        e_abs = origin_east_m + p["y"]
        d_abs = origin_down_m + p["z"]
        lat_deg, lon_deg, rel_alt = ned_from_home_to_wgs84_relative_alt(
            home_lat_deg, home_lon_deg, n_abs, e_abs, d_abs
        )
        lat_i = int(round(lat_deg * 1.0e7))
        lon_i = int(round(lon_deg * 1.0e7))
        heading = float(p["yaw"])
        items.append(
            MissionItem(
                seq,
                MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                MAV_CMD_NAV_WAYPOINT,
                1 if seq == 0 else 0,
                1,
                0.0,
                float(acceptance_radius_m),
                0.0,
                float(heading),
                lat_i,
                lon_i,
                float(rel_alt),
                MAV_MISSION_TYPE_MISSION,
            )
        )
    return items
