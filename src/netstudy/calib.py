"""Camera calibration: where each camera stands and what part of the road it measures. Setup data, as an
installer measures for a real camera (docs/architecture-next.md 1.5): for a simulated city it follows from the
camera pose the recorder uses (carla_world.camera_pose), with no CARLA needed here.

  Calib.pose(cam)          position (CARLA axes, metres) and pitch / yaw (degrees)
  Calib.to_ground(cam, u, v)   a pixel -> the road point under it (SUMO axes), or None above the horizon
  Calib.to_pixel(cam, x, y)    a road point (SUMO axes) -> pixel
  Zone                     the camera's approach lanes up to view_m from the stop line: is a road point on
                           them, on which lane, and how far before the stop line
"""
from __future__ import annotations

import math

import numpy as np

from .carla_world import LANE_W, pose_settings

# Speed lines: a camera's speed is measured between these distances before its stop line (as the speed labels).
MEASURE = {"line_m": 5.0, "speed_lines_m": [5.0, 35.0]}


def measure_settings(sim: dict) -> dict:
    return {**MEASURE, **(sim.get("measure") or {})}


def pose_values(cam: dict, pose: dict) -> tuple[np.ndarray, float, float]:
    """(position in CARLA axes, pitch, yaw) of carla_world.camera_pose, as plain numbers."""
    up = np.array(cam["upstream"], dtype=float)
    travel = -up
    right = np.array([travel[1], -travel[0]])
    stop = np.array(cam["stop"], dtype=float)
    half_width = len(cam["lanes"]) * LANE_W / 2
    pos = stop + travel * float(pose["camera_ahead_m"]) + right * (half_width + float(pose["camera_kerb_m"]))
    aim = stop + up * float(pose["camera_aim_m"])
    (px, py), (ax, ay) = (pos[0], -pos[1]), (aim[0], -aim[1])
    h = float(pose["camera_height_m"])
    pitch = -math.degrees(math.atan2(h, math.hypot(ax - px, ay - py)))
    yaw = math.degrees(math.atan2(ay - py, ax - px))
    return np.array([px, py, h]), pitch, yaw


class Calib:
    def __init__(self, sim: dict):
        self.pose = pose_settings(sim)
        self.w, self.h = int(self.pose["width"]), int(self.pose["height"])
        self.f = self.w / (2 * math.tan(math.radians(float(self.pose["fov"])) / 2))
        self._cache: dict[str, tuple] = {}

    def axes(self, cam: dict):
        """Camera position and its forward, right and up vectors (CARLA axes: x forward, y right, z up)."""
        cid = cam["camera_id"]
        if cid not in self._cache:
            pos, pitch, yaw = pose_values(cam, self.pose)
            p, y = math.radians(pitch), math.radians(yaw)
            fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
            right = np.array([-math.sin(y), math.cos(y), 0.0])
            up = np.cross(fwd, right)                       # forward (1,0,0) x right (0,1,0) = up (0,0,1)
            self._cache[cid] = (pos, fwd, right, up)
        return self._cache[cid]

    def to_ground(self, cam: dict, u: float, v: float):
        pos, fwd, right, up = self.axes(cam)
        d = fwd + right * ((u - self.w / 2) / self.f) - up * ((v - self.h / 2) / self.f)
        if d[2] >= -1e-6:
            return None
        t = -pos[2] / d[2]
        g = pos + t * d
        return float(g[0]), float(-g[1])                     # back to SUMO axes

    def to_pixel(self, cam: dict, x: float, y: float):
        pos, fwd, right, up = self.axes(cam)
        c = np.array([x, -y, 0.0]) - pos
        depth = c @ fwd
        if depth <= 0.5:
            return None
        return self.w / 2 + self.f * (c @ right) / depth, self.h / 2 - self.f * (c @ up) / depth


class Zone:
    """A camera's approach lanes, the last `view_m` metres before the stop line."""

    def __init__(self, cam: dict, net, view_m: float):
        self.lanes = [net.getLane(ln) for ln in cam["lanes"]]
        self.view_m = view_m

    def locate(self, x: float, y: float, margin: float = 0.6):
        """(lane index, metres before the stop line) of a road point on the approach, else None."""
        from sumolib.geomhelper import distancePointToPolygon, polygonOffsetWithMinimumDistanceToPoint

        best = None
        for i, ln in enumerate(self.lanes):
            shape = ln.getShape()
            d = distancePointToPolygon((x, y), shape)
            if d <= ln.getWidth() / 2 + margin and (best is None or d < best[0]):
                best = (d, i, ln.getLength() - polygonOffsetWithMinimumDistanceToPoint((x, y), shape))
        if best is None or not (-2.0 <= best[2] <= self.view_m):
            return None
        return best[1], best[2]

    def outline(self) -> list[tuple[float, float]]:
        """The zone's border on the road (SUMO axes), for drawing."""
        from .carla_world import _offset, _resample

        left = _offset(self.lanes[-1].getShape(), -self.lanes[-1].getWidth() / 2)
        right = _offset(self.lanes[0].getShape(), self.lanes[0].getWidth() / 2)

        def tail(p):
            p = _resample(p, 1.0)
            return p[-int(self.view_m):] if len(p) > self.view_m else p

        return [tuple(q) for q in tail(right)] + [tuple(q) for q in tail(left)[::-1]]
