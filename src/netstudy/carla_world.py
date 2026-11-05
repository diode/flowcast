"""A city's CARLA world, built from its OpenDRIVE file, dressed as a street, and its CCTV cameras.

  build_world      CARLA builds the roads of network/net.xodr (plain roads, no buildings), without its signals:
                   SUMO runs them, and CARLA's signal poles stand in front of the cameras. (Destroying the
                   traffic-light actors afterwards crashed the server.)
  dress            lane markings, stop lines and kerb lines (drawn on the road), trees, bus stops, benches
                   and bins along the kerbs: CARLA's OpenDRIVE roads have none of these
  camera_pose      a CCTV pole at the kerb by the junction corner, looking back down its approach at the
                   oncoming traffic
  to_carla         SUMO (x, y) -> CARLA (x, y); a world built from our own net.xodr needs no offset

Pose settings come from the city config's `recording` section (defaults in POSE).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

POSE = {"camera_height_m": 10.0, "camera_kerb_m": 2.0, "camera_ahead_m": 4.0, "camera_aim_m": 18.0,
        "fov": 90, "width": 1280, "height": 720,         # framed like v1's Town05 recordings
        # a CCTV camera, not a film camera: no glow round the painted lines, no flare, no motion blur, and
        # auto-exposure held lower (the grey ground beyond the verge otherwise drives it to over-expose)
        "exposure_compensation": -1.0, "bloom_intensity": 0.0, "lens_flare_intensity": 0.0,
        "motion_blur_intensity": 0.0}
CAMERA_ATTRS = ("exposure_compensation", "bloom_intensity", "lens_flare_intensity", "motion_blur_intensity")

# Light per time of day (a window's `light`): CARLA weather settings. Exposure per light is calibrated by
# scripts/check_carla_world.py --calibrate and kept in the city config (recording.exposure).
LIGHTS = {
    "morning": {"sun_altitude_angle": 18.0, "sun_azimuth_angle": 100.0, "cloudiness": 20.0, "fog_density": 3.0},
    "noon": {"sun_altitude_angle": 65.0, "sun_azimuth_angle": 180.0, "cloudiness": 25.0},
    "evening": {"sun_altitude_angle": 6.0, "sun_azimuth_angle": 260.0, "cloudiness": 30.0},
    "night": {"sun_altitude_angle": -30.0, "sun_azimuth_angle": 0.0, "cloudiness": 10.0},
}
# Camera exposure per light. Day: auto-exposure with a compensation. Night: a fixed exposure for every camera
# (exposure_min_bright = exposure_max_bright, EV): auto-exposure will not brighten a dark street, and a camera
# that sees many painted lines (drawn at full brightness, day or night) darkened its whole picture to suit them.
EXPOSURE = {"morning": {"exposure_compensation": -0.5}, "noon": {"exposure_compensation": -0.5},
            "evening": {"exposure_compensation": 1.5},
            "night": {"exposure_compensation": 0.0, "exposure_min_bright": 3.0, "exposure_max_bright": 3.0}}
TARGET_GREY = {"morning": 100, "noon": 100, "evening": 95, "night": 80}      # v1's Town05 day video: ~98


def weather(light: str):
    import carla

    w = carla.WeatherParameters()
    for k, v in LIGHTS[light].items():
        setattr(w, k, v)
    return w


def exposure(sim: dict, light: str) -> dict:
    """Camera attributes for this light: the default, overridden by the city config's recording.exposure."""
    return {**EXPOSURE[light], **(((sim.get("recording") or {}).get("exposure") or {}).get(light) or {})}


def headlights_on(client, world):
    """Night: every vehicle drives with its lights on (SUMO does not switch them). One batch, so a vehicle
    that has just left the road does no harm."""
    import carla

    state = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
    client.apply_batch([carla.command.SetVehicleLightState(v.id, state) for v in world.get_actors().filter("vehicle.*")])
LANE_W = 3.2


def pose_settings(sim: dict) -> dict:
    return {**POSE, **{k: v for k, v in (sim.get("recording") or {}).items() if k in POSE}}


def to_carla(x: float, y: float) -> tuple[float, float]:
    return x, -y                                         # CARLA's y axis points the other way


def build_world(client, xodr: Path, keep_traffic_lights: bool = False):
    import carla

    text = Path(xodr).read_text(encoding="utf-8")
    if not keep_traffic_lights:
        text = re.sub(r"\s*<signals>.*?</signals>", "", text, flags=re.S)
        text = re.sub(r"\s*<controller\b.*?</controller>", "", text, flags=re.S)
    return client.generate_opendrive_world(
        text, carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
                                                  additional_width=0.6, smooth_junctions=True,
                                                  enable_mesh_visibility=True))


def _offset(shape, d: float) -> np.ndarray:
    """A polyline moved `d` metres to its right (SUMO axes)."""
    p = np.asarray(shape, dtype=float)[:, :2]
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return p + d * np.column_stack([t[:, 1], -t[:, 0]])


def _resample(p: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    at = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(at, s, p[:, 0]), np.interp(at, s, p[:, 1])])


def dress(world, net, seed: int = 0) -> list:
    """Markings and roadside props for every edge. Returns the spawned props (markings are persistent
    debug lines, gone with the world)."""
    import carla

    rng = np.random.default_rng(seed)
    lib = world.get_blueprint_library()
    trees = [lib.find(b) for b in ("static.prop.aporosatree", "static.prop.cypresstree")]
    street = [lib.find(b) for b in ("static.prop.bench01", "static.prop.bin", "static.prop.kiosk_01",
                                    "static.prop.streetsign")]
    stop = lib.find("static.prop.busstop")
    white, yellow = carla.Color(235, 235, 235), carla.Color(230, 190, 40)
    props = []

    def line(p, colour, thick=0.15, dash=None):
        z = 0.03
        pts = _resample(p, 1.0)
        for i in range(len(pts) - 1):
            if dash and (i // dash) % 2:
                continue
            a, b = to_carla(*pts[i]), to_carla(*pts[i + 1])
            world.debug.draw_line(carla.Location(a[0], a[1], z), carla.Location(b[0], b[1], z), thickness=thick,
                                  color=colour, life_time=0.0, persistent_lines=True)

    def spawn(bp, x, y, yaw):
        cx, cy = to_carla(x, y)
        a = world.try_spawn_actor(bp, carla.Transform(carla.Location(cx, cy, 0.0), carla.Rotation(yaw=yaw)))
        if a:
            props.append(a)

    for e in net.getEdges():
        lanes = e.getLanes()                                # index 0 = the kerb lane (right-hand traffic)
        if not lanes:
            continue
        n = len(lanes)
        half = lanes[0].getWidth() / 2
        line(_offset(lanes[-1].getShape(), -half + 0.12), yellow, 0.12)            # centre line
        for k in range(1, n):
            line(_offset(lanes[k].getShape(), half), white, 0.12, dash=3)           # lane separators
        kerb_shape = lanes[0].getShape()
        line(_offset(kerb_shape, half - 0.2), white, 0.15)                          # road edge
        end = np.asarray(kerb_shape[-1][:2])
        tvec = end - np.asarray(kerb_shape[-2][:2])
        tvec /= np.linalg.norm(tvec) or 1
        right = np.array([tvec[1], -tvec[0]])
        if e.getToNode().getType() == "traffic_light":                              # stop line
            a = end - tvec * 1.0 + right * half
            line(np.array([a, a - right * n * 2 * half]), white, 0.45)
        side = _resample(_offset(kerb_shape, half + 4.0), 1.0)                      # verge, 4 m off the kerb
        if len(side) < 60:
            continue
        yaw = -math.degrees(math.atan2(tvec[1], tvec[0]))
        for i in range(25, len(side) - 25, 22):                                    # a tree every ~22 m
            spawn(trees[rng.integers(len(trees))], *(side[i] + right * rng.uniform(-0.5, 0.5)), rng.uniform(0, 360))
            if rng.random() < 0.25:
                spawn(street[rng.integers(len(street))], *(side[i + 6] - right * 1.5), yaw)
        if len(side) > 200:                                                         # bus stop half-way
            mid = side[len(side) // 2] - right * 1.0
            spawn(stop, *mid, yaw + 180)
    return props


def camera_pose(cam: dict, pose: dict):
    """A CCTV pole on the kerb at the junction corner: `camera_ahead_m` past the stop line, `camera_kerb_m`
    beyond the kerb, `camera_height_m` up, looking at the middle of the approach `camera_aim_m` upstream."""
    import carla

    up = np.array(cam["upstream"], dtype=float)                 # SUMO axes, junction -> upstream
    travel = -up
    right = np.array([travel[1], -travel[0]])
    stop = np.array(cam["stop"], dtype=float)                   # end of the edge's shape: the middle of its lanes
    half_width = len(cam["lanes"]) * LANE_W / 2
    pos = stop + travel * float(pose["camera_ahead_m"]) + right * (half_width + float(pose["camera_kerb_m"]))
    aim = stop + up * float(pose["camera_aim_m"])
    (px, py), (ax, ay) = to_carla(*pos), to_carla(*aim)
    h = float(pose["camera_height_m"])
    dist = math.hypot(ax - px, ay - py)
    return carla.Transform(carla.Location(x=px, y=py, z=h),
                           carla.Rotation(pitch=-math.degrees(math.atan2(h, dist)),
                                          yaw=math.degrees(math.atan2(ay - py, ax - px))))


def camera_blueprint(world, pose: dict, exposure_attrs: dict | None = None):
    """CARLA's RGB camera as a CCTV camera; `exposure_attrs` (from exposure()) set the light's exposure."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(pose["width"]))
    bp.set_attribute("image_size_y", str(pose["height"]))
    bp.set_attribute("fov", str(pose["fov"]))
    attrs = {**{k: pose[k] for k in CAMERA_ATTRS}, **(exposure_attrs or {})}
    for k, v in attrs.items():
        if bp.has_attribute(k):
            bp.set_attribute(k, str(v))
    return bp


def overlay(bgr: np.ndarray, cam_label: str, when) -> np.ndarray:
    """CCTV on-screen text: camera name top left, date and time top right."""
    import cv2

    h, w = bgr.shape[:2]
    stamp = when.strftime("%Y-%m-%d  %H:%M:%S")
    for text, org in ((cam_label, (14, 30)), (stamp, (w - 14 - 11 * len(stamp), 30))):
        cv2.putText(bgr, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr
