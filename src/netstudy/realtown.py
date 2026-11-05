"""A real town: its map, its cameras and their calibration, with no SUMO or CARLA (netstudy onboard).

Town config (towns/<town>.yaml, `source.kind: real`, `network.source: osm`):

  network.osm.file / bbox      the OpenStreetMap extract (relative to the config), or a box to fetch it for
  cameras:                     one entry per camera, as the installer measured it
    - id: cam01
      name: High Street northbound at Mill Lane
      at: {lat: 51.5012, lon: -0.1203}     # a point on the road it watches, near the stop line
      heading: 15                          # optional: the traffic's direction, degrees from north
      # edge: J3_J1                        # or the edge id from map.png, instead of at / heading
      stop_m: 8                            # stop line this far before the junction centre (default 8)
      calibration:                         # 4 or more points seen in the image and found on the map
        image_size: [1920, 1080]
        points:
          - {px: [412, 980], lat: 51.50118, lon: -0.12035}
          - ...

  onboard(sim, out_dir)        map -> nodes / edges / shapes / names (osm.py), then place_cameras, then map.png
  place_cameras(sim, out_dir)  cameras.json (the same fields as a simulated town's) and edges.camera_id
  HomographyCalib              pixel <-> road point from calibration.H (the ground-plane homography)
  ShapeZone                    the camera's lanes along the edge's shape, up to view_m before its stop line
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from . import osm

LANE_W = 3.2
STOP_M = 8.0                     # OSM road lines end at the junction centre; the stop line is short of it


def is_real(sim: dict) -> bool:
    return (sim.get("source") or {}).get("kind") == "real" or (sim.get("network") or {}).get("source") == "osm"


def config_dir(sim: dict) -> Path:
    return Path(sim.get("_config_dir") or ".")


def osm_file(sim: dict, out_dir: Path) -> Path:
    f = ((sim.get("network") or {}).get("osm") or {}).get("file")
    if not f:
        return out_dir / "network" / "map.osm"
    p = Path(f)
    return p if p.is_absolute() else config_dir(sim) / p


def onboard(sim: dict, out_dir: Path, fetch: bool = False, log=print) -> dict:
    """Map, cameras and map.png of a real town; the map is fetched (Overpass) when asked or when the file is
    missing and a bbox is set."""
    net_cfg = sim.get("network") or {}
    cfg = osm.settings(net_cfg)
    f = osm_file(sim, out_dir)
    if fetch or not f.exists():
        if not cfg["bbox"]:
            raise SystemExit(f"no map: put the OpenStreetMap extract at {f} (network.osm.file), or set network.osm.bbox "
                             "[south, west, north, east] and run with --fetch")
        log(f"fetching the roads in {cfg['bbox']} from the Overpass API -> {f}")
        osm.fetch(cfg["bbox"], f, cfg["highways"])
    town = osm.build(f, net_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    osm.save(town, out_dir)
    j = town["nodes"]
    log(f"map {f.name}: {int((j.kind == 'junction').sum())} junctions ({int(j.signal.sum())} signalised), "
        f"{int((j.kind == 'boundary').sum())} boundary points, {len(town['edges'])} road directions")
    cams = place_cameras(sim, out_dir) if sim.get("cameras") else []
    if cams:
        log(f"cameras: {len(cams)} placed, {sum(1 for c in cams if c.get('calibration'))} calibrated")
    title = sim.get("name") or out_dir.name.replace("_", " ").title()
    png = osm.draw(town, cams, title, out_dir / "map.png")
    log(f"map for review: {png}")
    return town


def _project(shape: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """(distance from the line, metres along it, metres to its right) of point p against polyline `shape`."""
    best = None
    s0 = 0.0
    for a, b in zip(shape, shape[1:]):
        d = b - a
        seg = float(np.hypot(*d))
        if seg < 1e-9:
            continue
        t = float(np.clip((p - a) @ d / seg ** 2, 0.0, 1.0))
        q = a + t * d
        dist = float(np.hypot(*(p - q)))
        right = float((d[1] * (p - a)[0] - d[0] * (p - a)[1]) / seg)      # + right of the driving direction
        if best is None or dist < best[0]:
            best = (dist, s0 + t * seg, right)
        s0 += seg
    return best


def _point_at(shape: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """The point `s` metres along the polyline and the unit direction there."""
    s0 = 0.0
    for a, b in zip(shape, shape[1:]):
        seg = float(np.hypot(*(b - a)))
        if seg > 1e-9 and s <= s0 + seg:
            u = (b - a) / seg
            return a + u * (s - s0), u
        s0 += seg
    a, b = shape[-2], shape[-1]
    return b.copy(), (b - a) / max(float(np.hypot(*(b - a))), 1e-9)


def _length(shape: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(shape, axis=0).T)))


def _pick_edge(edges, shapes: dict, p: np.ndarray, heading: float | None) -> str:
    """The edge a camera at road point p watches: the nearest road line; of its two directions the one going the
    way of `heading` (degrees from north), else the one that ends nearer p (traffic coming up to the junction)."""
    cand = []
    for eid in edges.edge_id:
        sh = np.asarray(shapes[eid], dtype=float)
        dist, s, _ = _project(sh, p)
        _, u = _point_at(sh, s)
        cand.append((dist, eid, s, _length(sh), u))
    near = min(c[0] for c in cand)
    close = [c for c in cand if c[0] <= near + 2.0]
    if heading is not None:
        h = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        return max(close, key=lambda c: float(c[4] @ h))[1]
    return min(close, key=lambda c: c[3] - c[2])[1]


def lateral(edge_id: str, lanes: int, edges, lefthand: bool) -> tuple[float, float]:
    """Metres right of the edge's line its lanes cover: the right-hand half of a two-way road (the left in
    left-hand traffic), the whole width of a one-way road."""
    e = edges.set_index("edge_id")
    a, b = e.at[edge_id, "from_node"], e.at[edge_id, "to_node"]
    two_way = ((e.from_node == b) & (e.to_node == a)).any()
    w = lanes * LANE_W
    if not two_way:
        return -w / 2, w / 2
    return (-w, 0.0) if lefthand else (0.0, w)


def homography(points: list[dict], centre: dict) -> tuple[list, float]:
    """(H pixel -> ground, rms reprojection error in metres) from >= 4 {px: [u, v], lat, lon | x, y} pairs."""
    import cv2

    if len(points) < 4:
        raise ValueError(f"{len(points)} calibration points; 4 or more are needed")
    src = np.array([p["px"] for p in points], dtype=float)
    dst = np.array([[p["x"], p["y"]] if "x" in p else osm.to_xy(centre, p["lat"], p["lon"]) for p in points],
                   dtype=float)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("the calibration points do not give a ground plane (are 3 of them on one line?)")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H.tolist(), float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))


def place_cameras(sim: dict, out_dir: Path) -> list[dict]:
    """cameras.json from the config's cameras (see the module doc), and edges.camera_id in traffic.db / graph.json."""
    import pandas as pd

    from .network import save_cameras

    with sqlite3.connect(out_dir / "traffic.db") as c:
        edges = pd.read_sql("SELECT * FROM edges", c)
        nodes = pd.read_sql("SELECT * FROM nodes", c)
    shapes = {k: np.asarray(v, dtype=float) for k, v in osm.load_shapes(out_dir).items()}
    ctr = osm.centre(out_dir)
    net_cfg = sim.get("network") or {}
    view_m = float(net_cfg.get("view_m", 60))
    lefthand = bool(net_cfg.get("lefthand", False))
    nxy = {r.node_id: (r.x, r.y) for r in nodes.itertuples()}
    cams, seen = [], set()
    for k, cfg in enumerate(sim.get("cameras") or [], start=1):
        cid = str(cfg.get("id") or f"cam{k:02d}")
        if cfg.get("edge"):
            eid = str(cfg["edge"])
            if eid not in shapes:
                raise SystemExit(f"cameras.{cid}: no edge {eid!r} (see map.png / the edges table)")
        elif cfg.get("at"):
            at = cfg["at"]
            p = np.array(osm.to_xy(ctr, at["lat"], at["lon"]) if "lat" in at else (at["x"], at["y"]), dtype=float)
            eid = _pick_edge(edges, shapes, p, cfg.get("heading"))
        else:
            raise SystemExit(f"cameras.{cid}: give `at` (lat / lon on the road) or `edge`")
        if eid in seen:
            raise SystemExit(f"cameras.{cid}: edge {eid} already has a camera (one camera per road direction)")
        seen.add(eid)
        e = edges.set_index("edge_id").loc[eid]
        sh = shapes[eid]
        L = _length(sh)
        stop_s = max(0.0, L - float(cfg.get("stop_m", STOP_M)))
        stop, u = _point_at(sh, stop_s)
        n = int(cfg.get("lanes", e.lanes))
        lo, hi = lateral(eid, n, edges, lefthand)
        cam = {"camera_id": cid, "name": cfg.get("name") or f"{e.to_node} - approach {eid}", "junction": e.to_node,
               "edge": eid, "lanes": [f"{eid}_{i}" for i in range(n)],
               "x": nxy[e.to_node][0], "y": nxy[e.to_node][1],
               "stop": [round(float(stop[0]), 2), round(float(stop[1]), 2)],
               "upstream": [round(float(-u[0]), 4), round(float(-u[1]), 4)], "stop_s": round(stop_s, 2),
               "lateral_m": [lo, hi], "view_m": float(cfg.get("view_m", view_m)),
               "capacity_pcu": round(n * min(float(cfg.get("view_m", view_m)), stop_s) / 6.0, 1),
               "source": {k: cfg[k] for k in ("stream", "external_id", "notes") if k in cfg}, "calibration": None}
        cal = cfg.get("calibration") or {}
        if cal.get("points"):
            try:
                H, rms = homography(cal["points"], ctr)
                cam["calibration"] = {"H": H, "rms_m": round(rms, 2), "points": cal["points"],
                                      "image_size": cal.get("image_size")}
            except ValueError as err:
                cam["calibration_error"] = str(err)
        cams.append(cam)
    save_cameras(cams, out_dir / "cameras.json")
    cam_of = {c["edge"]: c["camera_id"] for c in cams}
    edges["camera_id"] = edges.edge_id.map(cam_of)
    from .graph import save_graph
    save_graph(out_dir / "traffic.db", out_dir / "graph.json", nodes, edges)
    return cams


class HomographyCalib:
    """Real cameras: the road plane seen through each camera's calibration.H (pixel -> town metres)."""

    def __init__(self):
        self._inv: dict[str, np.ndarray] = {}

    @staticmethod
    def ok(cam: dict) -> bool:
        return bool((cam.get("calibration") or {}).get("H"))

    def to_ground(self, cam: dict, u: float, v: float):
        H = np.asarray(cam["calibration"]["H"], dtype=float)
        g = H @ np.array([u, v, 1.0])
        if abs(g[2]) < 1e-12:
            return None
        return float(g[0] / g[2]), float(g[1] / g[2])

    def to_pixel(self, cam: dict, x: float, y: float):
        cid = cam["camera_id"]
        if cid not in self._inv:
            self._inv[cid] = np.linalg.inv(np.asarray(cam["calibration"]["H"], dtype=float))
        p = self._inv[cid] @ np.array([x, y, 1.0])
        if p[2] <= 1e-12:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])


class ShapeZone:
    """A real camera's approach: its lanes along the edge's map shape, the last `view_m` metres before the stop
    line (the same answers as calib.Zone, without SUMO's lane shapes)."""

    def __init__(self, cam: dict, shape, view_m: float | None = None):
        self.shape = np.asarray(shape, dtype=float)
        self.n = len(cam["lanes"])
        self.lo, self.hi = cam["lateral_m"]
        self.stop_s = float(cam["stop_s"])
        self.view_m = float(view_m if view_m is not None else cam.get("view_m", 60))

    def locate(self, x: float, y: float, margin: float = 0.6):
        _, s, r = _project(self.shape, np.array([x, y], dtype=float))
        if not (self.lo - margin <= r <= self.hi + margin):
            return None
        m = self.stop_s - s
        if not (-2.0 <= m <= self.view_m):
            return None
        return int(np.clip((r - self.lo) // LANE_W, 0, self.n - 1)), float(m)

    def outline(self) -> list[tuple[float, float]]:
        side = []
        for off in (self.hi, self.lo):
            pts = []
            for s in np.linspace(max(0.0, self.stop_s - self.view_m), self.stop_s, 25):
                p, u = _point_at(self.shape, s)
                pts.append(tuple(p + off * np.array([u[1], -u[0]])))          # right of the driving direction
            side.append(pts)
        return side[0] + side[1][::-1]
