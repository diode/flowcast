"""Record a city's day (docs/architecture-next.md 1, 2): SUMO moves the traffic, CARLA films every camera.

Only recordings are kept, in the layout a real recorder writes:

  recordings/<date>/<period>/<camera>_<HHMMSS>.mp4          one per camera per minute (`segment_s`): H.264,
                                                            5 fps, 1280 x 720, CCTV text (camera, date, time)
  recordings/<date>/<period>/<camera>_<HHMMSS>.labels.json  with skip_render_read: SUMO's labels for it
  recordings/<date>/incidents.labels.json                   with skip_render_read: the day's incidents

SUMO's labels use the same definitions people label with (docs/architecture-next.md 1.3, 1.5):
  frame_labels     vehicles of each type in the camera's view (view_m of the stop line) on the frame at each
                   5-minute slot
  crossings        vehicles crossing the counting line, `line_m` before the stop line
  speeds           each vehicle's time between the two speed lines (`speed_lines_m` before the stop line)

Between filmed windows SUMO runs alone. Its work folder (trips, outputs) is deleted when the day is done.
The CARLA world is the city's own (carla_world.build_world); CARLA's SUMO bridge drives it (carla_bridge).
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .calib import MEASURE, measure_settings  # noqa: F401  (re-exported)
from .carla_world import (LIGHTS, build_world, camera_blueprint, camera_pose, dress, exposure, headlights_on,
                          overlay, pose_settings, weather)
from .carla_bridge import connect, detach, import_cosim, use_blueprints
from .simulate import DEFAULT_LEVELS, DEFAULT_PCU, CameraMeter, plan_day
from .tools import ffmpeg as ffmpeg_exe

log = logging.getLogger(__name__)

WINDOWS = [{"period": "morning", "start": "07:30", "end": "09:30", "light": "morning"},
           {"period": "evening", "start": "16:30", "end": "18:30", "light": "evening"}]


def light_of(win: dict) -> str:
    """A window's light (carla_world.LIGHTS): its own, else its period's name, else morning."""
    return win.get("light") or (win["period"] if win.get("period") in LIGHTS else "morning")


def _s(hhmm: str) -> int:
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 3600 + int(m) * 60


def recording_settings(sim: dict) -> dict:
    rc = dict(sim.get("recording") or {})
    rc.setdefault("windows", WINDOWS)
    rc.setdefault("fps", 5)
    rc.setdefault("segment_s", 60)
    rc.setdefault("attach_s", 20)                        # CARLA joins this early, so vehicles exist when filming starts
    return rc


class Mp4:
    """An H.264 MP4 written by ffmpeg from BGR frames (plays in any player and browser)."""

    def __init__(self, path: Path, w: int, h: int, fps: int, ffmpeg: str | None = None):
        import subprocess

        exe = ffmpeg_exe(ffmpeg)
        self.proc = subprocess.Popen(
            [exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, bgr: np.ndarray):
        self.proc.stdin.write(bgr.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def camera_labels(sim: dict, cameras: list[dict], net) -> dict[str, str]:
    """On-screen camera names: 'CAM06  Elm Park > Ashford Centre'."""
    names = (sim["network"].get("town") or {}).get("names") or {}

    def place(node: str) -> str:
        return f"{names.get(node[2:], node[2:])} local" if node.startswith("F_") else names.get(node, node)

    out = {}
    for c in cameras:
        e = net.getEdge(c["edge"])
        out[c["camera_id"]] = f"{c['camera_id'].upper()}  {place(e.getFromNode().getID())} > {place(e.getToNode().getID())}"
    return out


class Labeller:
    """SUMO's labels for one camera segment after another (skip_render_read)."""

    def __init__(self, traci, cameras: list[dict], meter: CameraMeter, lane_len: dict, ms: dict):
        self.traci, self.cameras, self.meter, self.lane_len = traci, cameras, meter, lane_len
        self.line = float(ms["line_m"])
        self.near, self.far = sorted(float(x) for x in ms["speed_lines_m"])
        self.prev: dict[str, float] = {}                  # vehicle -> distance to its stop line last tick
        self.at_far: dict[str, float] = {}                # vehicle -> time it passed the far speed line
        self.reset()

    def reset(self):
        self.seg = {c["camera_id"]: {"crossings": [], "speeds": [], "frame_labels": []} for c in self.cameras}

    def tick(self, t: float):
        traci = self.traci
        seen = {}
        for c in self.cameras:
            for ln in c["lanes"]:
                for v in traci.lane.getLastStepVehicleIDs(ln):
                    d = self.lane_len[ln] - traci.vehicle.getLanePosition(v)      # metres to the stop line
                    seen[v] = d
                    before = self.prev.get(v)
                    if before is None:
                        continue
                    vt = traci.vehicle.getTypeID(v)
                    if before > self.far >= d:
                        self.at_far[v] = t
                    if before > self.near >= d and v in self.at_far:
                        dt = t - self.at_far.pop(v)
                        if dt > 0:
                            self.seg[c["camera_id"]]["speeds"].append(
                                {"t": round(t, 2), "type": vt, "kmh": round(3.6 * (self.far - self.near) / dt, 1)})
                    if before > self.line >= d:
                        self.seg[c["camera_id"]]["crossings"].append({"t": round(t, 2), "type": vt})
        self.prev = seen

    def frame(self, t: float, frame_no: int):
        for c in self.cameras:
            counts: dict[str, int] = {}
            for v in self.meter.in_view(c):
                vt = self.traci.vehicle.getTypeID(v)
                counts[vt] = counts.get(vt, 0) + 1
            self.seg[c["camera_id"]]["frame_labels"].append({"t": round(t, 2), "frame": frame_no, "counts": counts})


def _incident_labels(plan: dict, midnight: datetime, car_lanes: dict) -> list[dict]:
    out = []
    for inc in plan["incidents"]:
        if not inc.get("lanes"):
            continue
        edge = inc["lanes"][0].rsplit("_", 1)[0]
        kind = "closure" if inc.get("block", 0) >= car_lanes.get(edge, 1) else \
            "lane_blocked" if inc.get("block", 0) else "slowdown"
        out.append({"edge": edge, "kind": kind,
                    "start": (midnight + timedelta(seconds=inc["start"])).isoformat(timespec="seconds"),
                    "end": (midnight + timedelta(seconds=inc["end"])).isoformat(timespec="seconds")})
    return out


def record_day(day: date, sim: dict, out_dir: Path, cameras: list[dict], windows: list[dict] | None = None,
               port: int = 2000, log_every_s: int = 60, film: list[str] | None = None) -> dict:
    """Film one day's windows. Returns timings and counts. `film`: only these camera ids are filmed (and labelled);
    the day is planned with every camera as always, so its traffic and incidents do not depend on the choice."""
    import sumolib

    carla, SimulationSynchronization, SumoSimulation, CarlaSimulation, BridgeHelper = import_cosim()
    rc, ms = recording_settings(sim), measure_settings(sim)
    pose = pose_settings(sim)
    labels_on = bool((sim.get("source") or {}).get("skip_render_read"))
    fps, seg_s = int(rc["fps"]), int(rc["segment_s"])
    step = 1.0 / fps
    w, h = int(pose["width"]), int(pose["height"])
    wins = sorted(windows or rc["windows"], key=lambda x: _s(x["start"]))
    net_file = out_dir / "network" / "net.net.xml"
    midnight = datetime.combine(day, datetime.min.time())
    day_dir = out_dir / "recordings" / f"{day:%Y-%m-%d}"
    work = out_dir / "work" / f"{day:%Y%m%d}"

    client = carla.Client("localhost", port)
    client.set_timeout(600.0)
    client.get_server_version()
    net = sumolib.net.readNet(str(net_file))
    dress(build_world(client, out_dir / "network" / "net.xodr"), net, seed=int(sim.get("seed", 0)))
    plan = plan_day(day, str(net_file), cameras, sim, str(work))
    use_blueprints(BridgeHelper, np.random.default_rng(plan["seed"]))
    labels = camera_labels(sim, cameras, net)
    car_lanes = {e.getID(): sum(ln.allows("passenger") for ln in e.getLanes()) for e in net.getEdges()}
    view_m = float(sim["network"].get("view_m", 60))
    stats = {"day": day.isoformat(), "segments": 0, "frames": 0, "film_s": 0.0, "windows": []}

    old_cwd = os.getcwd()
    os.chdir(plan["cfg"].parent)                  # CARLA's SumoSimulation resolves the cfg against the cwd
    traci = None
    try:
        sumo = SumoSimulation(str(plan["cfg"].resolve()), step, None, None, False, 1)
        traci = sys.modules["traci"]
        meter = CameraMeter(traci, cameras, plan, view_m, DEFAULT_PCU, DEFAULT_LEVELS)
        filmed = [c for c in cameras if film is None or c["camera_id"] in set(film)]
        labeller = Labeller(traci, filmed, meter, plan["lane_len"], ms) if labels_on else None
        for win in wins:
            start_s, end_s = _s(win["start"]), _s(win["end"])
            _run_alone(traci, meter, start_s - float(rc["attach_s"]))
            csim = connect(CarlaSimulation, "localhost", port, step)
            world = csim.world
            light = light_of(win)
            world.set_weather(weather(light))
            sync = SimulationSynchronization(sumo, csim, "none", False, False)
            BridgeHelper.offset = (0.0, 0.0)          # the world was built from our own net.xodr: no shift
            existing = set(traci.vehicle.getIDList())

            def tick(real=sumo.tick, existing=existing):   # vehicles already on the road never "depart" again
                real()
                sumo.spawned_actors |= existing & set(traci.vehicle.getIDList())   # not those that left this step
                existing.clear()

            sumo.tick = tick
            t0 = datetime.now()
            try:
                n = _film(client, world, sync, traci, meter, labeller, filmed, labels, pose, exposure(sim, light),
                          light == "night", day_dir / win["period"], midnight, start_s, end_s, step, seg_s, w, h,
                          rc.get("ffmpeg"), log_every_s)
            finally:
                del sumo.tick
                detach(sync, traci)
            secs = (datetime.now() - t0).total_seconds()
            stats["segments"] += n["segments"]
            stats["frames"] += n["frames"]
            stats["film_s"] += secs
            stats["windows"].append({**win, "wall_s": round(secs), "x_real_time": round((end_s - start_s) / secs, 2)})
        if labels_on:
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "incidents.labels.json").write_text(
                json.dumps(_incident_labels(plan, midnight, car_lanes), indent=1), encoding="utf-8")
    finally:
        if traci is not None:
            try:
                traci.close()
            except Exception:  # noqa: BLE001 - SUMO already gone
                pass
        os.chdir(old_cwd)
        shutil.rmtree(work, ignore_errors=True)     # only recordings are kept
    return stats


def _run_alone(traci, meter: CameraMeter, until: float):
    """SUMO without CARLA up to `until`, applying incidents minute by minute."""
    t = traci.simulation.getTime()
    while t < until - 1e-6:
        t = min(until, (math.floor(t / 60) + 1) * 60)
        traci.simulationStep(float(t))
        meter.apply_incidents(t)


def _film(client, world, sync, traci, meter, labeller, cameras, labels, pose, exposure_attrs, night: bool,
          video_dir: Path, midnight, start_s, end_s, step, seg_s, w, h, ffmpeg, log_every_s) -> dict:
    bp = camera_blueprint(world, pose, exposure_attrs)
    video_dir.mkdir(parents=True, exist_ok=True)
    sensors, queues, writers = {}, {}, {}
    seg_start, k, n_segments, n_frames = None, 0, 0, 0
    per_slot = int(round(300 / step))

    def close_segment():
        nonlocal n_segments
        for cid, wr in writers.items():
            wr.release()
            if labeller is not None:
                c = next(c for c in cameras if c["camera_id"] == cid)
                (video_dir / f"{cid}_{seg_start:%H%M%S}.labels.json").write_text(json.dumps(
                    {"camera_id": cid, "edge": c["edge"], "segment_start": seg_start.isoformat(),
                     "frames": k, "fps": round(1 / step), **labeller.seg[cid]}), encoding="utf-8")
        n_segments += len(writers)
        writers.clear()
        if labeller is not None:
            labeller.reset()

    try:
        while traci.simulation.getTime() < end_s - step / 2:
            if not sensors and traci.simulation.getTime() + step >= start_s - step / 2:
                for c in cameras:
                    q: queue.Queue = queue.Queue()
                    s = world.spawn_actor(bp, camera_pose(c, pose))
                    s.listen(q.put)
                    sensors[c["camera_id"]], queues[c["camera_id"]] = s, q
            sync.tick()
            t = round(traci.simulation.getTime(), 3)
            if night and int(round(t / step)) % round(1 / step) == 0:      # new vehicles get their lights each second
                headlights_on(client, world)
            if labeller is not None:
                labeller.tick(t)
            if int(round(t / step)) % int(60 / step) == 0:
                meter.apply_incidents(t)
            if int(round(t / step)) % int(log_every_s / step) == 0:
                log.info("%s: %d vehicles in CARLA", (midnight + timedelta(seconds=t)).time(), len(sync.sumo2carla_ids))
            if not sensors or t < start_s - step / 2:
                continue
            if int(round(t / step)) % int(round(seg_s / step)) == 0 or seg_start is None:
                if writers:
                    close_segment()
                seg_start, k = midnight + timedelta(seconds=round(t)), 0
                for cid in sensors:
                    writers[cid] = Mp4(video_dir / f"{cid}_{seg_start:%H%M%S}.mp4", w, h, round(1 / step), ffmpeg)
            frame_no = world.get_snapshot().frame
            now = midnight + timedelta(seconds=t)
            for cid, q in queues.items():
                image = q.get(timeout=60.0)
                while image.frame < frame_no:
                    image = q.get(timeout=60.0)
                bgr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
                writers[cid].write(overlay(bgr, labels[cid], now))
            if labeller is not None and int(round(t / step)) % per_slot == 0:
                labeller.frame(t, k)
            k += 1
            n_frames += 1
        if writers:
            close_segment()
    finally:
        for s in sensors.values():
            s.stop()
            s.destroy()
        for wr in writers.values():
            wr.release()
    return {"segments": n_segments, "frames": n_frames}
