"""Ingest: recorded video -> what the cameras measured (docs/architecture-next.md 1.4, 3). The same for a
simulated city's CARLA video and a real camera's: every closed segment of recordings/<date>/<period>/, once,
in time order (table `ingested`).

SAM 3.1 runs on a few frames per segment, not all of them (about 2.5 frames/s on a 12-16 GB consumer GPU):

  samples      at every `pair_every_s` seconds on the clock, the frame then and the frame `pair_gap_s` later.
               Each vehicle is placed on the road (calib: pixel -> ground) and kept if it is on the camera's
               approach (calib.Zone): its lane, metres before the stop line, type. Vehicles of the first frame
               matched in the second give speeds.            -> table vision_samples
  observations per camera and 5-minute slot ending at ts (slot rows, rebuilt from the samples):
     total, counts, pcu, occupancy, load_ratio, level   vehicles on the approach in the sample nearest ts
                                                       (the snapshot v1 counted)
     speed     mean speed of the matched vehicles between the speed lines (as the speed labels), km/h
     flow      vehicles per 5 minutes, by Edie's definition: at each sample, the sum of the speeds of the vehicles
               on the stretch over its length (vehicles per second), averaged over the slot's samples x 300.
               Every vehicle counts at its own matched speed, so a queue at a red light adds about nothing;
               one not found in the second frame counts at the slot's median matched speed
     source    'vision'
A slot with fewer than `min_samples` samples gets no row: a gap, not zero traffic.

A simulated city's cameras are placed by its SUMO network and calibrated from the CARLA pose (calib); a real town's
by its map and the installer's point pairs (realtown: ShapeZone, HomographyCalib), with no SUMO at all. A real
camera without a calibration is skipped (its segments wait until it has one).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .calib import measure_settings
from .simulate import DEFAULT_LEVELS, DEFAULT_PCU, congestion_level

log = logging.getLogger(__name__)

INGEST = {"pair_every_s": 12.0, "pair_gap_s": 0.6, "min_samples": 10, "slot_s": 300}
LENGTH_M = {"car": 4.5, "two_wheeler": 2.0, "bus": 11.0, "truck": 8.0}
MAX_STEP_M = 12.0                 # furthest a matched vehicle moves towards the stop line between the two frames

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested (path TEXT PRIMARY KEY, camera_id TEXT, start TEXT, samples INTEGER,
    seconds REAL, processed_at TEXT);
CREATE TABLE IF NOT EXISTS vision_samples (camera_id TEXT NOT NULL, t TEXT NOT NULL, vehicles TEXT NOT NULL,
    speeds TEXT NOT NULL, PRIMARY KEY (camera_id, t));
CREATE TABLE IF NOT EXISTS observations (
    camera_id TEXT NOT NULL, ts TEXT NOT NULL, image_path TEXT, ts_source TEXT,
    total INTEGER NOT NULL, pcu REAL NOT NULL, occupancy REAL NOT NULL, load_ratio REAL NOT NULL, level TEXT,
    counts_json TEXT, mean_score REAL, processed_at TEXT, period TEXT,
    flow REAL, speed REAL, samples INTEGER, source TEXT NOT NULL,
    PRIMARY KEY (camera_id, ts));
"""
OBS_COLUMNS = ["camera_id", "ts", "image_path", "ts_source", "total", "pcu", "occupancy", "load_ratio", "level",
               "counts_json", "mean_score", "processed_at", "period", "flow", "speed", "samples", "source"]


def ingest_settings(sim: dict) -> dict:
    return {**INGEST, **(sim.get("ingest") or {})}


VIDEO = (".mp4", ".mkv", ".avi", ".mov")


def segments(rec_dir: Path, day: str | None = None) -> list[tuple[Path, str, datetime]]:
    """(path, camera, start) of every recorded segment, in time order."""
    out = []
    for mp4 in Path(rec_dir).glob(f"{day or '*'}/*/*"):
        if mp4.suffix.lower() not in VIDEO:
            continue
        cam, hhmmss = mp4.stem.rsplit("_", 1)
        out.append((mp4, cam, datetime.strptime(f"{mp4.parent.parent.name} {hhmmss}", "%Y-%m-%d %H%M%S")))
    return sorted(out, key=lambda s: (s[2], s[1]))


def _match(a: list[dict], b: list[dict], gap_s: float, band: tuple[float, float] | None = None) -> list[float]:
    """Speeds (km/h) of the vehicles of frame a found again in frame b: same or next lane, a little nearer the
    stop line. Greedy on the smallest jump. With `band` (near, far metres before the stop line) only vehicles
    between the speed lines count, as in the speed labels."""
    if band:
        a = [v for v in a if band[0] <= v["m"] <= band[1]]
    pairs = []
    for i, va in enumerate(a):
        for j, vb in enumerate(b):
            dd = va["m"] - vb["m"]
            if -1.0 <= dd <= MAX_STEP_M and abs(va["lane"] - vb["lane"]) <= 1:
                pairs.append((abs(va["lane"] - vb["lane"]) * 3 + abs(dd - 3.0), i, j, dd))
    used_a, used_b, speeds = set(), set(), []
    for _, i, j, dd in sorted(pairs):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        speeds.append(round(3.6 * max(dd, 0.0) / gap_s, 1))
    return speeds


class Ingest:
    def __init__(self, sim: dict, out_dir: Path, cameras: list[dict], detector=None):
        from .realtown import HomographyCalib, ShapeZone, is_real

        self.sim, self.out_dir, self.cfg = sim, out_dir, ingest_settings(sim)
        self.view_m = float(sim["network"].get("view_m", 60))
        if is_real(sim):                               # the map and the installer's calibration
            from .osm import load_shapes

            shapes = load_shapes(out_dir)
            self.skipped = [c["camera_id"] for c in cameras if not HomographyCalib.ok(c)]
            cameras = [c for c in cameras if HomographyCalib.ok(c)]
            self.zones = {c["camera_id"]: ShapeZone(c, shapes[c["edge"]]) for c in cameras}
            self.calib = HomographyCalib()
            self.quiet_s = 0.0                         # pickup moved only closed files into recordings/
        else:                                          # SUMO's lanes and the CARLA camera pose
            import sumolib

            from .calib import Calib, Zone

            net = sumolib.net.readNet(str(out_dir / "network" / "net.net.xml"))
            self.skipped = []
            self.zones = {c["camera_id"]: Zone(c, net, self.view_m) for c in cameras}
            self.calib = Calib(sim)
            self.quiet_s = 5.0                         # the live recorder writes into recordings/ itself
        self.cams = {c["camera_id"]: c for c in cameras}
        self.band = tuple(sorted(float(x) for x in measure_settings(sim)["speed_lines_m"]))
        self.db = out_dir / "traffic.db"
        with sqlite3.connect(self.db) as c:
            c.executescript(SCHEMA)
        self._detector = detector

    @property
    def detector(self):
        if self._detector is None:
            from .vision import Sam3Detector
            self._detector = Sam3Detector()
        return self._detector

    def on_approach(self, cid: str, bgr: np.ndarray) -> list[dict]:
        cam, zone, out = self.cams[cid], self.zones[cid], []
        for d in self.detector.detect(bgr):
            g = self.calib.to_ground(cam, *d.bottom_center)
            where = zone.locate(*g) if g else None
            if where:
                out.append({"type": d.category, "lane": where[0], "m": round(where[1], 1), "score": round(d.score, 2)})
        return out

    def segment(self, path: Path, cid: str, start: datetime) -> list[tuple]:
        """Samples of one segment: (camera, time, vehicles, speeds). Its length is kept in self.last_seconds."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_seconds = n / fps
        every, gap = float(self.cfg["pair_every_s"]), max(1, round(float(self.cfg["pair_gap_s"]) * fps))
        secs0 = (start - start.replace(hour=0, minute=0, second=0)).total_seconds()
        firsts = [k for k in range(n) if abs((secs0 + k / fps) / every - round((secs0 + k / fps) / every)) * every < 0.5 / fps]
        wanted = set(firsts) | {k + gap for k in firsts if k + gap < n}
        frames, k = {}, 0
        while k <= max(wanted, default=-1):
            ok, img = cap.read()
            if not ok:
                break
            if k in wanted:
                frames[k] = img
            k += 1
        cap.release()
        rows = []
        for k in firsts:
            if k not in frames:
                continue
            a = self.on_approach(cid, frames[k])
            b = self.on_approach(cid, frames[k + gap]) if k + gap in frames else []
            speeds = {"band": _match(a, b, gap / fps, self.band) if b else [],    # between the speed lines: speed
                      "all": _match(a, b, gap / fps) if b else []}               # every vehicle, queued too: flow
            t = start + timedelta(seconds=round(k / fps, 3))
            rows.append((cid, t.isoformat(timespec="seconds"), json.dumps(a), json.dumps(speeds)))
        return rows

    def run(self, day: str | None = None, follow: bool = False, poll_s: float = 10.0, log_fn=print) -> int:
        """Every segment not ingested yet; with `follow`, then wait for new ones (live)."""
        done_total = 0
        while True:
            with sqlite3.connect(self.db) as c:
                done = {r[0] for r in c.execute("SELECT path FROM ingested")}
            todo = [s for s in segments(self.out_dir / "recordings", day)
                    if str(s[0]) not in done and s[1] in self.cams and _closed(s[0], self.quiet_s)]
            for path, cid, start in todo:
                t0 = time.time()
                rows = self.segment(path, cid, start)
                with sqlite3.connect(self.db) as c:
                    c.executemany("INSERT OR REPLACE INTO vision_samples VALUES (?, ?, ?, ?)", rows)
                    c.execute("INSERT OR REPLACE INTO ingested VALUES (?, ?, ?, ?, ?, ?)",
                              (str(path), cid, start.isoformat(), len(rows), round(time.time() - t0, 1),
                               datetime.now().isoformat(timespec="seconds")))
                self.slot_rows(cid, start, start + timedelta(seconds=max(1.0, getattr(self, "last_seconds", 60.0))))
                done_total += 1
                log_fn(f"{cid} {start:%Y-%m-%d %H:%M:%S}: {len(rows)} samples, {time.time() - t0:.0f} s")
            if not follow:
                return done_total
            time.sleep(poll_s)

    def slot_rows(self, cid: str, t0: datetime, t1: datetime):
        """Rebuild the observations of the slots touching [t0, t1] from the samples."""
        slot = int(self.cfg["slot_s"])
        cam = self.cams[cid]
        first = _slot_end(t0, slot)
        last = _slot_end(t1, slot)
        with sqlite3.connect(self.db) as c:
            ts = first
            while ts <= last:
                lo = ts - timedelta(seconds=slot)
                rows = c.execute("SELECT t, vehicles, speeds FROM vision_samples WHERE camera_id = ? AND t > ? AND t <= ?",
                                 (cid, lo.isoformat(timespec="seconds"), ts.isoformat(timespec="seconds"))).fetchall()
                c.execute("DELETE FROM observations WHERE camera_id = ? AND ts = ?", (cid, ts.isoformat(timespec="seconds")))
                if len(rows) >= int(self.cfg["min_samples"]):
                    c.execute("INSERT INTO observations (camera_id, ts, image_path, ts_source, total, pcu, occupancy, "
                              "load_ratio, level, counts_json, mean_score, processed_at, period, flow, speed, samples, "
                              "source) VALUES (?, ?, '', 'video', ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, 'vision')",
                              (cid, ts.isoformat(timespec="seconds"), *self._slot(cam, rows, ts, slot)))
                ts += timedelta(seconds=slot)

    def _slot(self, cam: dict, rows: list, ts: datetime, slot: int) -> tuple:
        samples = [(datetime.fromisoformat(t), json.loads(v), json.loads(s)) for t, v, s in rows]
        return slot_values(cam, samples, ts, slot, self.view_m)


def slot_values(cam: dict, samples: list[tuple], ts: datetime, slot: int, view_m: float) -> tuple:
    """One slot's observation values from its samples (time, vehicles, speeds), as described at the top:
    (total, pcu, occupancy, load_ratio, level, counts_json, mean_score, processed_at, flow, speed, samples).
    Shared by vision (ingest) and SUMO's direct observations (direct.py), so both measure the same way."""
    snap = min(samples, key=lambda x: abs((ts - x[0]).total_seconds()))[1]
    counts = {k: 0 for k in LENGTH_M}
    for v in snap:
        counts[v["type"]] = counts.get(v["type"], 0) + 1
    pcu = sum(n * DEFAULT_PCU.get(k, 1.0) for k, n in counts.items())
    occupied = sum(n * LENGTH_M.get(k, 4.5) for k, n in counts.items())
    ratio = pcu / cam["capacity_pcu"] if cam["capacity_pcu"] else 0.0
    band = [s for _, _, sp in samples for s in sp["band"]]
    mean_speed = float(np.mean(band)) if band else None
    every = [s for _, _, sp in samples for s in sp["all"]]
    fallback = float(np.median(every)) if every else 0.0   # for a vehicle not found in the second frame
    q = []                                               # vehicles per second past a point, per sample
    for _, vehicles, sp in samples:
        matched = sp["all"]
        q.append((sum(matched) + (len(vehicles) - len(matched)) * fallback) / 3.6 / view_m)
    scores = [v["score"] for v in snap if "score" in v]        # direct observations have no detection score
    return (sum(counts.values()), round(pcu, 3),
            round(min(1.0, occupied / (view_m * max(1, len(cam["lanes"])))), 4), round(ratio, 4),
            congestion_level(ratio, DEFAULT_LEVELS), json.dumps(counts), float(np.mean(scores)) if scores else None,
            datetime.now().isoformat(timespec="seconds"),
            round(float(np.mean(q)) * slot, 1), None if mean_speed is None else round(mean_speed, 1), len(samples))


def _slot_end(t: datetime, slot: int) -> datetime:
    s = (t - t.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return t.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=-(-s // slot) * slot)


def _closed(path: Path, quiet_s: float = 5.0) -> bool:
    """A segment is closed once its file has not changed for a few seconds (a live recorder's open segment
    keeps growing)."""
    return time.time() - path.stat().st_mtime > quiet_s
