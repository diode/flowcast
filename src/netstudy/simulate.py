"""Headless SUMO runs over many days: what every camera sees, every sample slot.

Each day gets a plan seeded by its date (trips from zones.py, incidents), so the same day always
replays the same traffic. Every `sample_minutes` each camera records vehicles by type within
`view_m` of its stop line, plus the true flow from induction loops. Random incidents slow a road to
walking pace for 20-60 minutes. A fixed incident list (phase 2) can replace them; such an incident
can also block lanes with a stopped vehicle, which others must pass. Days run in parallel processes.

Rows use the reference project's `observations` table layout, so its tools can read them.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import pandas as pd

from .network import ensure_sumo, sumo_binary
from .zones import write_zone_trips

log = logging.getLogger(__name__)

CATEGORIES = ["car", "two_wheeler", "auto_rickshaw", "bus", "truck"]

# Vehicle types share the category names, so SUMO type id == category.
# Two-wheelers and autos weave between lanes (sublane model, `latAlignment="arbitrary"`).
VTYPES = {
    "car": 'vClass="passenger" length="4.3" width="1.8" maxSpeed="16.7" minGap="2.0"',
    "two_wheeler": 'vClass="motorcycle" length="2.0" width="0.8" maxSpeed="16.7" minGap="1.0" '
                   'latAlignment="arbitrary" minGapLat="0.3"',
    "auto_rickshaw": 'vClass="passenger" length="2.9" width="1.4" maxSpeed="12.5" accel="1.5" minGap="1.2" '
                     'latAlignment="arbitrary" minGapLat="0.4"',
    "bus": 'vClass="bus" length="11.0" width="2.5" maxSpeed="13.9" accel="1.0" minGap="2.5"',
    "truck": 'vClass="truck" length="8.0" width="2.5" maxSpeed="13.9" accel="1.0" minGap="2.5"',
}

DEFAULT_PCU = {"car": 1.0, "two_wheeler": 0.5, "auto_rickshaw": 1.2, "bus": 3.0, "truck": 3.0}
DEFAULT_LEVELS = [("free", 0.35), ("moderate", 0.65), ("heavy", 0.9), ("jammed", float("inf"))]

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    camera_id   TEXT NOT NULL,
    ts          TEXT NOT NULL,
    image_path  TEXT,
    ts_source   TEXT,
    total       INTEGER NOT NULL,
    pcu         REAL NOT NULL,
    occupancy   REAL NOT NULL,
    load_ratio  REAL NOT NULL,
    level       TEXT,
    counts_json TEXT,
    mean_score  REAL,
    processed_at TEXT,
    period      TEXT,
    PRIMARY KEY (camera_id, ts)
);
"""
COLUMNS = ["camera_id", "ts", "image_path", "ts_source", "total", "pcu", "occupancy", "load_ratio",
           "level", "counts_json", "mean_score", "processed_at"]

# The true state of every edge per slot (sim only); same columns as edge_state plus veh_h.
TRUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_truth (
    edge_id     TEXT NOT NULL,
    ts          TEXT NOT NULL,
    total       INTEGER NOT NULL,   -- vehicles on the whole edge at slot end
    pcu         REAL NOT NULL,
    occupancy   REAL NOT NULL,      -- vehicle length / (lanes x length)
    load_ratio  REAL NOT NULL,      -- pcu / (lanes x length / 6 m)
    level       TEXT,
    flow        INTEGER,            -- vehicles that left (or ended their trip on) the edge in the slot
    speed       REAL,               -- mean speed over the slot, km/h (empty: no vehicles)
    wait_s      REAL,               -- vehicle-seconds stopped in the slot
    time_loss_s REAL,               -- vehicle-seconds lost against driving at the allowed speed
    veh_h       REAL,               -- vehicle-hours spent on the edge in the slot
    incident    INTEGER,            -- an incident slowed this edge during the slot
    observed    INTEGER,
    PRIMARY KEY (edge_id, ts)
);
CREATE INDEX IF NOT EXISTS edge_truth_ts ON edge_truth (ts);
"""
TRUTH_COLUMNS = ["edge_id", "ts", "total", "pcu", "occupancy", "load_ratio", "level", "flow", "speed",
                 "wait_s", "time_loss_s", "veh_h", "incident", "observed"]


def car_lanes(edge) -> int:
    """Lanes cars may use (CARLA networks also have sidewalk and kerb lanes); at least 1."""
    return max(1, sum(ln.allows("passenger") for ln in edge.getLanes()))


def congestion_level(ratio: float, levels) -> str:
    for name, max_ratio in levels:
        if ratio <= max_ratio:
            return name
    return levels[-1][0]


def _traci():
    ensure_sumo()
    try:
        import libsumo as traci   # same API as traci, in-process and much faster
    except ImportError:
        import traci
    return traci


def write_sumocfg(path: Path, net_file: Path, route_file: Path, begin_s: float, end_s: float,
                  additional: Path | list[Path] | None = None, step_length: float = 1.0,
                  teleport_s: float = 300):
    files = [additional] if isinstance(additional, Path) else (additional or [])
    add = f'<additional-files value="{",".join(f.name for f in files)}"/>' if files else ""
    path.write_text(f"""<configuration>
  <input>
    <net-file value="{os.path.relpath(net_file, path.parent)}"/>
    <route-files value="{route_file.name}"/>
    {add}
  </input>
  <time><begin value="{begin_s:.0f}"/><end value="{end_s:.0f}"/><step-length value="{step_length}"/></time>
  <processing>
    <ignore-route-errors value="true"/>
    <time-to-teleport value="{teleport_s:.0f}"/>
    <lateral-resolution value="0.8"/>
  </processing>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>
""", encoding="utf-8")


def write_loops(path: Path, cameras: list[dict], lane_lengths: dict[str, float], out_file: str,
                terminal: set[str] = frozenset()):
    """Induction loops 3 m before each camera lane's stop line: true flow per 5 minutes.
    On `terminal` edges (trips end there) vehicles vanish at the lane end before clearing a loop
    there, and TraCI only counts vehicles that left the loop, so those loops sit mid-lane."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("<additional>\n")
        for cam in cameras:
            for lane in cam["lanes"]:
                pos = lane_lengths[lane] / 2 if cam["edge"] in terminal else max(0.0, lane_lengths[lane] - 3.0)
                f.write(f'  <inductionLoop id={quoteattr(cam["camera_id"] + "|" + lane)} lane={quoteattr(lane)} '
                        f'pos="{pos:.1f}" period="300" file="{out_file}"/>\n')
        f.write("</additional>\n")


def write_edge_data(path: Path, out_file: str, period_s: int = 300):
    """SUMO edgeData: per edge and interval, vehicles entering/leaving, mean speed, waiting time."""
    path.write_text(f'<additional>\n  <edgeData id="truth" period="{period_s}" file="{out_file}" '
                    f'excludeEmpty="false"/>\n</additional>\n', encoding="utf-8")


def read_edge_data(path: Path) -> pd.DataFrame:
    """flow (vehicles that left or ended their trip on the edge), speed km/h, wait_s, time_loss_s,
    veh_h per interval."""
    import xml.etree.ElementTree as ET

    rows = []
    for interval in ET.parse(path).getroot().iter("interval"):
        end = float(interval.get("end"))
        for e in interval.iter("edge"):
            get = lambda k: float(e.get(k, 0) or 0)  # noqa: E731
            speed = e.get("speed")
            rows.append({"edge_id": e.get("id"), "t": end, "flow": int(get("left") + get("arrived")),
                         "speed": round(float(speed) * 3.6, 2) if speed not in (None, "-1.00") else None,
                         "wait_s": round(get("waitingTime"), 1), "time_loss_s": round(get("timeLoss"), 1),
                         "veh_h": round(get("sampledSeconds") / 3600, 4)})
    return pd.DataFrame(rows)


def plan_day(day: date, net_file: str, cameras: list[dict], sim: dict, work_dir: str,
             end_s: float = 86400, incidents: list[dict] | None = None, tag: str | None = None) -> dict:
    """Trips, loops, incidents and the SUMO config for one day, seeded by its date.
    `incidents` (edge, start_s, end_s, block, slow_kmh; or car = [from, to] for a placebo: one extra
    car at start_s) replaces the random ones; the trips stay the same. `tag` names the work files (several runs of one day)."""
    import sumolib

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    seed = int(sim.get("seed", 7)) * 10_000 + day.toordinal() % 10_000
    rng = np.random.default_rng(seed)
    net_path = Path(net_file)
    net = sumolib.net.readNet(net_file)
    lane_len = {ln: net.getLane(ln).getLength() for cam in cameras for ln in cam["lanes"]}

    tag = tag or f"{day:%Y%m%d}"
    routes = work / f"trips_{tag}.rou.xml"
    loops = work / f"loops_{tag}.add.xml"
    edge_data = work / f"edges_{tag}.add.xml"
    cfg = work / f"day_{tag}.sumocfg"
    n_trips = write_zone_trips(routes, day, sim["demand"], rng, VTYPES)
    terminal = {cam["edge"] for cam in cameras if not net.getEdge(cam["edge"]).getOutgoing()}
    write_loops(loops, cameras, lane_len, f"loops_{tag}.out.xml", terminal)
    write_edge_data(edge_data, f"edges_{tag}.out.xml", 60 * int(sim.get("sample_minutes", 5)))
    write_sumocfg(cfg, net_path, routes, 0, end_s, additional=[loops, edge_data],
                  teleport_s=float(sim.get("time_to_teleport_s", 300)))   # -1: vehicles never teleport
    edges = {e.getID(): (car_lanes(e), e.getLength()) for e in net.getEdges()}

    # Incidents: an approach or the road leaving a camera junction crawls at walking pace.
    inc_cfg = sim.get("incidents", {})
    fixed, incidents = incidents, []
    for _ in range(rng.poisson(float(inc_cfg.get("per_day", 0.5)))):
        cam = cameras[rng.integers(len(cameras))]
        edge = net.getEdge(cam["edge"])
        targets = [edge] + [e for e in edge.getOutgoing()]
        target = targets[rng.integers(len(targets))]
        lo, hi = inc_cfg.get("duration_min", [20, 60])
        start = float(rng.uniform(6 * 3600, 21 * 3600))
        incidents.append({"start": start, "end": start + 60 * float(rng.uniform(lo, hi)),
                          "lanes": [ln.getID() for ln in target.getLanes()], "block": 0, "slow_ms": 1.0})
    if fixed is not None:
        incidents = fixed_incidents(fixed, net)
    return {"cfg": cfg, "seed": seed, "lane_len": lane_len, "incidents": incidents, "n_trips": n_trips,
            "edges": edges, "edge_data_out": work / f"edges_{tag}.out.xml", "routes": routes}


def fixed_incidents(fixed: list[dict], net) -> list[dict]:
    """Incident dicts (see plan_day) in the meter's form; only lanes cars may use are blocked or slowed."""
    def lanes(edge_id):
        e = net.getEdge(edge_id)
        return [ln.getID() for ln in e.getLanes() if ln.allows("passenger")] or [ln.getID() for ln in e.getLanes()]
    return [{"start": float(i["start_s"]), "end": float(i["end_s"]),
             "lanes": [] if i.get("car") else lanes(i["edge"]), "car": i.get("car"),
             "block": int(i.get("block", 0)), "slow_ms": float(i.get("slow_kmh", 3.6)) / 3.6,
             "pos": float(i.get("pos", 0.5))} for i in fixed]


class CameraMeter:
    """What each camera's sample shows, read from SUMO: vehicles by type within `view_m` of the
    stop line on its approach lanes, plus the true flow from the induction loops."""

    def __init__(self, traci, cameras: list[dict], plan: dict, view_m: float, pcu, levels):
        self.traci, self.cameras, self.view_m = traci, cameras, view_m
        self.lane_len, self.incidents = plan["lane_len"], plan["incidents"]
        self.pcu, self.levels = pcu, levels
        self._saved_speed: dict[str, float] = {}
        self._blockers: dict[str, float] = {}         # stopped vehicle -> release time
        self._cars_added: set[int] = set()

    def apply_incidents(self, t: float):
        """The first `block` lanes get a stopped vehicle (until the end), the others crawl."""
        traci = self.traci
        for k, inc in enumerate(self.incidents):
            if inc.get("car") and inc["start"] <= t and k not in self._cars_added:   # placebo
                self._cars_added.add(k)
                traci.route.add(f"placebo{k}_route", list(traci.simulation.findRoute(*inc["car"]).edges))
                traci.vehicle.add(f"placebo{k}", f"placebo{k}_route", typeID="car", depart="now")
            for i, ln in enumerate(inc["lanes"]):
                if inc["start"] <= t < inc["end"] and ln not in self._saved_speed:
                    self._saved_speed[ln] = traci.lane.getMaxSpeed(ln)
                    if i < inc["block"]:
                        self._block(f"incident{k}_{i}", ln, inc)
                    else:
                        traci.lane.setMaxSpeed(ln, inc["slow_ms"])
                elif t >= inc["end"] and ln in self._saved_speed:
                    traci.lane.setMaxSpeed(ln, self._saved_speed.pop(ln))
        for vid, end in list(self._blockers.items()):   # setStop's `until` alone does not release them
            if t >= end:
                if traci.vehicle.isStopped(vid):
                    traci.vehicle.resume(vid)
                elif vid not in traci.vehicle.getIDList():
                    traci.vehicle.remove(vid)               # never got onto the road
                del self._blockers[vid]

    def _block(self, vid: str, lane: str, inc: dict):
        traci = self.traci
        edge, index = lane.rsplit("_", 1)
        pos = inc.get("pos", 0.5) * traci.lane.getLength(lane)
        traci.route.add(f"{vid}_route", [edge])
        traci.vehicle.add(vid, f"{vid}_route", typeID="car", depart="now", departLane=index,
                          departPos=f"{pos - 5:.1f}", departSpeed="0")
        traci.vehicle.setStop(vid, edge, pos=pos, laneIndex=int(index), until=inc["end"])
        self._blockers[vid] = inc["end"]

    def in_view(self, cam: dict) -> list[str]:
        """Vehicles whose front is within view_m of the camera's stop line."""
        traci = self.traci
        return [v for ln in cam["lanes"] for v in traci.lane.getLastStepVehicleIDs(ln)
                if self.lane_len[ln] - traci.vehicle.getLanePosition(v) <= self.view_m]

    def rows(self, ts: datetime, t: float) -> list[dict]:
        traci = self.traci
        rows = []
        for cam in self.cameras:
            counts = {c: 0 for c in CATEGORIES}
            occupied = 0.0
            for v in self.in_view(cam):
                vt = traci.vehicle.getTypeID(v)
                counts[vt if vt in counts else "car"] += 1
                occupied += traci.vehicle.getLength(v)
            loops = [f"{cam['camera_id']}|{ln}" for ln in cam["lanes"]]
            passed = [(traci.inductionloop.getLastIntervalVehicleNumber(lp),
                       traci.inductionloop.getLastIntervalMeanSpeed(lp)) for lp in loops]
            flow = sum(n for n, _ in passed)
            speed = round(3.6 * sum(n * v for n, v in passed if n > 0) / flow, 1) if flow else None
            total_pcu = sum(n * self.pcu.get(c, 1.0) for c, n in counts.items())
            ratio = total_pcu / cam["capacity_pcu"] if cam["capacity_pcu"] else 0.0
            rows.append({
                "camera_id": cam["camera_id"], "ts": ts.isoformat(), "image_path": "", "ts_source": "sumo",
                "total": sum(counts.values()), "pcu": round(total_pcu, 3),
                "occupancy": round(min(1.0, occupied / (self.view_m * max(1, len(cam["lanes"])))), 4),
                "load_ratio": round(ratio, 4), "level": congestion_level(ratio, self.levels),
                "counts_json": json.dumps(counts), "mean_score": 1.0,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "flow": flow, "speed": speed, "incident": any(i["start"] <= t < i["end"] for i in self.incidents),
            })
        return rows


class EdgeMeter:
    """The true state of every edge at the end of each slot: all vehicles on the whole edge
    (vehicles inside junction areas are on no edge)."""

    def __init__(self, traci, plan: dict, pcu, levels, slot_s: int):
        self.traci, self.edges, self.incidents = traci, plan["edges"], plan["incidents"]
        self.pcu, self.levels, self.slot_s = pcu, levels, slot_s

    def rows(self, ts: datetime, t: float) -> list[dict]:
        traci = self.traci
        hit = {ln.rsplit("_", 1)[0] for i in self.incidents if i["start"] < t and i["end"] > t - self.slot_s
               for ln in i["lanes"]}
        rows = []
        for edge_id, (lanes, length) in self.edges.items():
            vehs = traci.edge.getLastStepVehicleIDs(edge_id)
            types = [traci.vehicle.getTypeID(v) for v in vehs]
            pcu = sum(self.pcu.get(vt, 1.0) for vt in types)
            occupied = sum(traci.vehicle.getLength(v) for v in vehs)
            ratio = pcu / (lanes * length / 6.0)
            rows.append({"edge_id": edge_id, "ts": ts.isoformat(), "t": float(t), "total": len(vehs),
                         "pcu": round(pcu, 3), "occupancy": round(min(1.0, occupied / (lanes * length)), 4),
                         "load_ratio": round(ratio, 4), "level": congestion_level(ratio, self.levels),
                         "incident": int(edge_id in hit), "observed": 1})
        return rows


def _measures(sim: dict):
    pcu = {**DEFAULT_PCU, **(sim.get("pcu") or {})}
    levels = [(lv["name"], float(lv["max_ratio"])) for lv in sim["congestion_levels"]] \
        if sim.get("congestion_levels") else DEFAULT_LEVELS
    return pcu, levels


def simulate_day(day: date, net_file: str, cameras: list[dict], sim: dict, work_dir: str,
                 incidents: list[dict] | None = None, tag: str | None = None, end_s: float = 86400,
                 demand: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one day. Returns (one row per camera per sample: observations layout + flow,
    one row per edge per sample: the edge's true state). `incidents`, `tag`: see plan_day.
    `end_s`: stop early. `demand`: change the day's trips first (demand.edit_trips)."""
    traci = _traci()
    pcu, levels = _measures(sim)
    plan = plan_day(day, net_file, cameras, sim, work_dir, end_s=end_s, incidents=incidents, tag=tag)
    if demand:
        from .demand import edit_trips
        plan["n_trips"] = edit_trips(plan["routes"], demand, seed=plan["seed"])
    meter = CameraMeter(traci, cameras, plan, float(sim.get("network", {}).get("view_m", 60)), pcu, levels)
    step = 60 * int(sim.get("sample_minutes", 5))
    edge_meter = EdgeMeter(traci, plan, pcu, levels, step)
    midnight = datetime.combine(day, datetime.min.time())
    traci.start([sumo_binary("sumo"), "-c", str(plan["cfg"]), "--seed", str(plan["seed"] % 2**31)])
    rows, truth = [], []
    try:
        for t in range(step, int(end_s) + 1, step):
            traci.simulationStep(float(t))
            meter.apply_incidents(t)
            ts = midnight + timedelta(seconds=t)
            rows += meter.rows(ts, t)
            truth += edge_meter.rows(ts, t)
    finally:
        traci.close()        # also writes the edgeData file
    truth = pd.DataFrame(truth).merge(read_edge_data(plan["edge_data_out"]), on=["edge_id", "t"], how="left")
    log.info("%s: %d trips, %d incidents", day, plan["n_trips"], len(plan["incidents"]))
    return pd.DataFrame(rows), truth[TRUTH_COLUMNS]


def run_days(sim: dict, net_file: Path, cameras: list[dict], out_dir: Path, db_path: Path,
             workers: int | None = None) -> pd.DataFrame:
    tr = sim.get("traffic", {})
    start = date.fromisoformat(str(tr.get("start", "2026-10-01")))
    days = [start + timedelta(days=i) for i in range(int(tr.get("days", 28)))]
    work = out_dir / "sumo_days"
    workers = workers or int(tr.get("workers", 4))
    frames = []
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA + TRUTH_SCHEMA)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(simulate_day, d, str(net_file), cameras, sim, str(work)) for d in days]
            for d, fut in zip(days, futures):
                df, truth = fut.result()
                conn.executemany(f"INSERT OR REPLACE INTO observations ({','.join(COLUMNS)}) "
                                 f"VALUES ({','.join('?' * len(COLUMNS))})",
                                 list(df[COLUMNS].itertuples(index=False, name=None)))
                conn.executemany(f"INSERT OR REPLACE INTO edge_truth ({','.join(TRUTH_COLUMNS)}) "
                                 f"VALUES ({','.join('?' * len(TRUTH_COLUMNS))})",
                                 list(truth.astype(object).where(truth.notna(), None)
                                      .itertuples(index=False, name=None)))
                conn.commit()
                frames.append(df)
                print(f"  {d}  {len(df)} samples")
    finally:
        conn.close()
    series = pd.concat(frames, ignore_index=True)
    series.to_csv(out_dir / "sumo_series.csv", index=False)
    return series
