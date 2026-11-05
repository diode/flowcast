"""Paired SUMO runs: the answer key for road-change predictions (docs/architecture-next.md 1.2). Simulation only,
scoring only: `labels.db` `whatif_labels`, `label_source = 'sumo'`, read by scripts/check_predict.py and never by
the graph model or Glimmer.

For each case (town config `whatif_labels.cases`) on each of the last `whatif_labels.days` working days:
  base    the day as SUMO makes it, random incidents off (one run per day, shared by its cases)
  change  the same day and seed, with the road change: busy / slow cap the road at that share of its speed limit
          (days.LEVELS), jammed at 5 km/h, closed blocks every lane. Drivers hear of it: from its start, every
          sample (12 s) while it lasts, each vehicle whose remaining route uses the road is rerouted once on the
          current travel times (SUMO's, refreshed every sample), in which the road takes length / level speed
          (closed: not usable). Only the change run reroutes,
          so the base run stays the sample day. Vehicles already on the road, or with no way round, wait; SUMO's
          teleport after `time_to_teleport_s` (as in the sample data) still applies and is counted.
Both runs are measured by the perfect camera (direct.py), in observations' columns, per camera and 5 minutes.

  labels.db  whatif_cases  (case_id, edge, level, day, start, minutes, rerouted, teleports_base, teleports_change:
                            teleports from the change's start to 2 hours after it ends, in each run)
             whatif_base   (day, camera_id, ts, flow, speed, total): the base runs, kept for the next cases
             whatif_teleports (day, run, t_s, edge)
             whatif_labels (case_id, day, camera_id, edge_id, ts, flow_base, flow_change, speed_base, speed_change,
                            total_base, total_change, label_source)
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .days import JAMMED_KMH, LEVELS, day_kind
from .labels import connect

AFTER_S = 2 * 3600                  # followed to 2 hours after the change ends
CLOSED_TT_S = 1e6                   # a closed road's travel time for rerouting


def case_id(c: dict) -> str:
    return f"{c['edge']}_{c['level']}_{c['start'].replace(':', '')}_{int(c['minutes'])}"


def _s(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return 3600.0 * h + 60 * m


def test_days(db: Path, n: int) -> list[date]:
    """The last n working days with observations."""
    with sqlite3.connect(db) as c:
        days = pd.read_sql("SELECT DISTINCT substr(ts, 1, 10) AS d FROM observations WHERE substr(ts, 12) > '00:00:00'",
                           c).d
    ds = sorted(date.fromisoformat(d) for d in days)
    return [d for d in ds if day_kind(d) == "working"][-n:]


class Rerouter:
    """Called after each step of the change run: tells drivers of the change once it starts."""

    def __init__(self, edge: str, start_s: float, end_s: float, travel_s: float):
        self.edge, self.start_s, self.end_s, self.travel_s = edge, start_s, end_s, travel_s
        self.tried: set[str] = set()
        self.rerouted = 0
        self.set = False

    def __call__(self, traci, t: float):
        if not self.start_s <= t <= self.end_s:
            return
        if not self.set:
            self.roads = [e for e in traci.edge.getIDList() if not e.startswith(":")]
            self.set = True
        for e in self.roads:              # current travel times, so later drivers avoid a detour that is filling up
            traci.edge.adaptTraveltime(e, self.travel_s if e == self.edge else traci.edge.getTraveltime(e))
        for vid in traci.vehicle.getIDList():
            if vid in self.tried or vid.startswith("incident"):
                continue
            route, i = traci.vehicle.getRoute(vid), traci.vehicle.getRouteIndex(vid)
            if self.edge not in route[i + 1:]:
                continue
            self.tried.add(vid)
            traci.vehicle.rerouteTraveltime(vid, False)
            if self.edge not in traci.vehicle.getRoute(vid)[traci.vehicle.getRouteIndex(vid) + 1:]:
                self.rerouted += 1


def _run(day: str, sim: dict, out_dir: Path, cams: list, case: dict | None) -> tuple:
    from .direct import direct_day

    d = date.fromisoformat(day)
    stats: dict = {}
    if case is None:
        rows, _ = direct_day(d, sim, out_dir, cams, incidents=[], tag=f"wl_base_{d:%Y%m%d}", stats=stats)
        return rows, stats, 0
    import sumolib

    net = sumolib.net.readNet(str(out_dir / "network" / "net.net.xml"))
    edge = net.getEdge(case["edge"])
    start_s = _s(case["start"])
    end_s = start_s + 60 * float(case["minutes"])
    lvl = case["level"]
    if lvl == "closed":
        inc, travel = {"block": len(edge.getLanes())}, CLOSED_TT_S
    else:
        kmh = JAMMED_KMH if lvl == "jammed" else 3.6 * edge.getSpeed() * LEVELS[lvl]
        inc, travel = {"block": 0, "slow_kmh": kmh}, edge.getLength() / (kmh / 3.6)
    incidents = [{"edge": case["edge"], "start_s": start_s, "end_s": end_s, **inc}]
    rr = Rerouter(case["edge"], start_s, end_s, travel)
    rows, _ = direct_day(d, sim, out_dir, cams, end_s=min(86400.0, end_s + AFTER_S), incidents=incidents,
                         tag=f"wl_{case_id(case)}_{d:%Y%m%d}", on_step=rr, stats=stats)
    return rows, stats, rr.rerouted


SCHEMA = """
CREATE TABLE IF NOT EXISTS whatif_cases (case_id TEXT, edge TEXT, level TEXT, day TEXT, start TEXT, minutes INTEGER,
    rerouted INTEGER, teleports_base INTEGER, teleports_change INTEGER, made_at TEXT, PRIMARY KEY (case_id, day));
CREATE TABLE IF NOT EXISTS whatif_base (day TEXT, camera_id TEXT, ts TEXT, flow REAL, speed REAL, total INTEGER,
    PRIMARY KEY (day, camera_id, ts));
CREATE TABLE IF NOT EXISTS whatif_teleports (day TEXT, run TEXT, t_s REAL, edge TEXT);
CREATE TABLE IF NOT EXISTS whatif_labels (case_id TEXT, day TEXT, camera_id TEXT, edge_id TEXT, ts TEXT,
    flow_base REAL, flow_change REAL, speed_base REAL, speed_change REAL, total_base INTEGER, total_change INTEGER,
    label_source TEXT, PRIMARY KEY (case_id, day, camera_id, ts));
"""


def make(sim: dict, out_dir: Path, cams: list, workers: int = 4, only: str | None = None, log=print) -> pd.DataFrame:
    if (sim.get("source") or {}).get("kind", "simulation") == "real":
        raise SystemExit("whatif-labels: simulation only (a real town has no paired runs; see section 5.5)")
    cfg = sim.get("whatif_labels") or {}
    cases = [c for c in cfg.get("cases", []) if only in (None, case_id(c))]
    if not cases:
        raise SystemExit("whatif-labels: no cases (town config whatif_labels.cases)")
    days = [d.isoformat() for d in test_days(out_dir / "traffic.db", int(cfg.get("days", 3)))]
    with connect(out_dir) as c:
        c.executescript(SCHEMA)
        done = {(a, b) for a, b in c.execute("SELECT case_id, day FROM whatif_cases")}
    todo = [(cs, d) for cs in cases for d in days if (case_id(cs), d) not in done]
    log(f"whatif-labels: {len(cases)} cases x {len(days)} days ({', '.join(days)}), {len(todo)} change runs to do")
    cam_edge = {c["camera_id"]: c["edge"] for c in cams}
    with connect(out_dir) as c:
        have = set(pd.read_sql("SELECT DISTINCT day FROM whatif_base", c).day)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        base_f = {d: pool.submit(_run, d, sim, out_dir, cams, None) for d in sorted({d for _, d in todo} - have)}
        change_f = [(cs, d, pool.submit(_run, d, sim, out_dir, cams, cs)) for cs, d in todo]
        for d, f in base_f.items():
            rows, stats, _ = f.result()
            with connect(out_dir) as c:
                c.execute("DELETE FROM whatif_base WHERE day = ?", (d,))
                c.execute("DELETE FROM whatif_teleports WHERE day = ? AND run = 'base'", (d,))
                rows[["camera_id", "ts", "flow", "speed", "total"]].assign(day=d).to_sql(
                    "whatif_base", c, if_exists="append", index=False)
                c.executemany("INSERT INTO whatif_teleports VALUES (?, 'base', ?, ?)",
                              [(d, t, e) for t, e in stats.get("teleports", [])])
            log(f"  base {d}: {len(rows)} camera rows, {len(stats.get('teleports', []))} teleports")
        for cs, d, f in change_f:
            rows, stats, rerouted = f.result()
            with connect(out_dir) as c:
                b = pd.read_sql("SELECT camera_id, ts, flow, speed, total FROM whatif_base WHERE day = ?", c, params=(d,))
                bt = [t for (t,) in c.execute("SELECT t_s FROM whatif_teleports WHERE day = ? AND run = 'base'", (d,))]
            lo, hi = _s(cs["start"]), _s(cs["start"]) + 60 * float(cs["minutes"]) + AFTER_S
            n_base = sum(lo <= t <= hi for t in bt)
            n_change = sum(lo <= t <= hi for t, _ in stats.get("teleports", []))
            k = ["camera_id", "ts"]
            m = b[k + ["flow", "speed", "total"]].merge(rows[k + ["flow", "speed", "total"]], on=k, how="inner",
                                                       suffixes=("_base", "_change"))
            m = m[m.ts <= rows.ts.max()]
            m.insert(0, "day", d)
            m.insert(0, "case_id", case_id(cs))
            m.insert(3, "edge_id", m.camera_id.map(cam_edge))
            m["label_source"] = "sumo"
            now = datetime.now().isoformat(timespec="seconds")
            with connect(out_dir) as c:
                c.execute("DELETE FROM whatif_labels WHERE case_id = ? AND day = ?", (case_id(cs), d))
                m.to_sql("whatif_labels", c, if_exists="append", index=False)
                c.execute("INSERT OR REPLACE INTO whatif_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (case_id(cs), cs["edge"], cs["level"], d, cs["start"], int(cs["minutes"]), rerouted,
                           n_base, n_change, now))
                c.execute("DELETE FROM whatif_teleports WHERE day = ? AND run = ?", (d, case_id(cs)))
                c.executemany("INSERT INTO whatif_teleports VALUES (?, ?, ?, ?)",
                              [(d, case_id(cs), t, e) for t, e in stats.get("teleports", [])])
            log(f"  {case_id(cs)} {d}: {rerouted} rerouted, teleports from its start to 2 h after: "
                f"{n_change} (base {n_base})")
    with connect(out_dir) as c:
        return pd.read_sql("SELECT * FROM whatif_cases", c)
