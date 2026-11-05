"""Phase 2: how incidents spread (docs/phase2.md).

Ground truth from paired runs: a weekday of the 28-day run that had no random incident is replayed
with one road closed (both lanes blocked by stopped vehicles). Replays are identical to the stored
run, so the stored day is the base, and the difference between the two is the incident's footprint.

Once a run differs from its base at all, small differences grow by chance. Each closure therefore
has a placebo run: the same day and time with one extra car instead of the closure. The placebos give,
per edge and time of day, how large chance differences get (the noise level).

  incident_list       incident_id, kind (closure | placebo), edge_id, day, start, end, block, car
  run_truth           edge_truth rows of each run (+ incident_id)
  run_cameras         what each camera saw in each run (+ incident_id)
  incident_noise      per edge and time of day: noise level of delay and flow (from the placebos)
  incident_footprint  per run and affected edge: hops (any direction, upstream, downstream), effect,
                      first slot, extra delay, flow change
"""
from __future__ import annotations

import sqlite3
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import load_edges
from .simulate import TRUTH_COLUMNS, simulate_day

CAMERA_COLUMNS = ["camera_id", "ts", "total", "pcu", "occupancy", "load_ratio", "level", "flow"]
AFTER_MIN = 60            # footprint window: incident start to end + 60 minutes
MARGIN = 2.0              # affected: more than MARGIN x the largest placebo change of that edge


def hops(edges: pd.DataFrame, origin: str) -> tuple[dict[str, int], dict[str, int]]:
    """Junctions between each edge and `origin`, following traffic: (upstream, downstream).
    Upstream 1 = edges that feed the origin. U-turns are not followed."""
    e = edges.set_index("edge_id")
    nxt = {a: [b for b in e.index if e.at[b, "from_node"] == e.at[a, "to_node"]
               and e.at[b, "to_node"] != e.at[a, "from_node"]] for a in e.index}
    prv = {b: [a for a in e.index if b in nxt[a]] for b in e.index}

    def bfs(links):
        dist, q = {origin: 0}, deque([origin])
        while q:
            a = q.popleft()
            for b in links[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    q.append(b)
        return dist
    return bfs(prv), bfs(nxt)


def hop_any(edges: pd.DataFrame, origin: str) -> dict[str, int]:
    """Hops ignoring direction: 1 = shares a junction with `origin`, 2 = with those, ..."""
    ends = {r.edge_id: {r.from_node, r.to_node} for r in edges.itertuples()}
    dist, frontier, seen = {origin: 0}, ends[origin], set(ends[origin])
    k = 0
    while len(dist) < len(ends):
        k += 1
        new = [e for e, n in ends.items() if e not in dist and n & frontier]
        if not new:
            break
        dist.update({e: k for e in new})
        frontier = set().union(*(ends[e] for e in new)) - seen
        seen |= frontier
    return dist




def placebo_car(edges: pd.DataFrame, edge_id: str | None = None, avoid: set[str] = frozenset()) -> list[str]:
    """A trip that enters and leaves town at junctions away from `edge_id`'s two ends (and `avoid`)."""
    e = edges.set_index("edge_id")
    ends = set(avoid) | ({e.at[edge_id, "from_node"], e.at[edge_id, "to_node"]} if edge_id else set())
    far = sorted(j for j in e.to_node[edges.kind.eq("in").to_numpy()] if j not in ends)
    return [f"in_{far[0]}", f"out_{far[-1]}"]


def clean_weekdays(db_path: Path) -> list[date]:
    """Weekdays of the 28-day run on which no random incident happened."""
    with sqlite3.connect(db_path) as conn:
        t = pd.read_sql("SELECT ts, MAX(incident) AS inc FROM edge_truth GROUP BY ts", conn)
    t["day"] = (pd.to_datetime(t.ts) - pd.Timedelta(seconds=1)).dt.date
    days = t.groupby("day").inc.max()
    return [d for d, inc in days.items() if inc == 0 and d.weekday() < 5]


def incident_set(sim: dict, edges: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """One closure per road and time in propagation.incidents, each on its own base weekday
    (weekdays of the 28-day run without a random incident, used in turn), plus its placebo."""
    cfg = (sim.get("propagation") or {}).get("incidents") or {}
    roads = edges.edge_id[edges.kind == "road"].tolist() if cfg.get("edges", "roads") == "roads" else cfg["edges"]
    clean = clean_weekdays(db_path)
    dur = float(cfg.get("duration_min", 30))
    rows = []
    for time in cfg.get("times", ["08:00", "17:30"]):
        h, m = map(int, time.split(":"))
        for edge in roads:
            day = clean[len(rows) // 2 % len(clean)]
            start = datetime.combine(day, datetime.min.time()) + timedelta(hours=h, minutes=m)
            common = {"edge_id": edge, "time": time, "day": day.isoformat(), "start": start.isoformat(),
                      "end": (start + timedelta(minutes=dur)).isoformat()}
            rows.append({"incident_id": f"{edge}@{time}", "kind": "closure", **common,
                         "block": int(cfg.get("block", 2)), "car": None})
            rows.append({"incident_id": f"{edge}@{time}~placebo", "kind": "placebo", **common,
                         "block": 0, "car": " ".join(placebo_car(edges, edge))})
    return pd.DataFrame(rows)


def run_incidents(sim: dict, net_file: Path, cameras: list[dict], db_path: Path, out_dir: Path,
                  workers: int | None = None, fresh: bool = False) -> pd.DataFrame:
    """Simulate every run of incident_set not stored yet (all with `fresh`) in out_dir/incidents.db."""
    with sqlite3.connect(db_path) as conn:
        edges = load_edges(conn)
    incs = incident_set(sim, edges, db_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or int(sim.get("traffic", {}).get("workers", 4))
    conn = sqlite3.connect(out_dir / "incidents.db")
    if fresh:
        for table in ("run_truth", "run_cameras"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    done = set(pd.read_sql("SELECT DISTINCT incident_id FROM run_truth", conn).incident_id) \
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'run_truth'").fetchone() else set()
    incs.to_sql("incident_list", conn, index=False, if_exists="replace")
    todo = incs[~incs.incident_id.isin(done)]
    print(f"{len(todo)} runs to simulate ({len(done)} stored)")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for r in todo.itertuples():
                midnight = datetime.fromisoformat(r.day)
                spec = [{"edge": r.edge_id, "block": r.block, "car": r.car.split() if isinstance(r.car, str) else None,
                         "start_s": (datetime.fromisoformat(r.start) - midnight).total_seconds(),
                         "end_s": (datetime.fromisoformat(r.end) - midnight).total_seconds()}]
                futures.append(pool.submit(simulate_day, date.fromisoformat(r.day), str(net_file), cameras, sim,
                                           str(out_dir / "sumo_runs"), spec,
                                           r.incident_id.replace(":", "").replace("~", "_")))
            for r, fut in zip(todo.itertuples(), futures):
                cams, truth = fut.result()
                truth.assign(incident_id=r.incident_id)[["incident_id"] + TRUTH_COLUMNS] \
                    .to_sql("run_truth", conn, index=False, if_exists="append")
                cams.assign(incident_id=r.incident_id)[["incident_id"] + CAMERA_COLUMNS] \
                    .to_sql("run_cameras", conn, index=False, if_exists="append")
                conn.commit()
                print(f"  {r.incident_id:22s} on {r.day}")
    finally:
        conn.close()
    return incs


def _diffs(conn_base, conn_inc, inc) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run minus base, per slot (rows) and edge (columns): (time loss veh-min, flow, load_ratio)."""
    day0 = datetime.fromisoformat(inc.day)
    lo, hi = day0.isoformat(), (day0 + timedelta(days=1)).isoformat()
    cols = ["time_loss_s", "flow", "load_ratio"]
    base = pd.read_sql(f"SELECT edge_id, ts, {', '.join(cols)} FROM edge_truth WHERE ts > ? AND ts <= ?",
                       conn_base, params=(lo, hi))
    run = pd.read_sql(f"SELECT edge_id, ts, {', '.join(cols)} FROM run_truth WHERE incident_id = ?",
                      conn_inc, params=(inc.incident_id,))
    m = run.merge(base, on=["edge_id", "ts"], suffixes=("", "_base"))
    m["ts"] = pd.to_datetime(m.ts)
    d = {c: m.pivot(index="ts", columns="edge_id", values=c) - m.pivot(index="ts", columns="edge_id", values=c + "_base")
         for c in cols}
    return d["time_loss_s"] / 60, d["flow"], d["load_ratio"]


def _window(d: pd.DataFrame, start: datetime, minutes: float) -> pd.DataFrame:
    return d[(d.index > start) & (d.index <= start + timedelta(minutes=minutes))]


def _smooth(d: pd.DataFrame) -> pd.DataFrame:
    return d.rolling(3, center=True, min_periods=1).mean()


def build_footprints(db_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Footprint of every run, closures and placebos. Window: start to end + 60 minutes.
    An edge is affected when its extra delay or its flow change over the window is more than MARGIN
    times the largest in the placebo runs of the same time of day (for a placebo itself: the other
    placebos). first_ts: first slot whose 15-minute mean is beyond the placebos' largest."""
    idb = out_dir / "incidents.db"
    with sqlite3.connect(db_path) as cb, sqlite3.connect(idb) as ci:
        edges = load_edges(cb)
        incs = pd.read_sql("SELECT * FROM incident_list", ci)
        diffs = {r.incident_id: _diffs(cb, ci, r) for r in incs.itertuples()}
    win = {}                                         # run -> (window sums, max |15-min mean| per edge)
    for r in incs.itertuples():
        start, end = datetime.fromisoformat(r.start), datetime.fromisoformat(r.end)
        length = (end - start).total_seconds() / 60 + AFTER_MIN
        loss, flow, load = (_window(x, start, length) for x in diffs[r.incident_id])
        sl, sf = (_window(_smooth(x), start, length) for x in diffs[r.incident_id][:2])
        win[r.incident_id] = {"loss": loss, "flow": flow, "load": load, "sl": sl, "sf": sf, "start": start}
    placebos = incs[incs.kind == "placebo"]

    def limits(time: str, leave_out: str | None = None) -> pd.DataFrame:
        ids = [i for i in placebos.incident_id[placebos.time == time] if i != leave_out]
        w = [win[i] for i in ids]
        return pd.DataFrame({
            "loss_sum": pd.concat([x["loss"].sum().abs() for x in w], axis=1).max(axis=1),
            "flow_sum": pd.concat([x["flow"].sum().abs() for x in w], axis=1).max(axis=1),
            "loss_slot": pd.concat([x["sl"].abs().max() for x in w], axis=1).max(axis=1),
            "flow_slot": pd.concat([x["sf"].abs().max() for x in w], axis=1).max(axis=1)})

    noise = pd.concat({t: limits(t) for t in placebos.time.unique()}, names=["time", "edge_id"]).reset_index()
    rows = []
    for r in incs.itertuples():
        lim = limits(r.time, r.incident_id if r.kind == "placebo" else None)
        w = win[r.incident_id]
        up, down = hops(edges, r.edge_id)
        near = hop_any(edges, r.edge_id)
        for e in w["loss"].columns:
            loss, flow = w["loss"][e].sum(), w["flow"][e].sum()
            by_loss = abs(loss) > MARGIN * lim.at[e, "loss_sum"]
            by_flow = abs(flow) > MARGIN * lim.at[e, "flow_sum"]
            if not (by_loss or by_flow):
                continue
            hit = w["sl"].index[(w["sl"][e].abs() > lim.at[e, "loss_slot"]) | (w["sf"][e].abs() > lim.at[e, "flow_slot"])]
            effect = ("slower" if loss > 0 else "faster") if by_loss else ("fewer" if flow < 0 else "more")
            rows.append({"incident_id": r.incident_id, "kind": r.kind, "edge_id": e, "hop": near.get(e),
                         "hop_up": up.get(e), "hop_down": down.get(e), "effect": effect,
                         "by_delay": int(by_loss), "by_flow": int(by_flow),
                         "first_ts": hit.min().isoformat() if len(hit) else None,
                         "delay_min": (hit.min() - w["start"]).total_seconds() / 60 if len(hit) else None,
                         "extra_delay_veh_h": round(loss / 60, 2), "flow_change": int(flow),
                         "peak_load_rise": round(float(w["load"][e].max()), 3)})
    fp = pd.DataFrame(rows)
    with sqlite3.connect(idb) as ci:
        fp.to_sql("incident_footprint", ci, index=False, if_exists="replace")
        noise.to_sql("incident_noise", ci, index=False, if_exists="replace")
    return fp, noise
