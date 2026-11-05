"""Predict (docs/predict.md): a future date and time, optionally with one road set to a traffic level,
simulated on demand with SUMO from the studied days.

Day kinds (a mock calendar of festivals and holidays, Nov-Dec 2026):
  working   a clean studied weekday of the same weekday          no run needed without a road change
  weekend   a clean studied Saturday / Sunday                    no run needed without a road change
  holiday   a studied Sunday, trips scaled hour by hour          one base run (HOLIDAY_FACTORS)
  festival  the date's own kind of day plus extra trips to and from a venue district around the event

A road change (busy / slow / jammed / closed for some minutes from the chosen time) is simulated on
REPLICATES base days (one studied day alone is too chaotic in busy hours: one extra car can move a
junction's delay by half), each with a placebo run (one extra car from the same time) for the chance
level, as in phase 4. A junction is affected in a 15-minute slot when the mean extra delay on its
roads in is more than MARGIN times the largest the mean placebo reaches from the change's start.

  data/<town>/predict/predict.db   runs (run_id, role, day, spec, status, seconds), run_truth
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .days import (CAL_FROM, CAL_TO, EVENTS, HOLIDAY_FACTORS, JAMMED_KMH, LEVELS, _s,  # noqa: F401
                   calendar, day_kind, event_on, time_context)
from .graph import state_table
from .simulate import TRUTH_COLUMNS, simulate_day

ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIGS = {"maple_falls": "sim/maple_falls.yaml", "ashford": "towns/ashford.yaml",
               "brookhaven": "towns/brookhaven.yaml", "clearwater": "towns/clearwater.yaml"}
MARGIN = 2.0
REPLICATES = 3                           # base days a road change is simulated on
SLOT = "15min"
AFTER_S = 2 * 3600                       # a case run goes on 2 hours past the change
VENUES = {"maple_falls": ["J2", "J1", "J3", "J4", "J5"], "westbrook": ["B1", "B0", "C0", "A1", "C1"]}


# ---------------------------------------------------------------- the base day
def stored_days(db_path: Path) -> pd.DataFrame:
    """The studied days, each with whether an incident happened: from the incident log in labels.db
    (people's or SUMO's labels) when there is one, else from the state table's incident column."""
    with sqlite3.connect(db_path) as conn:
        t = pd.read_sql(f"SELECT ts, MAX(incident) AS inc FROM {state_table(conn)} GROUP BY ts", conn)
    t["day"] = (pd.to_datetime(t.ts) - pd.Timedelta(seconds=1)).dt.date
    d = t.groupby("day").inc.max().fillna(0).rename("incident").reset_index()
    labels = Path(db_path).parent / "labels.db"
    if labels.exists():
        with sqlite3.connect(labels) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'incident_log'").fetchone():
                logged = {pd.Timestamp(s).date() for (s,) in conn.execute("SELECT start FROM incident_log")}
                d["incident"] = d.day.isin(logged).astype(int)
    d["weekday"] = [x.weekday() for x in d.day]
    return d


def base_days(db_path: Path, d: date, n: int = REPLICATES) -> list[date]:
    """The studied days a future date is simulated from, best first: no random incident and the same
    weekday (holiday: a Sunday), latest first; then other clean days of the same type."""
    days = stored_days(db_path)
    want = 6 if day_kind(d) == "holiday" else d.weekday()
    out = []
    for pick in (days[(days.incident == 0) & (days.weekday == want)],
                 days[(days.incident == 0) & ((days.weekday >= 5) == (want >= 5))],
                 days[days.weekday == want], days):
        out += [x for x in sorted(pick.day, reverse=True) if x not in out]
    return out[:n]


def base_day(db_path: Path, d: date) -> date:
    return base_days(db_path, d, 1)[0]


def venue_junctions(town: str, nodes: pd.DataFrame) -> list[str]:
    if town in VENUES:
        return VENUES[town]
    j = nodes[nodes.kind == "junction"].copy()            # from the centre outwards (places.Places)
    j["d"] = (j.x - j.x.mean()) ** 2 + (j.y - j.y.mean()) ** 2
    return j.sort_values(["d", "node_id"]).node_id.tolist()


def venue_edges(edges: pd.DataFrame, kinds: dict, venue: str) -> tuple[list[str], list[str]]:
    """(edges to send event trips to, edges they leave from): the venue's own local streets if it has
    them, else its roads in / out."""
    out_local = edges[(edges.from_node == venue) & edges.to_node.map(kinds).eq("boundary")].edge_id.tolist()
    in_local = edges[(edges.to_node == venue) & edges.from_node.map(kinds).eq("boundary")].edge_id.tolist()
    to = out_local or edges[edges.to_node == venue].edge_id.tolist()
    frm = in_local or edges[edges.from_node == venue].edge_id.tolist()
    return to, frm


def demand_for(town: str, d: date, edges: pd.DataFrame, nodes: pd.DataFrame) -> dict | None:
    kind, e = day_kind(d), event_on(d)
    if kind == "holiday":
        return {"hour_factors": HOLIDAY_FACTORS}
    if kind != "festival":
        return None
    venue = venue_junctions(town, nodes)[e["venue"] % len(venue_junctions(town, nodes))]
    to, frm = venue_edges(edges, dict(zip(nodes.node_id, nodes.kind)), venue)
    s0, s1 = _s(e["start"]), _s(e["end"])
    return {"venue": {"junction": venue, "to": to, "from": frm, "arrive": [max(0, s0 - 5400), s1 - 3600],
                      "leave": [max(s0, s1 - 1800), min(86400, s1 + 3600)], "share": e["share"]}}


# ----------------------------------------------------------------- the case
def case_spec(town: str, d: date, hhmm: str, change: dict | None, db_path: Path) -> dict:
    """Everything that defines a prediction. change: {edge, level, minutes} from the chosen time."""
    with sqlite3.connect(db_path) as conn:
        edges = pd.read_sql("SELECT * FROM edges", conn)
        nodes = pd.read_sql("SELECT * FROM nodes", conn)
    spec = {"town": town, "date": d.isoformat(), "time": hhmm, "kind": day_kind(d),
            "event": (event_on(d) or {}).get("name"), "base_day": base_day(db_path, d).isoformat(),
            "base_days": [x.isoformat() for x in base_days(db_path, d)],
            "demand": demand_for(town, d, edges, nodes), "change": None}
    if change:
        e = edges.set_index("edge_id").loc[change["edge"]]
        start = _s(hhmm)
        end = min(86400.0, start + 60 * float(change["minutes"]))
        level = change["level"]
        inc = {"edge": change["edge"], "start_s": start, "end_s": end}
        if level == "closed":
            inc["block"] = 99
        else:
            inc["slow_kmh"] = JAMMED_KMH if level == "jammed" else LEVELS[level] * float(e.speed_kmh)
        spec["change"] = {**change, "start_s": start, "end_s": end, "incident": inc,
                          "end_run_s": float(min(86400, int((end + AFTER_S) // 300) * 300))}
    return spec


def _id(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def runs_for(spec: dict) -> list[dict]:
    """The SUMO runs a case needs, per base day: the base (only when the day's trips change), the
    case and its placebo. Without a road change only the first base day."""
    runs = []
    c = spec["change"]
    for day in (spec["base_days"] if c else [spec["base_day"]]):
        base_key = {"town": spec["town"], "day": day, "demand": spec["demand"]}
        if spec["demand"]:
            runs.append({"run_id": "base_" + _id(base_key), "role": "base", "day": day,
                         "demand": spec["demand"], "incidents": None, "end_s": 86400.0})
        if c:
            runs.append({"run_id": "case_" + _id({**base_key, "inc": c["incident"], "end": c["end_run_s"]}),
                         "role": "case", "day": day, "demand": spec["demand"], "incidents": [c["incident"]],
                         "end_s": c["end_run_s"]})
            runs.append({"run_id": "placebo_" + _id({**base_key, "start": c["start_s"], "end": c["end_run_s"]}),
                         "role": "placebo", "day": day, "demand": spec["demand"], "incidents": "placebo",
                         "start_s": c["start_s"], "end_s": c["end_run_s"]})
    return runs


def run_ids(spec: dict, role: str) -> dict[str, str]:
    """{base day: run id} of one role."""
    return {r["day"]: r["run_id"] for r in runs_for(spec) if r["role"] == role}


def base_run_id(spec: dict, day: str | None = None) -> str | None:
    return run_ids(spec, "base").get(day or spec["base_day"])


def placebo_trip(out_dir: Path, day: str, start_s: float, avoid: set[str], sim: dict) -> list[str]:
    """[from, to] of a trip of that day departing just after start_s, away from the changed road."""
    import re

    trips = out_dir / "sumo_days" / f"trips_{day.replace('-', '')}.rou.xml"
    if not trips.exists():                     # recording keeps no work files: the same trips again, by the day's seed
        from .network import load_cameras
        from .simulate import plan_day

        plan_day(date.fromisoformat(day), str(out_dir / "network" / "net.net.xml"),
                 load_cameras(out_dir / "cameras.json"), sim, str(trips.parent))
    text = trips.read_text(encoding="utf-8")
    best = None
    for m in re.finditer(r'depart="([0-9.]+)" from="([^"]+)" to="([^"]+)"', text):
        dep, a, b = float(m.group(1)), m.group(2), m.group(3)
        if dep >= start_s and a not in avoid and b not in avoid:
            return [a, b]
        best = best or [a, b]
    return best


# ---------------------------------------------------------------- running
class Store:
    def __init__(self, out_dir: Path):
        self.dir = out_dir / "predict"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = self.dir / "predict.db"
        with self.conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, role TEXT, day TEXT, spec TEXT, "
                      "status TEXT, started TEXT, seconds REAL)")
            c.execute(f"CREATE TABLE IF NOT EXISTS run_truth (run_id TEXT, {', '.join(TRUTH_COLUMNS)})")
            c.execute("CREATE INDEX IF NOT EXISTS run_truth_id ON run_truth(run_id, ts)")

    def conn(self):
        return sqlite3.connect(self.db, timeout=30)

    def status(self, run_ids: list[str]) -> dict[str, str]:
        with self.conn() as c:
            q = pd.read_sql(f"SELECT run_id, status FROM runs WHERE run_id IN ({','.join('?' * len(run_ids))})",
                            c, params=run_ids) if run_ids else pd.DataFrame(columns=["run_id", "status"])
        return dict(zip(q.run_id, q.status))

    def truth(self, run_id: str) -> pd.DataFrame:
        with self.conn() as c:
            return pd.read_sql("SELECT * FROM run_truth WHERE run_id = ?", c, params=(run_id,))


def _one(args):
    r, net_file, cameras, sim, work = args
    t0 = time.time()
    _, truth = simulate_day(date.fromisoformat(r["day"]), net_file, cameras, sim, work, incidents=r["incidents"],
                            end_s=r["end_s"], demand=r["demand"])
    return r["run_id"], truth, time.time() - t0


def run_case(spec: dict, sim: dict, out_dir: Path, workers: int = 3, log=print) -> list[str]:
    """Simulate the case's runs not stored yet (in parallel); returns their ids."""
    from .network import load_cameras

    store = Store(out_dir)
    runs = runs_for(spec)
    done = {k for k, v in store.status([r["run_id"] for r in runs]).items() if v == "done"}
    todo = [r for r in runs if r["run_id"] not in done]
    if not todo:
        log("all runs stored")
        return [r["run_id"] for r in runs]
    cameras = load_cameras(out_dir / "cameras.json")
    net_file = str(out_dir / "network" / "net.net.xml")
    avoid = {spec["change"]["edge"]} if spec["change"] else set()
    for r in todo:
        if r["incidents"] == "placebo":
            car = placebo_trip(out_dir, r["day"], r["start_s"], avoid, sim)
            r["incidents"] = [{"edge": None, "car": car, "start_s": r["start_s"], "end_s": r["start_s"] + 300}]
    now = datetime.now().isoformat(timespec="seconds")
    with store.conn() as c:
        for r in todo:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, 'running', ?, NULL)",
                      (r["run_id"], r["role"], r["day"], json.dumps(r), now))
            c.execute("DELETE FROM run_truth WHERE run_id = ?", (r["run_id"],))
    log(f"simulating {len(todo)} run(s): " + ", ".join(r["role"] for r in todo))
    jobs = [(r, net_file, cameras, sim, str(store.dir / "sumo_runs" / r["run_id"])) for r in todo]
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for run_id, truth, secs in pool.map(_one, jobs):
                with store.conn() as c:
                    truth.assign(run_id=run_id)[["run_id"] + TRUTH_COLUMNS].to_sql("run_truth", c, index=False,
                                                                                    if_exists="append")
                    c.execute("UPDATE runs SET status = 'done', seconds = ? WHERE run_id = ?", (round(secs, 1), run_id))
                log(f"done {run_id} ({secs:.0f} s)")
    except Exception:
        with store.conn() as c:
            c.execute(f"UPDATE runs SET status = 'failed' WHERE status = 'running' AND run_id IN "
                      f"({','.join('?' * len(todo))})", [r["run_id"] for r in todo])
        raise
    return [r["run_id"] for r in runs]


# ---------------------------------------------------------------- results
def stored_truth(db_path: Path, day: str, end_s: float = 86400) -> pd.DataFrame:
    lo = datetime.fromisoformat(day)
    with sqlite3.connect(db_path) as c:
        return pd.read_sql(f"SELECT * FROM {state_table(c)} WHERE ts > ? AND ts <= ?", c,
                           params=(lo.isoformat(), (lo + timedelta(seconds=end_s)).isoformat()))


def base_truth(spec: dict, db_path: Path, out_dir: Path, day: str | None = None) -> pd.DataFrame:
    rid = base_run_id(spec, day)
    return Store(out_dir).truth(rid) if rid else stored_truth(db_path, day or spec["base_day"])


def _junction_delay(t: pd.DataFrame, edges: pd.DataFrame, day: str) -> pd.DataFrame:
    """Delay (veh-min) and vehicles on the roads into each junction, per 15-minute slot of the day."""
    j = t.merge(edges[["edge_id", "to_node"]], on="edge_id")
    j["slot"] = (pd.to_datetime(j.ts) - pd.Timestamp(day) - pd.Timedelta(minutes=5)).dt.floor(SLOT)
    j["delay"] = j.time_loss_s / 60
    return j.groupby(["to_node", "slot"])[["delay", "flow"]].sum()


def impact(spec: dict, db_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per junction, per edge, per junction and slot) effect of the road change against the base day."""
    with sqlite3.connect(db_path) as c:
        edges = pd.read_sql("SELECT * FROM edges", c)
        nodes = pd.read_sql("SELECT * FROM nodes", c)
    ch = spec["change"]
    store = Store(out_dir)
    cases, placebos = run_ids(spec, "case"), run_ids(spec, "placebo")
    dcs, dps, per_day = [], [], []
    for day in cases:
        base = base_truth(spec, db_path, out_dir, day)
        base = base[pd.to_datetime(base.ts) <= pd.Timestamp(day) + pd.Timedelta(seconds=ch["end_run_s"])]
        case, plac = store.truth(cases[day]), store.truth(placebos[day])
        jb, jc, jp = (_junction_delay(t, edges, day) for t in (base, case, plac))
        dcs.append((jc - jb).fillna(0))
        dps.append((jp - jb).fillna(0))
        per_day.append(_edge_change(base, case, plac, pd.Timestamp(day) + pd.Timedelta(seconds=ch["start_s"])))
    n = len(dcs)
    dc, dp = sum(dcs) / n, sum(dps) / n                          # mean over the base days
    start = pd.Timedelta(seconds=ch["start_s"]).floor(SLOT)
    rows, slots = [], []
    for j in nodes.node_id[nodes.kind == "junction"]:
        if j not in dc.index.get_level_values(0):
            continue
        s, p = dc.loc[j], dp.loc[j] if j in dp.index.get_level_values(0) else dc.loc[j] * 0
        after = s.index >= start
        lim = MARGIN * float(p.delay[p.index >= start].abs().max() or 0)
        hit = s.index[after & (s.delay.abs() > lim) & (s.delay.abs() >= 1.0)]       # at least 1 veh-min
        slots.append(s[after].assign(junction=j, limit=lim).reset_index())
        row = {"junction": j, "affected": int(len(hit) > 0), "limit_veh_min": round(lim, 1), "days": n}
        if len(hit):
            sl = s.loc[hit]
            total = sl.delay.sum()
            agree = sum(np.sign(float(d.loc[j].delay.reindex(hit).fillna(0).sum())) == np.sign(total)
                        for d in dcs if j in d.index.get_level_values(0))
            row.update(from_ts=_hhmm(hit.min()), to_ts=_hhmm(hit.max() + pd.Timedelta(SLOT)), minutes=15 * len(hit),
                       extra_delay_veh_h=round(float(total) / 60, 2),
                       extra_vehicles=round(float(sl.flow.sum()), 1),
                       peak_ts=_hhmm(sl.delay.abs().idxmax()), days_agree=int(agree))
        rows.append(row)
    junction = pd.DataFrame(rows)
    for col in ("from_ts", "to_ts", "minutes", "extra_delay_veh_h", "extra_vehicles", "peak_ts", "days_agree"):
        if col not in junction:
            junction[col] = None
    days = pd.concat(per_day, keys=range(n), names=["d"])
    e = days.groupby("edge_id").mean()
    same = days.groupby("edge_id").delay.apply(lambda x: bool((np.sign(x) == np.sign(x.mean())).all()))
    e["affected"] = ((((e.delay.abs() > MARGIN * e.delay_placebo.abs()) & (e.delay.abs() >= 5)) & same)
                     | ((e.vehicles.abs() > MARGIN * e.vehicles_placebo.abs()) & (e.vehicles.abs() >= 10))).astype(int)
    e = e.reset_index().rename(columns={"delay": "extra_delay_veh_min", "vehicles": "extra_vehicles"})
    return junction, e.sort_values("extra_delay_veh_min", ascending=False), pd.concat(slots, ignore_index=True)


def _edge_change(base: pd.DataFrame, case: pd.DataFrame, plac: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Per edge, from `after` on: extra delay (veh-min) and vehicles of the case and of the placebo."""
    b = base[pd.to_datetime(base.ts) > after][["edge_id", "ts", "time_loss_s", "flow"]]
    per = []
    for suffix, t in (("", case), ("_placebo", plac)):
        m = t.merge(b, on=["edge_id", "ts"], suffixes=("", "_b"))
        m["delay"], m["vehicles"] = (m.time_loss_s - m.time_loss_s_b) / 60, m.flow - m.flow_b
        per.append(m.groupby("edge_id")[["delay", "vehicles"]].sum().add_suffix(suffix))
    return per[0].join(per[1], how="left").fillna(0)


def _hhmm(td: pd.Timedelta) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def usual_state(db_path: Path, d: date, hhmm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every edge at this time of day on the studied days of the same type (weekday / weekend):
    (median per edge, all days' rows)."""
    tod = f"{hhmm}:00" if hhmm != "24:00" else "00:00:00"
    with sqlite3.connect(db_path) as c:
        t = pd.read_sql(f"SELECT * FROM {state_table(c)} WHERE substr(ts, 12) = ?", c, params=(tod,))
    t["day"] = (pd.to_datetime(t.ts) - pd.Timedelta(minutes=5)).dt.date
    weekend = d.weekday() >= 5 or day_kind(d) == "holiday"
    t = t[np.array([(x.weekday() >= 5) == weekend for x in t.day], dtype=bool)]   # a mask, even when empty
    num = ["total", "pcu", "load_ratio", "flow", "speed", "time_loss_s", "wait_s", "veh_h"]
    t = t.assign(**{c: pd.to_numeric(t[c], errors="coerce") for c in num})   # NULLs from cameras: numbers
    med = t.groupby("edge_id")[num].median().reset_index()
    return med, t


def state_from(truth: pd.DataFrame, ts: str) -> pd.DataFrame:
    return truth[truth.ts == ts].copy()


def run_ts(spec: dict, hhmm: str) -> str:
    """The slot time inside a run (runs replay the base day's date)."""
    h, m = map(int, hhmm.split(":"))
    return (pd.Timestamp(spec["base_day"]) + pd.Timedelta(hours=h, minutes=m)).isoformat()


def precompute(town: str, sim: dict, out_dir: Path, workers: int = 3, log=print):
    """Base runs of every holiday and festival in the calendar, so they open without waiting."""
    db = out_dir / "traffic.db"
    specs = [case_spec(town, d, "12:00", None, db) for d in calendar().date if day_kind(d) in ("holiday", "festival")]
    seen = set()
    for s in specs:
        rid = base_run_id(s)
        if rid and rid not in seen:
            seen.add(rid)
            log(f"{s['date']} {s['kind']} {s['event']}: {rid}")
            run_case(s, sim, out_dir, workers, log)


def holiday_vs_usual(spec: dict, db_path: Path, out_dir: Path) -> pd.DataFrame:
    """For a holiday / festival: each junction's delay per slot on that day against the studied days
    of the same type (the base day's type): above the usual range (90th percentile) = busier."""
    with sqlite3.connect(db_path) as c:
        edges = pd.read_sql("SELECT * FROM edges", c)
    t = base_truth(spec, db_path, out_dir)
    jb = _junction_delay(t, edges, spec["base_day"]).reset_index()
    with sqlite3.connect(db_path) as c:
        all_t = pd.read_sql(f"SELECT edge_id, ts, time_loss_s, flow FROM {state_table(c)}", c)
    all_t["day"] = (pd.to_datetime(all_t.ts) - pd.Timedelta(minutes=5)).dt.date
    bd = date.fromisoformat(spec["base_day"])
    same = all_t[np.array([(x.weekday() >= 5) == (bd.weekday() >= 5) for x in all_t.day], dtype=bool)]
    rows = []
    for dday, g in same.groupby("day"):
        rows.append(_junction_delay(g, edges, dday.isoformat()).reset_index().assign(day=dday))
    u = pd.concat(rows).groupby(["to_node", "slot"]).delay.quantile([0.5, 0.9]).unstack()
    u.columns = ["usual", "p90"]
    out = jb.merge(u.reset_index(), on=["to_node", "slot"], how="left")
    out["above"] = out.delay > np.maximum(out.p90 * 1.15, out.p90 + 5)
    return out.rename(columns={"to_node": "junction"})
