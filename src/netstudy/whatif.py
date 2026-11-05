"""Phase 4: what-if scenarios (docs/phase4.md).

A scenario (sim config `whatif.scenarios`) is replayed on `whatif.days` weekdays without a random
incident and compared with the stored days (replays are identical, phase 2). A placebo run per day
(one extra car through town from when the change starts) gives the chance level.

  closure  {kind: closure, edge, from, to}   both lanes blocked by stopped vehicles
  demand   {kind: demand, zone, factor}      the zone's peak_veh_per_hour x factor, all day

Impact per junction: the extra delay on all roads into it, per 15 minutes, averaged over the days.
A junction is affected in a slot when that is more than MARGIN times the largest its placebo runs
reach from the change's start on. An edge is affected when its extra delay or extra vehicles over the
change (closures: to 2 hours after it) are beyond MARGIN times its placebo change and the days agree
(mean at least MIN_T standard errors from zero).

  whatif.db  run_truth (edge_truth rows + run_id), runs (run_id, scenario, kind, day, start)
  traffic.db scenario_impact (per junction), scenario_edges (per edge)
"""
from __future__ import annotations

import copy
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import load_edges
from .propagation import clean_weekdays, placebo_car
from .simulate import TRUTH_COLUMNS, simulate_day

MARGIN = 2.0
MIN_T = 3.0                # per edge: the days' mean change at least 3 standard errors from zero
SLOT = "15min"
EDGE_AFTER_MIN = 120       # closure: per-edge totals from its start to 2 hours after its end
IMPACT_COLUMNS = ["scenario", "junction", "affected", "from_ts", "to_ts", "minutes", "extra_delay_veh_h",
                  "extra_vehicles", "days_agree", "days", "limit_veh_min"]


def _seconds(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    return 3600 * h + 60 * m


def scenario_runs(sim: dict, name: str, edges: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """One row per run: the scenario and its placebo on each day."""
    cfg = sim["whatif"]
    sc = cfg["scenarios"][name]
    days = clean_weekdays(db_path)[:int(cfg.get("days", 5))]
    if sc["kind"] == "closure":
        start_s, end_s = _seconds(sc["from"]), _seconds(sc["to"])
        car = placebo_car(edges, sc["edge"])
    elif sc["kind"] == "demand":
        start_s, end_s = 300.0, 86400.0
        car = placebo_car(edges, avoid={sc["zone"]})
    else:
        raise ValueError(f"unknown scenario kind {sc['kind']!r}")
    rows = []
    for d in days:
        rows.append({"run_id": f"{name}@{d}", "scenario": name, "kind": "scenario", "day": d.isoformat(),
                     "start_s": start_s, "end_s": end_s})
        rows.append({"run_id": f"placebo@{d}@{int(start_s)}", "scenario": name, "kind": "placebo",
                     "day": d.isoformat(), "start_s": start_s, "end_s": start_s + 300, "car": " ".join(car)})
    return pd.DataFrame(rows)


def _run_args(sim: dict, sc: dict, r) -> tuple[dict, list[dict] | None]:
    if r.kind == "placebo":
        return sim, [{"edge": None, "car": r.car.split(), "start_s": r.start_s, "end_s": r.end_s}]
    if sc["kind"] == "closure":
        return sim, [{"edge": sc["edge"], "block": 2, "start_s": r.start_s, "end_s": r.end_s}]
    s = copy.deepcopy(sim)                                 # demand: scale the zone, keep random incidents off
    zone = s["demand"]["zones"][sc["zone"]]
    zone["peak_veh_per_hour"] = float(zone["peak_veh_per_hour"]) * float(sc["factor"])
    return s, []


def run_scenario(sim: dict, name: str, net_file: Path, cameras: list[dict], db_path: Path, out_dir: Path,
                 workers: int = 4) -> pd.DataFrame:
    """Simulate the scenario's runs (placebos already stored are reused) into out_dir/whatif.db."""
    with sqlite3.connect(db_path) as conn:
        edges = load_edges(conn)
    runs = scenario_runs(sim, name, edges, db_path)
    sc = sim["whatif"]["scenarios"][name]
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_dir / "whatif.db")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'run_truth'").fetchone():
        conn.execute("DELETE FROM run_truth WHERE run_id IN (SELECT run_id FROM runs WHERE scenario = ? AND kind = 'scenario')",
                     (name,))
        done = set(pd.read_sql("SELECT DISTINCT run_id FROM run_truth", conn).run_id)
    else:
        done = set()
    old = pd.read_sql("SELECT * FROM runs", conn) if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'runs'").fetchone() else pd.DataFrame()
    pd.concat([old[old.scenario != name] if len(old) else old, runs]).to_sql("runs", conn, index=False, if_exists="replace")
    todo = runs[~runs.run_id.isin(done)]
    print(f"{name}: {len(todo)} runs to simulate ({len(runs) - len(todo)} placebos reused)")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for r in todo.itertuples():
                s, spec = _run_args(sim, sc, r)
                futures.append(pool.submit(simulate_day, datetime.fromisoformat(r.day).date(), str(net_file), cameras,
                                           s, str(out_dir / "sumo_runs"), spec,
                                           r.run_id.replace(":", "").replace("@", "_")))
            for r, fut in zip(todo.itertuples(), futures):
                _, truth = fut.result()
                truth.assign(run_id=r.run_id)[["run_id"] + TRUTH_COLUMNS] \
                    .to_sql("run_truth", conn, index=False, if_exists="append")
                conn.commit()
                print(f"  {r.run_id}")
    finally:
        conn.close()
    return runs


def _diff(conn_base, conn_run, run_id: str, day: str) -> pd.DataFrame:
    """Run minus base per 5-minute slot and edge: extra delay (veh-min) and extra vehicles."""
    lo = datetime.fromisoformat(day)
    q = "SELECT edge_id, ts, time_loss_s, flow FROM {} WHERE {}"
    base = pd.read_sql(q.format("edge_truth", "ts > ? AND ts <= ?"), conn_base,
                       params=(lo.isoformat(), (lo + timedelta(days=1)).isoformat()))
    run = pd.read_sql(q.format("run_truth", "run_id = ?"), conn_run, params=(run_id,))
    m = run.merge(base, on=["edge_id", "ts"], suffixes=("", "_base"))
    m["delay"] = (m.time_loss_s - m.time_loss_s_base) / 60
    m["vehicles"] = m.flow - m.flow_base
    m["tod"] = pd.to_datetime(m.ts) - pd.Timestamp(lo)            # time of day, so days line up
    return m[["edge_id", "tod", "delay", "vehicles"]]


def impact(name: str, db_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per junction, per edge) impact of a simulated scenario; written to traffic.db."""
    with sqlite3.connect(db_path) as cb, sqlite3.connect(out_dir / "whatif.db") as cr:
        edges = load_edges(cb)
        nodes = pd.read_sql("SELECT node_id, kind FROM nodes", cb)
        runs = pd.read_sql("SELECT * FROM runs WHERE scenario = ?", cr, params=(name,))
        d = pd.concat([_diff(cb, cr, r.run_id, r.day).assign(kind=r.kind, day=r.day) for r in runs.itertuples()])
    start = pd.Timedelta(seconds=float(runs.start_s.iloc[0]))
    end = pd.Timedelta(seconds=float(runs.end_s[runs.kind == "scenario"].iloc[0]))
    n_days = runs.day.nunique()
    d["slot"] = (d.tod - pd.Timedelta(minutes=5)).dt.floor(SLOT)        # slot ending 08:05 belongs to 08:00
    d = d.merge(edges[["edge_id", "to_node"]], on="edge_id")
    junctions = nodes.node_id[nodes.kind == "junction"].tolist()
    per_day = d[d.to_node.isin(junctions)].groupby(["kind", "day", "to_node", "slot"])[["delay", "vehicles"]].sum()
    mean = per_day.groupby(["kind", "to_node", "slot"]).sum() / n_days          # mean over days (missing = 0)
    rows = []
    for j in junctions:
        s = mean.loc[("scenario", j)] if ("scenario", j) in mean.index.droplevel(2) else None
        p = mean.loc[("placebo", j)] if ("placebo", j) in mean.index.droplevel(2) else None
        if s is None:
            continue
        lim = MARGIN * (p.delay[p.index >= start.floor(SLOT)].abs().max() if p is not None else 0.0)
        hit = s.index[(s.delay.abs() > lim) & (s.index >= start.floor(SLOT))]
        row = {"scenario": name, "junction": j, "limit_veh_min": round(float(lim), 1), "affected": int(len(hit) > 0)}
        if len(hit):
            sl = s.loc[hit]
            days = per_day.loc["scenario"].xs(j, level="to_node")
            days = days[days.index.get_level_values("slot").isin(hit)].groupby(level="day").delay.sum()
            row.update(from_ts=_hhmm(hit.min()), to_ts=_hhmm(hit.max() + pd.Timedelta(SLOT)),
                       minutes=15 * len(hit), extra_delay_veh_h=round(float(sl.delay.sum()) / 60, 2),
                       extra_vehicles=round(float(sl.vehicles.sum()), 1),
                       days_agree=int((np.sign(days) == np.sign(sl.delay.sum())).sum()), days=n_days)
        rows.append(row)
    junction = pd.DataFrame(rows).reindex(columns=IMPACT_COLUMNS)

    win = (d.tod > start) & (d.tod <= (end + pd.Timedelta(minutes=EDGE_AFTER_MIN) if end < pd.Timedelta(days=1)
                                         else pd.Timedelta(days=1)))
    by_day = d[win].groupby(["kind", "edge_id", "day"])[["delay", "vehicles"]].sum()
    e = by_day.groupby(["kind", "edge_id"]).sum() / n_days
    sd = by_day.loc["scenario"].groupby("edge_id").std()
    t = e.loc["scenario"] / (sd / np.sqrt(n_days)).replace(0, np.nan)        # days agree on the change
    per_edge = e.loc["scenario"].join(e.loc["placebo"].abs().add_suffix("_placebo"), how="left").fillna(0)
    beyond = lambda c: (per_edge[c].abs() > MARGIN * per_edge[f"{c}_placebo"]) & (t[c].abs() >= MIN_T)  # noqa: E731
    per_edge["affected"] = (beyond("delay") | beyond("vehicles")).astype(int)
    per_edge = per_edge.reset_index().assign(scenario=name).rename(
        columns={"delay": "extra_delay_veh_min", "vehicles": "extra_vehicles"})
    per_edge = per_edge[["scenario", "edge_id", "extra_delay_veh_min", "extra_vehicles", "affected",
                         "delay_placebo", "vehicles_placebo"]].sort_values("extra_delay_veh_min", ascending=False)
    with sqlite3.connect(db_path) as conn:
        for table, df in (("scenario_impact", junction), ("scenario_edges", per_edge)):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                conn.execute(f"DELETE FROM {table} WHERE scenario = ?", (name,))
            df.to_sql(table, conn, index=False, if_exists="append")
    return junction, per_edge


def _hhmm(td: pd.Timedelta) -> str:
    minutes = int(td.total_seconds() // 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
